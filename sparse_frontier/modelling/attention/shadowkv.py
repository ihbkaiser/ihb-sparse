"""Accuracy-first ShadowKV attention with optional fused CUDA primitives.

This module mirrors ByteDance-Seed/ShadowKV's ``ShadowKVCache`` algorithm.
It keeps the normal vLLM KV cache dense and does not use the official
CPU-offload cache layout.  Optional fused CUDA primitives cover the
reconstruction and landmark operations without changing the algorithm or
claiming the upstream CPU-offload memory savings.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Callable, Optional

import torch
import torch.nn.functional as F

if os.getenv("SF_SHADOWKV_CUDA", "auto").lower() in {"0", "off", "false"}:
    _shadowkv_cuda = None
else:
    try:
        # Optional CUDA extension containing the fused primitives modeled after
        # ByteDance-Seed/ShadowKV's batch_gemm_softmax and batch_gather_gemm
        # kernels.  The package remains usable without a CUDA rebuild.
        from sparse_frontier import _shadowkv_cuda
    except (ImportError, OSError):
        _shadowkv_cuda = None

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    tensor_model_parallel_all_gather,
)
from vllm.vllm_flash_attn.flash_attn_interface import (
    flash_attn_varlen_func,
)

from .abstract_attention import AbstractAttention, AttentionUtils


RoPEForward = Callable[..., tuple[torch.Tensor, Optional[torch.Tensor]]]


class ShadowKVAttention(AbstractAttention):
    """ShadowKV's accuracy path for single-request vLLM inference.

    The state layout follows ``ShadowKVCache`` for batch size one:

    * ``U`` is shared across local KV heads and comes from an SVD over all
      tensor-parallel KV heads in their global concatenated representation.
    * ``SV`` retains only this worker's KV-head slices.
    * landmarks and outlier indices are computed from post-RoPE keys.
    * values are always gathered from the existing dense vLLM cache.
    """

    # ShadowKVCache's retrieval implementation is written for the Llama head
    # size and uses math.sqrt(128) rather than deriving the scale dynamically.
    RETRIEVAL_SCALE_DIM = 128

    def __init__(
        self,
        sparse_budget: int = 2048,
        chunk_size: int = 8,
        rank: int = 160,
        svd_backend: str = "randomized",
        svd_oversample: int = 32,
        svd_niter: int = 2,
        fused_retrieval: bool = False,
        local_chunk: int = 4,
        outlier_chunk: int = 48,
        num_layers: int = 1,
        num_q_heads: int = 1,
        num_kv_heads: int = 1,
        tp_size: int = 1,
        block_size: int = 16,
        tp_rank: Optional[int] = None,
    ) -> None:
        super().__init__()

        integer_args = {
            "sparse_budget": sparse_budget,
            "chunk_size": chunk_size,
            "rank": rank,
            "local_chunk": local_chunk,
            "outlier_chunk": outlier_chunk,
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "tp_size": tp_size,
            "block_size": block_size,
        }
        for name, value in integer_args.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"ShadowKV {name} must be a positive integer, got {value!r}")
        if svd_backend not in {"exact", "gesvdj", "randomized"}:
            raise ValueError(
                "ShadowKV svd_backend must be 'exact', 'gesvdj', or 'randomized', "
                f"got {svd_backend!r}"
            )
        if (
            not isinstance(svd_oversample, int)
            or isinstance(svd_oversample, bool)
            or svd_oversample < 0
        ):
            raise ValueError(
                "ShadowKV svd_oversample must be a non-negative integer, "
                f"got {svd_oversample!r}"
            )
        if not isinstance(svd_niter, int) or isinstance(svd_niter, bool) or svd_niter < 0:
            raise ValueError(
                "ShadowKV svd_niter must be a non-negative integer, "
                f"got {svd_niter!r}"
            )
        if not isinstance(fused_retrieval, bool):
            raise ValueError(
                "ShadowKV fused_retrieval must be a boolean, "
                f"got {fused_retrieval!r}"
            )

        if sparse_budget % chunk_size != 0:
            raise ValueError(
                "ShadowKV sparse_budget must be divisible by chunk_size: "
                f"{sparse_budget} % {chunk_size} != 0"
            )
        if num_q_heads % tp_size != 0 or num_kv_heads % tp_size != 0:
            raise ValueError(
                "ShadowKV requires query and KV head counts divisible by TP size; "
                f"got q={num_q_heads}, kv={num_kv_heads}, tp={tp_size}"
            )
        if num_q_heads % num_kv_heads != 0:
            raise ValueError(
                "ShadowKV requires num_q_heads divisible by num_kv_heads; "
                f"got q={num_q_heads}, kv={num_kv_heads}"
            )
        self.sparse_budget = sparse_budget
        self.chunk_size = chunk_size
        self.rank = rank
        self.svd_backend = svd_backend
        self.svd_oversample = svd_oversample
        self.svd_niter = svd_niter
        self.fused_retrieval = fused_retrieval
        self.local_chunk = local_chunk
        self.outlier_chunk = outlier_chunk
        self.num_layers = num_layers
        self.num_attention_heads = num_q_heads
        self.num_key_value_heads = num_kv_heads
        self.tp_size = tp_size
        self.block_size = block_size
        self.num_key_value_groups = num_q_heads // num_kv_heads
        self.local_q_heads = num_q_heads // tp_size
        self.local_kv_heads = num_kv_heads // tp_size
        self.select_sets = sparse_budget // chunk_size
        if tp_rank is not None:
            self.tp_rank = tp_rank
        elif tp_size == 1:
            self.tp_rank = 0
        else:
            try:
                self.tp_rank = get_tensor_model_parallel_rank()
            except Exception as exc:
                raise RuntimeError(
                    "ShadowKV needs an initialized vLLM tensor-parallel group "
                    "to determine the local KV-head slice"
                ) from exc
        if not 0 <= self.tp_rank < tp_size:
            raise ValueError(f"ShadowKV TP rank must be in [0, {tp_size}), got {self.tp_rank}")

        self.U: Optional[torch.Tensor] = None
        self.SV: Optional[torch.Tensor] = None
        self.k_landmark: Optional[torch.Tensor] = None
        self.k_landmark_idx: Optional[torch.Tensor] = None
        self.outlier_chunk_idx: Optional[torch.Tensor] = None
        self.selected_chunk_idx: Optional[torch.Tensor] = None
        self._rope_positions: list[Optional[torch.Tensor]] = [None] * num_layers
        self._rope_forward: list[Optional[RoPEForward]] = [None] * num_layers
        self._rope_kernel_metadata: list[Optional[dict[str, object]]] = [None] * num_layers
        self._kernel_warning_emitted = False
        self.initialized = False
        self.prompt_len = 0
        self.prefill_local = 0
        self.chunks = 0
        # Decode uses one two-entry cumulative-length tensor per distinct
        # assembled length.  Reusing these tiny device tensors avoids an
        # allocation in every layer while keeping the values immutable while
        # FlashAttention is running.
        self._flash_cu_q: dict[torch.device, torch.Tensor] = {}
        self._flash_cu_k: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._assembly_key_scratch: Optional[torch.Tensor] = None
        self._assembly_value_scratch: Optional[torch.Tensor] = None
        self._assembly_capacity = 0

    @property
    def minimum_prompt_len(self) -> int:
        """Smallest context for which the official top-k layout is defined."""
        return (
            self.local_chunk + self.outlier_chunk + self.select_sets
        ) * self.chunk_size

    def reset(self) -> None:
        """Clear all per-request and per-layer ShadowKV state."""
        self.U = None
        self.SV = None
        self.k_landmark = None
        self.k_landmark_idx = None
        self.outlier_chunk_idx = None
        self.selected_chunk_idx = None
        self._rope_positions = [None] * self.num_layers
        self._rope_forward = [None] * self.num_layers
        self._rope_kernel_metadata = [None] * self.num_layers
        self._kernel_warning_emitted = False
        self.initialized = False
        self.prompt_len = 0
        self.prefill_local = 0
        self.chunks = 0
        self._flash_cu_q.clear()
        self._flash_cu_k.clear()
        self._assembly_key_scratch = None
        self._assembly_value_scratch = None
        self._assembly_capacity = 0

    def _validate_layer(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise ValueError(
                f"ShadowKV layer_idx must be in [0, {self.num_layers}), got {layer_idx}"
            )

    def _validate_prefill_shapes(
        self,
        pre_rope_keys: torch.Tensor,
        post_rope_keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[int, int, int, torch.Tensor]:
        if pre_rope_keys.ndim != 3 or post_rope_keys.ndim != 3 or values.ndim != 3:
            raise ValueError(
                "ShadowKV expects pre-RoPE K, post-RoPE K, and V as "
                "[tokens, local_kv_heads, head_dim]"
            )
        if pre_rope_keys.shape != post_rope_keys.shape or pre_rope_keys.shape != values.shape:
            raise ValueError(
                "ShadowKV prefill K/V shapes must match; got "
                f"pre={tuple(pre_rope_keys.shape)}, post={tuple(post_rope_keys.shape)}, "
                f"v={tuple(values.shape)}"
            )
        tokens, local_kv_heads, head_dim = pre_rope_keys.shape
        if local_kv_heads != self.local_kv_heads:
            raise ValueError(
                "ShadowKV local KV-head count does not match the configured TP layout; "
                f"got tensor={local_kv_heads}, expected={self.local_kv_heads}"
            )
        positions = positions.flatten()
        if positions.numel() != tokens:
            raise ValueError(
                "ShadowKV position count must equal prefill token count; "
                f"got positions={positions.numel()}, tokens={tokens}"
            )
        if tokens < self.minimum_prompt_len:
            raise ValueError(
                "ShadowKV prompt is too short for the official layout: got "
                f"{tokens} tokens, minimum is {self.minimum_prompt_len} for "
                f"budget={self.sparse_budget}, chunk={self.chunk_size}, "
                f"local_chunk={self.local_chunk}, outlier_chunk={self.outlier_chunk}"
            )
        global_width = self.num_key_value_heads * head_dim
        if self.rank > min(tokens, global_width):
            raise ValueError(
                "ShadowKV rank exceeds the available SVD dimension: "
                f"rank={self.rank}, min(tokens, global_kv*head_dim)={min(tokens, global_width)}"
            )
        if not torch.is_floating_point(pre_rope_keys):
            raise ValueError("ShadowKV requires floating-point K/V tensors")
        return tokens, local_kv_heads, head_dim, positions

    def _gather_global_pre_rope_keys(self, pre_rope_keys: torch.Tensor) -> torch.Tensor:
        """All-gather KV heads for the official global SVD."""
        if self.tp_size == 1:
            return pre_rope_keys.contiguous()
        gathered = tensor_model_parallel_all_gather(pre_rope_keys.contiguous(), dim=1)
        if gathered.ndim != 3 or gathered.shape[1] != self.num_key_value_heads:
            raise RuntimeError(
                "ShadowKV global pre-RoPE KV all-gather returned an unexpected shape; "
                f"got {tuple(gathered.shape)}, expected [tokens, {self.num_key_value_heads}, head_dim]"
            )
        return gathered

    def _compute_svd(
        self,
        matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the official or truncated SVD used by ShadowKV.

        ``exact`` intentionally calls the same ``torch.svd`` operation as the
        official cache.  ``gesvdj`` computes the same reduced full SVD through
        cuSOLVER's Jacobi driver and converts its V^H output to the V
        convention used below.  ``randomized`` uses a rank-plus-oversampling
        randomized subspace iteration; it returns the same factors needed by
        the rest of the algorithm, but is an explicit approximation.
        """
        if self.svd_backend == "exact":
            return torch.svd(matrix)
        if self.svd_backend == "gesvdj":
            try:
                u, s, vh = torch.linalg.svd(matrix, full_matrices=False, driver="gesvdj")
            except RuntimeError as exc:
                warnings.warn(
                    "ShadowKV gesvdj failed; falling back to the official torch.svd "
                    f"backend ({exc})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return torch.svd(matrix)
            return u, s, vh.transpose(-2, -1)

        min_dim = min(matrix.shape[-2:])
        q = min(self.rank + self.svd_oversample, min_dim)
        if q == min_dim:
            # Avoid making the randomized mode less accurate when the caller
            # has requested the complete available subspace.
            return torch.svd(matrix)
        return torch.svd_lowrank(matrix, q=q, niter=self.svd_niter)

    def prefill_state(
        self,
        pre_rope_keys: torch.Tensor,
        post_rope_keys: torch.Tensor,
        values: torch.Tensor,
        positions: torch.Tensor,
        layer_idx: int,
        rope_forward: Optional[RoPEForward],
        rope_metadata: Optional[dict[str, object]] = None,
    ) -> None:
        """Build one layer of official ShadowKV state during dense prefill."""
        self._validate_layer(layer_idx)
        tokens, local_kv_heads, head_dim, positions = self._validate_prefill_shapes(
            pre_rope_keys, post_rope_keys, values, positions
        )
        if rope_forward is None:
            raise RuntimeError(
                "ShadowKV requires the original scalar RoPE dispatch method to "
                "reconstruct selected keys"
            )

        if layer_idx == 0:
            self.prompt_len = tokens
            self.chunks = tokens // self.chunk_size - self.local_chunk
            self.prefill_local = tokens - self.chunks * self.chunk_size
            if self.chunks < self.outlier_chunk:
                raise ValueError(
                    "ShadowKV prompt does not contain enough complete chunks for "
                    f"outlier_chunk={self.outlier_chunk}; got {self.chunks}"
                )
            num_landmarks = self.chunks - self.outlier_chunk
            self.U = torch.empty(
                self.num_layers,
                1,
                tokens,
                self.rank,
                device=pre_rope_keys.device,
                dtype=pre_rope_keys.dtype,
            )
            self.SV = torch.empty(
                self.num_layers,
                1,
                self.local_kv_heads,
                self.rank,
                head_dim,
                device=pre_rope_keys.device,
                dtype=pre_rope_keys.dtype,
            )
            self.k_landmark = torch.empty(
                self.num_layers,
                1,
                self.local_kv_heads,
                num_landmarks,
                head_dim,
                device=post_rope_keys.device,
                dtype=post_rope_keys.dtype,
            )
            self.k_landmark_idx = torch.empty(
                self.num_layers,
                1,
                self.local_kv_heads,
                num_landmarks,
                device=post_rope_keys.device,
                dtype=torch.long,
            )
            self.outlier_chunk_idx = torch.empty(
                self.num_layers,
                1,
                self.local_kv_heads,
                self.outlier_chunk,
                device=post_rope_keys.device,
                dtype=torch.long,
            )
            self.selected_chunk_idx = torch.zeros(
                self.num_layers,
                1,
                self.local_kv_heads,
                self.select_sets,
                device=post_rope_keys.device,
                dtype=torch.long,
            )
        elif self.U is None or self.SV is None or self.k_landmark is None:
            raise RuntimeError(
                "ShadowKV received a nonzero layer before layer 0 initialized its state"
            )
        elif tokens != self.prompt_len or head_dim != self.SV.shape[-1]:
            raise ValueError(
                "ShadowKV requires the same prompt length and head dimension for every layer"
            )

        global_pre_rope = self._gather_global_pre_rope_keys(pre_rope_keys)
        # The official cache receives [batch, kv_heads, tokens, head_dim] and
        # transposes to [batch, tokens, kv_heads, head_dim] before flattening.
        # Our handler stores tokens first, so flatten directly in that order;
        # transposing [tokens, heads, dim] here would interleave the wrong
        # dimensions and corrupt reconstruction.
        matrix = global_pre_rope.reshape(1, tokens, -1).float()
        u, s, v = self._compute_svd(matrix)
        self.U[layer_idx, 0].copy_(u[0, :, :self.rank].to(self.U.dtype))

        v = v.transpose(1, 2)
        sv_global = torch.matmul(
            torch.diag_embed(s[:, : self.rank]), v[:, : self.rank]
        )[0]
        sv_global = sv_global.view(self.rank, self.num_key_value_heads, head_dim)
        sv_global = sv_global.permute(1, 0, 2)
        local_start = self.tp_rank * self.local_kv_heads
        local_end = local_start + self.local_kv_heads
        self.SV[layer_idx, 0].copy_(sv_global[local_start:local_end].to(self.SV.dtype))

        post_rope_ctx = post_rope_keys[: self.chunks * self.chunk_size]
        post_rope_ctx = post_rope_ctx.transpose(0, 1).reshape(
            local_kv_heads, self.chunks, self.chunk_size, head_dim
        )
        landmark_candidates = post_rope_ctx.mean(dim=-2)
        cosine = F.cosine_similarity(
            landmark_candidates.unsqueeze(2).expand(
                -1, -1, self.chunk_size, -1
            ),
            post_rope_ctx,
            dim=-1,
        )
        outlier_indices = cosine.min(dim=-1).values.topk(
            self.outlier_chunk, largest=False
        ).indices

        all_indices = torch.arange(self.chunks, device=post_rope_keys.device)
        all_indices = all_indices.unsqueeze(0).expand(local_kv_heads, -1)
        mask = torch.ones_like(all_indices, dtype=torch.bool)
        mask.scatter_(dim=-1, index=outlier_indices, value=False)
        landmark_indices = all_indices.masked_select(mask).view(
            local_kv_heads, self.chunks - self.outlier_chunk
        )
        landmarks = landmark_candidates.gather(
            dim=1,
            index=landmark_indices.unsqueeze(-1).expand(-1, -1, head_dim),
        )

        self.k_landmark[layer_idx, 0].copy_(landmarks.contiguous())
        self.k_landmark_idx[layer_idx, 0].copy_(landmark_indices.contiguous())
        self.outlier_chunk_idx[layer_idx, 0].copy_(outlier_indices.contiguous())
        self._rope_positions[layer_idx] = positions.detach().clone()
        self._rope_forward[layer_idx] = rope_forward
        self._rope_kernel_metadata[layer_idx] = rope_metadata
        self.initialized = True
        if layer_idx == self.num_layers - 1:
            from sparse_frontier.utils.sparsity_server import set_shadowkv_state

            set_shadowkv_state(ready=True, layers_built=self.num_layers)

    def __call__(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Exact dense prefill attention, matching the official baseline."""
        del layer_idx
        return AttentionUtils.flash_attention(queries, keys, values)

    def select_chunks(
        self,
        query: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select official landmark chunks and return their cache positions."""
        self._validate_layer(layer_idx)
        if not self.initialized or self.k_landmark is None or self.k_landmark_idx is None:
            raise RuntimeError("ShadowKV retrieval requested before prefill initialization")
        if query.ndim != 3 or query.shape[0] != 1:
            raise ValueError("ShadowKV decode query must have shape [1, num_q_heads, head_dim]")
        if query.shape[1] != self.local_q_heads:
            raise ValueError(
                f"ShadowKV decode query has {query.shape[1]} heads, expected {self.local_q_heads}"
            )

        head_dim = query.shape[-1]
        q = query.view(1, self.local_kv_heads, self.num_key_value_groups, 1, head_dim)
        landmarks = self.k_landmark[layer_idx]
        chunk_attn = self._landmark_softmax(query, landmarks)
        chunk_attn = chunk_attn.sum(dim=-2)
        if self.num_key_value_groups > 1:
            chunk_attn = chunk_attn.max(dim=-2).values
        merged_results = torch.topk(
            chunk_attn,
            k=self.select_sets,
            dim=-1,
        ).indices
        selected_chunks = self.k_landmark_idx[layer_idx].gather(
            dim=-1, index=merged_results
        )
        if self.selected_chunk_idx is not None:
            self.selected_chunk_idx[layer_idx].copy_(selected_chunks)
        positions = (
            selected_chunks.unsqueeze(-1) * self.chunk_size
            + torch.arange(
                self.chunk_size,
                device=query.device,
                dtype=selected_chunks.dtype,
            ).view(1, 1, -1)
        ).view(1, self.local_kv_heads, -1)
        return selected_chunks, positions

    def _landmark_softmax(
        self,
        query: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """Return official per-GQA-group landmark softmax scores.

        The optional CUDA path fuses the GEMM and softmax stages used by the
        upstream CPU-offload implementation.  It returns ``[batch, kv_heads,
        groups, query_length, chunks]`` so the official GQA sum/max/top-k
        order remains in Python and is numerically auditable.
        """
        if (
            self.fused_retrieval
            and _shadowkv_cuda is not None
            and hasattr(_shadowkv_cuda, "retrieval_softmax")
            and query.dtype == torch.bfloat16
            and landmarks.dtype == torch.bfloat16
        ):
            try:
                scores = _shadowkv_cuda.retrieval_softmax(
                    query.contiguous(), landmarks.contiguous()
                )
                compact_shape = (
                    query.shape[0],
                    self.local_kv_heads,
                    self.num_key_value_groups,
                    landmarks.shape[-2],
                )
                full_shape = compact_shape[:-1] + (1, compact_shape[-1])
                if scores.shape == compact_shape:
                    scores = scores.unsqueeze(-2)
                elif scores.shape != full_shape:
                    raise RuntimeError(
                        "unexpected retrieval_softmax output shape "
                        f"{tuple(scores.shape)}"
                    )
                return scores
            except RuntimeError as exc:
                self._warn_kernel_fallback("retrieval_softmax", exc)

        q = query.view(1, self.local_kv_heads, self.num_key_value_groups, 1, -1)
        scores = torch.einsum(
            "bhgqd,bhdc->bhgqc",
            q,
            landmarks.transpose(2, 3),
        ).squeeze(2) / math.sqrt(self.RETRIEVAL_SCALE_DIM)
        return F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

    def _warn_kernel_fallback(self, operation: str, exc: RuntimeError) -> None:
        if not self._kernel_warning_emitted:
            warnings.warn(
                f"ShadowKV CUDA {operation} kernel failed; using the verified "
                f"PyTorch fallback ({exc})",
                RuntimeWarning,
                stacklevel=2,
            )
            self._kernel_warning_emitted = True

    def _reconstruct_selected_keys(
        self,
        selected_positions: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Reconstruct selected pre-RoPE K and apply its original RoPE.

        When the optional extension is available, this is the fused
        ``batch_gather_gemm`` equivalent from the upstream implementation.
        Otherwise the existing scalar-RoPE path is retained exactly.
        """
        metadata = self._rope_kernel_metadata[layer_idx]
        if (
            _shadowkv_cuda is not None
            and hasattr(_shadowkv_cuda, "gather_gemm_rope")
            and metadata is not None
            and self.U is not None
            and self.SV is not None
            and self.U.dtype == torch.bfloat16
            and self.SV.dtype == torch.bfloat16
            and metadata.get("head_size") == self.SV.shape[-1]
            and metadata.get("rotary_dim") == self.SV.shape[-1]
            and metadata.get("is_neox_style") is True
        ):
            cos_sin_cache = metadata.get("cos_sin_cache")
            if isinstance(cos_sin_cache, torch.Tensor):
                if cos_sin_cache.device != self.U.device or cos_sin_cache.dtype != self.U.dtype:
                    cos_sin_cache = cos_sin_cache.to(self.U.device, dtype=self.U.dtype)
                    metadata["cos_sin_cache"] = cos_sin_cache
                try:
                    result = _shadowkv_cuda.gather_gemm_rope(
                        self.U[layer_idx, 0].contiguous(),
                        self.SV[layer_idx, 0].contiguous(),
                        selected_positions.contiguous(),
                        cos_sin_cache.contiguous(),
                    )
                    expected_shape = (
                        self.local_kv_heads,
                        selected_positions.shape[-1],
                        self.SV.shape[-1],
                    )
                    if result.shape != expected_shape:
                        raise RuntimeError(
                            "unexpected gather_gemm_rope output shape "
                            f"{tuple(result.shape)}"
                        )
                    return result
                except RuntimeError as exc:
                    self._warn_kernel_fallback("gather_gemm_rope", exc)

        if self.U is None or self.SV is None:
            raise RuntimeError("ShadowKV reconstruction requested before SVD state exists")
        u = self.U[layer_idx, 0]
        sv = self.SV[layer_idx, 0]
        u_expanded = u.unsqueeze(0).expand(self.local_kv_heads, -1, -1)
        selected_u = torch.gather(
            u_expanded,
            dim=1,
            index=selected_positions.unsqueeze(-1).expand(-1, -1, self.rank),
        )
        selected_pre_rope = torch.einsum("hnr,hrd->hnd", selected_u, sv)
        rope_positions = self._rope_positions[layer_idx][selected_positions]
        return self._apply_rope_per_head(
            selected_pre_rope,
            rope_positions,
            layer_idx,
        )

    @staticmethod
    def _copy_cache_tokens(
        cache: torch.Tensor,
        positions: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        """Gather a head-major cache directly into ``[tokens, heads, dim]``."""
        if cache.ndim != 3 or positions.ndim not in (1, 2) or destination.ndim != 3:
            raise ValueError(
                "ShadowKV cache gather expects cache [heads, tokens, dim], positions "
                "[tokens] or [heads, tokens], and token-major destination"
            )
        num_heads, _, head_dim = cache.shape
        expected_tokens = positions.shape[-1]
        if destination.shape != (expected_tokens, num_heads, head_dim):
            raise ValueError(
                "ShadowKV cache gather destination shape does not match positions/cache; "
                f"got destination={tuple(destination.shape)}, positions={tuple(positions.shape)}, "
                f"cache={tuple(cache.shape)}"
            )
        cache_token_major = cache.transpose(0, 1)
        if positions.ndim == 1:
            torch.index_select(cache_token_major, 0, positions, out=destination)
            return
        if positions.shape[0] != num_heads:
            raise ValueError(
                "ShadowKV per-head cache positions must have one row per KV head; "
                f"got positions={tuple(positions.shape)}, heads={num_heads}"
            )
        position_index = positions.transpose(0, 1).contiguous()
        torch.gather(
            cache_token_major,
            dim=0,
            index=position_index.unsqueeze(-1).expand(-1, -1, head_dim),
            out=destination,
        )

    def _apply_rope_per_head(
        self,
        pre_rope_keys: torch.Tensor,
        rope_positions: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        rope_forward = self._rope_forward[layer_idx]
        if rope_forward is None:
            raise RuntimeError(
                "ShadowKV has no captured RoPE dispatch method for layer "
                f"{layer_idx}; this model's RoPE architecture is unsupported"
            )
        if self._rope_positions[layer_idx] is None:
            raise RuntimeError(f"ShadowKV has no captured RoPE positions for layer {layer_idx}")

        # vLLM's scalar RoPE API accepts one position per token.  Flattening
        # the KV-head dimension turns the per-head position map into one
        # batched scalar dispatch, preserving the exact position-dependent
        # transform while avoiding one kernel launch per KV head.
        flat_positions = rope_positions.reshape(-1).contiguous()
        flat_keys = pre_rope_keys.reshape(-1, pre_rope_keys.shape[-1]).contiguous()
        dummy_query = torch.zeros_like(flat_keys)
        _, rotated = rope_forward(flat_positions, dummy_query, flat_keys)
        if rotated is None:
            raise RuntimeError("ShadowKV RoPE dispatch returned no rotated key")
        return rotated.view_as(pre_rope_keys)

    def _flash_attention_assembled(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Run decode attention directly on the assembled ShadowKV tokens.

        Quest can pass its selected canonical blocks directly to
        ``flash_attn_with_kvcache``.  ShadowKV reconstructs a different key
        tensor, so packing that tensor into a padded block cache would add a
        device allocation and a full copy at every layer and decode step.
        vLLM's varlen FlashAttention interface already supports GQA, and its
        one-query-token layout lets us feed the assembled keys/values without
        that intermediate cache.
        """
        if query.ndim != 3 or query.shape[0] != 1:
            raise ValueError("ShadowKV decode query must have shape [1, num_q_heads, head_dim]")
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("ShadowKV assembled K/V must have shape [tokens, kv_heads, head_dim]")
        if query.shape[-1] != keys.shape[-1]:
            raise ValueError("ShadowKV assembled K and query head dimensions must match")
        if query.shape[1] != self.local_q_heads or keys.shape[1] != self.local_kv_heads:
            raise ValueError("ShadowKV assembled K/Q head counts do not match the TP layout")
        if output.shape != query.shape:
            raise ValueError(
                "ShadowKV FlashAttention output must have the same shape as the query; "
                f"got output={tuple(output.shape)}, query={tuple(query.shape)}"
            )

        sequence_len = keys.shape[0]
        device = query.device
        cu_q = self._flash_cu_q.get(device)
        if cu_q is None:
            cu_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
            self._flash_cu_q[device] = cu_q
        cu_k_key = (device, sequence_len)
        cu_k = self._flash_cu_k.get(cu_k_key)
        if cu_k is None:
            cu_k = torch.tensor([0, sequence_len], device=device, dtype=torch.int32)
            self._flash_cu_k[cu_k_key] = cu_k

        q_var = query.contiguous()
        k_var = keys.contiguous()
        v_var = values.contiguous()
        out_var = output if output.is_contiguous() else torch.empty_like(q_var)
        flash_attn_varlen_func(
            q=q_var,
            k=k_var,
            v=v_var,
            max_seqlen_q=1,
            cu_seqlens_q=cu_q,
            max_seqlen_k=sequence_len,
            cu_seqlens_k=cu_k,
            causal=True,
            fa_version=2,
            out=out_var,
        )
        if not output.is_contiguous():
            output.copy_(out_var)
        return output

    def _assemble_decode_from_cache(
        self,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        positions: tuple[torch.Tensor, ...],
        selected_keys: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill reusable token-major buffers directly from the dense cache."""
        if len(positions) != 4:
            raise ValueError("ShadowKV decode assembly requires four position groups")
        if selected_keys.ndim != 3 or selected_keys.shape[0] != self.local_kv_heads:
            raise ValueError("ShadowKV reconstructed keys must have shape [kv_heads, tokens, dim]")
        if selected_keys.shape[1] != positions[2].shape[-1]:
            raise ValueError("ShadowKV reconstructed key count does not match selected positions")
        total_tokens = sum(position.shape[-1] for position in positions)
        num_heads, _, head_dim = cache_k.shape
        if num_heads != self.local_kv_heads or cache_v.shape != cache_k.shape:
            raise ValueError("ShadowKV cache shape does not match the configured local KV heads")
        if selected_keys.shape[-1] != head_dim:
            raise ValueError("ShadowKV reconstructed key dimension does not match the cache")

        if (
            self._assembly_key_scratch is None
            or self._assembly_value_scratch is None
            or self._assembly_capacity < total_tokens
            or self._assembly_key_scratch.shape[1] != num_heads
            or self._assembly_key_scratch.shape[2] != head_dim
            or self._assembly_key_scratch.device != cache_k.device
            or self._assembly_key_scratch.dtype != cache_k.dtype
        ):
            capacity = max(total_tokens, max(1, self._assembly_capacity * 2))
            self._assembly_key_scratch = torch.empty(
                capacity,
                num_heads,
                head_dim,
                device=cache_k.device,
                dtype=cache_k.dtype,
            )
            self._assembly_value_scratch = torch.empty_like(self._assembly_key_scratch)
            self._assembly_capacity = capacity

        key_target = self._assembly_key_scratch[:total_tokens]
        value_target = self._assembly_value_scratch[:total_tokens]
        offset = 0
        for group_idx, position_group in enumerate(positions):
            group_len = position_group.shape[-1]
            end = offset + group_len
            if group_idx == 2:
                key_target[offset:end].copy_(selected_keys.transpose(0, 1))
            else:
                self._copy_cache_tokens(cache_k, position_group, key_target[offset:end])
            self._copy_cache_tokens(cache_v, position_group, value_target[offset:end])
            offset = end
        return key_target, value_target

    def decode(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        tokens_per_head: torch.Tensor,
        output: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """Perform official retrieval/reconstruction followed by FlashAttention."""
        del keys, values
        self._validate_layer(layer_idx)
        if not self.initialized or self.U is None or self.SV is None:
            raise RuntimeError("ShadowKV decode requested before prefill initialization")
        if k_cache.shape != v_cache.shape:
            raise ValueError("ShadowKV K/V cache shapes must match")
        if k_cache.ndim != 4:
            raise ValueError(
                "ShadowKV expects reshaped vLLM caches as "
                "[local_kv_heads, blocks, block_size, head_dim]"
            )

        _, positions = self.select_chunks(query, layer_idx)
        selected_positions = positions[0]
        cache_k = k_cache.view(self.local_kv_heads, -1, k_cache.shape[-1])
        cache_v = v_cache.view(self.local_kv_heads, -1, v_cache.shape[-1])
        cache_len = int(tokens_per_head[0].item())
        if cache_len < self.prompt_len:
            raise RuntimeError(
                f"ShadowKV cache length {cache_len} is shorter than prompt {self.prompt_len}"
            )

        local_positions = torch.arange(
            self.prompt_len - self.prefill_local,
            self.prompt_len,
            device=query.device,
            dtype=torch.long,
        )
        outlier_chunks = self.outlier_chunk_idx[layer_idx, 0].to(torch.long)
        outlier_positions = (
            outlier_chunks.unsqueeze(-1) * self.chunk_size
            + torch.arange(
                self.chunk_size,
                device=query.device,
                dtype=outlier_chunks.dtype,
            ).view(1, 1, -1)
        ).view(1, self.local_kv_heads, -1)[0]
        generated_positions = torch.arange(
            self.prompt_len,
            cache_len,
            device=query.device,
            dtype=torch.long,
        )

        selected_keys = self._reconstruct_selected_keys(selected_positions, layer_idx)

        assembled_keys, assembled_values = self._assemble_decode_from_cache(
            cache_k,
            cache_v,
            (local_positions, outlier_positions, selected_positions, generated_positions),
            selected_keys,
        )
        self._flash_attention_assembled(query, assembled_keys, assembled_values, output)
        if layer_idx == self.num_layers - 1:
            from sparse_frontier.utils.sparsity_server import mark_shadowkv_decode

            mark_shadowkv_decode()
        return output

    def preallocate_memory(self, keys: torch.Tensor) -> None:
        """No extra persistent allocation; the vLLM cache remains dense."""
        del keys
