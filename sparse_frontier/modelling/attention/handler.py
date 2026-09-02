import os
from typing import Callable, Optional

import torch
from .abstract_attention import AttentionUtils
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)

from sparse_frontier.utils.sparsity_server import (
    ENV_ENABLED,
    set_prefill_sparsity,
    add_decode_access,
)

DECODE_REPORT_EVERY_N_STEPS = 100


def update_kv_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    is_prefilling: bool,
    tokens_per_head: torch.Tensor,
    q_heads_per_kv: int,
    head_indices: torch.Tensor = None,
    queries: torch.Tensor = None,
):
    """Update the KV cache with new key/value tensors.
    
    Args:
        key: [num_tokens, num_kv_heads, head_size]
        value: [num_tokens, num_kv_heads, head_size]
        k_cache: [num_kv_heads, num_blocks, block_size, head_size]
        v_cache: [num_kv_heads, num_blocks, block_size, head_size]
        is_prefilling: Whether we're in prefilling phase
        tokens_per_head: Tensor tracking token counts per head for current layer
        q_heads_per_kv: Number of query heads per key/value head
        head_indices: Precomputed indices for heads to avoid recreating tensor each time
        queries: [num_tokens, num_heads, head_size], required for compression
    """
    if is_prefilling:
        from sparse_frontier.modelling.attention.registry import get_attention
        key, value, seq_lens = get_attention().kv_compress(
            queries=queries,
            keys=key,
            values=value,
        )
        
        num_kv_heads, num_tokens, _ = key.shape
    
        # View the cache as a contiguous vector by merging block dimensions
        k_cache_flat = k_cache.view(num_kv_heads, -1, k_cache.shape[-1])
        v_cache_flat = v_cache.view(num_kv_heads, -1, v_cache.shape[-1])
        
        # Fill the prefix of the flattened cache
        k_cache_flat[:, :num_tokens] = key
        v_cache_flat[:, :num_tokens] = value
        
        tokens_per_head += seq_lens.repeat_interleave(q_heads_per_kv)
    else:
        num_tokens, num_kv_heads, _ = key.shape

        # Decoding case: single token update, use tokens_per_head to place correctly
        assert num_tokens == 1, "Decoding should only add one token at a time"
        
        # View the cache as a contiguous vector by merging block dimensions
        k_cache_flat = k_cache.view(num_kv_heads, -1, k_cache.shape[-1])
        v_cache_flat = v_cache.view(num_kv_heads, -1, v_cache.shape[-1])

        # Update the cache at the specific positions for all heads at once
        k_cache_flat[head_indices, tokens_per_head[::q_heads_per_kv]] = key[0]
        v_cache_flat[head_indices, tokens_per_head[::q_heads_per_kv]] = value[0]
        
        # Update token counts
        tokens_per_head += 1


class AttentionHandler:
    def __init__(
        self,
        tp_size: int,
        model_q_heads: int,
        model_kv_heads: int,
        model_layers: int,
        max_input_tokens: int,
        max_output_tokens: int,
        block_size: int
    ):
        """Initialize the attention handler.
        
        Args:
            tp_size: Tensor parallelism size
            model_q_heads: Total number of query heads
            model_kv_heads: Total number of key/value heads
            model_layers: Number of transformer layers
            max_input_tokens: Maximum number of input tokens
            max_output_tokens: Maximum number of output tokens
            block_size: default size of each block in the KV cache
        """
        self.tp_size = tp_size
        self.model_q_heads = model_q_heads
        self.model_kv_heads = model_kv_heads
        self.q_heads_per_gpu = model_q_heads // tp_size
        self.kv_heads_per_gpu = model_kv_heads // tp_size
        self.q_heads_per_kv = model_q_heads // model_kv_heads
        self.model_layers = model_layers
        self.max_seq_len = max_input_tokens + max_output_tokens
        self.max_blocks = (self.max_seq_len + block_size - 1) // block_size
        self.block_size = block_size
        
        self.current_layer = 0
        self.tokens_per_layer_head = torch.zeros(
            (model_layers, self.q_heads_per_gpu), 
            dtype=torch.int32, 
            device="cpu"
        )
        self.head_indices = None
        self._prefill_sparsity_sum = 0.0
        self._prefill_sparsity_count = 0
        self._prompt_len: Optional[int] = None
        self._dense_seq_len: Optional[int] = None
        self._decode_step: int = 0
        self._decode_access_sum_pending: float = 0.0
        self._decode_dense_sum_pending: float = 0.0
        self._pre_rope_key: Optional[torch.Tensor] = None
        self._pre_rope_positions: Optional[torch.Tensor] = None
        self._pre_rope_forward: Optional[Callable] = None
        self._pre_rope_metadata: Optional[dict[str, object]] = None

    @staticmethod
    def _should_report() -> bool:
        return os.environ.get(ENV_ENABLED) == "1" and get_tensor_model_parallel_rank() == 0

    def report_prefill_sparsity(self, sparsity: float) -> None:
        """Record one per-layer prefill sparsity value."""
        self._prefill_sparsity_sum += float(sparsity)
        self._prefill_sparsity_count += 1

    def _begin_prefill(self, prompt_len: int) -> None:
        from sparse_frontier.modelling.attention.registry import get_attention

        attention = get_attention()
        if hasattr(attention, "reset"):
            attention.reset()
        self._prompt_len = int(prompt_len)
        self._dense_seq_len = int(prompt_len)
        self._decode_step = 0
        self._prefill_sparsity_sum = 0.0
        self._prefill_sparsity_count = 0
        self._decode_access_sum_pending = 0.0
        self._decode_dense_sum_pending = 0.0
        self._pre_rope_metadata = None
        self.tokens_per_layer_head.zero_()

    def capture_pre_rope(
        self,
        key: torch.Tensor,
        positions: torch.Tensor,
        rope_forward: Callable,
    ) -> None:
        """Save pre-RoPE K before vLLM's in-place rotary operation mutates it."""
        if positions.ndim != 1:
            raise RuntimeError(
                "ShadowKV supports only scalar-position RoPE architectures; "
                f"received positions with shape {tuple(positions.shape)}"
            )
        if key.ndim == 2:
            if key.shape[-1] % self.kv_heads_per_gpu != 0:
                raise RuntimeError(
                    "ShadowKV cannot reshape flattened pre-RoPE K into local KV heads; "
                    f"shape={tuple(key.shape)}, local_kv_heads={self.kv_heads_per_gpu}"
                )
            key = key.view(key.shape[0], self.kv_heads_per_gpu, -1)
        elif key.ndim != 3:
            raise RuntimeError(
                "ShadowKV supports vLLM RoPE K with shape [tokens, kv*head_dim] "
                "or [tokens, kv_heads, head_dim]; received "
                f"{tuple(key.shape)}"
            )
        if key.shape[0] != positions.numel():
            raise RuntimeError(
                "ShadowKV pre-RoPE K and RoPE position counts differ; "
                f"K tokens={key.shape[0]}, positions={positions.numel()}"
            )
        self._pre_rope_key = key.detach().clone().contiguous()
        self._pre_rope_positions = positions.detach().clone().contiguous()
        self._pre_rope_forward = rope_forward

    def capture_rope_metadata(self, rotary_embedding: object) -> None:
        """Capture metadata needed by the optional fused ShadowKV kernels."""
        cos_sin_cache = getattr(rotary_embedding, "cos_sin_cache", None)
        head_size = getattr(rotary_embedding, "head_size", None)
        rotary_dim = getattr(rotary_embedding, "rotary_dim", None)
        is_neox_style = getattr(rotary_embedding, "is_neox_style", None)
        if (
            isinstance(cos_sin_cache, torch.Tensor)
            and isinstance(head_size, int)
            and isinstance(rotary_dim, int)
            and isinstance(is_neox_style, bool)
        ):
            self._pre_rope_metadata = {
                "cos_sin_cache": cos_sin_cache,
                "head_size": head_size,
                "rotary_dim": rotary_dim,
                "is_neox_style": is_neox_style,
            }
        else:
            # The scalar RoPE callable remains sufficient for the correctness
            # fallback, but the fused kernel cannot safely infer its layout.
            self._pre_rope_metadata = None

    def consume_rope_metadata(self) -> Optional[dict[str, object]]:
        """Return and clear optional fused-RoPE metadata for one layer."""
        metadata = self._pre_rope_metadata
        self._pre_rope_metadata = None
        return metadata

    def consume_pre_rope(self) -> tuple[torch.Tensor, torch.Tensor, Callable]:
        """Return and clear the current layer's pre-RoPE capture."""
        if (
            self._pre_rope_key is None
            or self._pre_rope_positions is None
            or self._pre_rope_forward is None
        ):
            raise RuntimeError(
                "ShadowKV did not capture pre-RoPE K and positions for the current "
                "attention layer; this RoPE architecture is unsupported"
            )
        result = (
            self._pre_rope_key,
            self._pre_rope_positions,
            self._pre_rope_forward,
        )
        self._pre_rope_key = None
        self._pre_rope_positions = None
        self._pre_rope_forward = None
        return result

    def _begin_decode_step(self) -> None:
        self._decode_step += 1
        if self._prompt_len is not None:
            self._dense_seq_len = int(self._prompt_len + self._decode_step)

    def _set_prompt_len_from_decode(self, prompt_len: int) -> None:
        """Recover the prompt length when vLLM used chunked prefill.

        vLLM exposes the current decode sequence length, while its chunked
        prefill metadata does not expose the original prompt length directly.
        The first decode sequence therefore gives us ``prompt_len + 1``.
        """
        if prompt_len < 1:
            raise RuntimeError(f"Invalid recovered prompt length {prompt_len}")
        self._prompt_len = int(prompt_len)
        self._dense_seq_len = int(prompt_len)
        self.tokens_per_layer_head.fill_(int(prompt_len))

    def _maybe_report_prefill_sparsity(self, device: torch.device) -> None:
        if self.current_layer != self.model_layers - 1:
            return

        if self._prefill_sparsity_count > 0:
            local = torch.tensor(
                [self._prefill_sparsity_sum / self._prefill_sparsity_count],
                device=device,
                dtype=torch.float32,
            )
        else:
            local = torch.tensor([0.0], device=device, dtype=torch.float32)

        if get_tensor_model_parallel_world_size() > 1:
            local = tensor_model_parallel_all_gather(local)

        if self._should_report():
            set_prefill_sparsity(float(local.mean().item()))

    def _maybe_report_decode_cost(self, device: torch.device) -> None:
        if self.current_layer != self.model_layers - 1:
            return

        if self._dense_seq_len is None:
            return

        dense_len = int(self._dense_seq_len)
        local_heads = int(self.q_heads_per_gpu)

        from sparse_frontier.modelling.attention.registry import get_attention

        attention = get_attention()
        attention_name = attention.__class__.__name__
        if attention_name == "QuestAttention":
            accessed_cap = min(dense_len, (attention.page_budget + 1) * attention.page_size)
        elif attention_name == "QueryRobustAttention":
            accessed_cap = min(dense_len, int(attention.last_accessed_tokens))
        elif attention_name == "ShadowKVAttention" and attention.initialized:
            generated_len = max(0, dense_len - attention.prompt_len)
            accessed_cap = min(
                dense_len,
                attention.prefill_local
                + attention.outlier_chunk * attention.chunk_size
                + attention.sparse_budget
                + generated_len,
            )
        else:
            accessed_cap = None

        if accessed_cap is None:
            accessed_sum = float(self.tokens_per_layer_head[self.current_layer].sum().item())
        else:
            accessed_sum = float(accessed_cap * local_heads)

        dense_sum = float(dense_len * local_heads)
        self._decode_access_sum_pending += accessed_sum
        self._decode_dense_sum_pending += dense_sum

        if (
            self._decode_step != 1
            and self._decode_step % DECODE_REPORT_EVERY_N_STEPS != 0
        ):
            return

        local = torch.tensor(
            [self._decode_access_sum_pending, self._decode_dense_sum_pending],
            device=device,
            dtype=torch.float32,
        )
        if get_tensor_model_parallel_world_size() > 1:
            local = tensor_model_parallel_all_gather(local)

        if self._should_report():
            totals = local if local.ndim == 1 else local.sum(dim=0)
            add_decode_access(accessed_sum=float(totals[0].item()), dense_sum=float(totals[1].item()))

        self._decode_access_sum_pending = 0.0
        self._decode_dense_sum_pending = 0.0

    def flush_decode_access(self, device: torch.device) -> None:
        """Publish a request's final bounded access counters off the hot path.

        The runner calls this only after vLLM has completed a request.  It
        keeps periodic online telemetry cheap while ensuring the persisted
        per-example total includes a short generation's trailing steps.
        """
        if (
            not self._should_report()
            or self._decode_access_sum_pending == 0.0
            and self._decode_dense_sum_pending == 0.0
        ):
            return
        local = torch.tensor(
            [self._decode_access_sum_pending, self._decode_dense_sum_pending],
            device=device,
            dtype=torch.float32,
        )
        if get_tensor_model_parallel_world_size() > 1:
            local = tensor_model_parallel_all_gather(local)
        totals = local if local.ndim == 1 else local.sum(dim=0)
        add_decode_access(
            accessed_sum=float(totals[0].item()),
            dense_sum=float(totals[1].item()),
        )
        self._decode_access_sum_pending = 0.0
        self._decode_dense_sum_pending = 0.0

    def advance_layer(self) -> None:
        self.current_layer = (self.current_layer + 1) % self.model_layers

    def _compute_attention_head_by_head(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention output by processing each head separately.
        
        Args:
            queries: [num_tokens, num_heads, head_size]
            keys: [num_tokens, num_kv_heads, head_size]
            values: [num_tokens, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size]
        """
        from sparse_frontier.modelling.attention.registry import get_attention

        # Get the attention implementation
        attention = get_attention()
        
        efficient_attention_classes = [
            "BlockSparseAttentionMInference", 
            "VerticalAndSlashAttentionMInference", 
            "FlexPrefill"
        ]
        
        if attention.__class__.__name__ not in efficient_attention_classes:
            output[:] = attention(
                queries=queries.transpose(0, 1).unsqueeze(0),
                keys=keys.transpose(0, 1).unsqueeze(0),
                values=values.transpose(0, 1).unsqueeze(0),
                layer_idx=self.current_layer,
            ).squeeze(0).transpose(0, 1)
            return
        
        # Otherwise, process head by head
        head_size = queries.shape[-1]
        q_heads_per_kv = queries.shape[1] // keys.shape[1]

        for head in range(self.q_heads_per_gpu):
            kv_head = head // q_heads_per_kv
            output[:, head, :] = attention(
                queries=queries[:, head, :].view(1, 1, -1, head_size),
                keys=keys[:, kv_head, :].view(1, 1, -1, head_size),
                values=values[:, kv_head, :].view(1, 1, -1, head_size),
                layer_idx=self.current_layer,
            ).view(-1, head_size)
        
    def __call__(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache: torch.Tensor,
        output: torch.Tensor,
        block_table: torch.Tensor = None,
    ) -> torch.Tensor:
        """Process attention for either prefill or decode phase.
        
        Args:
            queries: [num_tokens, num_heads, head_size]
            keys: [num_tokens, num_kv_heads, head_size]
            values: [num_tokens, num_kv_heads, head_size]
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            output: [num_tokens, num_heads, head_size]
        
        Returns:
            attention_output: [num_tokens, num_heads, head_size]
        """
        num_tokens = queries.shape[0]
        is_prefilling = num_tokens > 1

        from sparse_frontier.modelling.attention.registry import get_attention

        attention = get_attention()
        is_shadowkv = attention.__class__.__name__ == "ShadowKVAttention"
        is_query_robust = attention.__class__.__name__ == "QueryRobustAttention"
        if is_shadowkv:
            pre_rope_key, pre_rope_positions, rope_forward = self.consume_pre_rope()
            rope_metadata = self.consume_rope_metadata()
        else:
            rope_metadata = None

        # Initialize head_indices if not already done
        if self.head_indices is None and keys.numel() > 0:
            num_kv_heads = keys.shape[1]
            self.head_indices = torch.arange(num_kv_heads, device=keys.device)
            self.tokens_per_layer_head = self.tokens_per_layer_head.to(keys.device)

        if is_prefilling:
            if is_shadowkv:
                attention.prefill_state(
                    pre_rope_keys=pre_rope_key,
                    post_rope_keys=keys,
                    values=values,
                    positions=pre_rope_positions,
                    layer_idx=self.current_layer,
                    rope_forward=rope_forward,
                    rope_metadata=rope_metadata,
                )
            self._compute_attention_head_by_head(
                queries=queries,
                keys=keys,
                values=values,
                output=output,
            )
            self._maybe_report_prefill_sparsity(device=keys.device)

        if is_query_robust:
            # vllm_patched_forward has already written the canonical paged
            # cache. Query-Robust gathers only selected chunks into its small
            # scratch cache, so materializing a full logical K/V copy here
            # would erase the intended online speedup.
            if is_prefilling:
                self.tokens_per_layer_head[self.current_layer] += num_tokens
            else:
                self.tokens_per_layer_head[self.current_layer] += 1
                attention.decode_paged(
                    query=queries,
                    kv_cache=kv_cache,
                    physical_block_table=block_table,
                    tokens_per_head=self.tokens_per_layer_head[self.current_layer],
                    total_tokens=int(self._dense_seq_len),
                    output=output,
                    layer_idx=self.current_layer,
                )
                self._maybe_report_decode_cost(device=keys.device)
            self.advance_layer()
            return output

        k_cache, v_cache = AttentionUtils.reshape_kv_cache(
            kv_cache,
            self.block_size,
            self.max_blocks,
            block_table=block_table,
        )

        # Then update cache (with compression if prefilling)
        update_kv_cache(
            keys,
            values,
            k_cache,
            v_cache,
            is_prefilling=is_prefilling,
            tokens_per_head=self.tokens_per_layer_head[self.current_layer],
            q_heads_per_kv=self.q_heads_per_kv,
            head_indices=self.head_indices,
            queries=queries if is_prefilling else None,
        )

        if not is_prefilling:
            ensure_page_reps = getattr(attention, "ensure_page_reps_from_cache", None)
            if ensure_page_reps is not None:
                ensure_page_reps(
                    k_cache=k_cache,
                    cache_len=int(self.tokens_per_layer_head[self.current_layer][0].item()),
                    layer_idx=self.current_layer,
                )
            attention.decode(
                query=queries,
                keys=keys,
                values=values,
                k_cache=k_cache,
                v_cache=v_cache,
                tokens_per_head=self.tokens_per_layer_head[self.current_layer],
                output=output,
                layer_idx=self.current_layer,
            )

            self._maybe_report_decode_cost(device=keys.device)

        self.advance_layer()
