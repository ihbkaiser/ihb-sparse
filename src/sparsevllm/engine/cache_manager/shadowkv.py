"""ShadowKV cache manager.

This is the reference, correctness-first integration of ShadowKV.  The
prefill path keeps the ordinary causal computation but stages one layer's
prompt KV on the GPU.  In CPU-shadow mode, values remain in pinned host
memory; decode selects landmark chunks and reconstructs only the selected keys
from a low-rank factorization.  In GPU-cache mode, exact post-RoPE K/V is kept
on the device and only the selected view is gathered.  The explicit decode
payload is indexed per KV head, so each head can retain its own selected
positions and effective length.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from sparsevllm.config import Config
from sparsevllm.configs.sparse import resolve_shadowkv_outlier_chunks
from sparsevllm.distributed import ParallelContext
from sparsevllm.engine.cache_manager.base import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    LayerBatchStates,
    PrefillComputeView,
    SparseSelection,
)
from sparsevllm.engine.decode_graph_contract import (
    CacheDecodeGraphState,
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.layers.rotary_embedding import apply_rotary_emb, get_rope
from sparsevllm.models.rope import resolve_rope_scaling, resolve_rope_theta
from sparsevllm.platforms import device_runtime
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import log_once, logger
from sparsevllm.utils.profiler import profiler


class ShadowKVCacheManager(StandardCacheManager):
    """CPU-shadowed explicit KV storage with ShadowKV decode selection."""

    def __init__(
        self,
        config: Config,
        parallel_context: ParallelContext,
        *,
        allocation_budget_bytes: int | None = None,
    ) -> None:
        self._shadow_entries: dict[tuple[int, int], dict[str, Any]] = {}
        self._shadow_prefill_k: torch.Tensor | None = None
        self._shadow_prefill_v: torch.Tensor | None = None
        self._shadow_decode_k: torch.Tensor | None = None
        self._shadow_decode_v: torch.Tensor | None = None
        self._shadow_decode_capacity = 0
        self._shadow_batch_capacity = 0
        self._shadow_decode_workspaces: dict[int, dict[str, torch.Tensor]] = {}
        self._shadow_workspace_signatures: dict[int, tuple[object, ...]] = {}
        self._shadow_host_pointer_signatures: dict[int, tuple[object, ...]] = {}
        self._shadow_offset_workspaces: dict[int, dict[str, Any]] = {}
        self._shadow_host_gather = None
        self._shadow_gpu_cache_gather = None
        self._shadow_offset_gather = None
        self._shadow_kernel_backend: str | None = None
        self._shadow_copy_stream = None
        self._shadow_offset_stream = None
        self._shadow_copy_ready_event = None
        self._shadow_copy_done_event = None
        self._shadow_offset_ready_event = None
        self._shadow_offset_done_event = None
        self._shadow_active_rows: list[int] = []
        super().__init__(
            config,
            parallel_context,
            allocation_budget_bytes=allocation_budget_bytes,
        )
        # These are logical CPU-shadow slots.  They are deliberately separate
        # from the tiny physical storage allocated during Standard init.
        # The inherited scheduler/admission path consumes ``free_slots_stack``
        # even though ShadowKV does not retain prompt KV in that device store.
        # Replace the tiny compatibility accounting with a virtual token pool;
        # the stack is only metadata and is negligible compared with the host
        # shadow tensors, but it allows large prompt batches to be admitted.
        self._shadow_virtual_capacity = int(self.max_buffer_rows) * int(self.max_model_len)
        self.free_slots_stack = torch.arange(
            self._shadow_virtual_capacity,
            dtype=torch.int32,
            device=self.device,
        )
        self._num_free_slots = self._shadow_virtual_capacity
        self.config.num_kvcache_slots = self._shadow_virtual_capacity
        self._shadow_rope = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max(1, int(self.max_model_len)),
            base=resolve_rope_theta(self.hf_config),
            rope_scaling=resolve_rope_scaling(self.hf_config, model_name="ShadowKV"),
            backend="torch",
        ).to(self.device)
        log_once(
            "ShadowKV uses a head-indexed compact decode payload; each KV head "
            "keeps its own selected positions and effective length.",
            level="INFO",
        )

    def allocate_kv_cache(self) -> None:
        """Allocate only a tiny compatibility cache; ShadowKV owns read state."""

        storage = self.attention_cache_storage
        # The current-token write path is intercepted by save_rope_kv_if_needed.
        # One slot per resident row keeps inherited diagnostics and storage
        # validation well-defined without allocating a full prompt KV cache.
        storage.allocate(
            num_layers=self.num_kv_layers,
            num_slots=max(1, int(self.max_buffer_rows)),
            device=self.device,
        )
        self.config.num_kvcache_slots = max(1, int(self.max_buffer_rows))
        self.kv_cache = getattr(storage, "kv_cache", None)

    def _validate_attention_slot_mapping(self, slot_mapping: torch.Tensor) -> None:
        """ShadowKV uses virtual host-shadow coordinates, not physical storage slots."""
        del slot_mapping

    @property
    def _shadow_dtype(self) -> torch.dtype:
        return self.hf_config.dtype

    @property
    def _shadow_pin_memory(self) -> bool:
        return bool(torch.cuda.is_available() and device_runtime.supports_pin_memory())

    @property
    def _shadow_gpu_cache_enabled(self) -> bool:
        return str(getattr(self.config, "shadowkv_storage", "cpu")).lower() == "gpu_cache"

    def _apply_shadow_rope(
        self,
        raw_keys: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the model's exact static RoPE to reconstructed raw keys.

        ShadowKV factorizes pre-RoPE keys.  Applying RoPE after reconstruction
        is therefore part of the method contract, not an optional formatting
        step.  The implementation is composed only of device tensor ops so it
        remains capturable by CUDA Graph.
        """

        if raw_keys.shape[-2:] != (self.num_kv_heads, self.head_dim):
            raise ValueError(
                "ShadowKV reconstructed keys have an invalid shape: "
                f"got {tuple(raw_keys.shape)}, expected [..., {self.num_kv_heads}, {self.head_dim}]."
            )
        if self._shadow_rope is None:
            raise RuntimeError("ShadowKV RoPE cache was not initialized.")
        flat = raw_keys.reshape(-1, self.num_kv_heads, self.head_dim)
        flat_positions = positions.to(device=self.device, dtype=torch.long).reshape(-1)
        if int(flat.numel()) // (self.num_kv_heads * self.head_dim) != int(flat_positions.numel()):
            raise ValueError(
                "ShadowKV key/position shape mismatch: "
                f"keys={tuple(raw_keys.shape)} positions={tuple(positions.shape)}."
            )
        max_position = int(self._shadow_rope.cos_sin_cache.shape[0]) - 1
        flat_positions = flat_positions.clamp_(0, max_position)
        cos_sin = self._shadow_rope.cos_sin_cache.index_select(0, flat_positions)
        cos, sin = cos_sin.squeeze(1).chunk(2, dim=-1)
        return apply_rotary_emb(
            flat,
            cos[:, None, :],
            sin[:, None, :],
        ).reshape_as(raw_keys)

    def _entry(self, row_idx: int, layer_idx: int) -> dict[str, Any]:
        key = (int(row_idx), int(layer_idx))
        entry = self._shadow_entries.get(key)
        if entry is None:
            entry = {
                "filled": 0,
                "raw_k": None,
                "rope_k": None,
                "v": None,
                "v_chunks": None,
                "prompt_len": 0,
                "shadow": False,
                "candidate_chunks": 0,
                "local_tokens": 0,
                "gpu_rope_k": None,
                "gpu_v": None,
            }
        self._shadow_entries[key] = entry
        return entry

    def _ensure_entry_capacity(
        self,
        entry: dict[str, Any],
        capacity: int,
        *,
        names: tuple[str, ...] = ("raw_k", "rope_k", "v"),
    ) -> None:
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"ShadowKV entry capacity must be positive, got {capacity}.")
        old_capacity = 0
        for name in names:
            tensor = entry[name]
            if tensor is not None:
                old_capacity = int(tensor.shape[0])
                break
        missing_tensor = any(entry[name] is None for name in names)
        if old_capacity >= capacity and not missing_tensor:
            return
        new_capacity = max(capacity, max(1, old_capacity * 2))
        for name in names:
            old = entry[name]
            if old is None:
                new = torch.empty(
                    (new_capacity, self.num_kv_heads, self.head_dim),
                    dtype=self._shadow_dtype,
                    device="cpu",
                    pin_memory=self._shadow_pin_memory,
                )
            else:
                new = torch.empty(
                    (new_capacity, *old.shape[1:]),
                    dtype=old.dtype,
                    device="cpu",
                    pin_memory=self._shadow_pin_memory,
                )
                copy_count = min(int(entry["filled"]), int(old.shape[0]))
                if copy_count > 0:
                    new[:copy_count].copy_(old[:copy_count])
            entry[name] = new

    def _ensure_v_chunk_capacity(
        self,
        entry: dict[str, Any],
        capacity: int,
    ) -> None:
        """Maintain the chunk-major pinned V view consumed by offset-copy."""
        chunk_size = int(self.config.shadowkv_chunk_size)
        chunk_dim = chunk_size * self.head_dim
        needed_chunks = (int(capacity) + chunk_size - 1) // chunk_size
        current = entry.get("v_chunks")
        if current is not None and int(current.shape[1]) >= needed_chunks:
            return
        old_chunks = 0 if current is None else int(current.shape[1])
        new_chunks = max(needed_chunks, max(1, old_chunks * 2))
        new = torch.zeros(
            (self.num_kv_heads, new_chunks, chunk_dim),
            dtype=self._shadow_dtype,
            device="cpu",
            pin_memory=self._shadow_pin_memory,
        )
        if current is not None and old_chunks:
            new[:, :old_chunks].copy_(current)
        entry["v_chunks"] = new

    def _sync_v_chunks(
        self,
        entry: dict[str, Any],
        start: int,
        end: int,
    ) -> None:
        """Pack only touched token chunks into contiguous per-head rows."""
        if end <= start or entry.get("v") is None:
            return
        self._ensure_v_chunk_capacity(entry, end)
        chunk_size = int(self.config.shadowkv_chunk_size)
        first_chunk = max(0, int(start) // chunk_size)
        last_chunk = (int(end) + chunk_size - 1) // chunk_size
        source = entry["v"]
        destination = entry["v_chunks"]
        for chunk_idx in range(first_chunk, last_chunk):
            token_start = chunk_idx * chunk_size
            token_end = min(token_start + chunk_size, int(source.shape[0]))
            token_count = max(0, token_end - token_start)
            destination[:, chunk_idx].zero_()
            if token_count:
                packed = (
                    source[token_start:token_end]
                    .permute(1, 0, 2)
                    .contiguous()
                    .view(self.num_kv_heads, -1)
                )
                destination[:, chunk_idx, : packed.shape[1]].copy_(packed)

    def _current_ranges(
        self, layer_idx: int, token_count: int
    ) -> list[tuple[int, int, int, int, int]]:
        state = self.layer_batch_state
        if state.req_indices is None or state.context_lens is None:
            raise RuntimeError("ShadowKV received KV before cache batch metadata was prepared.")
        context = get_context()
        if context.is_prefill:
            cu = context.cu_seqlens_q
            if cu is None or int(cu.numel()) != int(state.req_indices.numel()) + 1:
                raise RuntimeError("ShadowKV prefill requires cu_seqlens_q for CPU shadow writes.")
            offsets = cu.detach().cpu().tolist()
        else:
            offsets = list(range(int(state.req_indices.numel()) + 1))
        rows = state.req_indices.detach().cpu().tolist()
        ends = state.context_lens.detach().cpu().tolist()
        ranges: list[tuple[int, int, int, int, int]] = []
        for batch_idx, row in enumerate(rows):
            start_offset = int(offsets[batch_idx])
            end_offset = int(offsets[batch_idx + 1])
            start = int(ends[batch_idx]) - (end_offset - start_offset)
            if start < 0 or end_offset > int(token_count):
                raise RuntimeError(
                    "ShadowKV CPU shadow write range is invalid: "
                    f"layer={layer_idx} row={row} start={start} end={ends[batch_idx]} "
                    f"token_count={token_count}."
                )
            ranges.append(
                (
                    int(row),
                    start,
                    int(ends[batch_idx]),
                    start_offset,
                    end_offset,
                )
            )
        return ranges

    @torch.no_grad()
    def _write_shadow(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kind: str,
    ) -> None:
        if k.ndim != 3 or v.ndim != 3 or k.shape != v.shape:
            raise ValueError(
                "ShadowKV expects flattened explicit KV tensors with shape "
                f"[tokens, kv_heads, head_dim], got k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        for row, start, end, source_start, source_end in self._current_ranges(
            layer_idx, int(k.shape[0])
        ):
            entry = self._entry(row, layer_idx)
            names = ("raw_k", "v") if kind == "raw_k" else (kind,)
            self._ensure_entry_capacity(entry, end, names=names)
            entry[kind][start:end].copy_(
                k[source_start:source_end].detach(), non_blocking=False
            )
            if kind == "raw_k":
                entry["v"][start:end].copy_(
                    v[source_start:source_end].detach(), non_blocking=False
                )
            self._sync_v_chunks(entry, start, end)
            entry["filled"] = max(int(entry["filled"]), end)

    def save_raw_kv_if_needed(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        # After the prefill factorization, only values need to be appended for
        # decode.  Avoid recreating the discarded raw-key shadow tensor.
        context = get_context()
        if not context.is_prefill:
            # GPU-cache mode receives the exact post-RoPE K/V through
            # save_rope_kv_if_needed below.  Keeping a second CPU copy of the
            # generated token would only add synchronous host traffic and is
            # not read by selection, reconstruction, or the GPU lookup path.
            if self._shadow_gpu_cache_enabled:
                return
            # CUDA Graph replay cannot perform the Python/CPU writes used by
            # the host shadow. Decode reads current-token KV from the
            # graph-stable recent_* buffers instead.
            if bool(getattr(self.config, "decode_graph", False)):
                if self._workspace_for_layer(layer_idx) is None:
                    raise RuntimeError(
                        "ShadowKV CUDA Graph raw-KV write reached decode before "
                        "the layer workspace was prepared."
                    )
                return
            ranges = self._current_ranges(layer_idx, int(k.shape[0]))
            needs_dense_write = False
            for row, start, end, source_start, source_end in ranges:
                entry = self._entry(row, layer_idx)
                if entry.get("shadow", False):
                    self._ensure_entry_capacity(entry, end, names=("v",))
                    entry["v"][start:end].copy_(v[source_start:source_end].detach())
                    self._sync_v_chunks(entry, start, end)
                    entry["filled"] = max(int(entry["filled"]), end)
                else:
                    needs_dense_write = True
            if needs_dense_write:
                for row, start, end, source_start, source_end in ranges:
                    entry = self._entry(row, layer_idx)
                    if entry.get("shadow", False):
                        continue
                    self._ensure_entry_capacity(entry, end, names=("raw_k", "v"))
                    entry["raw_k"][start:end].copy_(k[source_start:source_end].detach())
                    entry["v"][start:end].copy_(v[source_start:source_end].detach())
                    self._sync_v_chunks(entry, start, end)
                    entry["filled"] = max(int(entry["filled"]), end)
            return
        self._write_shadow(layer_idx, k, v, kind="raw_k")

    def save_rope_kv_if_needed(self, layer_idx: int, k_post_rope: torch.Tensor, v: torch.Tensor):
        # Post-RoPE keys are retained as the exact fallback for outlier/local
        # chunks.  The low-rank factorization itself is built over raw pre-RoPE
        # keys and RoPE is applied after reconstruction at decode time.
        context = get_context()
        if not context.is_prefill and bool(getattr(self.config, "decode_graph", False)):
            workspace = self._workspace_for_layer(layer_idx)
            if workspace is None:
                raise RuntimeError(
                    "ShadowKV CUDA Graph RoPE-KV write reached decode before "
                    "the layer workspace was prepared."
                )
            if k_post_rope.ndim != 3 or v.ndim != 3 or k_post_rope.shape != v.shape:
                raise ValueError(
                    "ShadowKV decode expects [batch, kv_heads, head_dim] KV tensors, "
                    f"got k={tuple(k_post_rope.shape)} v={tuple(v.shape)}."
                )
            batch_size = int(k_post_rope.shape[0])
            context_lens = self.layer_batch_state.context_lens
            if context_lens is None or int(context_lens.numel()) != batch_size:
                raise RuntimeError(
                    "ShadowKV CUDA Graph current-token metadata does not match KV batch: "
                    f"context_lens={None if context_lens is None else context_lens.numel()} "
                    f"batch={batch_size}."
                )
            recent_width = int(workspace["recent_k"].shape[1])
            if recent_width <= 0:
                raise RuntimeError("ShadowKV CUDA Graph recent-token workspace is empty.")
            prompt_lens = workspace["prompt_lens"][:batch_size]
            recent_index = (
                context_lens[:batch_size].to(dtype=torch.int64)
                - prompt_lens.to(dtype=torch.int64)
                - 1
            ).clamp_(0, recent_width - 1).to(dtype=torch.long)
            index = recent_index[:, None, None, None].expand(
                -1, 1, self.num_kv_heads, self.head_dim
            )
            k_value = k_post_rope.reshape(
                batch_size, 1, self.num_kv_heads, self.head_dim
            ).to(dtype=workspace["recent_k"].dtype)
            v_value = v.reshape(
                batch_size, 1, self.num_kv_heads, self.head_dim
            ).to(dtype=workspace["recent_v"].dtype)
            workspace["recent_k"][:batch_size].scatter_(1, index, k_value)
            workspace["recent_v"][:batch_size].scatter_(1, index, v_value)
            return
        if self._shadow_gpu_cache_enabled:
            ranges = self._current_ranges(layer_idx, int(k_post_rope.shape[0]))
            if context.is_prefill:
                gpu_capacity = int(self.config.shadowkv_gpu_cache_tokens)
                for row, start, end, source_start, source_end in ranges:
                    if end > gpu_capacity:
                        raise RuntimeError(
                            "ShadowKV GPU cache capacity is smaller than the prompt: "
                            f"end={end} capacity={gpu_capacity}. Increase "
                            "shadowkv_gpu_cache_tokens or use shadowkv_storage='cpu'."
                        )
                    entry = self._entry(row, layer_idx)
                    gpu_k = entry.get("gpu_rope_k")
                    gpu_v = entry.get("gpu_v")
                    if gpu_k is None or gpu_v is None:
                        gpu_k = torch.empty(
                            (gpu_capacity, self.num_kv_heads, self.head_dim),
                            dtype=self._shadow_dtype,
                            device=self.device,
                        )
                        gpu_v = torch.empty_like(gpu_k)
                        entry["gpu_rope_k"] = gpu_k
                        entry["gpu_v"] = gpu_v
                    gpu_k[start:end].copy_(
                        k_post_rope[source_start:source_end], non_blocking=True
                    )
                    gpu_v[start:end].copy_(
                        v[source_start:source_end], non_blocking=True
                    )
                self._write_shadow(layer_idx, k_post_rope, v, kind="rope_k")
                return
            for row, start, end, source_start, source_end in ranges:
                del start, source_end
                entry = self._entry(row, layer_idx)
                gpu_k = entry.get("gpu_rope_k")
                gpu_v = entry.get("gpu_v")
                if gpu_k is None or gpu_v is None:
                    raise RuntimeError(
                        "ShadowKV GPU cache entry was not materialized for eager decode: "
                        f"row={row} layer={layer_idx}."
                    )
                gpu_k[end - 1].copy_(k_post_rope[source_start], non_blocking=True)
                gpu_v[end - 1].copy_(v[source_start], non_blocking=True)
            # The GPU cache is authoritative in this mode.  Do not mirror
            # generated K/V into the pinned CPU shadow on every decode step.
            for row, _start, end, _source_start, _source_end in ranges:
                entry = self._entry(row, layer_idx)
                entry["filled"] = max(int(entry["filled"]), end)
            return
        self._write_shadow(layer_idx, k_post_rope, v, kind="rope_k")

    def _virtual_slots(self, row_idx: int, start: int, size: int) -> torch.Tensor:
        base = int(row_idx) * int(self.max_model_len) + int(start)
        return torch.arange(
            base,
            base + int(size),
            dtype=torch.int32,
            device=self.device,
        )

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        row_idx = self.seq_id_to_row.get(int(seq_id))
        if row_idx is None:
            row_idx = self._get_free_row(int(seq_id))
            for key in [key for key in self._shadow_entries if key[0] == row_idx]:
                self._shadow_entries.pop(key, None)
        start = int(self.row_seq_lens[row_idx])
        end = start + int(size)
        if end > int(self.max_model_len):
            raise RuntimeError(
                f"ShadowKV sequence length exceeds max_model_len: end={end} "
                f"max_model_len={self.max_model_len}."
            )
        self.buffer_req_to_token_slots[row_idx, start:end] = self._virtual_slots(
            row_idx, start, size
        )
        self.row_seq_lens[row_idx] = end
        self.row_logical_lens[row_idx] = end
        return self.buffer_req_to_token_slots[row_idx, start:end]

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        if int(size) != 1:
            raise ValueError("ShadowKV decode allocation supports one token per sequence.")
        slots = []
        for seq_id in seq_ids:
            row_idx = self.seq_id_to_row.get(int(seq_id))
            if row_idx is None:
                row_idx = self._get_free_row(int(seq_id))
                for key in [key for key in self._shadow_entries if key[0] == row_idx]:
                    self._shadow_entries.pop(key, None)
            start = int(self.row_seq_lens[row_idx])
            if start >= int(self.max_model_len):
                raise RuntimeError("ShadowKV decode exceeded max_model_len.")
            slot = self._virtual_slots(row_idx, start, 1)[0]
            self.buffer_req_to_token_slots[row_idx, start] = slot
            self.row_seq_lens[row_idx] = start + 1
            self.row_logical_lens[row_idx] = start + 1
            slots.append(slot)
        return torch.stack(slots)

    def free_seq(self, seq_id: int):
        row_idx = self.seq_id_to_row.pop(int(seq_id), None)
        if row_idx is None:
            raise ValueError(f"Unknown ShadowKV sequence id={seq_id}.")
        for key in [key for key in self._shadow_entries if key[0] == row_idx]:
            self._shadow_entries.pop(key, None)
        self.buffer_req_to_token_slots[row_idx].zero_()
        self.row_seq_lens[row_idx] = 0
        self.row_logical_lens[row_idx] = 0
        self.free_rows.append(row_idx)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        result = super().prepare_step(seqs, is_prefill)
        if not is_prefill:
            self._shadow_active_rows = [
                int(self.seq_id_to_row[int(seq.seq_id)]) for seq in seqs
            ]
        else:
            self._shadow_active_rows = []
        return result

    @torch.no_grad()
    def _finalize_entry(
        self,
        row_idx: int,
        layer_idx: int,
        prompt_len: int,
        *,
        factorization: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        entry = self._entry(row_idx, layer_idx)
        if entry.get("shadow", False) and int(entry.get("prompt_len", 0)) == int(prompt_len):
            return
        if int(entry["filled"]) < int(prompt_len):
            raise RuntimeError(
                "ShadowKV prefill did not populate the complete CPU shadow: "
                f"row={row_idx} layer={layer_idx} filled={entry['filled']} prompt={prompt_len}."
            )
        rope = entry["rope_k"][:prompt_len]
        raw = entry["raw_k"][:prompt_len]
        chunk_size = int(self.config.shadowkv_chunk_size)
        local_chunks = int(self.config.shadowkv_local_chunks)
        budget = int(self.config.shadowkv_sparse_budget)
        select_sets = budget // chunk_size
        chunks = max(0, prompt_len // chunk_size - local_chunks)
        chunks -= chunks % 8
        outlier_count = min(int(self.config.shadowkv_outlier_chunks), chunks)

        # GPU-cache mode already has to materialize the exact post-RoPE K/V on
        # the device.  Reuse that copy to compute the small selection metadata
        # on GPU as well; doing the landmark mean, cosine reduction, and
        # outlier top-k on CPU otherwise makes long-context TTFT CPU-bound.
        gpu_cache_fast_path = (
            self._shadow_gpu_cache_enabled and self.device.type == "cuda"
        )
        gpu_k: torch.Tensor | None = None
        gpu_v: torch.Tensor | None = None
        if gpu_cache_fast_path:
            gpu_capacity = int(self.config.shadowkv_gpu_cache_tokens)
            if prompt_len > gpu_capacity:
                raise RuntimeError(
                    "ShadowKV GPU cache capacity is smaller than the prompt: "
                    f"prompt_len={prompt_len} capacity={gpu_capacity}. Increase "
                    "shadowkv_gpu_cache_tokens or use shadowkv_storage='cpu'."
                )
            gpu_k = entry.get("gpu_rope_k")
            gpu_v = entry.get("gpu_v")
            if gpu_k is None or gpu_v is None:
                gpu_k = torch.empty(
                    (gpu_capacity, self.num_kv_heads, self.head_dim),
                    dtype=self._shadow_dtype,
                    device=self.device,
                )
                gpu_v = torch.empty_like(gpu_k)
            gpu_k[:prompt_len].copy_(rope, non_blocking=True)
            gpu_v[:prompt_len].copy_(
                entry["v"][:prompt_len], non_blocking=True
            )
            entry["gpu_rope_k"] = gpu_k
            entry["gpu_v"] = gpu_v

        if chunks > 0:
            context_source = gpu_k[: chunks * chunk_size] if gpu_cache_fast_path else rope
            context = (
                context_source[: chunks * chunk_size]
                .view(chunks, chunk_size, self.num_kv_heads, self.head_dim)
                .permute(2, 0, 1, 3)
                .contiguous()
            )
            landmarks = context.mean(dim=2)
            if outlier_count > 0:
                cosine = F.cosine_similarity(
                    landmarks.unsqueeze(2), context, dim=-1
                )
                outlier_chunks = cosine.amin(dim=-1).topk(
                    outlier_count, largest=False, dim=-1
                ).indices
            else:
                outlier_chunks = torch.empty(
                    (self.num_kv_heads, 0), dtype=torch.long, device=landmarks.device
                )
        else:
            landmarks = torch.empty(
                (self.num_kv_heads, 0, self.head_dim),
                dtype=rope.dtype,
                device=self.device if gpu_cache_fast_path else rope.device,
            )
            outlier_chunks = torch.empty(
                (self.num_kv_heads, 0),
                dtype=torch.long,
                device=landmarks.device,
            )

        # The rest of finalization builds CPU-side per-head index tables.  A
        # single small metadata transfer is intentional and avoids retaining
        # GPU tensors in the host shadow entry.
        if gpu_cache_fast_path:
            landmarks = landmarks.detach().to(device="cpu").contiguous()
            outlier_chunks = outlier_chunks.detach().to(device="cpu").contiguous()

        # GPU-cache mode reads exact K/V directly during decode, so it needs
        # the landmark table and outlier metadata but never consumes U/SV.
        # Avoiding SVD here is important for long-context TTFT: the mode is an
        # explicit speed/memory point, not the CPU-shadow reconstruction path.
        matrix: torch.Tensor | None = None
        rank = 0
        if not gpu_cache_fast_path:
            # The original implementation factorizes flattened *pre-RoPE*
            # keys. RoPE is position-dependent and is applied after low-rank
            # reconstruction at decode time. Factorizing post-RoPE keys
            # destroys the shared low-rank representation and is especially
            # harmful for long contexts.
            if factorization is None:
                matrix = raw.float().reshape(prompt_len, -1).to(self.device)
                matrix_rows = int(matrix.shape[0])
                matrix_cols = int(matrix.shape[1])
            else:
                u, s, vh = factorization
                matrix_rows = int(u.shape[-2])
                matrix_cols = int(vh.shape[-1])
            rank = min(int(self.config.shadowkv_rank), matrix_rows, matrix_cols)
            if factorization is None:
                with profiler.record("shadowkv_prefill_svd"):
                    if str(getattr(self.config, "shadowkv_svd_method", "exact")) == "exact":
                        # Exact SVD is the reference accuracy path.  The optional
                        # low-rank path is explicit because it changes the basis
                        # and therefore can change selected-key values.
                        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
                    else:
                        q = min(
                            min(matrix_rows, matrix_cols),
                            rank + int(getattr(self.config, "shadowkv_svd_oversample", 16)),
                        )
                        u, s, v = torch.svd_lowrank(
                            matrix,
                            q=q,
                            niter=int(getattr(self.config, "shadowkv_svd_niter", 2)),
                        )
                        vh = v.transpose(-2, -1)
            sv = (
                (vh[..., :rank, :] * s[..., :rank, None])
                .reshape(*vh.shape[:-2], rank, self.num_kv_heads, self.head_dim)
                .movedim(-3, -2)
                .contiguous()
            )
            if sv.ndim != 3:
                raise RuntimeError(
                    "ShadowKV entry factorization must be unbatched, got "
                    f"u={tuple(u.shape)} s={tuple(s.shape)} vh={tuple(vh.shape)}."
                )
        def host_tensor(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.detach().to(device="cpu", dtype=self._shadow_dtype).contiguous()
            return tensor.pin_memory() if self._shadow_pin_memory else tensor

        # Keep outliers and landmark indices per KV head.  The decode payload
        # is flattened as (request, KV-head) rows, so each row can use its own
        # landmark candidate set without taking a union across heads.
        target_rank = int(self.config.shadowkv_rank)
        if gpu_cache_fast_path:
            entry["u"] = None
            entry["sv"] = None
            entry["sv_column_major"] = None
        else:
            u_store = torch.zeros(
                (prompt_len, target_rank), dtype=u.dtype, device=self.device
            )
            u_store[:, :rank].copy_(u[:, :rank])
            sv_store = torch.zeros(
                (self.num_kv_heads, target_rank, self.head_dim),
                dtype=sv.dtype,
                device=self.device,
            )
            sv_store[:, :rank].copy_(sv)
            entry["u"] = host_tensor(u_store)
            entry["sv"] = host_tensor(sv_store)
            # The fused CUTLASS gather iterator consumes the logical [rank, D]
            # factor as a column-major matrix. Keep its physical [D, rank]
            # view pinned so decode staging does not transpose or allocate.
            entry["sv_column_major"] = host_tensor(
                sv_store.permute(0, 2, 1).contiguous()
            )
        all_chunk_indices = torch.arange(chunks, dtype=torch.long)
        if outlier_count > 0:
            per_head_indices: list[torch.Tensor] = []
            per_head_landmarks: list[torch.Tensor] = []
            for head_idx in range(self.num_kv_heads):
                head_mask = torch.zeros(chunks, dtype=torch.bool)
                head_mask[outlier_chunks[head_idx]] = True
                head_indices = all_chunk_indices[~head_mask]
                per_head_indices.append(head_indices)
                per_head_landmarks.append(
                    landmarks[head_idx].index_select(0, head_indices)
                )
            landmark_indices = torch.stack(per_head_indices, dim=0)
            landmarks = torch.stack(per_head_landmarks, dim=0)
        else:
            landmark_indices = all_chunk_indices.unsqueeze(0).expand(
                self.num_kv_heads, -1
            )
        entry["landmark_indices"] = landmark_indices
        entry["landmarks"] = host_tensor(landmarks)
        entry["outlier_chunks"] = (
            outlier_chunks.to(torch.int32)
            if outlier_count > 0
            else torch.empty(
                (self.num_kv_heads, 0), dtype=torch.int32
            )
        )
        entry["candidate_chunks"] = int(landmarks.shape[1])
        entry["local_tokens"] = int(prompt_len - chunks * chunk_size)
        entry["prompt_len"] = int(prompt_len)
        entry["shadow"] = True
        entry["raw_k"] = None
        if self._shadow_gpu_cache_enabled:
            if gpu_k is None or gpu_v is None:
                gpu_capacity = int(self.config.shadowkv_gpu_cache_tokens)
                if prompt_len > gpu_capacity:
                    raise RuntimeError(
                        "ShadowKV GPU cache capacity is smaller than the prompt: "
                        f"prompt_len={prompt_len} capacity={gpu_capacity}. Increase "
                        "shadowkv_gpu_cache_tokens or use shadowkv_storage='cpu'."
                    )
                gpu_k = torch.empty(
                    (gpu_capacity, self.num_kv_heads, self.head_dim),
                    dtype=self._shadow_dtype,
                    device=self.device,
                )
                gpu_v = torch.empty_like(gpu_k)
                gpu_k[:prompt_len].copy_(rope)
                gpu_v[:prompt_len].copy_(entry["v"][:prompt_len])
            entry["gpu_rope_k"] = gpu_k
            entry["gpu_v"] = gpu_v
        if matrix is not None:
            del matrix
        if not gpu_cache_fast_path:
            del u, s, vh, sv

    @torch.no_grad()
    def _finalize_entries_batched(
        self,
        rows: list[int],
        layer_idx: int,
        prompt_len: int,
    ) -> None:
        """Factorize equal-length prompts together, preserving exact SVD math."""
        pending = [
            int(row)
            for row in rows
            if not (
                self._entry(int(row), layer_idx).get("shadow", False)
                and int(self._entry(int(row), layer_idx).get("prompt_len", 0))
                == int(prompt_len)
            )
        ]
        if not pending:
            return
        if self._shadow_gpu_cache_enabled and self.device.type == "cuda":
            # GPU-cache decode never consumes U/SV. Finalize rows independently
            # so the batched CPU/GPU SVD workspace is not materialized at all.
            for row_idx in pending:
                self._finalize_entry(row_idx, layer_idx, prompt_len)
            return
        for row_idx in pending:
            entry = self._entry(row_idx, layer_idx)
            if int(entry["filled"]) < int(prompt_len):
                raise RuntimeError(
                    "ShadowKV prefill did not populate the complete CPU shadow: "
                    f"row={row_idx} layer={layer_idx} filled={entry['filled']} "
                    f"prompt={prompt_len}."
                )
        matrices = torch.stack(
            [
                self._entry(row_idx, layer_idx)["raw_k"][:prompt_len]
                .float()
                .reshape(prompt_len, -1)
                for row_idx in pending
            ],
            dim=0,
        ).to(self.device)
        # The default is still the exact full SVD used by the single-request
        # path; only independent requests are evaluated in one batched cuSOLVER
        # call.  The caller limits the group size to bound workspace memory.
        with profiler.record("shadowkv_prefill_svd_batched"):
            if str(getattr(self.config, "shadowkv_svd_method", "exact")) == "exact":
                u, s, vh = torch.linalg.svd(matrices, full_matrices=False)
            else:
                matrix_rows = int(matrices.shape[-2])
                matrix_cols = int(matrices.shape[-1])
                rank = min(int(self.config.shadowkv_rank), matrix_rows, matrix_cols)
                q = min(
                    min(matrix_rows, matrix_cols),
                    rank + int(getattr(self.config, "shadowkv_svd_oversample", 16)),
                )
                u, s, v = torch.svd_lowrank(
                    matrices,
                    q=q,
                    niter=int(getattr(self.config, "shadowkv_svd_niter", 2)),
                )
                vh = v.transpose(-2, -1)
        for batch_idx, row_idx in enumerate(pending):
            self._finalize_entry(
                row_idx,
                layer_idx,
                prompt_len,
                factorization=(u[batch_idx], s[batch_idx], vh[batch_idx]),
            )
        del matrices, u, s, vh

    @torch.no_grad()
    def finalize_shadowkv_prefill(self, seqs: list[Sequence]) -> None:
        if not seqs:
            return
        pending: list[tuple[int, int]] = []
        for seq in seqs:
            if not seq.is_last_chunk_prefill:
                continue
            row_idx = self.seq_id_to_row.get(int(seq.seq_id))
            if row_idx is None:
                raise RuntimeError(f"ShadowKV row is missing for seq_id={seq.seq_id}.")
            prompt_len = int(seq.num_prompt_tokens)
            pending.append((int(row_idx), prompt_len))
        if not pending:
            return
        batch_limit = max(1, int(getattr(self.config, "shadowkv_svd_batch_size", 4)))
        for layer_idx in range(self.num_layers):
            if not self.is_full_attention_layer(layer_idx):
                continue
            by_length: dict[int, list[int]] = {}
            for row_idx, prompt_len in pending:
                by_length.setdefault(int(prompt_len), []).append(int(row_idx))
            for prompt_len, rows in by_length.items():
                for start in range(0, len(rows), batch_limit):
                    group = rows[start : start + batch_limit]
                    if len(group) == 1:
                        self._finalize_entry(group[0], layer_idx, prompt_len)
                    else:
                        self._finalize_entries_batched(group, layer_idx, prompt_len)

    def _ensure_gpu_buffer(self, kind: str, rows: int, width: int) -> torch.Tensor:
        needed = int(rows) * int(width)
        attr = "_shadow_prefill_k" if kind == "prefill_k" else "_shadow_prefill_v"
        if kind.startswith("decode"):
            attr = "_shadow_decode_k" if kind == "decode_k" else "_shadow_decode_v"
        current = getattr(self, attr)
        if current is None or int(current.shape[0]) < needed:
            tensor = torch.empty(
                (max(1, needed), self.num_kv_heads, self.head_dim),
                dtype=self._shadow_dtype,
                device=self.device,
            )
            setattr(self, attr, tensor)
        return getattr(self, attr)

    def _ensure_decode_workspace(
        self,
        layer_idx: int,
        batch_size: int,
        context_capacity: int,
    ) -> dict[str, torch.Tensor]:
        """Allocate one reusable, graph-stable ShadowKV decode workspace.

        All tensors used by the batched selector and reconstruction live here.
        In particular, no ``topk`` result or explicit KV payload is allocated
        on the decode hot path.  The workspace is intentionally per layer:
        this bounds peak temporary memory while keeping every captured layer's
        addresses stable.
        """
        batch_size = int(batch_size)
        context_capacity = int(context_capacity)
        chunk_size = int(self.config.shadowkv_chunk_size)
        budget_tokens = int(self.config.shadowkv_sparse_budget)
        select_sets = max(1, budget_tokens // chunk_size)
        raw_candidate_chunks = max(
            select_sets,
            (context_capacity + chunk_size - 1) // chunk_size,
        )
        max_outlier_chunks = resolve_shadowkv_outlier_chunks(
            budget_tokens,
            getattr(self.config, "shadowkv_outlier_chunks", None),
        )
        max_local_tokens = min(
            context_capacity,
            int(self.config.shadowkv_local_chunks) * chunk_size + chunk_size - 1,
        )
        max_recent_tokens = min(context_capacity, int(self.config.shadowkv_recent_tokens))
        query_heads = int(getattr(self.hf_config, "num_attention_heads", self.num_kv_heads))
        raw_width = (
            max_outlier_chunks * chunk_size
            + budget_tokens
            + max_local_tokens
            + max_recent_tokens
        )
        decode_page_size = int(
            getattr(self.config, "shadowkv_decode_page_size", 16) or 16
        )
        width = ((raw_width + decode_page_size - 1) // decode_page_size) * decode_page_size
        cutlass_requested = str(
            getattr(self.config, "shadowkv_kernel_backend", "auto")
        ).lower() == "cutlass" or (
            str(getattr(self.config, "shadowkv_kernel_backend", "auto")).lower()
            == "auto"
            and bool(getattr(self.config, "shadowkv_cutlass_root", None))
        )
        # ShadowKV's original CUTLASS BatchGemmSoftmax uses a 256-column
        # threadblock tile and rejects a tail such as 532 columns.  Pad only
        # the candidate workspace; candidate_counts still carries the true
        # landmark count, so the extra zero rows are excluded from top-k.
        max_candidate_chunks = (
            ((raw_candidate_chunks + 255) // 256) * 256
            if cutlass_requested
            else raw_candidate_chunks
        )
        # Eager decode visits layers serially, so one workspace is sufficient
        # and avoids retaining a full sparse payload for every transformer
        # layer.  CUDA Graph replay needs one address-stable workspace per
        # layer because all layers are captured together.
        workspace_key = int(layer_idx) if bool(getattr(self.config, "decode_graph", False)) else -1
        current = self._shadow_decode_workspaces.get(workspace_key)
        expected = {
            "landmarks": (
                batch_size,
                self.num_kv_heads,
                max_candidate_chunks,
                self.head_dim,
            ),
            "score_workspace": (
                batch_size,
                self.num_kv_heads,
                max(
                    1,
                    query_heads // int(self.num_kv_heads),
                ),
                max_candidate_chunks,
            ),
            "score_query": (
                batch_size,
                self.num_kv_heads,
                max(
                    1,
                    query_heads // int(self.num_kv_heads),
                ),
                self.head_dim,
            ),
            "score_probabilities": (
                batch_size,
                self.num_kv_heads,
                max(
                    1,
                    query_heads // int(self.num_kv_heads),
                ),
                max_candidate_chunks,
            ),
            "score_norm": (
                batch_size,
                self.num_kv_heads,
                ((max_candidate_chunks + 255) // 256)
                * max(1, query_heads // int(self.num_kv_heads)),
            ),
            "score_sum": (
                batch_size,
                self.num_kv_heads,
                ((max_candidate_chunks + 255) // 256)
                * max(1, query_heads // int(self.num_kv_heads)),
            ),
            "sv": (batch_size, self.num_kv_heads, int(self.config.shadowkv_rank), self.head_dim),
            "outlier_chunks": (
                batch_size,
                self.num_kv_heads,
                max_outlier_chunks,
            ),
            "outlier_counts": (batch_size, self.num_kv_heads),
            "candidate_counts": (batch_size, self.num_kv_heads),
            "candidate_chunk_indices": (
                batch_size,
                self.num_kv_heads,
                max_candidate_chunks,
            ),
            "prompt_lens": (batch_size,),
            "local_tokens": (batch_size,),
            "positions": (batch_size, width),
            "active_slots": (batch_size, width),
            "local_req_indices": (batch_size,),
            "selected_lens": (batch_size,),
            "selected_chunks": (batch_size, select_sets),
            "selected_valid": (batch_size, self.num_kv_heads, select_sets),
            "outlier_mask": (batch_size, max_candidate_chunks),
            "k_cache": (batch_size * width, self.num_kv_heads, self.head_dim),
            "v_cache": (batch_size * width, self.num_kv_heads, self.head_dim),
            "recent_k": (
                batch_size,
                min(int(self.config.shadowkv_recent_tokens), context_capacity),
                self.num_kv_heads,
                self.head_dim,
            ),
            "recent_v": (
                batch_size,
                min(int(self.config.shadowkv_recent_tokens), context_capacity),
                self.num_kv_heads,
                self.head_dim,
            ),
            "head_positions": (batch_size, self.num_kv_heads, width),
            "head_v_positions": (batch_size, self.num_kv_heads, width),
            "head_context_lens": (batch_size, self.num_kv_heads),
            "head_source_lens": (batch_size, self.num_kv_heads),
            "head_prompt_lens": (batch_size, self.num_kv_heads),
            "head_selected_positions": (
                batch_size,
                self.num_kv_heads,
                select_sets * chunk_size,
            ),
            "head_selected_valid": (
                batch_size,
                self.num_kv_heads,
                select_sets * chunk_size,
            ),
            "head_selected_dest": (
                batch_size,
                self.num_kv_heads,
                select_sets * chunk_size,
            ),
            "head_k_cache": (
                batch_size,
                self.num_kv_heads,
                width,
                self.head_dim,
            ),
            "head_v_cache": (
                batch_size,
                self.num_kv_heads,
                width,
                self.head_dim,
            ),
            "head_reconstructed": (
                batch_size,
                self.num_kv_heads,
                select_sets * chunk_size,
                self.head_dim,
            ),
            "host_k_ptrs": (batch_size,),
            "host_v_ptrs": (batch_size,),
            "host_u_ptrs": (batch_size,),
            "head_host_u_ptrs": (batch_size * self.num_kv_heads,),
            # Reused integer ranges keep the graph/eager hot path from
            # allocating a new arange tensor for every layout segment.
            "candidate_ids": (max_candidate_chunks,),
            "chunk_offsets": (chunk_size,),
            "outlier_token_offsets": (max_outlier_chunks * chunk_size,),
            "selected_token_offsets": (select_sets * chunk_size,),
            "local_token_offsets": (max_local_tokens,),
            "recent_token_offsets": (max_recent_tokens,),
        }
        if cutlass_requested:
            expected.update(
                {
                    "u_device": (
                        batch_size,
                        context_capacity,
                        int(self.config.shadowkv_rank),
                    ),
                    "sv_column_major": (
                        batch_size,
                        self.num_kv_heads,
                        self.head_dim,
                        int(self.config.shadowkv_rank),
                    ),
                }
            )
        else:
            expected["head_u_selected"] = (
                batch_size,
                self.num_kv_heads,
                select_sets * chunk_size,
                int(self.config.shadowkv_rank),
            )
        offset_copy_enabled = bool(
            getattr(self.config, "shadowkv_gather_copy_with_offsets", True)
        ) and not self._shadow_gpu_cache_enabled
        offset_workspace_in_manager = offset_copy_enabled and bool(
            getattr(self.config, "decode_graph", False)
        )
        if offset_workspace_in_manager:
            expected.update(
                {
                    "offset_current_ids": (
                        batch_size,
                        self.num_kv_heads,
                        select_sets,
                    ),
                    "offset_cached_ids": (
                        batch_size,
                        self.num_kv_heads,
                        select_sets,
                    ),
                    "offset_reordered_ids": (
                        batch_size,
                        self.num_kv_heads,
                        select_sets,
                    ),
                    "offsets": (batch_size * self.num_kv_heads * select_sets,),
                    "offset_counts": (batch_size * self.num_kv_heads,),
                    "offset_signals": (batch_size * self.num_kv_heads,),
                    "offset_v_cache": (
                        batch_size,
                        self.num_kv_heads,
                        select_sets,
                        int(self.config.shadowkv_chunk_size) * self.head_dim,
                    ),
                    "offset_temp": (
                        batch_size,
                        self.num_kv_heads,
                        select_sets,
                        int(self.config.shadowkv_chunk_size) * self.head_dim,
                    ),
                    "offset_host_v_ptrs": (batch_size * self.num_kv_heads,),
                }
            )
        if self._shadow_gpu_cache_enabled:
            expected.update(
                {
                    # Each request/layer owns its GPU cache entry.  Keep only
                    # device pointer tables in the graph workspace; copying
                    # the full 128K K/V cache into every layer workspace would
                    # duplicate roughly 64 GiB at batch 4.
                    "gpu_cache_k_ptrs": (batch_size,),
                    "gpu_cache_v_ptrs": (batch_size,),
                }
            )
        if current is not None and all(
            name in current and tuple(current[name].shape) == shape
            for name, shape in expected.items()
        ):
            return current
        workspace: dict[str, torch.Tensor] = {}
        for name, shape in expected.items():
            dtype = torch.int32 if name in {
                "outlier_chunks", "outlier_counts", "candidate_counts", "prompt_lens", "local_tokens",
                "candidate_chunk_indices", "positions", "active_slots", "local_req_indices", "selected_lens", "selected_chunks", "outlier_mask",
                "candidate_ids", "chunk_offsets", "local_token_offsets", "recent_token_offsets",
                "selected_valid", "head_positions", "head_v_positions", "head_context_lens", "head_source_lens", "head_prompt_lens",
                "head_selected_positions", "head_selected_dest",
                "head_selected_valid",
                "offsets", "offset_counts", "offset_signals",
            } else torch.float32 if name in {"score_norm", "score_sum"} else torch.int64 if (
                name.endswith("_ptrs")
                or name.endswith("_offsets")
                or name in {"offset_current_ids", "offset_cached_ids", "offset_reordered_ids"}
            ) else self._shadow_dtype
            if name in {"outlier_mask", "selected_valid", "head_selected_valid"}:
                dtype = torch.bool
            workspace[name] = torch.empty(shape, dtype=dtype, device=self.device)
        self._shadow_decode_workspaces[workspace_key] = workspace
        # ``k_cache``/``v_cache`` are flattened as [batch * width, ...].  Each
        # request therefore needs its own row-local page table; sharing
        # [0..width) across rows silently makes every request attend to row 0
        # and can make FlashInfer read an unrelated page during graph replay.
        active_slots = torch.arange(
            batch_size * width, dtype=torch.int32, device=self.device
        ).view(batch_size, width)
        workspace["active_slots"].copy_(active_slots)
        workspace["local_req_indices"].copy_(
            torch.arange(batch_size, dtype=torch.int32, device=self.device)
        )
        workspace["candidate_ids"].copy_(
            torch.arange(max_candidate_chunks, dtype=torch.int32, device=self.device)
        )
        workspace["chunk_offsets"].copy_(
            torch.arange(chunk_size, dtype=torch.int32, device=self.device)
        )
        workspace["outlier_token_offsets"].copy_(
            torch.arange(max_outlier_chunks * chunk_size, dtype=torch.int64, device=self.device)
        )
        workspace["selected_token_offsets"].copy_(
            torch.arange(select_sets * chunk_size, dtype=torch.int64, device=self.device)
        )
        workspace["local_token_offsets"].copy_(
            torch.arange(max_local_tokens, dtype=torch.int32, device=self.device)
        )
        workspace["recent_token_offsets"].copy_(
            torch.arange(max_recent_tokens, dtype=torch.int32, device=self.device)
        )
        workspace["head_positions"].fill_(-1)
        workspace["head_v_positions"].fill_(-1)
        workspace["head_context_lens"].zero_()
        workspace["head_source_lens"].zero_()
        workspace["head_prompt_lens"].zero_()
        workspace["head_selected_positions"].fill_(-1)
        workspace["head_selected_valid"].zero_()
        workspace["head_selected_dest"].zero_()
        if offset_workspace_in_manager:
            workspace["offset_current_ids"].zero_()
            workspace["offset_cached_ids"].fill_(-1)
            workspace["offset_reordered_ids"].fill_(-1)
            workspace["offsets"].zero_()
            workspace["offset_counts"].zero_()
            workspace["offset_signals"].zero_()
            workspace["offset_v_cache"].zero_()
            workspace["offset_temp"].zero_()
        if self._shadow_gpu_cache_enabled:
            workspace["gpu_cache_k_ptrs"].zero_()
            workspace["gpu_cache_v_ptrs"].zero_()
        return workspace

    def _load_shadowkv_kernel_backend(self):
        """Resolve the requested host-gather/CUTLASS implementation once.

        ``auto`` is intentionally conservative: it selects CUTLASS only when
        the user supplied a CUTLASS root, otherwise it keeps the tested host
        gather extension.  An explicit ``cutlass`` request never falls back.
        """
        if self._shadow_host_gather is not None:
            return self._shadow_host_gather
        configured = str(getattr(self.config, "shadowkv_kernel_backend", "auto")).lower()
        root = getattr(self.config, "shadowkv_cutlass_root", None)
        if configured == "cutlass" or (configured == "auto" and root):
            from sparsevllm.kernels.shadowkv_cutlass import load_shadowkv_cutlass

            self._shadow_host_gather = load_shadowkv_cutlass(root)
            self._shadow_kernel_backend = "cutlass"
            return self._shadow_host_gather
        from sparsevllm.kernels.shadowkv_host_gather import load_shadowkv_host_gather

        self._shadow_host_gather = load_shadowkv_host_gather()
        self._shadow_kernel_backend = "torch_host_gather"
        return self._shadow_host_gather

    def _load_shadowkv_offset_gather(self):
        """Load the ShadowKV-origin offset-copy extension when enabled."""
        if not bool(getattr(self.config, "shadowkv_gather_copy_with_offsets", True)):
            return None
        select_sets = int(self.config.shadowkv_sparse_budget) // int(
            self.config.shadowkv_chunk_size
        )
        if select_sets not in {128, 256, 512, 1024}:
            raise ValueError(
                "ShadowKV offset-copy requires sparse_budget/chunk_size to be "
                f"one of 128, 256, 512, or 1024; got {select_sets}. "
                "Disable shadowkv_gather_copy_with_offsets for another budget."
            )
        if self._shadow_offset_gather is None:
            from sparsevllm.kernels.shadowkv_host_gather import load_shadowkv_host_gather

            self._shadow_offset_gather = load_shadowkv_host_gather()
            configure = getattr(
                self._shadow_offset_gather,
                "configure_gather_copy_with_offsets",
                None,
            )
            if configure is not None:
                configure(
                    int(self.config.shadowkv_chunk_size),
                    int(self.head_dim),
                    int(select_sets),
                )
        return self._shadow_offset_gather

    def _load_shadowkv_gpu_cache_gather(self):
        """Load the graph-safe direct GPU-cache lookup kernel once."""
        if self._shadow_gpu_cache_gather is None:
            from sparsevllm.kernels.shadowkv_host_gather import load_shadowkv_host_gather

            self._shadow_gpu_cache_gather = load_shadowkv_host_gather()
        return self._shadow_gpu_cache_gather

    def _ensure_offset_workspace(
        self,
        layer_idx: int,
        rows: list[int],
    ) -> dict[str, Any]:
        """Allocate the per-layer selected-V cache used by offset-copy."""
        batch_size = len(rows)
        select_sets = int(self.config.shadowkv_sparse_budget) // int(
            self.config.shadowkv_chunk_size
        )
        chunk_dim = int(self.config.shadowkv_chunk_size) * self.head_dim
        current = self._shadow_offset_workspaces.get(int(layer_idx))
        if current is not None and current.get("rows") == tuple(rows):
            return current
        device = self.device
        state: dict[str, Any] = {
            "rows": tuple(rows),
            "offset_current_ids": torch.empty(
                batch_size, self.num_kv_heads, select_sets,
                dtype=torch.int64, device=device
            ),
            "offset_cached_ids": torch.full(
                (batch_size, self.num_kv_heads, select_sets), -1,
                dtype=torch.int64, device=device
            ),
            "offset_reordered_ids": torch.full(
                (batch_size, self.num_kv_heads, select_sets), -1,
                dtype=torch.int64, device=device
            ),
            "offsets": torch.empty(
                batch_size * self.num_kv_heads * select_sets,
                dtype=torch.int32, device=device
            ),
            "offset_counts": torch.empty(
                batch_size * self.num_kv_heads,
                dtype=torch.int32, device=device
            ),
            "offset_signals": torch.zeros(
                batch_size * self.num_kv_heads,
                dtype=torch.int32, device=device
            ),
            "offset_v_cache": torch.zeros(
                batch_size, self.num_kv_heads, select_sets, chunk_dim,
                dtype=self._shadow_dtype, device=device
            ),
            "offset_temp": torch.empty(
                batch_size, self.num_kv_heads, select_sets, chunk_dim,
                dtype=self._shadow_dtype, device=device
            ),
            "offset_host_v_ptrs": torch.empty(
                batch_size * self.num_kv_heads,
                dtype=torch.int64, device=device
            ),
        }
        self._shadow_offset_workspaces[int(layer_idx)] = state
        return state

    def _launch_async_value_gather(
        self,
        workspace: dict[str, torch.Tensor],
        positions: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        *,
        per_head: bool = False,
    ):
        """Overlap mapped-host V gather with current-stream K reconstruction.

        CUDA Graph replay stays on one stream.  Eager decode opts into a
        dedicated copy stream and joins it only after the reconstruction work
        that can overlap with the value gather has been enqueued.
        """
        if (
            bool(getattr(self.config, "decode_graph", False))
            or not bool(getattr(self.config, "shadowkv_multistream_gather", True))
        ):
            gather = (
                self._shadow_host_gather.gather_host_per_head
                if per_head
                else self._shadow_host_gather.gather_host
            )
            gather(workspace["host_v_ptrs"], positions, lengths, output)
            return None
        if self._shadow_copy_stream is None:
            self._shadow_copy_stream = torch.cuda.Stream(device=self.device)
            self._shadow_copy_ready_event = torch.cuda.Event()
            self._shadow_copy_done_event = torch.cuda.Event()
        current = torch.cuda.current_stream(device=self.device)
        ready = self._shadow_copy_ready_event
        done = self._shadow_copy_done_event
        if ready is None or done is None:
            raise RuntimeError("ShadowKV copy-stream events were not initialized.")
        ready.record(current)
        self._shadow_copy_stream.wait_event(ready)
        with torch.cuda.stream(self._shadow_copy_stream):
            gather = (
                self._shadow_host_gather.gather_host_per_head
                if per_head
                else self._shadow_host_gather.gather_host
            )
            gather(workspace["host_v_ptrs"], positions, lengths, output)
        done.record(self._shadow_copy_stream)
        return done

    def _workspace_for_layer(self, layer_idx: int) -> dict[str, torch.Tensor] | None:
        if bool(getattr(self.config, "decode_graph", False)):
            return self._shadow_decode_workspaces.get(int(layer_idx))
        return self._shadow_decode_workspaces.get(-1)

    def _workspace_content_signature(
        self,
        layer_idx: int,
        rows: list[int],
    ) -> tuple[object, ...]:
        """Identify the host-side factorization currently staged for ``rows``."""
        signature: list[object] = []
        for row in rows:
            entry = self._entry(row, layer_idx)
            signature.append(
                (
                    int(row),
                    id(entry),
                    int(entry.get("prompt_len", 0)),
                    int(entry["u"].data_ptr()) if entry.get("u") is not None else 0,
                    int(entry["sv"].data_ptr()) if entry.get("sv") is not None else 0,
                    int(entry["landmarks"].data_ptr())
                    if entry.get("landmarks") is not None
                    else 0,
                    int(entry["outlier_chunks"].data_ptr())
                    if entry.get("outlier_chunks") is not None
                    else 0,
                    int(entry["gpu_rope_k"].data_ptr())
                    if entry.get("gpu_rope_k") is not None
                    else 0,
                    int(entry["gpu_v"].data_ptr())
                    if entry.get("gpu_v") is not None
                    else 0,
                )
            )
        return tuple(signature)

    def _refresh_host_pointers(
        self,
        layer_idx: int,
        rows: list[int],
        workspace: dict[str, torch.Tensor],
        *,
        offset_workspace: dict[str, Any] | None = None,
    ) -> None:
        """Publish the current layer's pinned-host addresses to the gather kernel.

        Eager decode reuses one workspace while visiting layers serially, and
        host KV tensors can grow when generated tokens are appended.  The
        pointer table must therefore be refreshed for every eager layer call;
        graph decode has one immutable workspace and pointer table per layer.
        """
        if self._shadow_host_gather is None:
            raise RuntimeError("ShadowKV host-gather extension is not loaded.")
        k_bases = [self._entry(row, layer_idx)["rope_k"] for row in rows]
        v_bases = [self._entry(row, layer_idx)["v"] for row in rows]
        u_bases = [self._entry(row, layer_idx)["u"] for row in rows]
        k_sources = k_bases
        v_sources = v_bases
        u_sources = [source.unsqueeze(1) for source in u_bases]
        head_u_sources = [
            source.unsqueeze(1)
            for source in u_bases
            for _ in range(self.num_kv_heads)
        ]
        offset_sources = None
        if self._shadow_offset_gather is not None:
            offset_sources = [
                self._entry(row, layer_idx)["v_chunks"][head].unsqueeze(1)
                for row in rows
                for head in range(self.num_kv_heads)
            ]
        # Pointer publication is a driver call plus a small H2D copy.  Host
        # tensors are appended in place during normal decode, so their
        # addresses remain valid until a growth/reallocation occurs.  Include
        # the pointer-table address so a newly allocated workspace cannot
        # accidentally inherit a cache entry for an older table.
        signature = (
            int(layer_idx),
            tuple(int(row) for row in rows),
            tuple((id(source), int(source.data_ptr())) for source in k_bases),
            tuple((id(source), int(source.data_ptr())) for source in v_bases),
            tuple((id(source), int(source.data_ptr())) for source in u_bases),
            int(workspace["host_k_ptrs"].data_ptr()),
            int(workspace["head_host_u_ptrs"].data_ptr()),
            tuple((id(source), int(source.data_ptr())) for source in offset_sources)
            if offset_sources is not None
            else None,
            int(offset_workspace["offset_host_v_ptrs"].data_ptr())
            if offset_workspace is not None
            else 0,
        )
        if self._shadow_host_pointer_signatures.get(int(layer_idx)) == signature:
            return
        self._shadow_host_gather.set_host_pointers(k_sources, workspace["host_k_ptrs"])
        self._shadow_host_gather.set_host_pointers(v_sources, workspace["host_v_ptrs"])
        self._shadow_host_gather.set_host_pointers(u_sources, workspace["host_u_ptrs"])
        self._shadow_host_gather.set_host_pointers(
            head_u_sources, workspace["head_host_u_ptrs"]
        )
        if offset_sources is not None:
            self._shadow_offset_gather.set_host_pointers(
                offset_sources, offset_workspace["offset_host_v_ptrs"]
            )
        self._shadow_host_pointer_signatures[int(layer_idx)] = signature

    def _shadow_graph_batch_rows(
        self,
        rows: list[int],
        context_capacity: int,
        batch_capacity: int,
    ) -> tuple[list[int], list[int]]:
        if not rows:
            raise RuntimeError("ShadowKV requires a non-empty decode batch.")
        rows = list(rows)
        lengths = [int(self.row_seq_lens[row]) for row in rows]
        if len(rows) < int(batch_capacity):
            rows.extend([rows[0]] * (int(batch_capacity) - len(rows)))
            lengths.extend([lengths[0]] * (int(batch_capacity) - len(lengths)))
        if any(length > int(context_capacity) for length in lengths):
            raise RuntimeError(
                "ShadowKV decode graph context capacity is too small: "
                f"capacity={context_capacity} lengths={lengths}."
            )
        return rows, lengths

    def _stage_decode_workspace(
        self,
        layer_idx: int,
        rows: list[int],
        context_lens: list[int],
        *,
        context_capacity: int,
    ) -> dict[str, torch.Tensor]:
        workspace = self._ensure_decode_workspace(
            layer_idx, len(rows), int(context_capacity)
        )
        max_chunks = int(workspace["landmarks"].shape[2])
        workspace["landmarks"].zero_()
        workspace["sv"].zero_()
        if "u_device" in workspace:
            workspace["u_device"].zero_()
            workspace["sv_column_major"].zero_()
        workspace["outlier_chunks"].fill_(-1)
        workspace["outlier_counts"].zero_()
        workspace["candidate_counts"].zero_()
        workspace["candidate_chunk_indices"].fill_(-1)
        workspace["prompt_lens"].zero_()
        workspace["local_tokens"].zero_()
        workspace["selected_valid"].zero_()
        workspace["head_positions"].fill_(-1)
        workspace["head_v_positions"].fill_(-1)
        workspace["head_context_lens"].zero_()
        workspace["head_source_lens"].zero_()
        workspace["head_prompt_lens"].zero_()
        workspace["head_selected_positions"].fill_(-1)
        workspace["head_selected_valid"].zero_()
        workspace["head_selected_dest"].zero_()
        workspace["head_k_cache"].zero_()
        workspace["head_v_cache"].zero_()
        workspace["recent_k"].zero_()
        workspace["recent_v"].zero_()
        if self._shadow_gpu_cache_enabled:
            workspace["gpu_cache_k_ptrs"].zero_()
            workspace["gpu_cache_v_ptrs"].zero_()
        for batch_idx, row_idx in enumerate(rows):
            entry = self._entry(row_idx, layer_idx)
            required_host_tokens = min(
                int(context_lens[batch_idx]), int(entry.get("prompt_len", 0))
            )
            if int(entry["filled"]) < required_host_tokens:
                raise RuntimeError("ShadowKV decode workspace requested unwritten CPU shadow KV.")
            if not entry.get("shadow", False):
                raise RuntimeError(
                    "ShadowKV CUDA Graph requires a factorized prompt; "
                    f"row={row_idx} layer={layer_idx} prompt_len={entry.get('prompt_len', 0)}."
                )
            landmarks = entry["landmarks"]
            if landmarks.ndim != 3 or int(landmarks.shape[0]) != self.num_kv_heads:
                raise RuntimeError(
                    "ShadowKV entry landmarks must have shape [kv_heads, chunks, dim], "
                    f"got {tuple(landmarks.shape)}."
                )
            candidate_count = min(max_chunks, int(landmarks.shape[1]))
            workspace["landmarks"][batch_idx, :, :candidate_count].copy_(
                landmarks[:, :candidate_count].to(self.device), non_blocking=True
            )
            workspace["candidate_chunk_indices"][
                batch_idx, :, :candidate_count
            ].copy_(
                entry["landmark_indices"][:, :candidate_count].to(self.device),
                non_blocking=True,
            )
            prompt_len = min(int(context_capacity), int(entry["prompt_len"]))
            if not self._shadow_gpu_cache_enabled:
                sv = entry["sv"]
                if sv is None:
                    raise RuntimeError(
                        "ShadowKV CPU-shadow entry is missing its SV payload."
                    )
                workspace["sv"][batch_idx, :, : sv.shape[1]].copy_(
                    sv.to(self.device), non_blocking=True
                )
                if "u_device" in workspace:
                    sv_column_major = entry.get("sv_column_major")
                    if (
                        sv_column_major is None
                        or sv_column_major.ndim != 3
                        or tuple(sv_column_major.shape)
                        != (self.num_kv_heads, self.head_dim, int(self.config.shadowkv_rank))
                    ):
                        raise RuntimeError(
                            "ShadowKV entry is missing the contiguous column-major SV payload."
                        )
                    workspace["u_device"][
                        batch_idx, :prompt_len, : entry["u"].shape[1]
                    ].copy_(entry["u"][:prompt_len], non_blocking=True)
                    workspace["sv_column_major"][batch_idx].copy_(
                        sv_column_major, non_blocking=True
                    )
            outliers = entry["outlier_chunks"]
            if outliers.ndim != 2 or int(outliers.shape[0]) != self.num_kv_heads:
                raise RuntimeError(
                    "ShadowKV entry outlier chunks must have shape [kv_heads, chunks], "
                    f"got {tuple(outliers.shape)}."
                )
            outlier_count = min(
                int(outliers.shape[1]),
                int(workspace["outlier_chunks"].shape[2]),
            )
            if outlier_count:
                workspace["outlier_chunks"][
                    batch_idx, :, :outlier_count
                ].copy_(
                    outliers[:, :outlier_count].to(self.device),
                    non_blocking=True,
                )
            workspace["outlier_counts"][batch_idx].fill_(outlier_count)
            workspace["candidate_counts"][batch_idx].fill_(candidate_count)
            workspace["prompt_lens"][batch_idx] = prompt_len
            workspace["head_prompt_lens"][batch_idx].fill_(prompt_len)
            workspace["local_tokens"][batch_idx] = min(
                int(entry.get("local_tokens", 0)), prompt_len
            )
            if self._shadow_gpu_cache_enabled:
                gpu_k = entry.get("gpu_rope_k")
                gpu_v = entry.get("gpu_v")
                if gpu_k is None or gpu_v is None:
                    raise RuntimeError(
                        "ShadowKV GPU cache entry was not materialized before decode: "
                        f"row={row_idx} layer={layer_idx}."
                    )
                workspace["gpu_cache_k_ptrs"][batch_idx] = int(gpu_k.data_ptr())
                workspace["gpu_cache_v_ptrs"][batch_idx] = int(gpu_v.data_ptr())

        # Reuse the same mapped host pointers for every captured replay.  The
        # actual positions are graph inputs, so value/key gathering remains
        # dynamic without a Python or CPU operation in the captured region.
        offset_workspace = None
        if not self._shadow_gpu_cache_enabled:
            self._load_shadowkv_kernel_backend()
            if self._load_shadowkv_offset_gather() is not None:
                offset_workspace = (
                    workspace
                    if bool(getattr(self.config, "decode_graph", False))
                    else self._ensure_offset_workspace(layer_idx, rows)
                )
            self._refresh_host_pointers(
                layer_idx,
                rows,
                workspace,
                offset_workspace=offset_workspace,
            )
        else:
            configured_backend = str(
                getattr(self.config, "shadowkv_kernel_backend", "auto")
            ).lower()
            cutlass_requested = configured_backend == "cutlass" or (
                configured_backend == "auto"
                and bool(getattr(self.config, "shadowkv_cutlass_root", None))
            )
            if cutlass_requested:
                # CUTLASS is still used for landmark scoring in GPU-cache mode.
                # The direct GPU gather extension is loaded separately below.
                self._load_shadowkv_kernel_backend()
        if self._shadow_gpu_cache_enabled:
            self._load_shadowkv_gpu_cache_gather()
        self._shadow_workspace_signatures[int(layer_idx)] = (
            tuple(rows), int(context_capacity)
        )
        return workspace

    def build_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        del k_current, v_current
        rows = list(self._shadow_active_rows)
        batch_size = int(selection.context_lens.numel())
        if len(rows) != batch_size:
            rows = selection.req_indices.detach().cpu().tolist()
        lengths = selection.context_lens.detach().cpu().tolist()
        batch_size = len(rows)
        max_context_len = max(lengths, default=0)
        if max_context_len <= 0:
            raise RuntimeError("ShadowKV prefill requires positive context lengths.")
        # B200's FlashInfer CuTe-DSL paged prefill requires 16-token pages.
        # Padding is only physical storage; context_lens remains the exact
        # logical length, so the final page is masked by the provider.
        page_size = 16
        padded_max_len = ((max_context_len + page_size - 1) // page_size) * page_size
        k_cache = self._ensure_gpu_buffer("prefill_k", batch_size, padded_max_len)
        v_cache = self._ensure_gpu_buffer("prefill_v", batch_size, padded_max_len)
        active_slots = torch.empty(
            (batch_size, padded_max_len), dtype=torch.int32, device=self.device
        )
        for batch_idx, (row_idx, length) in enumerate(zip(rows, lengths)):
            entry = self._entry(row_idx, layer_idx)
            if int(entry["filled"]) < int(length):
                raise RuntimeError("ShadowKV prefill view requested unwritten CPU shadow KV.")
            start = batch_idx * padded_max_len
            k_cache[start : start + length].copy_(
                entry["rope_k"][:length], non_blocking=True
            )
            v_cache[start : start + length].copy_(
                entry["v"][:length], non_blocking=True
            )
            active_slots[batch_idx] = torch.arange(
                start,
                start + padded_max_len,
                dtype=torch.int32,
                device=self.device,
            )
        # The payload and its page table are compacted to this prefill batch;
        # the cache row IDs in ``selection.req_indices`` must not be used to
        # index this local table (startup/profile batches can reuse sparse row
        # IDs after earlier temporary allocations).
        local_req_indices = torch.arange(
            batch_size, dtype=torch.int32, device=self.device
        )
        return PrefillComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=local_req_indices,
                context_lens=selection.context_lens,
                max_context_len=max_context_len,
                attn_score=selection.attn_score,
            ),
            payload=ExplicitKVPayload(k_cache=k_cache, v_cache=v_cache),
        )

    def _decode_positions_and_payload(
        self,
        layer_idx: int,
        q: torch.Tensor,
        rows: list[int],
        context_lens: list[int],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        q = q.view(len(rows), -1, self.head_dim)
        groups = int(q.shape[1]) // int(self.num_kv_heads)
        if groups <= 0 or int(q.shape[1]) % int(self.num_kv_heads):
            raise RuntimeError("ShadowKV requires query heads divisible by KV heads.")
        position_rows: list[torch.Tensor] = []
        key_rows: list[torch.Tensor] = []
        value_rows: list[torch.Tensor] = []
        del value_rows
        for batch_idx, (row_idx, context_len) in enumerate(zip(rows, context_lens)):
            entry = self._entry(row_idx, layer_idx)
            if int(entry["filled"]) < int(context_len):
                raise RuntimeError("ShadowKV decode view requested unwritten CPU shadow KV.")
            if not entry.get("shadow", False):
                positions = torch.arange(context_len, dtype=torch.long)
                keys = entry["rope_k"][:context_len]
            else:
                landmarks = entry["landmarks"].to(self.device)
                query = q[batch_idx].view(self.num_kv_heads, groups, self.head_dim).float()
                scores = torch.einsum("hgd,hnd->hgn", query, landmarks.float())
                scores = torch.softmax(scores * (self.head_dim ** -0.5), dim=-1)
                scores = scores.amax(dim=1).amax(dim=0)
                select_sets = min(
                    int(self.config.shadowkv_sparse_budget) // int(self.config.shadowkv_chunk_size),
                    int(scores.shape[-1]),
                )
                selected = torch.topk(scores, select_sets, dim=-1).indices.cpu()
                selected_chunks = entry["landmark_indices"].index_select(1, selected)
                selected_chunks = selected_chunks.reshape(-1).unique(sorted=True)
                outlier_chunks = entry["outlier_chunks"].reshape(-1).unique(sorted=True).cpu()
                chunk_size = int(self.config.shadowkv_chunk_size)
                prompt_len = int(entry["prompt_len"])
                aligned_chunks = max(0, prompt_len // chunk_size - int(self.config.shadowkv_local_chunks))
                aligned_chunks -= aligned_chunks % 8
                local_start = prompt_len - aligned_chunks * chunk_size
                def expand(chunks: torch.Tensor) -> torch.Tensor:
                    if chunks.numel() == 0:
                        return torch.empty(0, dtype=torch.long)
                    return (chunks[:, None] * chunk_size + torch.arange(chunk_size)[None, :]).reshape(-1)
                outlier_pos = expand(outlier_chunks)
                selected_pos = expand(selected_chunks)
                local_pos = torch.arange(local_start, prompt_len, dtype=torch.long)
                recent_start = max(prompt_len, int(context_len) - int(self.config.shadowkv_recent_tokens))
                recent_pos = torch.arange(recent_start, context_len, dtype=torch.long)
                positions = torch.cat((outlier_pos, selected_pos, local_pos, recent_pos)).unique(sorted=True)
                direct_mask = ~torch.isin(positions, selected_pos)
                keys = entry["rope_k"].index_select(0, positions[direct_mask])
                u = entry["u"].index_select(0, positions[~direct_mask]).to(self.device).float()
                sv = entry["sv"].to(self.device).float()
                reconstructed_raw = torch.einsum("pr,hrd->phd", u, sv).to(self._shadow_dtype)
                reconstructed = self._apply_shadow_rope(
                    reconstructed_raw,
                    positions[~direct_mask],
                )
                key_full = torch.empty(
                    (int(positions.numel()), self.num_kv_heads, self.head_dim),
                    dtype=self._shadow_dtype,
                    device=self.device,
                )
                key_full[direct_mask.to(self.device)] = keys.to(self.device)
                key_full[(~direct_mask).to(self.device)] = reconstructed
                keys = key_full
            position_rows.append(positions)
            key_rows.append(keys)
            # Values are gathered below from the CPU shadow, using the same
            # merged token set as the reconstructed keys.
        max_len = max((int(row.numel()) for row in position_rows), default=0)
        k_cache = self._ensure_gpu_buffer("decode_k", len(rows), max_len)
        v_cache = self._ensure_gpu_buffer("decode_v", len(rows), max_len)
        for batch_idx, (row_idx, positions, keys) in enumerate(zip(rows, position_rows, key_rows)):
            entry = self._entry(row_idx, layer_idx)
            length = int(positions.numel())
            k_cache[batch_idx * max_len : batch_idx * max_len + length].copy_(keys)
            v_cache[batch_idx * max_len : batch_idx * max_len + length].copy_(
                entry["v"].index_select(0, positions), non_blocking=True
            )
        return [
            torch.arange(i * max_len, (i + 1) * max_len, dtype=torch.int32, device=self.device)
            for i in range(len(rows))
        ], [torch.tensor(int(row.numel()), dtype=torch.int32, device=self.device) for row in position_rows]

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if self.device.type == "cuda":
            return self._build_batched_cuda_decode_view(
                layer_idx,
                q,
                selection,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        del num_heads, num_kv_heads
        rows = selection.req_indices.detach().cpu().tolist()
        full_context_lens = self.layer_batch_state.context_lens.detach().cpu().tolist()
        slot_rows, selected_lens = self._decode_positions_and_payload(
            layer_idx, q, rows, full_context_lens
        )
        max_len = max((int(length.item()) for length in selected_lens), default=0)
        active_slots = torch.empty((len(rows), max_len), dtype=torch.int32, device=self.device)
        for idx, row in enumerate(slot_rows):
            active_slots[idx] = row
        payload = ExplicitKVPayload(
            k_cache=self._shadow_decode_k,
            v_cache=self._shadow_decode_v,
        )
        # The materialized payload is local to this decode batch.  Its slot
        # table has exactly one row per query batch item, so using the cache's
        # global request indices here would index beyond the local table when
        # rows were allocated non-contiguously.
        local_req_indices = torch.arange(
            len(rows), dtype=torch.int32, device=self.device
        )
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=local_req_indices,
                context_lens=torch.stack(selected_lens),
                max_context_len=max_len,
                attn_score=selection.attn_score,
                is_sparse=True,
            ),
            payload=payload,
        )

    def init_decode_graph_state(
        self,
        contract: DecodeGraphContract,
        inputs: DecodeGraphInputs,
    ) -> CacheDecodeGraphState:
        """Use the standard graph inputs; ShadowKV owns only private workspaces."""
        inputs.validate(contract)
        return CacheDecodeGraphState(contract=contract, inputs=inputs)

    def decode_graph_path_capacity(self, is_long_text: bool) -> int:
        del is_long_text
        return int(self.config.max_model_len)

    def decode_graph_force_eager_for_batch(
        self,
        seqs: list[Sequence],
        *,
        is_long_text: bool,
    ) -> bool:
        del seqs, is_long_text
        # ShadowKV has one fixed-width per-head payload contract for both
        # short and long requests.  Its graph path does not depend on the
        # sparse controller's short/long token boundary, so forcing short
        # batches eager would make the registry's single ``shadowkv`` graph
        # path impossible to capture during startup profiling.
        return False

    @torch.no_grad()
    def prepare_decode_graph_step(
        self,
        seqs: list[Sequence],
        state: CacheDecodeGraphState,
    ):
        result = super().prepare_decode_graph_step(seqs, state)
        rows = [self.seq_id_to_row[int(seq.seq_id)] for seq in seqs]
        graph_rows, graph_lengths = self._shadow_graph_batch_rows(
            rows,
            int(state.contract.context_capacity),
            int(state.contract.batch_capacity),
        )
        signature = (
            tuple(graph_rows),
            int(state.contract.batch_capacity),
            int(state.contract.context_capacity),
            tuple(
                self._workspace_content_signature(layer_idx, graph_rows)
                for layer_idx in self.kv_transformer_layer_indices()
            ),
        )
        if self._shadow_workspace_signatures.get(-1) != signature:
            for layer_idx in self.kv_transformer_layer_indices():
                self._stage_decode_workspace(
                    layer_idx,
                    graph_rows,
                    graph_lengths,
                    context_capacity=int(state.contract.context_capacity),
                )
            self._shadow_workspace_signatures[-1] = signature
        return result

    def prepare_decode_graph_in(self, state: CacheDecodeGraphState) -> None:
        # The base implementation publishes graph-stable physical reservations.
        # ShadowKV's explicit view uses independent virtual slots, so no
        # additional metadata kernel is needed here.
        del state

    def decode_graph_state_keepalive_tensors(
        self,
        state: CacheDecodeGraphState,
    ) -> list[torch.Tensor]:
        del state
        tensors: list[torch.Tensor] = []
        for workspace in self._shadow_decode_workspaces.values():
            tensors.extend(workspace.values())
        return tensors

    @torch.no_grad()
    def _build_batched_cuda_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        """Build a fixed-width batched view using only graph-capturable GPU ops."""
        if int(num_kv_heads) != int(self.num_kv_heads):
            raise ValueError(
                "ShadowKV explicit view head mismatch: "
                f"operator={num_kv_heads} cache={self.num_kv_heads}."
            )
        batch_size = int(q.shape[0])
        trace = os.environ.get("SPARSEVLLM_SHADOWKV_TRACE", "0") == "1"
        if trace:
            logger.info(
                "ShadowKV trace layer={} enter batch={} q_shape={} graph={}",
                layer_idx,
                batch_size,
                tuple(q.shape),
                bool(torch.cuda.is_current_stream_capturing()),
            )
        if q.ndim != 3 or int(q.shape[1]) != int(num_heads):
            raise ValueError(
                "ShadowKV decode query must have shape [batch, query_heads, head_dim], "
                f"got {tuple(q.shape)} expected heads={num_heads}."
            )
        context_lens = selection.context_lens
        if context_lens.numel() != batch_size:
            raise ValueError(
                "ShadowKV batched decode metadata disagrees with the query: "
                f"q_batch={batch_size} lengths={context_lens.numel()}."
            )
        capturing = bool(torch.cuda.is_current_stream_capturing())
        rows = None if capturing else list(getattr(self, "_shadow_active_rows", ()))
        if not capturing and len(rows) != batch_size:
            rows = selection.req_indices.detach().cpu().tolist()
        context_capacity = int(getattr(self, "_decode_static_max_context_len", 0) or 0)
        if context_capacity <= 0:
            context_capacity = int(selection.max_context_len or max(
                int(value) for value in context_lens.detach().cpu().tolist()
            ))
        workspace = self._workspace_for_layer(layer_idx)
        workspace_needs_stage = (
            workspace is None or int(workspace["positions"].shape[0]) != batch_size
        )
        if workspace_needs_stage:
            if capturing or rows is None:
                raise RuntimeError(
                    "ShadowKV CUDA Graph replay reached an unprepared decode workspace."
                )
            context_lens_host = [int(value) for value in context_lens.detach().cpu().tolist()]
            workspace = self._stage_decode_workspace(
                layer_idx,
                rows,
                context_lens_host,
                context_capacity=context_capacity,
            )
        if not capturing:
            # Metadata is immutable over decode, but a new row can reuse the
            # same workspace between requests.  The signature avoids paying
            # the large H2D copies on every generated token.
            signature = (
                tuple(rows),
                int(context_capacity),
                self._workspace_content_signature(layer_idx, rows),
            )
            if (
                workspace_needs_stage
                or self._shadow_workspace_signatures.get(int(layer_idx)) != signature
            ):
                if not workspace_needs_stage:
                    workspace = self._stage_decode_workspace(
                        layer_idx,
                        rows,
                        [int(value) for value in context_lens.detach().cpu().tolist()],
                        context_capacity=context_capacity,
                    )
                self._shadow_workspace_signatures[int(layer_idx)] = signature

        if int(workspace["positions"].shape[0]) != batch_size:
            raise RuntimeError(
                "ShadowKV CUDA Graph workspace batch capacity changed after capture: "
                f"workspace={workspace['positions'].shape[0]} batch={batch_size}."
            )
        if int(workspace["positions"].shape[1]) <= 0:
            raise RuntimeError("ShadowKV decode workspace has zero token capacity.")
        if (
            not bool(getattr(self.config, "decode_graph", False))
            and not self._shadow_gpu_cache_enabled
        ):
            # The eager path shares workspace=-1 across all transformer
            # layers.  Refreshing here is required both for layer ownership
            # and for host tensors reallocated by append/grow.
            self._refresh_host_pointers(
                layer_idx,
                rows,
                workspace,
                offset_workspace=self._shadow_offset_workspaces.get(int(layer_idx)),
            )

        q_heads = q.reshape(batch_size, int(num_heads), self.head_dim)
        groups = int(num_heads) // int(self.num_kv_heads)
        if groups <= 0 or int(num_heads) % int(self.num_kv_heads):
            raise RuntimeError("ShadowKV requires query heads divisible by KV heads.")
        query_bf16 = q_heads.view(batch_size, self.num_kv_heads, groups, self.head_dim)
        query = query_bf16.float()
        landmarks = workspace["landmarks"]
        candidate_count = workspace["candidate_counts"][:batch_size]
        candidate_width = int(landmarks.shape[2])
        candidate_ids = workspace["candidate_ids"][:candidate_width]
        if self._shadow_kernel_backend == "cutlass":
            if query_bf16.dtype != torch.bfloat16:
                raise RuntimeError(
                    "ShadowKV CUTLASS score GEMM currently requires BF16 activations."
                )
            # Q is a view into the model projection output.  Its base pointer
            # is not part of the model contract, while the original CUTLASS
            # score operator uses 128-bit aligned vector loads.  Copy into a
            # graph-stable workspace whose allocator alignment is controlled
            # by us before entering the vectorized kernel.
            score_query = workspace["score_query"][:batch_size]
            score_query.copy_(query_bf16)
            logits = workspace["score_workspace"][:batch_size, :, :, :candidate_width]
            probabilities = workspace["score_probabilities"][
                :batch_size, :, :, :candidate_width
            ]
            self._shadow_host_gather.batch_gemm_softmax_exact(
                score_query,
                landmarks[:batch_size],
                logits,
                probabilities,
                workspace["score_norm"][:batch_size],
                workspace["score_sum"][:batch_size],
                float(self.head_dim ** -0.5),
            )
            scores = probabilities
        else:
            scores = torch.einsum(
                "bhgd,bhnd->bhgn",
                query,
                landmarks[:batch_size].float(),
            ) * float(self.head_dim ** -0.5)
        candidate_mask = candidate_ids[None, None, :] < candidate_count[:, :, None]
        if self._shadow_kernel_backend == "cutlass":
            # The CUTLASS entry point already emits row-softmax probabilities.
            # Masking and applying softmax again would change the selector's
            # ranking, especially when the static workspace has padding.
            scores = scores.masked_fill(~candidate_mask[:, :, None, :], 0)
            # Keep the KV-head axis: each KV head has its own query-group
            # ranking and therefore its own compact payload positions.
            chunk_scores = scores.amax(dim=2)
        else:
            scores = scores.masked_fill(
                ~candidate_mask[:, :, None, :], torch.finfo(scores.dtype).min
            )
            chunk_scores = torch.softmax(scores, dim=-1).amax(dim=2)
        if trace:
            logger.info(
                "ShadowKV trace layer={} score_done candidates={} backend={}",
                layer_idx,
                candidate_width,
                self._shadow_kernel_backend,
            )
        select_sets = int(self.config.shadowkv_sparse_budget) // int(self.config.shadowkv_chunk_size)
        selected = torch.topk(chunk_scores, k=select_sets, dim=-1, largest=True).indices
        selected_valid = torch.gather(
            candidate_mask,
            2,
            selected,
        )
        # Masked candidates receive the lowest representable score, while a
        # real softmax score is strictly positive.  Therefore topk already
        # returns a valid prefix followed by invalid entries; an argsort here
        # would add a full per-request sort to every generated token.
        workspace["selected_chunks"][:batch_size].copy_(selected[:, 0].to(torch.int32))
        workspace["selected_valid"][:batch_size].copy_(selected_valid)
        candidate_chunk_indices = workspace["candidate_chunk_indices"][:batch_size]
        selected_chunk_ids = torch.gather(
            candidate_chunk_indices,
            2,
            selected.to(dtype=torch.long),
        )
        head_count = int(self.num_kv_heads)
        offset_workspace = None
        if (
            self._shadow_offset_gather is not None
            and not self._shadow_gpu_cache_enabled
        ):
            offset_workspace = (
                workspace
                if bool(getattr(self.config, "decode_graph", False))
                else self._shadow_offset_workspaces.get(int(layer_idx))
            )
            if offset_workspace is None:
                raise RuntimeError(
                    "ShadowKV offset-copy workspace was not prepared for decode."
                )
            offset_workspace["offset_current_ids"][:batch_size].copy_(
                selected_chunk_ids
            )
            offset_workspace["offset_signals"].zero_()
            self._shadow_offset_gather.reorder_shadowkv_chunk_offsets(
                offset_workspace["offset_cached_ids"][:batch_size],
                offset_workspace["offset_current_ids"][:batch_size],
                offset_workspace["offset_reordered_ids"][:batch_size],
                offset_workspace["offsets"],
                offset_workspace["offset_counts"],
                batch_size,
                head_count,
                select_sets,
            )
            offset_workspace["offset_cached_ids"][:batch_size].copy_(
                offset_workspace["offset_reordered_ids"][:batch_size]
            )
            selected_chunk_ids = offset_workspace["offset_cached_ids"][:batch_size]
            selected_valid = selected_chunk_ids >= 0
        # Keep compatibility/debugging metadata in the original int32 layout.
        workspace["selected_chunks"][:batch_size].copy_(
            selected_chunk_ids[:, 0].to(torch.int32)
        )
        workspace["selected_valid"][:batch_size].copy_(selected_valid)

        chunk_size = int(self.config.shadowkv_chunk_size)
        offsets = workspace["chunk_offsets"][:chunk_size]
        outlier_slots = int(workspace["outlier_chunks"].shape[2]) * chunk_size
        budget_slots = select_sets * chunk_size
        local_slots = min(
            int(workspace["positions"].shape[1]) - outlier_slots - budget_slots,
            int(self.config.shadowkv_local_chunks) * chunk_size + chunk_size - 1,
        )
        recent_slots = min(int(self.config.shadowkv_recent_tokens), int(workspace["positions"].shape[1]))
        head_positions = workspace["head_positions"][:batch_size]
        head_positions.fill_(-1)

        outlier_ids = workspace["outlier_chunks"][:batch_size].clamp_min(0).to(torch.int32)
        outlier_width = int(outlier_ids.shape[2]) * chunk_size
        outlier_pos = (
            outlier_ids[..., None] * chunk_size
            + offsets[None, None, None, :]
        ).reshape(batch_size, head_count, outlier_width)
        outlier_token_mask = (
            workspace["outlier_token_offsets"][:outlier_width][None, None, :]
            < workspace["outlier_counts"][:batch_size, :, None] * chunk_size
        )
        outlier_dest = workspace["outlier_token_offsets"][:outlier_width][
            None, None, :
        ].expand(batch_size, head_count, -1)
        head_positions.scatter_(
            2, outlier_dest, outlier_pos.masked_fill(~outlier_token_mask, -1)
        )

        selected_width = select_sets * chunk_size
        selected_valid_tokens = workspace["head_selected_valid"][:batch_size]
        selected_valid_tokens.copy_(
            selected_valid[..., None]
            .expand(-1, -1, -1, chunk_size)
            .reshape(batch_size, head_count, selected_width)
        )
        selected_pos = (
            selected_chunk_ids.clamp_min(0).to(torch.int32)[..., None] * chunk_size
            + offsets[None, None, None, :]
        ).reshape(batch_size, head_count, selected_width)
        selected_token_mask = selected_valid_tokens
        # Invalid topk tail slots must not cause host memory reads.  The
        # gather kernels treat -1 as zero-fill, and this mask also excludes
        # them from the compact per-head lengths below.
        selected_pos = selected_pos.masked_fill(~selected_token_mask, -1)
        selected_start = workspace["outlier_counts"][:batch_size] * chunk_size
        selected_dest = selected_start[..., None].to(torch.long) + workspace[
            "selected_token_offsets"
        ][:selected_width][None, None, :]
        head_positions.scatter_(
            2, selected_dest, selected_pos
        )
        workspace["head_selected_positions"][:batch_size].copy_(selected_pos)
        workspace["head_selected_dest"][:batch_size].copy_(selected_dest.to(torch.int32))

        local_tokens = workspace["local_tokens"][:batch_size].clamp_min(0).clamp_max(local_slots)
        local_tokens_head = local_tokens[:, None].expand(-1, head_count)
        local_start = workspace["prompt_lens"][:batch_size, None] - local_tokens_head
        local_pos = local_start[..., None] + workspace["local_token_offsets"][:local_slots][None, None, :]
        local_mask = workspace["local_token_offsets"][:local_slots][None, None, :] < local_tokens_head[..., None]
        local_offset = selected_start + selected_valid.sum(dim=2, dtype=torch.int32) * chunk_size
        local_dest = local_offset[..., None].to(torch.long) + workspace[
            "local_token_offsets"
        ][:local_slots].to(torch.long)[None, None, :]
        head_positions.scatter_(2, local_dest, local_pos.masked_fill(~local_mask, -1))

        context_lens_device = context_lens.to(device=self.device, dtype=torch.int32)
        source_lens_head = workspace["head_source_lens"][:batch_size]
        source_lens = (
            workspace["prompt_lens"][:batch_size]
            if bool(getattr(self.config, "decode_graph", False))
            else context_lens_device
        )
        source_lens_head.copy_(source_lens[:, None].expand(-1, head_count))
        recent_count = (
            context_lens_device - workspace["prompt_lens"][:batch_size]
        ).clamp_min(0).clamp_max(recent_slots)
        recent_count_head = recent_count[:, None].expand(-1, head_count)
        recent_start = torch.maximum(
            workspace["prompt_lens"][:batch_size],
            context_lens_device - recent_count,
        )[:, None].expand(-1, head_count)
        recent_pos = recent_start[..., None] + workspace["recent_token_offsets"][:recent_slots][None, None, :]
        recent_mask = workspace["recent_token_offsets"][:recent_slots][None, None, :] < recent_count_head[..., None]
        recent_offset = local_offset + local_tokens_head
        recent_dest = recent_offset[..., None].to(torch.long) + workspace[
            "recent_token_offsets"
        ][:recent_slots].to(torch.long)[None, None, :]
        head_positions.scatter_(2, recent_dest, recent_pos.masked_fill(~recent_mask, -1))

        selected_lens = (
            workspace["outlier_counts"][:batch_size] * chunk_size
            + selected_valid.sum(dim=2, dtype=torch.int32) * chunk_size
            + local_tokens_head
            + recent_count_head
        )
        workspace["head_context_lens"][:batch_size].copy_(selected_lens)
        workspace["selected_lens"][:batch_size].copy_(selected_lens.amax(dim=1))
        # Retain a representative shared table for generic diagnostics. The
        # per-head provider consumes head_positions/head_context_lens instead.
        workspace["positions"][:batch_size].copy_(head_positions[:, 0])

        head_k_view = workspace["head_k_cache"][:batch_size]
        head_v_view = workspace["head_v_cache"][:batch_size]
        head_prompt_lens = workspace["head_prompt_lens"][:batch_size]
        gather_events = []
        gather_event = None
        if self._shadow_gpu_cache_enabled:
            gpu_cache_gather = self._load_shadowkv_gpu_cache_gather()
            positions = workspace["head_positions"][:batch_size]
            fused_gpu_gather = getattr(
                gpu_cache_gather, "gather_gpu_cache_per_head_kv_ptrs", None
            )
            if fused_gpu_gather is None:
                raise RuntimeError(
                    "ShadowKV GPU-cache gather extension is missing the pointer-table "
                    "gather_gpu_cache_per_head_kv_ptrs entry point. Rebuild the CUDA extension."
                )
            fused_gpu_gather(
                workspace["gpu_cache_k_ptrs"][:batch_size],
                workspace["gpu_cache_v_ptrs"][:batch_size],
                positions,
                source_lens,
                head_k_view,
                head_v_view,
                int(self.config.shadowkv_gpu_cache_tokens),
            )
        else:
            self._shadow_host_gather.gather_host_per_head(
                workspace["host_k_ptrs"],
                workspace["head_positions"][:batch_size],
                source_lens_head,
                head_k_view,
            )
            if trace:
                logger.info("ShadowKV trace layer={} key_gather_done", layer_idx)
            if offset_workspace is None:
                gather_event = self._launch_async_value_gather(
                    workspace,
                    workspace["head_positions"][:batch_size],
                    source_lens_head,
                    head_v_view,
                    per_head=True,
                )
                if gather_event is not None:
                    gather_events.append(gather_event)
                if trace:
                    logger.info("ShadowKV trace layer={} value_gather_enqueued", layer_idx)
            else:
                # Let the original host gather fill outlier/local/recent
                # tokens while offset-copy handles the selected chunks.
                # Invalidating only the selected destinations avoids a second
                # host read for the large sparse budget.
                v_positions = workspace["head_v_positions"][:batch_size]
                v_positions.copy_(workspace["head_positions"][:batch_size])
                v_positions.scatter_(
                    2,
                    workspace["head_selected_dest"][:batch_size].to(torch.long),
                    torch.full_like(
                        workspace["head_selected_dest"][:batch_size], -1
                    ),
                )
                host_event = self._launch_async_value_gather(
                    workspace,
                    v_positions,
                    source_lens_head,
                    head_v_view,
                    per_head=True,
                )
                if host_event is not None:
                    gather_events.append(host_event)
                current_stream = torch.cuda.current_stream(device=self.device)
                offset_stream = current_stream
                if (
                    not bool(getattr(self.config, "decode_graph", False))
                    and bool(getattr(self.config, "shadowkv_multistream_gather", True))
                ):
                    if self._shadow_offset_stream is None:
                        self._shadow_offset_stream = torch.cuda.Stream(device=self.device)
                        self._shadow_offset_ready_event = torch.cuda.Event()
                        self._shadow_offset_done_event = torch.cuda.Event()
                    ready = self._shadow_offset_ready_event
                    offset_event = self._shadow_offset_done_event
                    if ready is None or offset_event is None:
                        raise RuntimeError(
                            "ShadowKV offset-stream events were not initialized."
                        )
                    ready.record(current_stream)
                    self._shadow_offset_stream.wait_event(ready)
                    offset_stream = self._shadow_offset_stream
                with torch.cuda.stream(offset_stream):
                    self._shadow_offset_gather.gather_copy_with_offsets(
                        offset_workspace["offset_host_v_ptrs"],
                        offset_workspace["offset_v_cache"],
                        offset_workspace["offset_temp"],
                        offset_workspace["offsets"],
                        offset_workspace["offset_counts"],
                        offset_workspace["offset_signals"],
                        batch_size,
                        head_count,
                        (
                            int(self.max_model_len)
                            + chunk_size
                            - 1
                        )
                        // chunk_size,
                        select_sets,
                        chunk_size,
                        self.head_dim,
                        select_sets,
                    )
                if offset_stream is not current_stream:
                    offset_event.record(offset_stream)
                    gather_events.append(offset_event)

        if bool(getattr(self.config, "decode_graph", False)):
            # Prompt KV is mapped from pinned host memory. Generated KV lives
            # in the graph-stable recent buffer and must replace host-gathered
            # zeros for positions beyond the prompt.
            recent_width = int(workspace["recent_k"].shape[1])
            recent_index = (
                workspace["head_positions"][:batch_size]
                - workspace["prompt_lens"][:batch_size, None, None]
            ).clamp_(0, recent_width - 1).to(dtype=torch.long)
            recent_index = recent_index[..., None].expand(
                -1, -1, -1, self.head_dim
            )
            recent_k_source = workspace["recent_k"][:batch_size].permute(0, 2, 1, 3)
            recent_v_source = workspace["recent_v"][:batch_size].permute(0, 2, 1, 3)
            recent_k = recent_k_source.gather(2, recent_index)
            recent_v = recent_v_source.gather(2, recent_index)
            recent_mask = (
                (workspace["head_positions"][:batch_size]
                 >= workspace["prompt_lens"][:batch_size, None, None])
                & (workspace["head_positions"][:batch_size]
                   < context_lens_device[:, None, None])
                & (workspace["head_positions"][:batch_size] >= 0)
            )
            head_k_view.copy_(torch.where(recent_mask[..., None], recent_k, head_k_view))
            head_v_view.copy_(torch.where(recent_mask[..., None], recent_v, head_v_view))

        if not self._shadow_gpu_cache_enabled:
            if offset_workspace is not None:
                for gather_event in gather_events:
                    torch.cuda.current_stream(device=self.device).wait_event(gather_event)
                selected_v = offset_workspace["offset_v_cache"][:batch_size].view(
                    batch_size,
                    head_count,
                    selected_width,
                    self.head_dim,
                )
                selected_dest_safe = workspace["head_selected_dest"][:batch_size].clamp(
                    0, int(workspace["head_positions"].shape[2]) - 1
                ).to(torch.long)
                existing_v = head_v_view.gather(
                    2,
                    selected_dest_safe[..., None].expand(-1, -1, -1, self.head_dim),
                )
                head_v_view.scatter_(
                    2,
                    selected_dest_safe[..., None].expand(-1, -1, -1, self.head_dim),
                    torch.where(
                        selected_valid_tokens[..., None].expand_as(selected_v),
                        selected_v,
                        existing_v,
                    ),
                )
            selected_positions = workspace["head_selected_positions"][:batch_size]
            selected_dest = workspace["head_selected_dest"][:batch_size]
            if self._shadow_kernel_backend == "cutlass":
                selected_width = int(selected_positions.shape[-1])
                if selected_width % 128:
                    raise RuntimeError(
                        "ShadowKV fused CUTLASS gather GEMM requires sparse budget "
                        f"aligned to 128 tokens, got width={selected_width}."
                    )
                fused_reconstruct = getattr(
                    self._shadow_host_gather,
                    "batch_gather_gemm_fused_rope",
                )
                fused_reconstruct(
                    workspace["u_device"][:batch_size],
                    workspace["sv_column_major"][:batch_size],
                    selected_chunk_ids.clamp_min(0).to(torch.int32),
                    selected_positions,
                    head_prompt_lens,
                    self._shadow_rope.cos_sin_cache.squeeze(1),
                    workspace["head_reconstructed"][:batch_size].reshape(
                        batch_size * self.num_kv_heads,
                        selected_width,
                        1,
                        self.head_dim,
                    ),
                    int(workspace["u_device"].shape[1]),
                    int(self.config.shadowkv_chunk_size),
                )
                if trace:
                    logger.info("ShadowKV trace layer={} reconstruct_done", layer_idx)
            else:
                reconstruct = getattr(
                    self._shadow_host_gather,
                    "gather_gemm_rope",
                )
                reconstruct(
                    workspace["head_host_u_ptrs"],
                    selected_positions.reshape(batch_size * self.num_kv_heads, -1),
                    head_prompt_lens.reshape(batch_size * self.num_kv_heads),
                    workspace["sv"][:batch_size].reshape(
                        batch_size * self.num_kv_heads,
                        int(self.config.shadowkv_rank),
                        1,
                        self.head_dim,
                    ),
                    self._shadow_rope.cos_sin_cache.squeeze(1),
                    workspace["head_u_selected"][:batch_size].reshape(
                        batch_size * self.num_kv_heads,
                        -1,
                        1,
                        int(self.config.shadowkv_rank),
                    ),
                    workspace["head_reconstructed"][:batch_size].reshape(
                        batch_size * self.num_kv_heads,
                        -1,
                        1,
                        self.head_dim,
                    ),
                )
            selected_dest_safe = selected_dest.clamp(
                0, int(workspace["head_positions"].shape[2]) - 1
            ).to(torch.long)
            gathered_selected = head_k_view.gather(
                2, selected_dest_safe[..., None].expand(-1, -1, -1, self.head_dim)
            )
            reconstructed = workspace["head_reconstructed"][:batch_size]
            head_k_view.scatter_(
                2,
                selected_dest_safe[..., None].expand(-1, -1, -1, self.head_dim),
                torch.where(
                    selected_valid_tokens[..., None].expand_as(reconstructed),
                    reconstructed,
                    gathered_selected,
                ),
            )
            if gather_event is not None:
                torch.cuda.current_stream(device=self.device).wait_event(gather_event)

        active_slots = workspace["active_slots"][:batch_size]
        payload = ExplicitKVPayload(
            k_cache=head_k_view,
            v_cache=head_v_view,
            backend="shadowkv_per_head",
            metadata={
                "head_context_lens": workspace["head_context_lens"][:batch_size],
            },
        )
        if trace:
            logger.info(
                "ShadowKV trace layer={} view_done head_lens={}",
                layer_idx,
                tuple(head_k_view.shape),
            )
        return DecodeComputeView(
            meta=AttentionViewMeta(
                active_slots=active_slots,
                req_indices=workspace["local_req_indices"][:batch_size],
                context_lens=workspace["selected_lens"][:batch_size],
                max_context_len=int(workspace["positions"].shape[1]),
                attn_score=selection.attn_score,
                # ShadowKV has already materialized its selected/reconstructed
                # tokens into the local decode payload. Ask FlashInfer for
                # its dense graph plan over that payload; applying sparse
                # paging a second time is incorrect.
                is_sparse=False,
            ),
            payload=payload,
        )


__all__ = ["ShadowKVCacheManager"]
