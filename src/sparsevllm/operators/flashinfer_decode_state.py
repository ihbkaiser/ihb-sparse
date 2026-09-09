from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from sparsevllm.engine.decode_graph_contract import (
    DecodeGraphContract,
    DecodeGraphInputs,
)
from sparsevllm.kernels.external.flashinfer.decode import (
    make_flashinfer_paged_decode_wrapper,
)
from sparsevllm.kernels.triton.flashinfer_decode_metadata import (
    pack_flashinfer_page_indices,
    pack_shadowkv_head_page_indices,
)

if TYPE_CHECKING:
    from sparsevllm.operators.decode_attention import DecodeAttentionOpSpec


class FlashInferPagedDecodeState:
    """Provider-owned eager plan state for FlashInfer paged decode."""

    def __init__(self, device: torch.device) -> None:
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(self.workspace)
        self.plan_key: tuple[object, ...] | None = None

    def plan(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
    ) -> None:
        if active_slots.dtype != torch.int32 or active_slots.ndim != 2:
            raise TypeError(
                "FlashInfer decode requires a rank-2 int32 physical-slot page table."
            )
        if req_indices.dtype != torch.int32 or context_lens.dtype != torch.int32:
            raise TypeError("FlashInfer decode requires int32 request metadata.")
        batch_size = int(context_lens.numel())
        if batch_size <= 0 or int(req_indices.numel()) != batch_size:
            raise ValueError("FlashInfer decode requires matched non-empty metadata.")
        max_context_len = int(max_context_len)
        page_size = int(spec.page_size)
        max_page_count = (max_context_len + page_size - 1) // page_size
        if max_context_len <= 0 or max_page_count > int(active_slots.shape[1]):
            raise ValueError(
                "FlashInfer decode context is outside the active slot table: "
                f"max_context_len={max_context_len} "
                f"max_page_count={max_page_count} "
                f"width={int(active_slots.shape[1])}."
            )
        rows = active_slots.index_select(0, req_indices.to(torch.long))[
            :, :max_page_count
        ]
        positions = torch.arange(
            max_page_count,
            device=context_lens.device,
            dtype=context_lens.dtype,
        )
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        valid = positions.unsqueeze(0) < page_counts.unsqueeze(1)
        indices = rows.masked_select(valid).to(torch.int32).contiguous()
        indptr = torch.cat(
            (
                torch.zeros(1, device=context_lens.device, dtype=torch.int32),
                page_counts.cumsum(0, dtype=torch.int32),
            )
        )
        last_page_len = context_lens - (page_counts - 1) * page_size
        self.wrapper.plan(
            indptr,
            indices,
            last_page_len,
            num_qo_heads=spec.num_query_heads,
            num_kv_heads=spec.num_kv_heads,
            head_dim=spec.head_dim,
            page_size=spec.page_size,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
            non_blocking=True,
        )


class FlashInferPagedDecodeGraphState:
    """Graph-stable FlashInfer plans and layer-varying page-index buffers."""

    def __init__(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        contract: DecodeGraphContract,
        inputs: DecodeGraphInputs,
    ) -> None:
        self.spec = spec
        self.contract = contract
        self.inputs = inputs
        device = inputs.context_lens.device
        batch_size = int(contract.batch_capacity)
        context_capacity = int(contract.context_capacity)
        self.workspace = torch.empty(
            128 * 1024 * 1024,
            dtype=torch.uint8,
            device=device,
        )
        self.indptr = torch.empty(
            batch_size + 1,
            dtype=torch.int32,
            device=device,
        )
        self.indices = torch.empty(
            batch_size * context_capacity,
            dtype=torch.int32,
            device=device,
        )
        self.last_page_len = torch.ones(
            batch_size,
            dtype=torch.int32,
            device=device,
        )
        pin_memory = bool(inputs.host.context_lens.is_pinned())
        self.host_indptr = torch.empty(
            batch_size + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_last_page_len = torch.ones(
            batch_size,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(
            self.workspace,
            use_cuda_graph=True,
            paged_kv_indptr_buffer=self.indptr,
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last_page_len,
        )
        self.sparse_indptr: torch.Tensor | None = None
        self.sparse_indices: torch.Tensor | None = None
        self.sparse_last_page_len: torch.Tensor | None = None
        self.host_sparse_indptr: torch.Tensor | None = None
        self.host_sparse_last_page_len: torch.Tensor | None = None
        self.sparse_wrapper = None
        if (
            spec.sparse_context_budget is not None
            and contract.topology_path_id == "long"
        ):
            self.sparse_indptr = torch.empty_like(self.indptr)
            self.sparse_indices = torch.empty_like(self.indices)
            self.sparse_last_page_len = torch.ones_like(self.last_page_len)
            self.host_sparse_indptr = torch.empty_like(self.host_indptr)
            self.host_sparse_last_page_len = torch.ones_like(
                self.host_last_page_len
            )
            self.sparse_wrapper = make_flashinfer_paged_decode_wrapper(
                self.workspace,
                use_cuda_graph=True,
                paged_kv_indptr_buffer=self.sparse_indptr,
                paged_kv_indices_buffer=self.sparse_indices,
                paged_kv_last_page_len_buffer=self.sparse_last_page_len,
            )
        self.planned = False
        self._eager_page_pack_keys: dict[bool, tuple[object, ...] | None] = {
            False: None,
            True: None,
        }
        self._captured_page_pack_keys: dict[bool, tuple[object, ...] | None] = {
            False: None,
            True: None,
        }
        self._capture_pack_started = False

    def _plan(
        self,
        wrapper: Any,
        host_indptr: torch.Tensor,
        indices: torch.Tensor,
        host_last_page_len: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> None:
        page_size = int(self.spec.page_size)
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        host_indptr[0] = 0
        torch.cumsum(
            page_counts, dim=0, dtype=torch.int32, out=host_indptr[1:]
        )
        host_last_page_len.copy_(
            context_lens - (page_counts - 1) * page_size
        )
        total_pages = int(host_indptr[-1])
        wrapper.plan(
            host_indptr,
            indices[:total_pages],
            host_last_page_len,
            num_qo_heads=self.spec.num_query_heads,
            num_kv_heads=self.spec.num_kv_heads,
            head_dim=self.spec.head_dim,
            page_size=self.spec.page_size,
            sm_scale=self.spec.softmax_scale,
            q_data_type=self.spec.activation_dtype,
            kv_data_type=self.spec.activation_dtype,
            non_blocking=True,
        )

    def prepare_out_graph(self) -> None:
        dense_context_lens = self.inputs.host.context_lens
        if torch.any(dense_context_lens <= 0):
            raise ValueError(
                "FlashInfer graph decode requires positive context lengths."
            )
        if torch.any(dense_context_lens > self.contract.context_capacity):
            raise ValueError(
                "FlashInfer graph decode context exceeds its captured capacity."
            )
        self._plan(
            self.wrapper,
            self.host_indptr,
            self.indices,
            self.host_last_page_len,
            dense_context_lens,
        )
        if self.sparse_wrapper is not None:
            self._plan_sparse(dense_context_lens)
        self.planned = True

    def _plan_sparse(self, dense_context_lens: torch.Tensor) -> None:
        assert self.spec.sparse_context_budget is not None
        assert self.sparse_wrapper is not None
        assert self.host_sparse_indptr is not None
        assert self.sparse_indices is not None
        assert self.host_sparse_last_page_len is not None
        page_size = int(self.spec.page_size)
        page_budget = max(
            3,
            int(self.spec.sparse_context_budget) // page_size,
        )
        prev_budget = min(
            page_budget - 1,
            (int(self.contract.context_capacity) + page_size - 1)
            // page_size
            - 1,
        )
        last_page_lens = torch.remainder(
            dense_context_lens - 1,
            page_size,
        ) + 1
        sparse_context_lens = prev_budget * page_size + last_page_lens
        self._plan(
            self.sparse_wrapper,
            self.host_sparse_indptr,
            self.sparse_indices,
            self.host_sparse_last_page_len,
            sparse_context_lens,
        )

    def begin_graph_in(self) -> None:
        """Reset page-pack deduplication at one warmup/capture boundary."""

        if torch.cuda.is_current_stream_capturing():
            self._captured_page_pack_keys = {False: None, True: None}
            self._capture_pack_started = True
        else:
            self._eager_page_pack_keys = {False: None, True: None}
            self._capture_pack_started = False

    def wrapper_for(self, is_sparse: bool) -> Any:
        if not is_sparse:
            return self.wrapper
        if self.sparse_wrapper is None:
            raise RuntimeError(
                "FlashInfer graph received a sparse paged view without a "
                "prepared sparse plan."
            )
        return self.sparse_wrapper

    @staticmethod
    def _page_pack_key(
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[object, ...]:
        return (
            active_slots.device,
            active_slots.dtype,
            active_slots.data_ptr(),
            tuple(active_slots.shape),
            tuple(active_slots.stride()),
            req_indices.device,
            req_indices.dtype,
            req_indices.data_ptr(),
            tuple(req_indices.shape),
            tuple(req_indices.stride()),
            context_lens.device,
            context_lens.dtype,
            context_lens.data_ptr(),
            tuple(context_lens.shape),
            tuple(context_lens.stride()),
        )

    def pack_page_indices_once(
        self,
        *,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        is_sparse: bool = False,
        force: bool = False,
    ) -> bool:
        """Pack unless the destination already holds this exact paged view."""

        if not self.planned:
            raise RuntimeError(
                "FlashInfer graph decode was not planned before forward."
            )
        is_capturing = torch.cuda.is_current_stream_capturing()
        if is_capturing:
            if not self._capture_pack_started:
                self._captured_page_pack_keys = {False: None, True: None}
                self._capture_pack_started = True
            keys = self._captured_page_pack_keys
        else:
            keys = self._eager_page_pack_keys
        key = self._page_pack_key(active_slots, req_indices, context_lens)
        if not force and key == keys[bool(is_sparse)]:
            return False
        packed_indices = self.indices
        if is_sparse:
            if self.sparse_indices is None:
                raise RuntimeError(
                    "FlashInfer sparse page-index storage is unavailable."
                )
            packed_indices = self.sparse_indices
        packed_indptr = getattr(self, "sparse_indptr", None) if is_sparse else getattr(
            self, "indptr", None
        )
        packed_last_page_len = (
            getattr(self, "sparse_last_page_len", None)
            if is_sparse
            else getattr(self, "last_page_len", None)
        )
        # ShadowKV materializes a shorter explicit payload than the request's
        # full context. Keep the wrapper's graph-owned range metadata aligned
        # with that payload; these fixed-address device buffers are safe to
        # update during graph capture/replay.
        pack_kwargs = {}
        if packed_indptr is not None and packed_last_page_len is not None:
            pack_kwargs = {
                "packed_indptr": packed_indptr,
                "packed_last_page_len": packed_last_page_len,
            }
        pack_flashinfer_page_indices(
            active_slots,
            req_indices,
            context_lens,
            packed_indices,
            context_capacity=min(
                int(self.contract.context_capacity),
                int(active_slots.shape[1]) * int(self.spec.page_size),
            ),
            page_size=int(self.spec.page_size),
            **pack_kwargs,
        )
        keys[bool(is_sparse)] = key
        return True

    def keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = [
            self.workspace,
            self.indptr,
            self.indices,
            self.last_page_len,
            self.host_indptr,
            self.host_last_page_len,
        ]
        for tensor in (
            self.sparse_indptr,
            self.sparse_indices,
            self.sparse_last_page_len,
            self.host_sparse_indptr,
            self.host_sparse_last_page_len,
        ):
            if tensor is not None:
                tensors.append(tensor)
        return tensors


class FlashInferShadowKVPerHeadState:
    """FlashInfer state for ShadowKV's flattened per-head page contract.

    A ShadowKV payload has different compact positions for each KV head.  The
    upstream batch wrapper cannot express that as one request, so this state
    flattens ``(batch, kv_head)`` into the wrapper batch and uses one KV head
    per flattened row.  The page table is packed on-device from head lengths
    for every layer, which keeps both eager execution and graph replay free of
    host synchronization.
    """

    def __init__(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        batch_capacity: int,
        payload_width: int,
        device: torch.device,
        use_cuda_graph: bool,
        workspace_bytes: int | None = None,
    ) -> None:
        if spec.num_query_heads % spec.num_kv_heads:
            raise ValueError("ShadowKV FlashInfer state requires GQA-compatible heads.")
        self.batch_capacity = int(batch_capacity)
        self.kv_heads = int(spec.num_kv_heads)
        self.group_size = int(spec.num_query_heads // spec.num_kv_heads)
        self.row_capacity = self.batch_capacity * self.kv_heads
        self.payload_width = int(payload_width)
        self.page_size = int(spec.shadowkv_page_size)
        if self.payload_width % self.page_size:
            raise ValueError(
                "ShadowKV FlashInfer compact payload width must be divisible by "
                f"page_size={self.page_size}, got {self.payload_width}."
            )
        self.page_capacity = self.payload_width // self.page_size
        if self.row_capacity <= 0 or self.payload_width <= 0:
            raise ValueError("ShadowKV FlashInfer state capacities must be positive.")
        self.device = device
        self.spec = spec
        requested_backend = str(
            getattr(spec, "shadowkv_flashinfer_backend", "auto")
        ).strip().lower()
        if requested_backend == "auto" and device.type == "cuda":
            capability = torch.cuda.get_device_capability(device)
            # CUDA-core FlashInfer plans its split topology from the exact
            # page counts passed to plan(). ShadowKV compact lengths change as
            # recent tokens grow. On Blackwell, the FlashInfer CuTe DSL path
            # consumes graph-stable sequence lengths instead.
            self.flashinfer_backend = (
                "cute-dsl" if int(capability[0]) >= 10 else "auto"
            )
        else:
            self.flashinfer_backend = requested_backend
        if self.flashinfer_backend == "cute-dsl" and device.type != "cuda":
            raise ValueError("FlashInfer CuTe DSL requires a CUDA device.")
        if self.flashinfer_backend == "cute-dsl" and self.page_size not in {
            8,
            16,
            32,
            64,
        }:
            raise ValueError(
                "FlashInfer CuTe DSL requires ShadowKV page_size in {8, 16, 32, 64}, "
                f"got {self.page_size}."
            )
        if use_cuda_graph and self.flashinfer_backend != "cute-dsl":
            raise ValueError(
                "ShadowKV FlashInfer CUDA Graph requires the graph-safe CuTe DSL "
                f"backend, got {self.flashinfer_backend!r}."
            )
        workspace_size = max(
            128 * 1024 * 1024,
            int(workspace_bytes) if workspace_bytes is not None else 0,
        )
        self.workspace = torch.empty(
            workspace_size,
            dtype=torch.uint8,
            device=device,
        )
        self.indptr = torch.empty(
            self.row_capacity + 1,
            dtype=torch.int32,
            device=device,
        )
        self.indices = torch.empty(
            self.row_capacity * self.page_capacity,
            dtype=torch.int32,
            device=device,
        )
        self.last_page_len = torch.ones(
            self.row_capacity,
            dtype=torch.int32,
            device=device,
        )
        row_ids = torch.arange(
            self.row_capacity * self.page_capacity,
            dtype=torch.int32,
            device=device,
        )
        self.indices.copy_(row_ids)
        self.output = torch.empty(
            self.row_capacity,
            self.group_size,
            spec.head_dim,
            dtype=spec.activation_dtype,
            device=device,
        )
        self.output_lse = torch.empty(
            self.row_capacity,
            self.group_size,
            dtype=torch.float32,
            device=device,
        )
        self.use_cuda_graph = bool(use_cuda_graph)
        pin_memory = bool(device.type == "cuda")
        self.host_indptr = torch.empty(
            self.row_capacity + 1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_last_page_len = torch.empty(
            self.row_capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.host_seq_lens = torch.empty(
            self.row_capacity,
            dtype=torch.int32,
            device="cpu",
            pin_memory=pin_memory,
        )
        self.wrapper = make_flashinfer_paged_decode_wrapper(
            self.workspace,
            use_cuda_graph=self.use_cuda_graph,
            paged_kv_indptr_buffer=self.indptr,
            paged_kv_indices_buffer=self.indices,
            paged_kv_last_page_len_buffer=self.last_page_len,
            backend=self.flashinfer_backend,
        )
        self.planned = False
        self.planned_rows = 0
        self.planned_page_counts: tuple[int, ...] = ()
        self.run_seq_lens: torch.Tensor | None = None

    def prepare_plan(
        self,
        spec: DecodeAttentionOpSpec,
        *,
        batch_size: int | None = None,
    ) -> None:
        rows = self.row_capacity if batch_size is None else int(batch_size) * self.kv_heads
        if rows <= 0 or rows > self.row_capacity:
            raise ValueError("ShadowKV FlashInfer plan batch is outside state capacity.")
        self.host_indptr[0] = 0
        self.host_indptr[1 : rows + 1] = torch.arange(
            self.page_capacity,
            (rows + 1) * self.page_capacity,
            self.page_capacity,
            dtype=torch.int32,
        )
        self.host_last_page_len[:rows].fill_(self.page_size)
        self.host_seq_lens[:rows].fill_(self.payload_width)
        # ``last_page_len`` remains physical page metadata for every
        # FlashInfer backend.  CuTe DSL additionally consumes the logical
        # per-row lengths through the explicit ``seq_lens`` keyword.  Passing
        # ``host_seq_lens`` positionally here is subtly wrong: the public
        # wrapper would interpret it as last-page lengths and derive inflated
        # sequence lengths from the page table.
        plan_lengths = self.host_last_page_len[:rows]
        plan_kwargs = (
            {
                "seq_lens": self.host_seq_lens[:rows],
            }
            if self.flashinfer_backend == "cute-dsl"
            else {}
        )
        self.wrapper.plan(
            self.host_indptr[: rows + 1],
            self.indices[: rows * self.page_capacity],
            plan_lengths,
            num_qo_heads=self.group_size,
            num_kv_heads=1,
            head_dim=spec.head_dim,
            page_size=self.page_size,
            sm_scale=spec.softmax_scale,
            q_data_type=spec.activation_dtype,
            kv_data_type=spec.activation_dtype,
            non_blocking=True,
            **plan_kwargs,
        )
        # Non-graph FlashInfer copies the metadata passed to plan() into
        # private device buffers. Keep those actual run-time buffers so the
        # per-layer device pack updates what the kernel will read.
        self.run_indptr = self.wrapper._paged_kv_indptr_buf
        self.run_indices = self.wrapper._paged_kv_indices_buf
        self.run_last_page_len = self.wrapper._paged_kv_last_page_len_buf
        self.run_seq_lens = getattr(self.wrapper, "_kv_lens_buffer", None)
        self.planned = True
        self.planned_rows = rows
        self.planned_page_counts = (self.page_capacity,) * rows

    def pack(self, head_lens: torch.Tensor, *, batch_size: int) -> None:
        if not self.planned:
            raise RuntimeError("ShadowKV FlashInfer state was not planned.")
        rows = int(batch_size) * self.kv_heads
        if rows != self.planned_rows:
            raise RuntimeError(
                "ShadowKV FlashInfer page-table batch changed without replanning: "
                f"rows={rows} planned={self.planned_rows}."
            )
        if tuple(head_lens.shape) != (batch_size, self.kv_heads):
            raise ValueError(
                "ShadowKV FlashInfer head lengths do not match graph capacity: "
                f"got={tuple(head_lens.shape)} expected={(batch_size, self.kv_heads)}."
            )
        pack_shadowkv_head_page_indices(
            head_lens,
            self.run_indices[: rows * self.page_capacity],
            self.run_indptr[: rows + 1],
            self.run_last_page_len[:rows],
            payload_width=self.payload_width,
            page_size=self.page_size,
        )
        if self.flashinfer_backend == "cute-dsl":
            if self.run_seq_lens is None:
                raise RuntimeError(
                    "FlashInfer CuTe DSL did not expose its graph-stable seq_lens buffer."
                )
            self.run_seq_lens[:rows].copy_(head_lens.reshape(-1))
            return
        if self.use_cuda_graph:
            raise RuntimeError(
                "ShadowKV CUDA Graph reached a non-CuTe FlashInfer backend; "
                "dynamic per-head page counts are not supported by that backend."
            )

        # CUDA-core FlashInfer bakes the page-count topology into plan_info.
        # Re-plan only when a row crosses a physical page boundary. Within a
        # page, updating last_page_len and page indices is safe and avoids a
        # per-token plan.
        page_counts = torch.div(
            head_lens + self.page_size - 1,
            self.page_size,
            rounding_mode="floor",
        ).reshape(-1)
        page_key = tuple(int(value) for value in page_counts.cpu().tolist())
        if page_key != self.planned_page_counts:
            self.wrapper.plan(
                self.run_indptr[: rows + 1],
                # Keep the private index buffer at full row capacity.  The
                # dynamic indptr limits what the kernel reads, while retaining
                # capacity lets the next page-boundary update repack in place.
                self.run_indices[: rows * self.page_capacity],
                self.run_last_page_len[:rows],
                num_qo_heads=self.group_size,
                num_kv_heads=1,
                head_dim=self.spec.head_dim,
                page_size=self.page_size,
                sm_scale=self.spec.softmax_scale,
                q_data_type=self.spec.activation_dtype,
                kv_data_type=self.spec.activation_dtype,
                non_blocking=True,
            )
            self.run_indptr = self.wrapper._paged_kv_indptr_buf
            self.run_indices = self.wrapper._paged_kv_indices_buf
            self.run_last_page_len = self.wrapper._paged_kv_last_page_len_buf
            self.run_seq_lens = getattr(self.wrapper, "_kv_lens_buffer", None)
            self.planned_page_counts = page_key

    def keepalive_tensors(self) -> list[torch.Tensor]:
        tensors = [
            self.workspace,
            self.indptr,
            self.indices,
            self.last_page_len,
            self.output,
            self.output_lse,
            self.host_indptr,
            self.host_last_page_len,
            self.host_seq_lens,
            self.run_indptr,
            self.run_indices,
            self.run_last_page_len,
        ]
        if self.run_seq_lens is not None:
            tensors.append(self.run_seq_lens)
        return tensors


__all__ = [
    "FlashInferPagedDecodeGraphState",
    "FlashInferPagedDecodeState",
    "FlashInferShadowKVPerHeadState",
]
