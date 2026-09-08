from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_page_indices_kernel(
    active_slots,
    request_indices,
    context_lens,
    packed_indices,
    active_slots_stride_0: tl.constexpr,
    active_slots_stride_1: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    PAGE_CAPACITY: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_block_idx = tl.program_id(1)

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    lengths = tl.load(context_lens + batch_offsets, mask=batch_offsets < BATCH_SIZE)
    page_counts = (lengths + PAGE_SIZE - 1) // PAGE_SIZE
    packed_start = tl.sum(
        tl.where(batch_offsets < batch_idx, page_counts, 0)
    )

    page_offsets = token_block_idx * PAGE_BLOCK + tl.arange(0, PAGE_BLOCK)
    context_len = tl.load(context_lens + batch_idx)
    page_count = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    request_idx = tl.load(request_indices + batch_idx)
    valid = (page_offsets < page_count) & (page_offsets < PAGE_CAPACITY)
    slots = tl.load(
        active_slots
        + request_idx * active_slots_stride_0
        + page_offsets * active_slots_stride_1,
        mask=valid,
    )
    tl.store(packed_indices + packed_start + page_offsets, slots, mask=valid)


def pack_flashinfer_page_indices(
    active_slots: torch.Tensor,
    request_indices: torch.Tensor,
    context_lens: torch.Tensor,
    packed_indices: torch.Tensor,
    *,
    context_capacity: int,
    page_size: int = 1,
    packed_indptr: torch.Tensor | None = None,
    packed_last_page_len: torch.Tensor | None = None,
) -> None:
    """Pack a canonical page table into graph-stable storage.

    The optional range buffers are updated from the same dynamic lengths as
    the packed indices.  Materialized sparse payloads use this because their
    effective attention length differs from the request's full context length.
    """

    if active_slots.ndim != 2 or active_slots.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires a rank-2 int32 slot table.")
    if request_indices.ndim != 1 or request_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 request indices.")
    if context_lens.ndim != 1 or context_lens.dtype != torch.int32:
        raise TypeError("FlashInfer graph decode requires int32 context lengths.")
    if request_indices.shape != context_lens.shape:
        raise ValueError("FlashInfer graph request indices and context lengths must match.")

    batch_size = int(context_lens.numel())
    context_capacity = int(context_capacity)
    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError(f"FlashInfer graph page_size must be positive, got {page_size}.")
    page_capacity = (context_capacity + page_size - 1) // page_size
    if page_capacity <= 0 or page_capacity > int(active_slots.shape[1]):
        raise ValueError(
            "FlashInfer graph context capacity is outside the slot table: "
            f"tokens={context_capacity} pages={page_capacity} "
            f"width={int(active_slots.shape[1])}."
        )
    if packed_indices.ndim != 1 or packed_indices.dtype != torch.int32:
        raise TypeError("FlashInfer graph packed indices must be a 1D int32 tensor.")
    required = batch_size * page_capacity
    if int(packed_indices.numel()) < required:
        raise ValueError(
            "FlashInfer graph packed-index buffer is too small: "
            f"required={required} actual={int(packed_indices.numel())}."
        )
    if packed_indptr is not None:
        if packed_indptr.ndim != 1 or packed_indptr.dtype != torch.int32:
            raise TypeError("FlashInfer graph packed indptr must be a 1D int32 tensor.")
        if int(packed_indptr.numel()) != batch_size + 1:
            raise ValueError(
                "FlashInfer graph packed indptr must have batch+1 entries: "
                f"expected={batch_size + 1} actual={int(packed_indptr.numel())}."
            )
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        ).to(torch.int32)
        packed_indptr.zero_()
        torch.cumsum(page_counts, dim=0, dtype=torch.int32, out=packed_indptr[1:])
    if packed_last_page_len is not None:
        if packed_last_page_len.ndim != 1 or packed_last_page_len.dtype != torch.int32:
            raise TypeError(
                "FlashInfer graph packed last-page lengths must be a 1D int32 tensor."
            )
        if int(packed_last_page_len.numel()) != batch_size:
            raise ValueError(
                "FlashInfer graph packed last-page lengths must match the batch: "
                f"expected={batch_size} actual={int(packed_last_page_len.numel())}."
            )
        page_counts = torch.div(
            context_lens + page_size - 1,
            page_size,
            rounding_mode="floor",
        )
        packed_last_page_len.copy_(
            context_lens - (page_counts - 1) * page_size
        )

    page_block = 128
    _pack_page_indices_kernel[
        (batch_size, triton.cdiv(page_capacity, page_block))
    ](
        active_slots,
        request_indices,
        context_lens,
        packed_indices,
        active_slots.stride(0),
        active_slots.stride(1),
        BATCH_SIZE=batch_size,
        BATCH_BLOCK=triton.next_power_of_2(batch_size),
        PAGE_CAPACITY=page_capacity,
        PAGE_SIZE=page_size,
        PAGE_BLOCK=page_block,
    )


@triton.jit
def _pack_shadowkv_head_page_indices_kernel(
    head_lens,
    packed_indptr,
    packed_indices,
    row_page_capacity: tl.constexpr,
    payload_width: tl.constexpr,
    page_size: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(packed_indptr + row)
    length = tl.load(head_lens + row)
    offsets = tl.arange(0, TOKEN_BLOCK)
    page_count = tl.cdiv(length, page_size)
    for block in range(0, tl.cdiv(row_page_capacity, TOKEN_BLOCK)):
        page_ids = block * TOKEN_BLOCK + offsets
        mask = page_ids < page_count
        tl.store(
            packed_indices + start + page_ids,
            row * row_page_capacity + page_ids,
            mask=mask,
        )


def pack_shadowkv_head_page_indices(
    head_lens: torch.Tensor,
    packed_indices: torch.Tensor,
    packed_indptr: torch.Tensor,
    packed_last_page_len: torch.Tensor,
    *,
    payload_width: int,
    page_size: int = 16,
) -> None:
    """Pack per-KV-head compact rows into FlashInfer's flat page table.

    ShadowKV treats ``(request, kv_head)`` as one FlashInfer request with one
    KV head. Each row points into the row-major
    ``[batch, kv_heads, payload_width, head_dim]`` payload, packed into fixed
    token pages so long compact payloads stay within FlashInfer's efficient
    decode workspace envelope.
    """

    if head_lens.ndim != 2 or head_lens.dtype != torch.int32:
        raise TypeError("ShadowKV head lengths must be a rank-2 int32 tensor.")
    if not head_lens.is_cuda or not head_lens.is_contiguous():
        raise ValueError("ShadowKV head lengths must be contiguous CUDA storage.")
    if packed_indices.ndim != 1 or packed_indices.dtype != torch.int32:
        raise TypeError("ShadowKV packed indices must be a 1D int32 tensor.")
    if packed_indptr.ndim != 1 or packed_indptr.dtype != torch.int32:
        raise TypeError("ShadowKV packed indptr must be a 1D int32 tensor.")
    if packed_last_page_len.ndim != 1 or packed_last_page_len.dtype != torch.int32:
        raise TypeError("ShadowKV last-page lengths must be a 1D int32 tensor.")
    if payload_width <= 0:
        raise ValueError(f"ShadowKV payload width must be positive, got {payload_width}.")
    page_size = int(page_size)
    if page_size <= 0 or payload_width % page_size:
        raise ValueError(
            "ShadowKV FlashInfer payload width must be divisible by its page size: "
            f"width={payload_width} page_size={page_size}."
        )
    rows = int(head_lens.numel())
    if int(packed_indptr.numel()) != rows + 1:
        raise ValueError("ShadowKV packed indptr must have rows+1 entries.")
    if int(packed_last_page_len.numel()) != rows:
        raise ValueError("ShadowKV last-page lengths must match flattened head rows.")
    # Do not convert a device reduction to a Python bool here: this function
    # runs once per layer and token in eager FlashInfer mode.  An asynchronous
    # device assertion preserves the boundary check without inserting a host
    # synchronization, and is also safe to elide during graph capture.
    if not torch.cuda.is_current_stream_capturing():
        torch._assert_async(
            torch.all(
                (head_lens > 0) & (head_lens <= int(payload_width))
            )
        )
    row_page_capacity = (int(payload_width) + page_size - 1) // page_size
    if int(packed_indices.numel()) < rows * row_page_capacity:
        raise ValueError(
            "ShadowKV packed-index storage is smaller than its page capacity."
        )
    packed_indptr.zero_()
    page_counts = torch.div(
        head_lens + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).to(torch.int32)
    torch.cumsum(
        page_counts.reshape(-1), dim=0, dtype=torch.int32, out=packed_indptr[1:]
    )
    packed_last_page_len.copy_(
        head_lens.reshape(-1)
        - (page_counts.reshape(-1) - 1) * page_size
    )
    _pack_shadowkv_head_page_indices_kernel[(rows,)](
        head_lens.reshape(-1),
        packed_indptr,
        packed_indices,
        row_page_capacity=row_page_capacity,
        payload_width=int(payload_width),
        TOKEN_BLOCK=128,
        page_size=page_size,
    )


__all__ = ["pack_flashinfer_page_indices", "pack_shadowkv_head_page_indices"]
