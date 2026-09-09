from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _score_query_robust_pages_kernel(
    query,
    landmark,
    bias,
    epsilon,
    metadata_valid,
    row_page_slots,
    output,
    query_stride_row,
    query_stride_head,
    landmark_stride_page,
    landmark_stride_head,
    scalar_stride_page,
    scalar_stride_head,
    page_table_stride,
    output_stride,
    scale,
    alpha,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    logical_page = tl.program_id(1)
    logical_slot = tl.load(row_page_slots + row * page_table_stride + logical_page)
    physical_page = tl.maximum(logical_slot, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < HEAD_DIM
    group_size: tl.constexpr = NUM_QUERY_HEADS // NUM_KV_HEADS
    best = -float("inf")
    all_valid = logical_slot >= 0
    for query_head in tl.static_range(0, NUM_QUERY_HEADS):
        kv_head = query_head // group_size
        q_values = tl.load(
            query + row * query_stride_row + query_head * query_stride_head + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        kbar_values = tl.load(
            landmark
            + physical_page * landmark_stride_page
            + kv_head * landmark_stride_head
            + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(q_values * kbar_values, axis=0)
        page_bias = tl.load(
            bias + physical_page * scalar_stride_page + kv_head * scalar_stride_head
        ).to(tl.float32)
        page_epsilon = tl.load(
            epsilon + physical_page * scalar_stride_page + kv_head * scalar_stride_head
        ).to(tl.float32)
        valid = tl.load(
            metadata_valid
            + physical_page * scalar_stride_page
            + kv_head * scalar_stride_head
        )
        all_valid = all_valid & (valid != 0)
        best = tl.maximum(best, dot * scale + page_bias + alpha * page_epsilon)
    score = tl.where(all_valid, best, float("inf"))
    tl.store(output + row * output_stride + logical_page, score)


def score_query_robust_pages(
    query: torch.Tensor,
    landmark: torch.Tensor,
    bias: torch.Tensor,
    epsilon: torch.Tensor,
    metadata_valid: torch.Tensor,
    row_page_slots: torch.Tensor,
    *,
    scale: float,
    alpha: float,
) -> torch.Tensor:
    """Fused FP32 QR scorer with one page route per request."""

    if not query.is_cuda:
        raise ValueError("Query-Robust Triton scoring requires CUDA tensors.")
    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("QR query must be contiguous [batch, query_heads, head_dim].")
    if landmark.ndim != 3 or not landmark.is_contiguous():
        raise ValueError("QR landmark must be contiguous [pages, kv_heads, head_dim].")
    if bias.ndim != 2 or epsilon.shape != bias.shape or not bias.is_contiguous() or not epsilon.is_contiguous():
        raise ValueError("QR scalar metadata must be contiguous [pages, kv_heads].")
    if metadata_valid.shape != bias.shape or not metadata_valid.is_contiguous():
        raise ValueError("QR metadata_valid must match contiguous scalar metadata.")
    if row_page_slots.ndim != 2 or row_page_slots.dtype != torch.int32 or not row_page_slots.is_contiguous():
        raise ValueError("QR row_page_slots must be contiguous int32 [batch, pages].")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported QR query dtype: {query.dtype}.")
    if landmark.dtype != torch.bfloat16:
        raise TypeError(f"QR landmarks must be BF16, got {landmark.dtype}.")
    if bias.dtype != torch.float32 or epsilon.dtype != torch.float32:
        raise TypeError("QR bias and epsilon must be FP32.")
    batch, query_heads, head_dim = map(int, query.shape)
    pages, kv_heads, landmark_dim = map(int, landmark.shape)
    if landmark_dim != head_dim or query_heads % kv_heads:
        raise ValueError("QR query heads must be divisible by KV heads and share head_dim.")
    tensors = (landmark, bias, epsilon, metadata_valid, row_page_slots)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("QR scoring tensors must share one CUDA device.")
    output = torch.empty(
        (batch, int(row_page_slots.shape[1])),
        dtype=torch.float32,
        device=query.device,
    )
    block_d = triton.next_power_of_2(head_dim)
    _score_query_robust_pages_kernel[(batch, int(row_page_slots.shape[1]))](
        query,
        landmark,
        bias,
        epsilon,
        metadata_valid,
        row_page_slots,
        output,
        query.stride(0),
        query.stride(1),
        landmark.stride(0),
        landmark.stride(1),
        bias.stride(0),
        bias.stride(1),
        row_page_slots.stride(0),
        output.stride(0),
        float(scale),
        float(alpha),
        NUM_QUERY_HEADS=query_heads,
        NUM_KV_HEADS=kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=min(max(block_d // 128, 1), 8),
        num_stages=2,
    )
    return output


__all__ = ["score_query_robust_pages"]
