"""Fused Query-Robust page-summary construction."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _build_query_robust_summary_kernel(
    keys,
    vertices,
    active_pages,
    page_slots,
    landmark_out,
    bias_out,
    epsilon_out,
    dual_out,
    gap_out,
    errors_out,
    keys_stride_page,
    keys_stride_token,
    keys_stride_head,
    keys_stride_dim,
    page_slots_stride,
    vertices_stride_head,
    vertices_stride_vertex,
    landmark_stride_page,
    landmark_stride_head,
    scalar_stride_page,
    errors_stride_page,
    scale,
    solver_lr,
    NUM_VERTICES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SOLVER_ITERS: tl.constexpr,
    UNIFORM_P: tl.constexpr,
    FROM_PAGE_CACHE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    page = tl.program_id(0)
    head = tl.program_id(1)
    vertex_offsets = tl.arange(0, BLOCK_M)
    token_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    vertex_mask = vertex_offsets < NUM_VERTICES
    token_mask = token_offsets < PAGE_SIZE
    dim_mask = dim_offsets < HEAD_DIM
    active = tl.load(active_pages + page)

    vertex_values = tl.load(
        vertices
        + head * vertices_stride_head
        + vertex_offsets[:, None] * vertices_stride_vertex
        + dim_offsets[None, :],
        mask=vertex_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if FROM_PAGE_CACHE:
        physical_page = tl.maximum(tl.load(page_slots + page * page_slots_stride), 0)
        page_base = physical_page * PAGE_SIZE * keys_stride_token
    else:
        page_base = page * keys_stride_page
    key_values = tl.load(
        keys
        + page_base
        + token_offsets[:, None] * keys_stride_token
        + head * keys_stride_head
        + dim_offsets[None, :] * keys_stride_dim,
        mask=token_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    valid_matrix = vertex_mask[:, None] & token_mask[None, :]
    logits = tl.dot(vertex_values, tl.trans(key_values)) * scale
    logits = tl.where(valid_matrix, logits, 0.0)

    row_max = tl.max(tl.where(valid_matrix, logits, -float("inf")), axis=1)
    row_exp = tl.where(valid_matrix, tl.exp(logits - row_max[:, None]), 0.0)
    row_sum = tl.sum(row_exp, axis=1)
    page_lse = tl.where(
        vertex_mask,
        row_max + tl.log(tl.maximum(row_sum, 1e-30)),
        0.0,
    )

    lambda_logits = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if UNIFORM_P:
        lam = tl.where(vertex_mask, 1.0 / NUM_VERTICES, 0.0)
        p = tl.where(token_mask, 1.0 / PAGE_SIZE, 0.0)
        bar_s = tl.sum(lam[:, None] * logits, axis=0)
    else:
        for step_index in tl.static_range(0, SOLVER_ITERS):
            lambda_max = tl.max(
                tl.where(vertex_mask, lambda_logits, -float("inf")), axis=0
            )
            lambda_exp = tl.where(
                vertex_mask,
                tl.exp(lambda_logits - lambda_max),
                0.0,
            )
            lambda_sum = tl.sum(lambda_exp, axis=0)
            lam = lambda_exp / tl.maximum(lambda_sum, 1e-30)
            bar_s = tl.sum(lam[:, None] * logits, axis=0)
            score_max = tl.max(tl.where(token_mask, bar_s, -float("inf")), axis=0)
            score_exp = tl.where(token_mask, tl.exp(bar_s - score_max), 0.0)
            score_sum = tl.sum(score_exp, axis=0)
            p = score_exp / tl.maximum(score_sum, 1e-30)
            grad = page_lse - tl.sum(logits * p[None, :], axis=1)
            grad = tl.where(vertex_mask, grad, 0.0)
            grad_mean = tl.sum(grad, axis=0) / NUM_VERTICES
            grad = grad - grad_mean
            grad_scale = tl.max(tl.where(vertex_mask, tl.abs(grad), 0.0), axis=0)
            grad_scale = tl.maximum(grad_scale, 1e-6)
            step = solver_lr / grad_scale / tl.sqrt(1.0 + step_index)
            lambda_logits = tl.where(
                vertex_mask,
                lambda_logits + step * grad,
                0.0,
            )

        lambda_max = tl.max(
            tl.where(vertex_mask, lambda_logits, -float("inf")), axis=0
        )
        lambda_exp = tl.where(
            vertex_mask,
            tl.exp(lambda_logits - lambda_max),
            0.0,
        )
        lambda_sum = tl.sum(lambda_exp, axis=0)
        lam = lambda_exp / tl.maximum(lambda_sum, 1e-30)
        bar_s = tl.sum(lam[:, None] * logits, axis=0)
        score_max = tl.max(tl.where(token_mask, bar_s, -float("inf")), axis=0)
        score_exp = tl.where(token_mask, tl.exp(bar_s - score_max), 0.0)
        score_sum = tl.sum(score_exp, axis=0)
        p = score_exp / tl.maximum(score_sum, 1e-30)

    landmark = tl.sum(p[:, None] * key_values, axis=0)
    if UNIFORM_P:
        entropy = tl.log(float(PAGE_SIZE))
    else:
        score_max = tl.max(tl.where(token_mask, bar_s, -float("inf")), axis=0)
        score_exp = tl.where(token_mask, tl.exp(bar_s - score_max), 0.0)
        score_sum = tl.sum(score_exp, axis=0)
        entropy = score_max + tl.log(tl.maximum(score_sum, 1e-30)) - tl.sum(p * bar_s)

    stored_landmark = landmark.to(tl.bfloat16).to(tl.float32)
    bound = scale * tl.sum(vertex_values * stored_landmark[None, :], axis=1) + entropy
    # Zero masked rows so the invalid-row values do not create 0 * -inf NaNs
    # in the dual reduction.  NUM_VERTICES is always at least two.
    errors = tl.where(vertex_mask, page_lse - bound, 0.0)
    epsilon_value = tl.maximum(tl.max(errors, axis=0), 0.0)
    dual_value = tl.sum(lam * errors, axis=0)
    gap_value = epsilon_value - dual_value

    output_landmark = tl.where(active, landmark.to(tl.bfloat16), 0.0)
    output_bias = tl.where(active, entropy, 0.0)
    output_epsilon = tl.where(active, epsilon_value, 0.0)
    output_dual = tl.where(active, dual_value, 0.0)
    output_gap = tl.where(active, gap_value, 0.0)
    output_errors = tl.where(active, errors, 0.0)
    tl.store(
        landmark_out
        + page * landmark_stride_page
        + head * landmark_stride_head
        + dim_offsets,
        output_landmark,
        mask=dim_mask,
    )
    tl.store(
        bias_out + page * scalar_stride_page + head,
        output_bias,
    )
    tl.store(
        epsilon_out + page * scalar_stride_page + head,
        output_epsilon,
    )
    tl.store(
        dual_out + page * scalar_stride_page + head,
        output_dual,
    )
    tl.store(
        gap_out + page * scalar_stride_page + head,
        output_gap,
    )
    tl.store(
        errors_out
        + page * errors_stride_page
        + head * NUM_VERTICES
        + vertex_offsets,
        output_errors,
        mask=vertex_mask,
    )


def build_query_robust_page_summaries(
    keys: torch.Tensor,
    vertices: torch.Tensor,
    active_pages: torch.Tensor,
    *,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build page summaries with one fixed Triton program per page/head."""

    if not keys.is_cuda or keys.ndim != 4 or not keys.is_contiguous():
        raise ValueError("QR Triton summary keys must be contiguous CUDA [P, N, H, D].")
    if vertices.ndim != 3 or not vertices.is_cuda or not vertices.is_contiguous():
        raise ValueError("QR Triton vertices must be contiguous CUDA [H, M, D].")
    if vertices.dtype != torch.float32:
        raise TypeError(f"QR Triton vertices must be FP32, got {vertices.dtype}.")
    if active_pages.shape != (int(keys.shape[0]),) or active_pages.dtype != torch.bool:
        raise TypeError("QR active_pages must be contiguous bool [pages].")
    if not active_pages.is_cuda or not active_pages.is_contiguous():
        raise ValueError("QR active_pages must be contiguous on the key device.")
    pages, page_size, num_heads, head_dim = map(int, keys.shape)
    if int(vertices.shape[0]) != num_heads or int(vertices.shape[2]) != head_dim:
        raise ValueError("QR vertex and key head dimensions do not match.")
    num_vertices = int(vertices.shape[1])
    solver_iters = int(solver_iters)
    solver_lr = float(solver_lr)
    scale = float(scale)
    if pages <= 0 or page_size <= 0 or num_vertices < 2 or solver_iters <= 0:
        raise ValueError("QR Triton summary dimensions and solver_iters must be positive.")
    if not math.isfinite(scale) or not math.isfinite(solver_lr) or solver_lr <= 0:
        raise ValueError("QR Triton summary scale and solver_lr must be finite and valid.")
    if keys.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported QR key dtype: {keys.dtype}.")
    if vertices.device != keys.device or active_pages.device != keys.device:
        raise ValueError("QR summary inputs must share one CUDA device.")

    landmark = torch.empty(
        (pages, num_heads, head_dim), dtype=torch.bfloat16, device=keys.device
    )
    bias = torch.empty((pages, num_heads), dtype=torch.float32, device=keys.device)
    epsilon = torch.empty_like(bias)
    dual = torch.empty_like(bias)
    gap = torch.empty_like(bias)
    errors = torch.empty(
        (pages, num_heads, num_vertices), dtype=torch.float32, device=keys.device
    )
    block_m = triton.next_power_of_2(num_vertices)
    block_n = triton.next_power_of_2(page_size)
    block_d = triton.next_power_of_2(head_dim)
    _build_query_robust_summary_kernel[(pages, num_heads)](
        keys,
        vertices,
        active_pages,
        active_pages,
        landmark,
        bias,
        epsilon,
        dual,
        gap,
        errors,
        keys.stride(0),
        keys.stride(1),
        keys.stride(2),
        keys.stride(3),
        active_pages.stride(0),
        vertices.stride(0),
        vertices.stride(1),
        landmark.stride(0),
        landmark.stride(1),
        bias.stride(0),
        errors.stride(0),
        scale,
        solver_lr,
        NUM_VERTICES=num_vertices,
        PAGE_SIZE=page_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        SOLVER_ITERS=solver_iters,
        UNIFORM_P=bool(uniform_p),
        FROM_PAGE_CACHE=False,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return landmark, bias, epsilon, dual, gap, errors


def build_query_robust_page_summaries_from_cache(
    k_cache: torch.Tensor,
    page_slots: torch.Tensor,
    vertices: torch.Tensor,
    active_pages: torch.Tensor,
    *,
    page_size: int,
    scale: float,
    solver_iters: int,
    solver_lr: float,
    uniform_p: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build page summaries by reading physical pages directly from KV cache."""

    if not k_cache.is_cuda or k_cache.ndim != 3 or not k_cache.is_contiguous():
        raise ValueError("QR cache summary expects contiguous CUDA KV [T, H, D].")
    if page_slots.ndim != 1 or page_slots.dtype not in {torch.int32, torch.int64}:
        raise TypeError("QR page_slots must be a 1D int32/int64 CUDA tensor.")
    if not page_slots.is_cuda or not page_slots.is_contiguous():
        raise ValueError("QR page_slots must be contiguous on the cache device.")
    if vertices.ndim != 3 or not vertices.is_cuda or not vertices.is_contiguous():
        raise ValueError("QR vertices must be contiguous CUDA [H, M, D].")
    if vertices.dtype != torch.float32:
        raise TypeError(f"QR Triton vertices must be FP32, got {vertices.dtype}.")
    pages = int(page_slots.shape[0])
    page_size = int(page_size)
    num_heads, head_dim = map(int, k_cache.shape[1:])
    if active_pages.shape != (pages,) or active_pages.dtype != torch.bool:
        raise TypeError("QR active_pages must be contiguous bool [pages].")
    if not active_pages.is_cuda or not active_pages.is_contiguous():
        raise ValueError("QR active_pages must be contiguous on the cache device.")
    if tuple(vertices.shape[::2]) != (num_heads, head_dim):
        raise ValueError("QR vertex and cache head dimensions do not match.")
    num_vertices = int(vertices.shape[1])
    solver_iters = int(solver_iters)
    solver_lr = float(solver_lr)
    scale = float(scale)
    if pages <= 0 or page_size <= 0 or num_vertices < 2 or solver_iters <= 0:
        raise ValueError("QR cache summary dimensions and solver_iters must be positive.")
    if not math.isfinite(scale) or not math.isfinite(solver_lr) or solver_lr <= 0:
        raise ValueError("QR cache summary scale and solver_lr must be finite and valid.")
    if k_cache.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported QR cache dtype: {k_cache.dtype}.")
    if vertices.device != k_cache.device or active_pages.device != k_cache.device:
        raise ValueError("QR cache summary inputs must share one CUDA device.")

    landmark = torch.empty(
        (pages, num_heads, head_dim), dtype=torch.bfloat16, device=k_cache.device
    )
    bias = torch.empty((pages, num_heads), dtype=torch.float32, device=k_cache.device)
    epsilon = torch.empty_like(bias)
    dual = torch.empty_like(bias)
    gap = torch.empty_like(bias)
    errors = torch.empty(
        (pages, num_heads, num_vertices), dtype=torch.float32, device=k_cache.device
    )
    block_m = triton.next_power_of_2(num_vertices)
    block_n = triton.next_power_of_2(page_size)
    block_d = triton.next_power_of_2(head_dim)
    _build_query_robust_summary_kernel[(pages, num_heads)](
        k_cache,
        vertices,
        active_pages,
        page_slots,
        landmark,
        bias,
        epsilon,
        dual,
        gap,
        errors,
        0,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        page_slots.stride(0),
        vertices.stride(0),
        vertices.stride(1),
        landmark.stride(0),
        landmark.stride(1),
        bias.stride(0),
        errors.stride(0),
        scale,
        solver_lr,
        NUM_VERTICES=num_vertices,
        PAGE_SIZE=page_size,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        SOLVER_ITERS=solver_iters,
        UNIFORM_P=bool(uniform_p),
        FROM_PAGE_CACHE=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return landmark, bias, epsilon, dual, gap, errors


__all__ = [
    "build_query_robust_page_summaries",
    "build_query_robust_page_summaries_from_cache",
]
