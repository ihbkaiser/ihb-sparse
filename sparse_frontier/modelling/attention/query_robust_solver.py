"""Empirical minimax/CVaR fitting for affine attention summaries.

This module deliberately contains no vLLM code.  A frozen empirical query
pool supplies rows of ``S`` for one key chunk; the solver returns the simplex
vector ``p`` defining the one-vector affine summary.  The only objectives
implemented here are the empirical worst case and weighted empirical CVaR.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only installs.
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _triton_full_support_minimax_kernel(
        scores_ptr,
        partitions_ptr,
        p_ptr,
        lower_ptr,
        upper_ptr,
        gap_ptr,
        converged_ptr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        max_iterations: tl.constexpr,
        armijo_backtracks: tl.constexpr,
        tolerance: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        """One full-support minimax solve per program.

        This is deliberately a pilot kernel: it fuses the fixed-full-support
        dual iterations, Armijo trials, and final full-pool certificate.  The
        production active-set schedule remains the reference implementation.
        """

        chunk = tl.program_id(0)
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        score_offsets = (
            chunk * num_queries * num_tokens
            + query_ids[:, None] * num_tokens
            + token_ids[None, :]
        )
        scores = tl.load(
            scores_ptr + score_offsets,
            mask=query_mask[:, None] & token_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partitions = tl.load(
            partitions_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        uniform = 1.0 / num_queries
        lam = tl.where(query_mask, uniform, 0.0)
        best_lam = lam
        best_value = -float("inf")

        for _ in range(max_iterations):
            z = tl.sum(scores * lam[:, None], axis=0)
            z_max = tl.max(tl.where(token_mask, z, -float("inf")), axis=0)
            exp_z = tl.exp(z - z_max)
            exp_z = tl.where(token_mask, exp_z, 0.0)
            log_partition = z_max + tl.log(tl.sum(exp_z, axis=0))
            p = exp_z / tl.sum(exp_z, axis=0)
            gradient = partitions - tl.sum(scores * p[None, :], axis=1)
            value = tl.sum(lam * partitions, axis=0) - log_partition

            step = 1.0
            accepted = 0
            accepted_lam = lam
            accepted_value = value
            last_candidate = lam
            last_candidate_value = value
            for _ in range(armijo_backtracks):
                log_lam = tl.log(tl.maximum(lam, 1e-30)) + step * gradient
                log_lam = tl.where(query_mask, log_lam, -float("inf"))
                candidate = tl.exp(log_lam - tl.max(log_lam, axis=0))
                candidate = candidate / tl.sum(candidate, axis=0)
                candidate = tl.where(query_mask, candidate, 0.0)
                candidate_z = tl.sum(scores * candidate[:, None], axis=0)
                candidate_z_max = tl.max(
                    tl.where(token_mask, candidate_z, -float("inf")), axis=0
                )
                candidate_exp_z = tl.exp(candidate_z - candidate_z_max)
                candidate_exp_z = tl.where(token_mask, candidate_exp_z, 0.0)
                candidate_log_partition = candidate_z_max + tl.log(
                    tl.sum(candidate_exp_z, axis=0)
                )
                candidate_value = (
                    tl.sum(candidate * partitions, axis=0) - candidate_log_partition
                )
                last_candidate = candidate
                last_candidate_value = candidate_value
                sufficient = candidate_value >= value + 1e-4 * tl.sum(
                    gradient * (candidate - lam), axis=0
                )
                take = (accepted == 0) & sufficient
                accepted_lam = tl.where(take, candidate, accepted_lam)
                accepted_value = tl.where(take, candidate_value, accepted_value)
                accepted = accepted | sufficient.to(tl.int32)
                step = tl.where(accepted != 0, step, step * 0.5)

            # The reference retains the last trial if Armijo never accepts.
            lam = tl.where(accepted != 0, accepted_lam, last_candidate)
            value = tl.where(accepted != 0, accepted_value, last_candidate_value)
            improved = value > best_value
            best_lam = tl.where(improved, lam, best_lam)
            best_value = tl.where(improved, value, best_value)

        z = tl.sum(scores * best_lam[:, None], axis=0)
        z_max = tl.max(tl.where(token_mask, z, -float("inf")), axis=0)
        exp_z = tl.exp(z - z_max)
        exp_z = tl.where(token_mask, exp_z, 0.0)
        p = exp_z / tl.sum(exp_z, axis=0)
        entropy = -tl.sum(p * tl.log(tl.maximum(p, 1e-30)), axis=0)
        primal = partitions - tl.sum(scores * p[None, :], axis=1) - entropy
        upper = tl.max(tl.where(query_mask, primal, -float("inf")), axis=0)
        gap = upper - best_value
        tl.store(p_ptr + chunk * num_tokens + token_ids, p, mask=token_mask)
        tl.store(lower_ptr + chunk, best_value)
        tl.store(upper_ptr + chunk, upper)
        tl.store(gap_ptr + chunk, gap)
        tl.store(converged_ptr + chunk, gap <= tolerance)


    @triton.jit
    def _triton_dual_state_kernel(
        scores_ptr,
        partitions_ptr,
        lam_ptr,
        p_ptr,
        value_ptr,
        gradient_ptr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        score_offsets = (
            chunk * num_queries * num_tokens
            + query_ids[:, None] * num_tokens
            + token_ids[None, :]
        )
        scores = tl.load(
            scores_ptr + score_offsets,
            mask=query_mask[:, None] & token_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partitions = tl.load(
            partitions_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        lam = tl.load(
            lam_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        z = tl.sum(scores * lam[:, None], axis=0)
        z_max = tl.max(tl.where(token_mask, z, -float("inf")), axis=0)
        exp_z = tl.exp(z - z_max)
        exp_z = tl.where(token_mask, exp_z, 0.0)
        normalizer = tl.sum(exp_z, axis=0)
        p = exp_z / normalizer
        value = tl.sum(lam * partitions, axis=0) - z_max - tl.log(normalizer)
        gradient = partitions - tl.sum(scores * p[None, :], axis=1)
        tl.store(p_ptr + chunk * num_tokens + token_ids, p, mask=token_mask)
        tl.store(value_ptr + chunk, value)
        tl.store(
            gradient_ptr + chunk * num_queries + query_ids,
            gradient,
            mask=query_mask,
        )


    @triton.jit
    def _triton_global_armijo_candidate_kernel(
        scores_ptr,
        partitions_ptr,
        lam_ptr,
        value_ptr,
        gradient_ptr,
        step_ptr,
        candidate_lam_ptr,
        candidate_p_ptr,
        candidate_value_ptr,
        candidate_gradient_ptr,
        sufficient_ptr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        score_offsets = (
            chunk * num_queries * num_tokens
            + query_ids[:, None] * num_tokens
            + token_ids[None, :]
        )
        scores = tl.load(
            scores_ptr + score_offsets,
            mask=query_mask[:, None] & token_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        partitions = tl.load(
            partitions_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        lam = tl.load(
            lam_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        value = tl.load(value_ptr + chunk).to(tl.float32)
        gradient = tl.load(
            gradient_ptr + chunk * num_queries + query_ids,
            mask=query_mask,
            other=0.0,
        ).to(tl.float32)
        step = tl.load(step_ptr).to(tl.float32)
        candidate_log_lam = tl.log(tl.maximum(lam, 1e-30)) + step * gradient
        candidate_log_lam = tl.where(query_mask, candidate_log_lam, -float("inf"))
        candidate = tl.exp(
            candidate_log_lam - tl.max(candidate_log_lam, axis=0)
        )
        candidate = candidate / tl.sum(candidate, axis=0)
        candidate = tl.where(query_mask, candidate, 0.0)
        z = tl.sum(scores * candidate[:, None], axis=0)
        z_max = tl.max(tl.where(token_mask, z, -float("inf")), axis=0)
        exp_z = tl.exp(z - z_max)
        exp_z = tl.where(token_mask, exp_z, 0.0)
        normalizer = tl.sum(exp_z, axis=0)
        candidate_p = exp_z / normalizer
        candidate_value = (
            tl.sum(candidate * partitions, axis=0) - z_max - tl.log(normalizer)
        )
        candidate_gradient = partitions - tl.sum(scores * candidate_p[None, :], axis=1)
        sufficient = candidate_value >= value + 1e-4 * tl.sum(
            gradient * (candidate - lam), axis=0
        )
        tl.store(
            candidate_lam_ptr + chunk * num_queries + query_ids,
            candidate,
            mask=query_mask,
        )
        tl.store(
            candidate_p_ptr + chunk * num_tokens + token_ids,
            candidate_p,
            mask=token_mask,
        )
        tl.store(candidate_value_ptr + chunk, candidate_value)
        tl.store(
            candidate_gradient_ptr + chunk * num_queries + query_ids,
            candidate_gradient,
            mask=query_mask,
        )
        tl.store(sufficient_ptr + chunk, sufficient)


    @triton.jit
    def _triton_global_armijo_control_kernel(
        trial_all_ptr, accepted_ptr, step_ptr, take_ptr
    ):
        accepted = tl.load(accepted_ptr).to(tl.int32)
        trial_all = tl.load(trial_all_ptr).to(tl.int32)
        take = (accepted == 0) & (trial_all != 0)
        tl.store(take_ptr, take)
        tl.store(accepted_ptr, accepted | take)
        tl.store(step_ptr, tl.where((accepted != 0) | take, tl.load(step_ptr), tl.load(step_ptr) * 0.5))


    @triton.jit
    def _triton_select_state_kernel(
        take_ptr,
        lam_ptr,
        p_ptr,
        value_ptr,
        gradient_ptr,
        candidate_lam_ptr,
        candidate_p_ptr,
        candidate_value_ptr,
        candidate_gradient_ptr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        take = tl.load(take_ptr) != 0
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        query_offset = chunk * num_queries + query_ids
        token_offset = chunk * num_tokens + token_ids
        lam = tl.load(lam_ptr + query_offset, mask=query_mask, other=0.0)
        candidate_lam = tl.load(candidate_lam_ptr + query_offset, mask=query_mask, other=0.0)
        gradient = tl.load(gradient_ptr + query_offset, mask=query_mask, other=0.0)
        candidate_gradient = tl.load(candidate_gradient_ptr + query_offset, mask=query_mask, other=0.0)
        p = tl.load(p_ptr + token_offset, mask=token_mask, other=0.0)
        candidate_p = tl.load(candidate_p_ptr + token_offset, mask=token_mask, other=0.0)
        value = tl.load(value_ptr + chunk)
        candidate_value = tl.load(candidate_value_ptr + chunk)
        tl.store(lam_ptr + query_offset, tl.where(take, candidate_lam, lam), mask=query_mask)
        tl.store(gradient_ptr + query_offset, tl.where(take, candidate_gradient, gradient), mask=query_mask)
        tl.store(p_ptr + token_offset, tl.where(take, candidate_p, p), mask=token_mask)
        tl.store(value_ptr + chunk, tl.where(take, candidate_value, value))


    @triton.jit
    def _triton_update_best_kernel(
        lam_ptr,
        p_ptr,
        value_ptr,
        best_lam_ptr,
        best_p_ptr,
        best_value_ptr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        query_offset = chunk * num_queries + query_ids
        token_offset = chunk * num_tokens + token_ids
        value = tl.load(value_ptr + chunk)
        best_value = tl.load(best_value_ptr + chunk)
        improved = value > best_value
        lam = tl.load(lam_ptr + query_offset, mask=query_mask, other=0.0)
        best_lam = tl.load(best_lam_ptr + query_offset, mask=query_mask, other=0.0)
        p = tl.load(p_ptr + token_offset, mask=token_mask, other=0.0)
        best_p = tl.load(best_p_ptr + token_offset, mask=token_mask, other=0.0)
        tl.store(best_lam_ptr + query_offset, tl.where(improved, lam, best_lam), mask=query_mask)
        tl.store(best_p_ptr + token_offset, tl.where(improved, p, best_p), mask=token_mask)
        tl.store(best_value_ptr + chunk, tl.where(improved, value, best_value))


    @triton.jit
    def _triton_certificate_kernel(
        scores_ptr,
        partitions_ptr,
        p_ptr,
        lower_ptr,
        upper_ptr,
        gap_ptr,
        converged_ptr,
        tolerance: tl.constexpr,
        num_queries: tl.constexpr,
        num_tokens: tl.constexpr,
        BLOCK_QUERIES: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
    ):
        chunk = tl.program_id(0)
        query_ids = tl.arange(0, BLOCK_QUERIES)
        token_ids = tl.arange(0, BLOCK_TOKENS)
        query_mask = query_ids < num_queries
        token_mask = token_ids < num_tokens
        score_offsets = (
            chunk * num_queries * num_tokens
            + query_ids[:, None] * num_tokens
            + token_ids[None, :]
        )
        scores = tl.load(scores_ptr + score_offsets, mask=query_mask[:, None] & token_mask[None, :], other=0.0)
        partitions = tl.load(partitions_ptr + chunk * num_queries + query_ids, mask=query_mask, other=0.0)
        p = tl.load(p_ptr + chunk * num_tokens + token_ids, mask=token_mask, other=0.0)
        entropy = -tl.sum(p * tl.log(tl.maximum(p, 1e-30)), axis=0)
        upper = tl.max(tl.where(query_mask, partitions - tl.sum(scores * p[None, :], axis=1) - entropy, -float("inf")), axis=0)
        lower = tl.load(lower_ptr + chunk)
        gap = upper - lower
        tl.store(upper_ptr + chunk, upper)
        tl.store(gap_ptr + chunk, gap)
        tl.store(converged_ptr + chunk, gap <= tolerance)


def triton_full_support_minimax_fit(
    scores: Tensor,
    partitions: Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    armijo_backtracks: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pilot fused full-support minimax fit with a final exact certificate.

    ``scores`` is ``[chunks, empirical_queries, chunk_tokens]``.  It is not
    wired into serving yet: callers must compare it against
    :func:`batched_independent_active_fit` before using it in a benchmark.
    """

    if triton is None:
        raise RuntimeError("Triton is required for the fused Query-Robust pilot")
    if not scores.is_cuda or not partitions.is_cuda:
        raise ValueError("Triton fused pilot requires CUDA scores and partitions")
    if scores.ndim != 3 or partitions.shape != scores.shape[:2]:
        raise ValueError("expected scores [chunks, queries, tokens] and matching partitions")
    if scores.dtype != torch.float32 or partitions.dtype != torch.float32:
        raise ValueError("Triton fused pilot currently requires float32 inputs")
    chunks, queries, tokens = scores.shape
    if not (1 <= queries <= 256 and 1 <= tokens <= 32):
        raise ValueError("pilot supports at most 256 empirical queries and 32 chunk tokens")
    if max_iterations < 1 or armijo_backtracks < 1 or tolerance <= 0:
        raise ValueError("invalid fused solver configuration")

    scores = scores.contiguous()
    partitions = partitions.contiguous()
    p = torch.empty((chunks, tokens), device=scores.device, dtype=torch.float32)
    lower = torch.empty(chunks, device=scores.device, dtype=torch.float32)
    upper = torch.empty_like(lower)
    gap = torch.empty_like(lower)
    converged = torch.empty(chunks, device=scores.device, dtype=torch.bool)
    _triton_full_support_minimax_kernel[(chunks,)](
        scores,
        partitions,
        p,
        lower,
        upper,
        gap,
        converged,
        num_queries=queries,
        num_tokens=tokens,
        max_iterations=max_iterations,
        armijo_backtracks=armijo_backtracks,
        tolerance=float(tolerance),
        BLOCK_QUERIES=triton.next_power_of_2(queries),
        BLOCK_TOKENS=triton.next_power_of_2(tokens),
        num_warps=4,
    )
    return p, lower, upper, gap, converged


@torch.no_grad()
def triton_full_support_minimax_retry_fit(
    scores: Tensor,
    partitions: Tensor,
    *,
    tolerance: float,
    fast_iterations: int,
    retry_iterations: int,
    armijo_backtracks: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    """Fail-closed fast/full-support solve followed by a hard-chunk retry.

    The retry starts from the deterministic original state, not an approximate
    tangent.  Thus an uncertified fast result can never leak into a serving
    decision; it is either replaced by the longer solve or remains explicitly
    uncertified for the caller's existing fallback policy.
    """

    if fast_iterations < 1 or retry_iterations < fast_iterations:
        raise ValueError("retry_iterations must be at least fast_iterations")
    p, lower, upper, gap, converged = triton_full_support_minimax_fit(
        scores,
        partitions,
        tolerance=tolerance,
        max_iterations=fast_iterations,
        armijo_backtracks=armijo_backtracks,
    )
    hard_indices = torch.nonzero(~converged, as_tuple=False).flatten()
    retry_count = int(hard_indices.numel())
    if retry_count == 0:
        return p, lower, upper, gap, converged, retry_count
    retry = triton_full_support_minimax_fit(
        scores.index_select(0, hard_indices),
        partitions.index_select(0, hard_indices),
        tolerance=tolerance,
        max_iterations=retry_iterations,
        armijo_backtracks=armijo_backtracks,
    )
    p.index_copy_(0, hard_indices, retry[0])
    lower.index_copy_(0, hard_indices, retry[1])
    upper.index_copy_(0, hard_indices, retry[2])
    gap.index_copy_(0, hard_indices, retry[3])
    converged.index_copy_(0, hard_indices, retry[4])
    return p, lower, upper, gap, converged, retry_count


@torch.no_grad()
def triton_global_armijo_full_support_fit(
    scores: Tensor,
    partitions: Tensor,
    *,
    tolerance: float,
    max_iterations: int,
    armijo_backtracks: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pilot a GPU-resident reproduction of batched global-Armijo semantics.

    This intentionally keeps the reference solver's batch-wide Armijo
    decision.  It removes host synchronizations but is limited to fixed full
    support; active-set growth is outside this pilot's contract.  It is an
    experimental diagnostic backend: a single hard chunk can reduce the
    shared step for the whole batch, so it is not the latency-oriented
    serving default.
    """

    if triton is None:
        raise RuntimeError("Triton is required for the global-Armijo pilot")
    if not scores.is_cuda or not partitions.is_cuda:
        raise ValueError("Triton global-Armijo pilot requires CUDA inputs")
    if scores.ndim != 3 or partitions.shape != scores.shape[:2]:
        raise ValueError("expected scores [chunks, queries, tokens] and matching partitions")
    if scores.dtype != torch.float32 or partitions.dtype != torch.float32:
        raise ValueError("Triton global-Armijo pilot currently requires float32 inputs")
    chunks, queries, tokens = scores.shape
    if not (1 <= queries <= 256 and 1 <= tokens <= 32):
        raise ValueError("pilot supports at most 256 empirical queries and 32 chunk tokens")
    if max_iterations < 1 or armijo_backtracks < 1 or tolerance <= 0:
        raise ValueError("invalid fused solver configuration")

    scores = scores.contiguous()
    partitions = partitions.contiguous()
    block_queries = triton.next_power_of_2(queries)
    block_tokens = triton.next_power_of_2(tokens)
    kernel_meta = {
        "num_queries": queries,
        "num_tokens": tokens,
        "BLOCK_QUERIES": block_queries,
        "BLOCK_TOKENS": block_tokens,
        "num_warps": 4,
    }
    lam = torch.full((chunks, queries), 1.0 / queries, device=scores.device, dtype=torch.float32)
    p = torch.empty((chunks, tokens), device=scores.device, dtype=torch.float32)
    value = torch.empty(chunks, device=scores.device, dtype=torch.float32)
    gradient = torch.empty_like(lam)
    _triton_dual_state_kernel[(chunks,)](
        scores, partitions, lam, p, value, gradient, **kernel_meta
    )
    best_lam, best_p, best_value = lam.clone(), p.clone(), value.clone()
    candidate_lam = torch.empty_like(lam)
    candidate_p = torch.empty_like(p)
    candidate_value = torch.empty_like(value)
    candidate_gradient = torch.empty_like(gradient)
    sufficient = torch.empty(chunks, device=scores.device, dtype=torch.bool)
    step = torch.empty(1, device=scores.device, dtype=torch.float32)
    accepted = torch.empty(1, device=scores.device, dtype=torch.uint8)
    take = torch.empty(1, device=scores.device, dtype=torch.uint8)

    for _ in range(max_iterations):
        step.fill_(1.0)
        accepted.zero_()
        for _ in range(armijo_backtracks):
            _triton_global_armijo_candidate_kernel[(chunks,)](
                scores,
                partitions,
                lam,
                value,
                gradient,
                step,
                candidate_lam,
                candidate_p,
                candidate_value,
                candidate_gradient,
                sufficient,
                **kernel_meta,
            )
            trial_all = torch.all(sufficient).reshape(1).to(torch.uint8)
            _triton_global_armijo_control_kernel[(1,)](
                trial_all, accepted, step, take, num_warps=1
            )
            _triton_select_state_kernel[(chunks,)](
                take,
                lam,
                p,
                value,
                gradient,
                candidate_lam,
                candidate_p,
                candidate_value,
                candidate_gradient,
                **kernel_meta,
            )

        # The reference keeps the final reduced-step candidate if none of the
        # Armijo trials accepts.  ``accepted`` stays GPU-resident here too.
        _triton_global_armijo_candidate_kernel[(chunks,)](
            scores,
            partitions,
            lam,
            value,
            gradient,
            step,
            candidate_lam,
            candidate_p,
            candidate_value,
            candidate_gradient,
            sufficient,
            **kernel_meta,
        )
        take.copy_(accepted == 0)
        _triton_select_state_kernel[(chunks,)](
            take,
            lam,
            p,
            value,
            gradient,
            candidate_lam,
            candidate_p,
            candidate_value,
            candidate_gradient,
            **kernel_meta,
        )
        _triton_update_best_kernel[(chunks,)](
            lam, p, value, best_lam, best_p, best_value, **kernel_meta
        )

    upper = torch.empty_like(best_value)
    gap = torch.empty_like(best_value)
    converged = torch.empty(chunks, device=scores.device, dtype=torch.bool)
    _triton_certificate_kernel[(chunks,)](
        scores,
        partitions,
        best_p,
        best_value,
        upper,
        gap,
        converged,
        tolerance=float(tolerance),
        **kernel_meta,
    )
    return best_p, best_value, upper, gap, converged


@dataclass(frozen=True)
class FitResult:
    """A robust fit and an auditable empirical certificate."""

    p: Tensor
    lower_bound: float | None
    upper_bound: float
    optimization_gap: float | None
    active_size: int
    converged: bool
    reason: str
    objective: str
    dual_weights: Tensor | None = None
    active_indices: Tensor | None = None


def _validate_matrix(S: Tensor, f: Tensor) -> None:
    if S.ndim != 2 or S.shape[0] < 1 or S.shape[1] < 1:
        raise ValueError("S must be a nonempty [samples, tokens] matrix")
    if f.ndim != 1 or f.shape[0] != S.shape[0]:
        raise ValueError("f must contain one log-partition value per sample")
    if not torch.is_floating_point(S) or not torch.is_floating_point(f):
        raise ValueError("S and f must be floating point")
    if not torch.isfinite(S).all() or not torch.isfinite(f).all():
        raise ValueError("S and f must be finite")


def _normalise_weights(weights: Tensor, n: int, device: torch.device) -> Tensor:
    if weights.ndim != 1 or weights.shape[0] != n:
        raise ValueError("weights must contain one value per empirical query")
    weights = weights.to(device=device, dtype=torch.float64)
    if not torch.isfinite(weights).all() or torch.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    total = weights.sum()
    if not bool(total > 0):
        raise ValueError("weights must have positive mass")
    return weights / total


def entropy(p: Tensor) -> Tensor:
    """Stable Shannon entropy for a simplex vector."""

    p = p.to(torch.float64)
    return -(p * p.clamp_min(torch.finfo(torch.float64).tiny).log()).sum()


def primal_gap(S: Tensor, f: Tensor, p: Tensor) -> Tensor:
    """Return ``g_i(p) = LSE(S_i) - S_i p - H(p)`` for every query."""

    _validate_matrix(S, f)
    if p.ndim != 1 or p.shape[0] != S.shape[1]:
        raise ValueError("p must have one simplex coordinate per key token")
    p_input = p.to(device=S.device, dtype=torch.float64)
    if not torch.isfinite(p_input).all() or torch.any(p_input < 0):
        raise ValueError("p must be finite and nonnegative")
    total = p_input.sum()
    if not bool(torch.isclose(total, torch.ones_like(total), atol=1e-9, rtol=1e-9)):
        raise ValueError("p must sum to one")
    work_dtype = torch.float64 if S.dtype == torch.float64 else torch.float32
    p = p_input.to(dtype=work_dtype)
    S_work, f_work = S.to(work_dtype), f.to(work_dtype)
    return f_work - S_work @ p - entropy(p).to(work_dtype)


def project_simplex(y: Tensor) -> Tensor:
    """Euclidean projection onto the probability simplex."""

    if y.ndim != 1 or y.numel() == 0 or not torch.isfinite(y).all():
        raise ValueError("simplex projection expects a finite nonempty vector")
    y64 = y.to(torch.float64)
    sorted_y, _ = torch.sort(y64, descending=True)
    cssv = torch.cumsum(sorted_y, dim=0) - 1.0
    index = torch.arange(1, y64.numel() + 1, device=y.device, dtype=torch.float64)
    feasible = sorted_y - cssv / index > 0
    if not bool(feasible.any()):
        rho = y64.numel() - 1
    else:
        rho = int(torch.nonzero(feasible, as_tuple=False)[-1].item())
    tau = cssv[rho] / float(rho + 1)
    return torch.clamp(y64 - tau, min=0.0)


def project_capped_simplex(y: Tensor, caps: Tensor, *, iterations: int = 80) -> Tensor:
    """Project onto ``{0 <= lambda <= caps, sum(lambda)=1}`` by bisection."""

    if y.ndim != 1 or caps.ndim != 1 or y.shape != caps.shape or y.numel() == 0:
        raise ValueError("capped-simplex inputs must be aligned nonempty vectors")
    y64 = y.to(torch.float64)
    caps64 = caps.to(device=y.device, dtype=torch.float64)
    if not torch.isfinite(y64).all() or not torch.isfinite(caps64).all():
        raise ValueError("capped-simplex inputs must be finite")
    if torch.any(caps64 < 0) or not bool(caps64.sum() >= 1.0 - 1e-12):
        raise ValueError("caps must be nonnegative and have total mass at least one")
    lo = torch.min(y64 - caps64)
    hi = torch.max(y64)
    for _ in range(max(1, int(iterations))):
        mid = (lo + hi) / 2
        mass = torch.clamp(y64 - mid, min=0.0).minimum(caps64).sum()
        # mass decreases monotonically as tau increases.
        if bool(mass > 1.0):
            lo = mid
        else:
            hi = mid
    result = torch.clamp(y64 - (lo + hi) / 2, min=0.0).minimum(caps64)
    # The bisection residual is tiny, but force exact feasibility without
    # changing the projection meaningfully when float64 is used.
    residual = 1.0 - result.sum()
    if bool(residual > 0):
        room = caps64 - result
        order = torch.argsort(room, descending=True)
        for idx in order.tolist():
            add = min(float(residual), float(room[idx]))
            result[idx] += add
            residual -= add
            if residual <= 1e-13:
                break
    elif bool(residual < 0):
        positive = result > 0
        order = torch.argsort(result.masked_fill(~positive, float("inf")))
        for idx in order.tolist():
            take = min(float(-residual), float(result[idx]))
            result[idx] -= take
            residual += take
            if residual >= -1e-13:
                break
    return result


def empirical_cvar(values: Tensor, weights: Tensor, alpha: float) -> Tensor:
    """Exact finite weighted CVaR of the upper tail."""

    if not 0.0 <= float(alpha) < 1.0:
        raise ValueError("CVaR alpha must lie in [0, 1)")
    w = _normalise_weights(weights, values.numel(), values.device)
    v = values.to(torch.float64)
    order = torch.argsort(v, descending=True, stable=True)
    sorted_v = v[order]
    sorted_w = w[order]
    tail_mass = 1.0 - float(alpha)
    cumulative_before = torch.cumsum(sorted_w, dim=0) - sorted_w
    take = torch.minimum(
        sorted_w,
        torch.clamp(torch.as_tensor(tail_mass, dtype=torch.float64, device=values.device) - cumulative_before, min=0.0),
    )
    covered = take.sum()
    if not bool(torch.isclose(covered, torch.as_tensor(tail_mass, dtype=torch.float64, device=values.device), atol=1e-10, rtol=1e-10)):
        raise RuntimeError("weighted CVaR tail could not be filled")
    return (take * sorted_v).sum() / tail_mass


def _dual_value(S: Tensor, f: Tensor, lam: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    S64, f64 = S.to(torch.float64), f.to(torch.float64)
    z = S64.T @ lam
    p = torch.softmax(z, dim=0)
    value = lam @ f64 - torch.logsumexp(z, dim=0)
    gradient = f64 - S64 @ p
    return value, p, gradient


def solve_simplex_dual(
    S: Tensor,
    f: Tensor,
    *,
    caps: Tensor | None = None,
    warm_start: Tensor | None = None,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> tuple[Tensor, Tensor, float, bool, int]:
    """Projected Armijo ascent on the finite empirical dual."""

    _validate_matrix(S, f)
    if max_iterations < 1 or tolerance <= 0:
        raise ValueError("max_iterations must be positive and tolerance must be positive")
    n = S.shape[0]
    if caps is None:
        caps64 = torch.ones(n, dtype=torch.float64, device=S.device)
        projection = project_simplex
    else:
        caps64 = caps.to(device=S.device, dtype=torch.float64)
        if caps64.ndim != 1 or caps64.shape[0] != n:
            raise ValueError("caps must align with dual samples")
        if torch.any(caps64 < 0) or not bool(caps64.sum() >= 1.0 - 1e-12):
            raise ValueError("dual caps are infeasible")
        projection = lambda value: project_capped_simplex(value, caps64)
    if warm_start is None:
        lam = projection(torch.zeros(n, dtype=torch.float64, device=S.device))
    else:
        if warm_start.shape != (n,):
            raise ValueError("dual warm start has the wrong shape")
        lam = projection(warm_start)
    value, p, gradient = _dual_value(S, f, lam)
    best_value, best_lam, best_p = value, lam.clone(), p.clone()
    converged = False
    for iteration in range(1, max_iterations + 1):
        direction = gradient
        step = 1.0
        accepted = False
        for _ in range(30):
            candidate = projection(lam + step * direction)
            delta = candidate - lam
            if bool(torch.linalg.vector_norm(delta) <= tolerance):
                accepted = True
                break
            candidate_value, candidate_p, candidate_gradient = _dual_value(S, f, candidate)
            if bool(candidate_value >= value + 1e-4 * torch.dot(direction, delta)):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
        lam = candidate
        value, p, gradient = _dual_value(S, f, lam)
        if bool(value > best_value):
            best_value, best_lam, best_p = value, lam.clone(), p.clone()
        projected_norm = torch.linalg.vector_norm(projection(lam + gradient) - lam)
        if bool(projected_norm <= tolerance):
            converged = True
            return best_lam, best_p, float(best_value.item()), converged, iteration
        if bool(torch.abs(value - best_value) <= tolerance and torch.linalg.vector_norm(gradient) <= 10 * tolerance):
            converged = True
            return best_lam, best_p, float(best_value.item()), converged, iteration
    return best_lam, best_p, float(best_value.item()), converged, max_iterations


def _default_initial_support(n: int, size: int) -> list[int]:
    if size < 1:
        raise ValueError("initial support must be positive")
    return list(range(min(n, size)))


def _augment_for_caps(active: list[int], caps: Tensor, weights: Tensor) -> list[int]:
    """Ensure an active CVaR support can represent one unit of original cap."""

    if float(caps[active].sum().item()) >= 1.0 - 1e-12:
        return active
    order = torch.argsort(weights, descending=True, stable=True).tolist()
    used = set(active)
    for index in order:
        if index not in used:
            active.append(int(index))
            used.add(int(index))
            if float(caps[active].sum().item()) >= 1.0 - 1e-12:
                break
    if float(caps[active].sum().item()) < 1.0 - 1e-12:
        raise RuntimeError("active support cannot carry one unit of the CVaR cap")
    return active


def active_robust_fit(
    S: Tensor,
    f: Tensor,
    *,
    objective: str,
    weights: Tensor | None = None,
    alpha: float = 0.95,
    initial_support: int | Sequence[int] = 64,
    max_support: int | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 500,
) -> FitResult:
    """Solve empirical minimax or CVaR using active constraints and scans."""

    _validate_matrix(S, f)
    objective = str(objective).lower()
    if objective not in {"minimax", "cvar"}:
        raise ValueError("objective must be 'minimax' or 'cvar'")
    if tolerance <= 0 or max_iterations < 1:
        raise ValueError("tolerance must be positive and max_iterations must be positive")
    n = S.shape[0]
    if weights is None:
        normalized_weights = torch.full((n,), 1.0 / n, dtype=torch.float64, device=S.device)
    else:
        normalized_weights = _normalise_weights(weights, n, S.device)
    if objective == "cvar" and not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must lie in [0, 1)")
    if isinstance(initial_support, int):
        active = _default_initial_support(n, initial_support)
    else:
        active = sorted(set(int(i) for i in initial_support))
        if not active:
            raise ValueError("initial support cannot be empty")
        if any(i < 0 or i >= n for i in active):
            raise ValueError("initial support contains an invalid query index")
    if max_support is None:
        max_support = n
    if max_support < len(active) or max_support > n:
        raise ValueError("max_support must contain the initial support and not exceed the pool")
    caps = None
    if objective == "cvar":
        caps = normalized_weights / (1.0 - float(alpha))
        active = _augment_for_caps(active, caps, normalized_weights)
        if len(active) > max_support:
            max_support = len(active)

    warm: Tensor | None = None
    while True:
        active_tensor = torch.tensor(active, device=S.device, dtype=torch.long)
        local_caps = caps[active_tensor] if caps is not None else None
        lam, p, lower, dual_converged, iterations = solve_simplex_dual(
            S[active_tensor],
            f[active_tensor],
            caps=local_caps,
            warm_start=warm,
            max_iterations=max_iterations,
            tolerance=min(1e-7, tolerance / 10),
        )
        gaps = primal_gap(S, f, p)
        if objective == "minimax":
            upper = float(gaps.max().item())
        else:
            upper = float(empirical_cvar(gaps, normalized_weights, alpha).item())
        gap = upper - lower
        if gap <= tolerance:
            return FitResult(
                p=p,
                lower_bound=lower,
                upper_bound=upper,
                optimization_gap=gap,
                active_size=len(active),
                converged=dual_converged,
                reason="gap" if dual_converged else "dual_iteration_limit",
                objective=objective,
                dual_weights=lam,
                active_indices=active_tensor,
            )
        if len(active) >= max_support:
            return FitResult(
                p=p,
                lower_bound=lower,
                upper_bound=upper,
                optimization_gap=gap,
                active_size=len(active),
                converged=False,
                reason="support_limit",
                objective=objective,
                dual_weights=lam,
                active_indices=active_tensor,
            )
        inactive = torch.ones(n, dtype=torch.bool, device=S.device)
        inactive[active_tensor] = False
        candidate_gaps = gaps.masked_fill(~inactive, float("-inf"))
        violator = int(torch.argmax(candidate_gaps).item())
        old_active = active
        active = active + [violator]
        warm = torch.cat([lam, torch.zeros(1, device=S.device, dtype=torch.float64)])
        if caps is not None and float(caps[active].sum().item()) < 1.0 - 1e-12:
            active = _augment_for_caps(active, caps, normalized_weights)
            # New coordinates are zero; retaining old weights is feasible after
            # the next projection and preserves the warm-start ordering.
            warm = torch.cat([warm, torch.zeros(len(active) - warm.numel(), device=S.device, dtype=torch.float64)])


def fit_p(
    S: Tensor,
    f: Tensor,
    weights: Tensor | None = None,
    *,
    objective: str,
    alpha: float = 0.95,
    tolerance: float = 1e-3,
    initial_support: int | Sequence[int] = 64,
    max_support: int | None = None,
    max_iterations: int = 500,
) -> FitResult:
    """Public objective dispatcher; only minimax and CVaR are accepted."""

    return active_robust_fit(
        S,
        f,
        objective=objective,
        weights=weights,
        alpha=alpha,
        initial_support=initial_support,
        max_support=max_support,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def _project_simplex_batch(y: Tensor) -> Tensor:
    """Row-wise simplex projection without Python/GPU synchronization."""

    sorted_y, _ = torch.sort(y, dim=-1, descending=True)
    cssv = torch.cumsum(sorted_y, dim=-1) - 1.0
    index = torch.arange(1, y.shape[-1] + 1, device=y.device, dtype=y.dtype)
    feasible = sorted_y - cssv / index > 0
    rho = feasible.sum(dim=-1).clamp_min(1).to(torch.long) - 1
    tau = cssv.gather(-1, rho.unsqueeze(-1)).squeeze(-1) / (rho + 1).to(y.dtype)
    return torch.clamp(y - tau.unsqueeze(-1), min=0.0)


def _project_capped_batch(y: Tensor, caps: Tensor, iterations: int = 50) -> Tensor:
    """Row-wise capped-simplex projection by vectorized bisection."""

    lo = (y - caps).amin(dim=-1)
    hi = y.amax(dim=-1)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        mass = torch.minimum(torch.clamp(y - mid.unsqueeze(-1), min=0.0), caps).sum(-1)
        lo = torch.where(mass > 1.0, mid, lo)
        hi = torch.where(mass > 1.0, hi, mid)
    result = torch.minimum(torch.clamp(y - ((lo + hi) / 2).unsqueeze(-1), min=0.0), caps)
    # Residual is below bisection precision; distribute it over available room
    # in a differentiability-irrelevant deterministic coordinate.
    residual = 1.0 - result.sum(-1)
    room = caps - result
    first_room = (room > 0).to(torch.long).argmax(-1)
    result.scatter_add_(-1, first_room.unsqueeze(-1), residual.clamp_min(0).unsqueeze(-1))
    return result


def _batched_dual_value(S: Tensor, f: Tensor, lam: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Dual value, recovered p, and gradient for [chunks, active, tokens]."""

    z = torch.einsum("can,ca->cn", S, lam)
    p = torch.softmax(z, dim=-1)
    value = (lam * f).sum(-1) - torch.logsumexp(z, dim=-1)
    gradient = f - torch.einsum("can,cn->ca", S, p)
    return value, p, gradient


def _mask_active_query_gaps(
    all_gaps: Tensor, active_indices: Tensor, active_mask: Tensor
) -> Tensor:
    """Mask query ids in the active support, ignoring stale support slots."""

    if all_gaps.ndim != 2 or active_indices.shape != active_mask.shape:
        raise ValueError("active query gaps expect matching [rows, queries] tensors")
    active_query = torch.zeros_like(all_gaps, dtype=torch.int64)
    active_query.scatter_add_(1, active_indices, active_mask.to(torch.int64))
    return all_gaps.masked_fill(active_query > 0, float("-inf"))


@torch.no_grad()
def batched_full_support_newton_fit(
    S: Tensor,
    f: Tensor,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 32,
    armijo_backtracks: int = 10,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Solve small full-support minimax problems with independent Newton steps.

    Query-Robust's online Pile configuration normally has only a handful of
    empirical queries per KV head.  For that regime, mirror ascent spends
    hundreds or thousands of iterations moving along a badly conditioned
    simplex.  The dual Hessian is a query covariance, so a constrained Newton
    step solves the local KKT system directly.  All chunk rows keep their own
    line-search state; there is no batch-wide synchronization or shared step.

    This is intentionally limited to small pools.  Larger pools should use
    the active-set reference path until a dedicated matrix-free Newton kernel
    is available.  The final primal scan is always performed, so ``converged``
    means the returned summary has an explicit minimax certificate.
    """

    if S.ndim != 3 or f.ndim != 2 or S.shape[:2] != f.shape:
        raise ValueError("full-support Newton fit expects S [chunks,queries,tokens] and matching f")
    if not torch.is_floating_point(S) or not torch.is_floating_point(f):
        raise ValueError("full-support Newton fit requires floating-point inputs")
    if not torch.isfinite(S).all() or not torch.isfinite(f).all():
        raise ValueError("full-support Newton fit requires finite inputs")
    if tolerance <= 0 or max_iterations < 1 or armijo_backtracks < 1:
        raise ValueError("invalid full-support Newton configuration")
    chunks, queries, tokens = S.shape
    if not (1 <= queries <= 32 and 1 <= tokens):
        raise ValueError("full-support Newton fit supports at most 32 empirical queries")

    # B200's production score range is roughly [-27, 27].  The extra
    # precision is used only for the tiny dual/KKT state; the input score
    # matrix and returned summary remain FP32 at the serving boundary.
    work_dtype = torch.float64
    work_S = S.to(device=S.device, dtype=work_dtype).contiguous()
    work_f = f.to(device=S.device, dtype=work_dtype).contiguous()
    identity = torch.eye(queries, device=S.device, dtype=work_dtype)
    ones = torch.ones(queries, device=S.device, dtype=work_dtype)
    active_indices = torch.zeros(
        chunks, queries, device=S.device, dtype=torch.long
    )
    active_indices[:, 0] = 0
    active_mask = torch.zeros(
        chunks, queries, device=S.device, dtype=torch.bool
    )
    active_mask[:, 0] = True
    active_size = torch.ones(chunks, device=S.device, dtype=torch.long)
    lam = active_mask.to(work_dtype)
    iterations_used = torch.zeros(chunks, device=S.device, dtype=torch.int32)
    best_p = torch.empty((chunks, tokens), device=S.device, dtype=work_dtype)
    best_lower = torch.full(
        (chunks,), -float("inf"), device=S.device, dtype=work_dtype
    )
    best_upper = torch.full(
        (chunks,), float("inf"), device=S.device, dtype=work_dtype
    )
    best_gap = torch.full(
        (chunks,), float("inf"), device=S.device, dtype=work_dtype
    )

    for _ in range(queries + 1):
        gather_index = active_indices.unsqueeze(-1).expand(-1, -1, tokens)
        local_S = work_S.gather(1, gather_index)
        local_f = work_f.gather(1, active_indices)
        mask = active_mask.to(work_dtype)

        def stats(current_lam: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            z = torch.bmm(local_S.transpose(1, 2), current_lam.unsqueeze(-1)).squeeze(-1)
            current_p = torch.softmax(z, dim=-1)
            current_value = (current_lam * local_f).sum(-1) - torch.logsumexp(z, dim=-1)
            current_gradient = local_f - torch.bmm(
                local_S, current_p.unsqueeze(-1)
            ).squeeze(-1)
            mean = torch.bmm(local_S, current_p.unsqueeze(-1)).squeeze(-1)
            centered = local_S - mean.unsqueeze(-1)
            covariance = torch.einsum(
                "ct,cit,cjt->cij", current_p, centered, centered
            )
            return current_value, current_p, current_gradient, -covariance

        def certificate(current_p: Tensor, current_value: Tensor) -> tuple[Tensor, Tensor]:
            entropy = -(
                current_p
                * current_p.clamp_min(torch.finfo(work_dtype).tiny).log()
            ).sum(-1)
            all_gaps = work_f - torch.bmm(
                work_S, current_p.unsqueeze(-1)
            ).squeeze(-1) - entropy.unsqueeze(-1)
            upper = all_gaps.amax(-1)
            return upper, upper - current_value

        value, p, gradient, hessian = stats(lam)
        for iteration in range(1, max_iterations + 1):
            scale = hessian.abs().amax(dim=(-2, -1)).clamp_min(1.0)
            active_hessian = hessian * mask.unsqueeze(-1) * mask.unsqueeze(-2)
            active_hessian = active_hessian - (
                1e-5 * scale
            ).view(-1, 1, 1) * torch.diag_embed(mask)
            inactive_diagonal = torch.diag_embed(1.0 - mask)
            system = torch.zeros(
                chunks, queries + 1, queries + 1,
                device=S.device,
                dtype=work_dtype,
            )
            system[:, :queries, :queries] = active_hessian - inactive_diagonal
            system[:, :queries, queries] = mask
            system[:, queries, :queries] = mask
            rhs = torch.cat(
                [
                    -(gradient * mask),
                    torch.zeros(chunks, 1, device=S.device, dtype=work_dtype),
                ],
                dim=-1,
            )
            solution = torch.linalg.solve(system, rhs.unsqueeze(-1)).squeeze(-1)
            direction = solution[:, :queries] * mask
            directional = (gradient * direction).sum(-1)
            fallback_direction = (gradient - gradient.mean(-1, keepdim=True)) * mask
            use_fallback = (~torch.isfinite(direction).all(-1)) | (directional <= 0)
            direction = torch.where(
                use_fallback.unsqueeze(-1), fallback_direction, direction
            )
            directional = (gradient * direction).sum(-1)

            negative = direction < 0
            ratios = torch.where(
                negative,
                -lam / direction.clamp_max(-torch.finfo(work_dtype).tiny),
                torch.full_like(direction, float("inf")),
            )
            feasible_step = ratios.amin(-1).clamp_min(0.0)
            step = torch.minimum(
                torch.ones_like(feasible_step), feasible_step * 0.99
            ).clamp_min(1e-6)
            accepted = torch.zeros(chunks, device=S.device, dtype=torch.bool)
            accepted_lam = lam
            accepted_value = value
            for _ in range(armijo_backtracks):
                candidate_lam = lam + step.unsqueeze(-1) * direction
                candidate_lam = candidate_lam.clamp_min(0.0) * mask
                candidate_lam = candidate_lam / candidate_lam.sum(
                    -1, keepdim=True
                ).clamp_min(1e-30)
                candidate_value, _, _, _ = stats(candidate_lam)
                sufficient = candidate_value >= value + 1e-4 * step * directional
                take = ~accepted & sufficient
                accepted_lam = torch.where(
                    take.unsqueeze(-1), candidate_lam, accepted_lam
                )
                accepted_value = torch.where(take, candidate_value, accepted_value)
                accepted |= sufficient
                step = torch.where(accepted, step, step * 0.5)

            lam = torch.where(accepted.unsqueeze(-1), accepted_lam, lam)
            value, p, gradient, hessian = stats(lam)
            iterations_used += 1
            tau = (lam * gradient).sum(-1, keepdim=True)
            positive = (lam > 1e-6) & active_mask
            positive_residual = (gradient - tau).abs().masked_fill(
                ~positive, 0.0
            ).amax(-1)
            boundary_residual = (gradient - tau).clamp_min(0.0).masked_fill(
                ~active_mask, 0.0
            ).amax(-1)
            residual = torch.maximum(positive_residual, boundary_residual)
            if bool((residual <= min(1e-5, tolerance / 10)).all()):
                break

        upper, gap = certificate(p, value)
        improved = gap < best_gap
        best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best_lower = torch.where(improved, value, best_lower)
        best_upper = torch.where(improved, upper, best_upper)
        best_gap = torch.where(improved, gap, best_gap)
        done = gap <= tolerance
        if bool(done.all()) or bool((active_size >= queries).all()):
            break

        # A Newton step approaches a boundary from the interior.  Keeping a
        # nearly-zero coordinate active would cap every subsequent step by
        # that tiny mass.  Drop only numerically boundary coordinates; the
        # full-pool scan below can re-admit one if it is still a violator, then
        # compact the per-row support so the next violator has a free slot.
        removable = active_mask & (lam <= max(1e-6, float(tolerance) / 10.0))
        removable &= active_size.unsqueeze(-1) > 1
        removed_any = bool(removable.any())
        if removed_any:
            kept = active_mask & ~removable
            order = torch.argsort(kept.to(torch.int64), dim=-1, descending=True)
            active_indices = torch.gather(active_indices, 1, order)
            lam = torch.gather(lam, 1, order)
            active_size = active_size - removable.sum(-1).to(torch.long)
            active_mask = (
                torch.arange(queries, device=S.device).unsqueeze(0)
                < active_size.unsqueeze(-1)
            )
            lam = lam * active_mask.to(work_dtype)
            # Re-solve the reduced support before scanning for a new
            # violator.  The current certificate used the pre-removal
            # tangent and would otherwise immediately re-add the coordinate
            # that was just removed.
            continue
        unfinished = ~done & (active_size < queries)
        if not bool(unfinished.any()):
            break
        all_gaps = work_f - torch.bmm(
            work_S, p.unsqueeze(-1)
        ).squeeze(-1) - (
            -(
                p * p.clamp_min(torch.finfo(work_dtype).tiny).log()
            ).sum(-1)
        ).unsqueeze(-1)
        # ``active_indices`` stores query ids in support slots; masking the
        # slot positions themselves would allow the same violator to be
        # inserted repeatedly and prevent the support from growing.  Ignore
        # inactive slots too: after a boundary removal, their old query ids
        # are stale and must not hide a valid violator.
        masked_gaps = _mask_active_query_gaps(
            all_gaps, active_indices, active_mask
        )
        candidates = masked_gaps.argmax(-1)
        rows = torch.nonzero(unfinished, as_tuple=False).flatten()
        slots = active_size[rows]
        active_indices[rows, slots] = candidates[rows]
        active_mask[rows, slots] = True
        active_size[rows] += 1
        lam = torch.where(
            active_mask,
            torch.where(
                active_mask & (lam > 0), lam, torch.zeros_like(lam)
            ),
            torch.zeros_like(lam),
        )

    converged = best_gap <= tolerance
    # The regular active Newton pass is cheap for interior solutions, but a
    # few high-dynamic-range rows can still land on a support boundary.  For
    # the serving pool (Q=4), solve only those rows by finite support
    # enumeration; this keeps the common path fast and makes the returned
    # certificate fail closed instead of exposing a stalled iterate.
    if queries <= 4 and bool((~converged).any()):
        hard_indices = torch.nonzero(~converged, as_tuple=False).flatten()
        hard = batched_exhaustive_newton_fit(
            S.index_select(0, hard_indices),
            f.index_select(0, hard_indices),
            tolerance=tolerance,
            max_iterations=min(max_iterations, 16),
            armijo_backtracks=armijo_backtracks,
        )
        best_p.index_copy_(0, hard_indices, hard[0].to(best_p.dtype))
        best_lower.index_copy_(0, hard_indices, hard[1].to(best_lower.dtype))
        best_upper.index_copy_(0, hard_indices, hard[2].to(best_upper.dtype))
        best_gap.index_copy_(0, hard_indices, hard[3].to(best_gap.dtype))
        iterations_used.index_add_(0, hard_indices, hard[5].to(iterations_used.dtype))
        converged = best_gap <= tolerance
    return (
        best_p.to(torch.float32),
        best_lower.to(torch.float32),
        best_upper.to(torch.float32),
        best_gap.to(torch.float32),
        converged,
        iterations_used,
    )


@torch.no_grad()
def batched_exhaustive_newton_fit(
    S: Tensor,
    f: Tensor,
    *,
    tolerance: float = 1e-3,
    max_iterations: int = 16,
    armijo_backtracks: int = 8,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Solve minimax duals by enumerating supports for very small query pools.

    For ``Q <= 4`` there are at most 15 nonempty supports.  Solving each
    support as an interior Newton problem makes boundary optima explicit
    instead of forcing an interior iterate to asymptotically reach zero.  The
    largest dual value is selected and certified against the complete pool.
    This is a bounded, deterministic hard-case path for the low-query serving
    configuration; larger pools deliberately remain on the reference route.
    """

    if S.ndim != 3 or f.ndim != 2 or S.shape[:2] != f.shape:
        raise ValueError("exhaustive Newton fit expects S [chunks,queries,tokens] and matching f")
    if not torch.is_floating_point(S) or not torch.is_floating_point(f):
        raise ValueError("exhaustive Newton fit requires floating-point inputs")
    if not torch.isfinite(S).all() or not torch.isfinite(f).all():
        raise ValueError("exhaustive Newton fit requires finite inputs")
    if tolerance <= 0 or max_iterations < 1 or armijo_backtracks < 1:
        raise ValueError("invalid exhaustive Newton configuration")
    chunks, queries, tokens = S.shape
    if not (1 <= queries <= 4 and 1 <= tokens):
        raise ValueError("exhaustive Newton fit supports at most 4 empirical queries")

    work_dtype = torch.float64
    work_S = S.to(device=S.device, dtype=work_dtype).contiguous()
    work_f = f.to(device=S.device, dtype=work_dtype).contiguous()
    best_p = torch.empty((chunks, tokens), device=S.device, dtype=work_dtype)
    best_lower = torch.full(
        (chunks,), -float("inf"), device=S.device, dtype=work_dtype
    )
    best_upper = torch.full(
        (chunks,), float("inf"), device=S.device, dtype=work_dtype
    )
    best_gap = torch.full(
        (chunks,), float("inf"), device=S.device, dtype=work_dtype
    )
    iterations_used = torch.zeros(chunks, device=S.device, dtype=torch.int32)

    for support_code in range(1, 1 << queries):
        indices = torch.tensor(
            [index for index in range(queries) if support_code & (1 << index)],
            device=S.device,
            dtype=torch.long,
        )
        local_S = work_S.index_select(1, indices)
        local_f = work_f.index_select(1, indices)
        support_size = indices.numel()
        lam = torch.full(
            (chunks, support_size),
            1.0 / support_size,
            device=S.device,
            dtype=work_dtype,
        )
        identity = torch.eye(support_size, device=S.device, dtype=work_dtype)
        ones = torch.ones(support_size, device=S.device, dtype=work_dtype)

        def stats(current_lam: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            z = torch.bmm(
                local_S.transpose(1, 2), current_lam.unsqueeze(-1)
            ).squeeze(-1)
            current_p = torch.softmax(z, dim=-1)
            current_value = (current_lam * local_f).sum(-1) - torch.logsumexp(
                z, dim=-1
            )
            current_gradient = local_f - torch.bmm(
                local_S, current_p.unsqueeze(-1)
            ).squeeze(-1)
            mean = torch.bmm(local_S, current_p.unsqueeze(-1)).squeeze(-1)
            centered = local_S - mean.unsqueeze(-1)
            covariance = torch.einsum(
                "ct,cit,cjt->cij", current_p, centered, centered
            )
            return current_value, current_p, current_gradient, -covariance

        value, p, gradient, hessian = stats(lam)
        for iteration in range(1, max_iterations + 1):
            if support_size == 1:
                break
            scale = hessian.abs().amax(dim=(-2, -1)).clamp_min(1.0)
            system = hessian - (
                1e-5 * scale
            ).view(-1, 1, 1) * identity
            kkt = torch.zeros(
                chunks,
                support_size + 1,
                support_size + 1,
                device=S.device,
                dtype=work_dtype,
            )
            kkt[:, :support_size, :support_size] = system
            kkt[:, :support_size, support_size] = ones
            kkt[:, support_size, :support_size] = ones
            rhs = torch.cat(
                [
                    -gradient,
                    torch.zeros(chunks, 1, device=S.device, dtype=work_dtype),
                ],
                dim=-1,
            )
            direction = torch.linalg.solve(kkt, rhs.unsqueeze(-1)).squeeze(-1)[
                :, :support_size
            ]
            directional = (gradient * direction).sum(-1)
            fallback = gradient - gradient.mean(-1, keepdim=True)
            use_fallback = (~torch.isfinite(direction).all(-1)) | (directional <= 0)
            direction = torch.where(use_fallback.unsqueeze(-1), fallback, direction)
            directional = (gradient * direction).sum(-1)
            ratios = torch.where(
                direction < 0,
                -lam / direction.clamp_max(-torch.finfo(work_dtype).tiny),
                torch.full_like(direction, float("inf")),
            )
            step = torch.minimum(
                torch.ones(chunks, device=S.device, dtype=work_dtype),
                ratios.amin(-1) * 0.99,
            ).clamp_min(1e-8)
            accepted = torch.zeros(chunks, device=S.device, dtype=torch.bool)
            accepted_lam = lam
            accepted_value = value
            for _ in range(armijo_backtracks):
                candidate = lam + step.unsqueeze(-1) * direction
                candidate = candidate.clamp_min(0.0)
                candidate = candidate / candidate.sum(-1, keepdim=True).clamp_min(1e-30)
                candidate_value, _, _, _ = stats(candidate)
                sufficient = candidate_value >= value + 1e-4 * step * directional
                take = ~accepted & sufficient
                accepted_lam = torch.where(take.unsqueeze(-1), candidate, accepted_lam)
                accepted_value = torch.where(take, candidate_value, accepted_value)
                accepted |= sufficient
                step = torch.where(accepted, step, step * 0.5)
            lam = torch.where(accepted.unsqueeze(-1), accepted_lam, lam)
            value, p, gradient, hessian = stats(lam)
            iterations_used += 1
            tau = (lam * gradient).sum(-1, keepdim=True)
            residual = (gradient - tau).abs().amax(-1)
            if bool((residual <= min(1e-7, tolerance / 10)).all()):
                break

        entropy = -(
            p * p.clamp_min(torch.finfo(work_dtype).tiny).log()
        ).sum(-1)
        all_gaps = work_f - torch.bmm(
            work_S, p.unsqueeze(-1)
        ).squeeze(-1) - entropy.unsqueeze(-1)
        upper = all_gaps.amax(-1)
        gap = upper - value
        improved = value > best_lower
        best_p = torch.where(improved.unsqueeze(-1), p, best_p)
        best_lower = torch.where(improved, value, best_lower)
        best_upper = torch.where(improved, upper, best_upper)
        best_gap = torch.where(improved, gap, best_gap)

    converged = best_gap <= tolerance
    return (
        best_p.to(torch.float32),
        best_lower.to(torch.float32),
        best_upper.to(torch.float32),
        best_gap.to(torch.float32),
        converged,
        iterations_used,
    )


def batched_active_robust_fit(
    S: Tensor,
    f: Tensor,
    *,
    objective: str,
    weights: Tensor | None = None,
    alpha: float = 0.95,
    initial_support: int = 64,
    max_support: int | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Fit many chunks together, sharing a growing active-index union.

    Args:
        S: ``[chunks, queries, tokens]`` score matrix.
        f: ``[chunks, queries]`` exact log-partition values.

    Returns ``(p, lower, upper, gap, active_size, converged, iterations)``;
    each first six item is per chunk.  A union support is only a computational
    device: every row retains zero dual mass on queries not active for it, and
    every upper bound scans the complete empirical pool.
    """

    if S.ndim != 3 or f.ndim != 2 or S.shape[:2] != f.shape:
        raise ValueError("batched robust fit expects S [chunks,queries,tokens] and f [chunks,queries]")
    if objective not in {"minimax", "cvar"}:
        raise ValueError("objective must be minimax or cvar")
    chunks, n, _ = S.shape
    if weights is None:
        w = torch.full((n,), 1.0 / n, dtype=torch.float32, device=S.device)
    else:
        w = _normalise_weights(weights, n, S.device).to(torch.float32)
    if objective == "cvar" and not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must lie in [0,1)")
    caps_all = w / (1.0 - float(alpha)) if objective == "cvar" else None
    initial_support = max(1, min(int(initial_support), n))
    max_support = n if max_support is None else int(max_support)
    if max_support < initial_support or max_support > n:
        raise ValueError("invalid batched robust support limit")
    active = list(range(initial_support))
    if caps_all is not None and float(caps_all[active].sum()) < 1.0 - 1e-6:
        for index in torch.argsort(w, descending=True, stable=True).tolist():
            if index not in active:
                active.append(int(index))
                if float(caps_all[active].sum()) >= 1.0 - 1e-6:
                    break
        if float(caps_all[active].sum()) < 1.0 - 1e-6:
            raise ValueError("CVaR active support cannot carry one unit of the original cap")
    if caps_all is not None and float(caps_all[active].sum()) < 1.0 - 1e-6:
        for index in torch.argsort(w, descending=True, stable=True).tolist():
            if index not in active:
                active.append(int(index))
                if float(caps_all[active].sum()) >= 1.0 - 1e-6:
                    break
    warm: Tensor | None = None
    converged = torch.zeros(chunks, dtype=torch.bool, device=S.device)
    lower = torch.full((chunks,), float("-inf"), dtype=torch.float32, device=S.device)
    upper = torch.full((chunks,), float("inf"), dtype=torch.float32, device=S.device)
    gap = torch.full((chunks,), float("inf"), dtype=torch.float32, device=S.device)
    iterations_used = torch.zeros(chunks, dtype=torch.int32, device=S.device)
    p = torch.full((chunks, S.shape[-1]), 1.0 / S.shape[-1], dtype=torch.float32, device=S.device)
    while True:
        active_tensor = torch.tensor(active, device=S.device, dtype=torch.long)
        local_S = S[:, active_tensor].float()
        local_f = f[:, active_tensor].float()
        local_caps = caps_all[active_tensor] if caps_all is not None else None
        project = (
            (lambda value: _project_capped_batch(value, local_caps))
            if local_caps is not None
            else _project_simplex_batch
        )
        if warm is None or warm.shape != (chunks, len(active)):
            lam = project(torch.zeros(chunks, len(active), device=S.device, dtype=torch.float32))
        else:
            lam = project(warm)
        value, p, gradient = _batched_dual_value(local_S, local_f, lam)
        best_value = value.clone()
        best_lam = lam.clone()
        best_p = p.clone()
        for iteration in range(1, max_iterations + 1):
            step = 1.0
            candidate = project(lam + step * gradient)
            candidate_value, candidate_p, candidate_gradient = _batched_dual_value(
                local_S, local_f, candidate
            )
            # Armijo backtracking is performed per batch with a shared step;
            # retaining rows that already satisfy the condition avoids a
            # Python loop over chunks while keeping ascent monotonic.
            for _ in range(12):
                delta = candidate - lam
                sufficient = candidate_value >= value + 1e-4 * (gradient * delta).sum(-1)
                if bool(sufficient.all()):
                    break
                step *= 0.5
                candidate = project(lam + step * gradient)
                candidate_value, candidate_p, candidate_gradient = _batched_dual_value(
                    local_S, local_f, candidate
                )
            lam, value, p, gradient = candidate, candidate_value, candidate_p, candidate_gradient
            improved = value > best_value
            best_value = torch.where(improved, value, best_value)
            best_lam = torch.where(improved.unsqueeze(-1), lam, best_lam)
            best_p = torch.where(improved.unsqueeze(-1), p, best_p)
            iterations_used[:] = iteration
            projected_norm = torch.linalg.vector_norm(project(lam + gradient) - lam, dim=-1)
            if bool((projected_norm <= min(1e-5, tolerance / 10)).all()):
                break
        lam, p = best_lam, best_p
        row_entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
        all_gaps = f.float() - torch.einsum("can,cn->ca", S.float(), p) - row_entropy.unsqueeze(-1)
        if objective == "minimax":
            upper = all_gaps.amax(-1)
        else:
            # The weighted tail is evaluated on the complete pool.
            upper = torch.stack([empirical_cvar(all_gaps[row], w, alpha).float() for row in range(chunks)])
        lower = best_value
        gap = upper - lower
        done = gap <= tolerance
        converged |= done
        if bool(done.all()) or len(active) >= max_support:
            return p, lower, upper, gap, torch.full((chunks,), len(active), dtype=torch.int32, device=S.device), converged, iterations_used
        # Add the largest inactive violation from every unfinished chunk.  The
        # union is finite and each row's dual remains feasible with zeros on
        # coordinates added for other rows.
        used = set(active)
        additions: list[int] = []
        for row in range(chunks):
            if bool(done[row]):
                continue
            order = torch.argsort(all_gaps[row], descending=True, stable=True).tolist()
            for index in order:
                if index not in used:
                    additions.append(int(index))
                    used.add(int(index))
                    break
        if not additions:
            return p, lower, upper, gap, torch.full((chunks,), len(active), dtype=torch.int32, device=S.device), converged, iterations_used
        additions = additions[: max_support - len(active)]
        old_len = len(active)
        active.extend(additions)
        warm = torch.cat([lam, torch.zeros(chunks, len(active) - old_len, device=S.device)], dim=-1)


def batched_independent_active_fit(
    S: Tensor,
    f: Tensor,
    *,
    objective: str,
    weights: Tensor | None = None,
    alpha: float = 0.95,
    initial_support: int = 64,
    max_support: int | None = None,
    tolerance: float = 1e-3,
    max_iterations: int = 100,
    active_additions_per_round: int = 1,
    compute_certificate: bool = True,
    armijo: bool = True,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Batched active-set solves with an independent support per chunk.

    Unlike the union-support accelerator above, this routine preserves the
    exact per-chunk active-set semantics: a violator discovered in one chunk
    does not consume another chunk's support budget.  The score matrix remains
    batched, so all dual updates and full-pool scans are vectorized.  More than
    one inactive violator may be admitted per outer round.  That only changes
    the active-set schedule (and reduces GPU synchronisation/dual solves), not
    the objective, support budget, or full-pool certificate.  An explicitly
    exploratory fixed-support route may set ``compute_certificate=False`` to
    omit the final full-pool scan.  This never changes ``p``; it simply
    declines to report a certificate and is invalid when support growth or a
    fail-closed decision is required.
    """

    if S.ndim != 3 or f.ndim != 2 or S.shape[:2] != f.shape:
        raise ValueError("batched robust fit expects S [chunks,queries,tokens] and f [chunks,queries]")
    if objective not in {"minimax", "cvar"}:
        raise ValueError("objective must be minimax or cvar")
    if int(active_additions_per_round) < 1:
        raise ValueError("active_additions_per_round must be positive")
    chunks, n, tokens = S.shape
    if weights is None:
        w = torch.full((n,), 1.0 / n, dtype=torch.float32, device=S.device)
    else:
        w = _normalise_weights(weights, n, S.device).to(torch.float32)
    if objective == "cvar" and not 0.0 <= alpha < 1.0:
        raise ValueError("CVaR alpha must lie in [0,1)")
    caps_all = w / (1.0 - float(alpha)) if objective == "cvar" else None
    required = (
        int(math.ceil((1.0 - float(alpha)) * n - 1e-12))
        if caps_all is not None
        else int(initial_support)
    )
    initial = min(n, max(1, int(initial_support), required))
    # A finite empirical source cannot expose more than ``n`` scenarios.  Treat
    # a larger configured cap as the natural finite-population cap so the
    # runtime remains fail-closed on the finite support it was given.
    limit = n if max_support is None else min(n, int(max_support))
    if limit < initial:
        raise ValueError("invalid batched robust support limit")
    if not compute_certificate and limit != initial:
        raise ValueError(
            "certificate-free fitting requires max_support equal to initial_support"
        )
    if not armijo and compute_certificate:
        raise ValueError("unguarded dual updates cannot report a certificate")
    seed_indices = list(range(initial))
    if caps_all is not None:
        # A count-based CVaR floor is insufficient for unequal weights.  Grow
        # the shared seed until the original empirical caps can carry one unit
        # of dual mass, or fail before allocating an infeasible projection.
        if float(caps_all[seed_indices].sum().item()) < 1.0 - 1e-6:
            for index in torch.argsort(w, descending=True, stable=True).tolist():
                if index not in seed_indices:
                    seed_indices.append(int(index))
                    if float(caps_all[seed_indices].sum().item()) >= 1.0 - 1e-6:
                        break
        if float(caps_all[seed_indices].sum().item()) < 1.0 - 1e-6:
            raise ValueError("CVaR active support cannot carry one unit of the original cap")
        if len(seed_indices) > limit:
            raise ValueError("max_support is too small for the original CVaR caps")
        initial = len(seed_indices)

    active_idx = torch.zeros(chunks, limit, dtype=torch.long, device=S.device)
    active_idx[:, :initial] = torch.tensor(seed_indices, device=S.device, dtype=torch.long)
    active_mask = torch.zeros(chunks, limit, dtype=torch.bool, device=S.device)
    active_mask[:, :initial] = True
    active_size = torch.full((chunks,), initial, dtype=torch.int32, device=S.device)
    warm: Tensor | None = None
    p = torch.full((chunks, tokens), 1.0 / tokens, dtype=torch.float32, device=S.device)
    lower = torch.full((chunks,), float("-inf"), dtype=torch.float32, device=S.device)
    upper = torch.full((chunks,), float("inf"), dtype=torch.float32, device=S.device)
    gap = torch.full((chunks,), float("inf"), dtype=torch.float32, device=S.device)
    converged = torch.zeros(chunks, dtype=torch.bool, device=S.device)
    iterations_used = torch.zeros(chunks, dtype=torch.int32, device=S.device)
    # ``width`` tracks the largest live support.  Keeping the backing storage
    # at ``limit`` is cheap, but all GEMMs/projections below use only this
    # prefix; the old implementation paid for the maximum support from the
    # first iteration even when every row still had the 32-query seed.
    width = initial
    while True:
        idx = active_idx[:, :width]
        mask = active_mask[:, :width]
        gather_idx = idx.unsqueeze(-1).expand(chunks, width, tokens)
        local_S = S.gather(1, gather_idx).float()
        local_f = f.gather(1, idx).float()
        if caps_all is None:
            local_caps = None
            project = lambda value: _project_simplex_batch(
                value.masked_fill(~mask, float("-inf"))
            ).masked_fill(~mask, 0.0)
        else:
            local_caps = caps_all[idx] * mask.to(torch.float32)
            project = lambda value: _project_capped_batch(value, local_caps)
        if warm is None:
            if caps_all is None:
                lam = mask.to(torch.float32) / active_size[:, None].to(torch.float32)
            else:
                lam = project(torch.zeros(chunks, width, dtype=torch.float32, device=S.device))
        else:
            lam = warm[:, :width]
            if caps_all is None:
                # Newly activated coordinates start at a tiny positive mass so
                # mirror-ascent can discover them; existing mass is preserved.
                lam = torch.where(mask & (lam <= 0), torch.full_like(lam, 1e-6), lam)
                lam = lam.masked_fill(~mask, 0.0)
                lam = lam / lam.sum(-1, keepdim=True).clamp_min(1e-30)
            else:
                lam = project(lam)
        value, p, gradient = _batched_dual_value(local_S, local_f, lam)
        best_value, best_lam, best_p = value.clone(), lam.clone(), p.clone()
        for iteration in range(1, max_iterations + 1):
            step = 1.0
            if caps_all is None:
                # Entropic mirror ascent is substantially cheaper than sorting
                # for an ordinary simplex and is exact-domain feasible.  The
                # full-pool certificate below remains unchanged.
                log_lam = lam.clamp_min(1e-30).log() + step * gradient
                log_lam = log_lam.masked_fill(~mask, float("-inf"))
                candidate = torch.softmax(log_lam, dim=-1).masked_fill(~mask, 0.0)
            else:
                candidate = project(lam + step * gradient)
            candidate_value, candidate_p, candidate_gradient = _batched_dual_value(
                local_S, local_f, candidate
            )
            step_delta = candidate - lam
            if armijo:
                for _ in range(12):
                    sufficient = candidate_value >= value + 1e-4 * (gradient * step_delta).sum(-1)
                    if bool(sufficient.all()):
                        break
                    step *= 0.5
                    if caps_all is None:
                        log_lam = lam.clamp_min(1e-30).log() + step * gradient
                        log_lam = log_lam.masked_fill(~mask, float("-inf"))
                        candidate = torch.softmax(log_lam, dim=-1).masked_fill(~mask, 0.0)
                    else:
                        candidate = project(lam + step * gradient)
                    candidate_value, candidate_p, candidate_gradient = _batched_dual_value(
                        local_S, local_f, candidate
                    )
                    step_delta = candidate - lam
            lam, value, p, gradient = candidate, candidate_value, candidate_p, candidate_gradient
            improved = value > best_value
            best_value = torch.where(improved, value, best_value)
            best_lam = torch.where(improved.unsqueeze(-1), lam, best_lam)
            best_p = torch.where(improved.unsqueeze(-1), p, best_p)
            iterations_used[:] = iteration
            if caps_all is None:
                # A small exponentiated-gradient step is not a KKT condition:
                # with hundreds of empirical rows, a visibly non-optimal
                # uniform dual can move only O(1/sqrt(N)) in one iteration.
                # Check the simplex dual residual instead.  ``tau`` is the
                # Lagrange multiplier induced by the current dual measure;
                # positive-mass coordinates must equal it and zero-mass
                # coordinates may not exceed it at a maximizer.
                tau = (lam * gradient).sum(-1, keepdim=True)
                positive = (lam > 1e-4) & mask
                positive_residual = (gradient - tau).abs().masked_fill(~positive, 0.0).amax(-1)
                boundary_residual = (gradient - tau).clamp_min(0.0).masked_fill(~mask, 0.0).amax(-1)
                projected_norm = torch.maximum(positive_residual, boundary_residual)
            else:
                projected_norm = torch.linalg.vector_norm(step_delta, dim=-1)
            if bool((projected_norm <= min(1e-5, tolerance / 10)).all()):
                break
        lam, p = best_lam, best_p
        if not compute_certificate:
            # ``p`` is already the exact result of the requested fixed-support
            # dual iterations.  A full-pool primal scan can neither alter it
            # nor trigger support growth in this mode, and no caller may treat
            # the placeholder values as a certificate.
            unavailable = torch.full(
                (chunks,), float("nan"), dtype=torch.float32, device=S.device
            )
            return (
                p,
                best_value,
                unavailable,
                unavailable,
                active_size,
                torch.zeros(chunks, dtype=torch.bool, device=S.device),
                iterations_used,
            )
        row_entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
        all_gaps = f.float() - torch.einsum("can,cn->ca", S.float(), p) - row_entropy.unsqueeze(-1)
        if objective == "minimax":
            upper = all_gaps.amax(-1)
        else:
            upper = torch.stack(
                [empirical_cvar(all_gaps[row], w, alpha).float() for row in range(chunks)]
            )
        lower = best_value
        gap = upper - lower
        done = gap <= tolerance
        converged |= done
        if bool(done.all()) or int(active_size.max().item()) >= limit:
            return p, lower, upper, gap, active_size, converged, iterations_used

        # Hide all currently active query coordinates.  Unused padded slots
        # repeat coordinate zero, which is harmless because coordinate zero is
        # already active from initialization.
        masked_gaps = all_gaps.clone()
        masked_gaps.scatter_(1, active_idx, float("-inf"))
        unfinished = ~done & (active_size < limit)
        if not bool(unfinished.any()):
            return p, lower, upper, gap, active_size, converged, iterations_used
        # Add the k largest *inactive* gaps for each unfinished chunk.  The
        # per-row top-k is deterministic for ties in current PyTorch builds;
        # the scalar solver remains the reference implementation for tests
        # requiring an explicit lower-index tie rule.  Each candidate is
        # masked from the current support, hence no row receives duplicates.
        additions = min(int(active_additions_per_round), limit - int(active_size.min().item()))
        candidates = torch.topk(masked_gaps, k=additions, dim=-1, largest=True, sorted=True).indices
        for offset in range(additions):
            rows = torch.nonzero(unfinished & (active_size < limit), as_tuple=False).flatten()
            if rows.numel() == 0:
                break
            slots = active_size[rows].to(torch.long)
            active_idx[rows, slots] = candidates[rows, offset]
            active_mask[rows, slots] = True
            active_size[rows] += 1
        width = min(limit, int(active_size.max().item()))
        warm = torch.zeros(chunks, width, dtype=torch.float32, device=S.device)
        old_width = min(width, best_lam.shape[1])
        warm[:, :old_width] = best_lam[:, :old_width]
