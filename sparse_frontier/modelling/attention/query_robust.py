"""Empirical affine chunk summaries and online Query-Robust routing.

The pure functions in this module intentionally have no vLLM dependency.  The
stateful attention implementation is added on top of this tested math contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


BiasMode = Literal["raw_entropy", "mean_residual"]


@dataclass(frozen=True)
class SummaryAudit:
    """Post-storage-cast error statistics for one summary build."""

    num_samples: int
    weighted_signed_mean: float
    weighted_abs_mean: float
    max_abs: float


@dataclass(frozen=True)
class SummaryBuildResult:
    """One affine vector/scalar summary for every input chunk."""

    summary_key: Tensor
    summary_bias: Tensor
    entropy: Tensor
    quantized_audit: SummaryAudit


def _accumulation_dtype(*tensors: Tensor) -> torch.dtype:
    return (
        torch.float64
        if any(tensor.dtype == torch.float64 for tensor in tensors)
        else torch.float32
    )


def weighted_centroid(queries: Tensor, weights: Tensor) -> Tensor:
    """Return the normalized nonnegative-weight query centroid."""

    if queries.ndim != 2 or weights.ndim != 1:
        raise ValueError(
            "weighted centroid expects queries [samples, dimension] and weights [samples]"
        )
    if queries.shape[0] == 0:
        raise ValueError("weighted centroid cannot use an empty empirical measure")
    if queries.shape[0] != weights.shape[0]:
        raise ValueError("query and weight sample counts differ")
    if not torch.is_floating_point(queries) or not torch.is_floating_point(weights):
        raise ValueError("queries and weights must be floating-point tensors")
    if not torch.isfinite(queries).all() or not torch.isfinite(weights).all():
        raise ValueError("queries and weights must be finite")
    if torch.any(weights < 0):
        raise ValueError("empirical weights must be nonnegative")

    dtype = _accumulation_dtype(queries, weights)
    queries_acc = queries.to(dtype=dtype)
    weights_acc = weights.to(device=queries.device, dtype=dtype)
    total = weights_acc.sum()
    if not bool(total > 0):
        raise ValueError("empirical weights must have a positive sum")
    normalized = weights_acc / total
    return normalized @ queries_acc


def tangent_gap(logits: Tensor, p: Tensor) -> Tensor:
    """Evaluate ``logsumexp(s) - s·p - H(p)`` stably.

    For simplex ``p`` this equals ``KL(p || softmax(logits))`` and is therefore
    the exact gap between log-sum-exp and its affine tangent.
    """

    if logits.ndim < 1 or p.ndim != 1 or logits.shape[-1] != p.shape[0]:
        raise ValueError("logits and tangent probabilities have incompatible dimensions")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(p):
        raise ValueError("logits and tangent probabilities must be floating point")
    if not torch.isfinite(logits).all() or not torch.isfinite(p).all():
        raise ValueError("logits and tangent probabilities must be finite")
    if torch.any(p < 0):
        raise ValueError("tangent probabilities must be nonnegative")

    dtype = _accumulation_dtype(logits, p)
    logits_acc = logits.to(dtype=dtype)
    p_acc = p.to(device=logits.device, dtype=dtype)
    total = p_acc.sum()
    tolerance = 1e-10 if dtype == torch.float64 else 1e-5
    if not bool(torch.isclose(total, torch.ones_like(total), rtol=tolerance, atol=tolerance)):
        raise ValueError("tangent probabilities must sum to one")
    p_acc = p_acc / total
    tiny = torch.finfo(dtype).tiny
    entropy = -(p_acc * p_acc.clamp_min(tiny).log()).sum()
    return torch.logsumexp(logits_acc, dim=-1) - (logits_acc * p_acc).sum(-1) - entropy


def _validate_summary_inputs(
    keys: Tensor,
    queries: Tensor,
    weights: Tensor,
    storage_dtype: torch.dtype,
    bias_mode: str,
    query_batch_size: int,
) -> None:
    if keys.ndim != 3 or keys.shape[0] == 0 or keys.shape[1] == 0 or keys.shape[2] == 0:
        raise ValueError("summary keys must have nonempty shape [chunks, tokens, dimension]")
    if queries.ndim != 2 or queries.shape[1] != keys.shape[2]:
        raise ValueError("query and key dimensions must match")
    if weights.ndim != 1 or weights.shape[0] != queries.shape[0]:
        raise ValueError("query and weight sample counts must match")
    if bias_mode not in {"raw_entropy", "mean_residual"}:
        raise ValueError("bias_mode must be 'raw_entropy' or 'mean_residual'")
    if not isinstance(query_batch_size, int) or isinstance(query_batch_size, bool) or query_batch_size < 1:
        raise ValueError("query batch size must be a positive integer")
    if not storage_dtype.is_floating_point:
        raise ValueError("summary storage dtype must be floating point")
    if not torch.is_floating_point(keys):
        raise ValueError("summary keys must be floating point")
    if not torch.isfinite(keys).all():
        raise ValueError("summary keys must be finite")


def build_mean_summaries(
    keys: Tensor,
    queries: Tensor,
    weights: Tensor,
    storage_dtype: torch.dtype,
    bias_mode: BiasMode,
    query_batch_size: int = 128,
) -> SummaryBuildResult:
    """Build empirical-mean tangent summaries and rescan their stored vectors.

    Args:
        keys: ``[chunks, tokens, head_dim]``.
        queries: ``[samples, head_dim]`` in the target score representation.
        weights: Nonnegative empirical sample weights.
        storage_dtype: Runtime summary-vector dtype.
        bias_mode: Raw tangent entropy or fitted empirical mean residual.
        query_batch_size: Maximum queries materialized in a residual scan.
    """

    _validate_summary_inputs(
        keys, queries, weights, storage_dtype, bias_mode, query_batch_size
    )
    mean_query = weighted_centroid(queries, weights)
    dtype = _accumulation_dtype(keys, queries, weights)
    keys_acc = keys.to(dtype=dtype)
    queries_acc = queries.to(device=keys.device, dtype=dtype)
    weights_acc = weights.to(device=keys.device, dtype=dtype)
    weights_acc = weights_acc / weights_acc.sum()
    mean_query = mean_query.to(device=keys.device, dtype=dtype)
    scale = keys.shape[-1] ** -0.5

    mean_logits = torch.einsum("d,cnd->cn", mean_query, keys_acc) * scale
    p = torch.softmax(mean_logits, dim=-1)
    summary_key_fp = torch.einsum("cn,cnd->cd", p, keys_acc)
    tiny = torch.finfo(dtype).tiny
    entropy = -(p * p.clamp_min(tiny).log()).sum(-1)

    if bias_mode == "mean_residual":
        residual_sum = torch.zeros(keys.shape[0], device=keys.device, dtype=torch.float64)
        for start in range(0, queries.shape[0], query_batch_size):
            stop = min(queries.shape[0], start + query_batch_size)
            query_batch = queries_acc[start:stop]
            weight_batch = weights_acc[start:stop].to(torch.float64)
            logits = torch.einsum("bd,cnd->bcn", query_batch, keys_acc) * scale
            exact = torch.logsumexp(logits, dim=-1)
            lower = torch.einsum("bd,cd->bc", query_batch, summary_key_fp) * scale
            residual = exact - lower - entropy
            residual_sum += (weight_batch[:, None] * residual.to(torch.float64)).sum(0)
        bias = entropy + residual_sum.to(dtype)
    else:
        bias = entropy

    summary_key = summary_key_fp.to(dtype=storage_dtype)
    summary_bias = bias.to(dtype=torch.float32)
    entropy_storage = entropy.to(dtype=torch.float32)

    signed_total = torch.zeros((), device=keys.device, dtype=torch.float64)
    absolute_total = torch.zeros((), device=keys.device, dtype=torch.float64)
    max_abs = torch.zeros((), device=keys.device, dtype=torch.float64)
    for start in range(0, queries.shape[0], query_batch_size):
        stop = min(queries.shape[0], start + query_batch_size)
        query_batch = queries_acc[start:stop]
        weight_batch = weights_acc[start:stop].to(torch.float64)
        logits = torch.einsum("bd,cnd->bcn", query_batch, keys_acc) * scale
        exact = torch.logsumexp(logits, dim=-1)
        predicted = (
            torch.einsum("bd,cd->bc", query_batch, summary_key.to(dtype=dtype)) * scale
            + summary_bias.to(dtype=dtype)
        )
        error = (exact - predicted).to(torch.float64)
        signed_total += (weight_batch[:, None] * error).sum()
        absolute_total += (weight_batch[:, None] * error.abs()).sum()
        max_abs = torch.maximum(max_abs, error.abs().max())

    chunk_count = keys.shape[0]
    audit = SummaryAudit(
        num_samples=int(queries.shape[0] * chunk_count),
        weighted_signed_mean=float((signed_total / chunk_count).item()),
        weighted_abs_mean=float((absolute_total / chunk_count).item()),
        max_abs=float(max_abs.item()),
    )
    return SummaryBuildResult(
        summary_key=summary_key,
        summary_bias=summary_bias,
        entropy=entropy_storage,
        quantized_audit=audit,
    )
