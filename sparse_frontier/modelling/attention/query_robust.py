"""Empirical affine chunk summaries and online Query-Robust routing.

The pure functions in this module intentionally have no vLLM dependency.  The
stateful attention implementation is added on top of this tested math contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

import torch
from torch import Tensor
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_with_kvcache

from .abstract_attention import AbstractAttention, AttentionUtils
from .query_pool import (
    QueryPool,
    QueryPoolExpectations,
    REPRESENTATION,
    load_query_pool,
)
from .query_robust_solver import (
    batched_independent_active_fit,
    batched_full_support_newton_fit,
    fit_p,
    triton_global_armijo_full_support_fit,
    triton_full_support_minimax_retry_fit,
)
from sparse_frontier.pile_query_capture import PileQueryPool, load_pile_query_pool
from .prompt_query_source import (
    load_prompt_query_support,
    validate_prompt_query_manifest,
)


BiasMode = Literal["raw_entropy", "mean_residual"]
RobustBiasMode = Literal["raw_entropy", "mean_residual", "minimax_midpoint"]
SharedChunkAggregation = Literal["sum", "max", "raw_score_sum"]


def _batched_query_chunk_logits(queries: Tensor, keys: Tensor, scale: float) -> Tensor:
    """Return scaled logits for all ``[head, chunk]`` pairs.

    ``einsum("hmd,hcnd->hcmn")`` expresses this operation clearly, but on the
    serving shapes it can select a less efficient contraction plan.  Flattening
    each chunk's token axis makes the same contraction an explicit strided
    batched GEMM.  ``reshape`` deliberately accepts the head-major strided
    view produced from vLLM prompt keys; it does not alter the score formula or
    the empirical solver's inputs.
    """

    if queries.ndim != 3 or keys.ndim != 4:
        raise ValueError("batched query/chunk logits require [heads,queries,dim] and [heads,chunks,tokens,dim]")
    if queries.shape[0] != keys.shape[0] or queries.shape[-1] != keys.shape[-1]:
        raise ValueError("batched query/chunk logits have incompatible head or dimension")
    heads, query_count, _ = queries.shape
    _, chunk_count, token_count, dimension = keys.shape
    product = torch.bmm(
        queries,
        keys.reshape(heads, chunk_count * token_count, dimension).transpose(1, 2),
    )
    return product.reshape(heads, query_count, chunk_count, token_count).permute(0, 2, 1, 3) * scale


def _transport_llama3_rope(
    queries: Tensor,
    source_positions: Tensor,
    target_position: int,
    parameters: dict[str, object],
) -> Tensor:
    """Undo captured RoPE at each source position and apply it at target."""

    if queries.ndim != 3 or source_positions.ndim != 2 or queries.shape[:2] != source_positions.shape:
        raise ValueError("RoPE transport expects [kv_heads,samples,dim] and aligned positions")
    base = float(parameters["rope_theta"])
    work = queries.to(dtype=torch.float32)
    dim = work.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=work.device, dtype=torch.float32) / dim))
    # Gradient's long-context checkpoint declares plain/default RoPE with a
    # large theta rather than Llama-3's piecewise frequency scaling. Both are
    # position-composable; only the frequency map differs.
    if {
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    } <= parameters.keys():
        factor = float(parameters["factor"])
        low = float(parameters["low_freq_factor"])
        high = float(parameters["high_freq_factor"])
        original = int(parameters["original_max_position_embeddings"])
        wavelength = 2.0 * torch.pi / inv
        low_wavelength = original / low
        high_wavelength = original / high
        smooth = (original / wavelength - low) / (high - low)
        inv = torch.where(
            wavelength < high_wavelength,
            inv,
            torch.where(
                wavelength > low_wavelength,
                inv / factor,
                (1.0 - smooth) * inv / factor + smooth * inv,
            ),
        )
    source_angle = source_positions.to(device=work.device, dtype=torch.float32).unsqueeze(-1) * inv
    target_angle = float(target_position) * inv
    first, second = work.chunk(2, dim=-1)
    source_cos, source_sin = source_angle.cos(), source_angle.sin()
    # Inverse of [x*c-y*s, y*c+x*s].
    unrot_first = first * source_cos + second * source_sin
    unrot_second = second * source_cos - first * source_sin
    target_cos, target_sin = target_angle.cos(), target_angle.sin()
    rotated_first = unrot_first * target_cos - unrot_second * target_sin
    rotated_second = unrot_second * target_cos + unrot_first * target_sin
    return torch.cat((rotated_first, rotated_second), dim=-1)


def _apply_llama3_rope_at_position(
    queries: Tensor,
    target_position: int,
    parameters: dict[str, object],
) -> Tensor:
    """Apply target-position RoPE to queries stored before RoPE.

    ``_transport_llama3_rope`` is for tensors that already contain source
    position RoPE and therefore first inverse-rotates them.  Prompt-local
    captures are explicitly pre-RoPE, so their source position must not be
    inverse-rotated a second time.
    """

    if queries.ndim != 3:
        raise ValueError(
            "pre-RoPE target application expects [kv_heads,samples,dim]"
        )
    zero_positions = torch.zeros(
        queries.shape[:2], device=queries.device, dtype=torch.int32
    )
    return _transport_llama3_rope(
        queries, zero_positions, target_position, parameters
    )


def select_balanced_pile_queries(
    queries: Tensor,
    positions: Tensor,
    query_head_ids: Tensor,
    weights: Tensor,
    budget: int | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Select an equal-mass deterministic empirical subpool per GQA group.

    Schema-2 capture uses an independent uniform-priority reservoir for every
    Q head.  Its retained order is descending random priority, so the first
    ``r`` rows of each head are themselves a deterministic uniform sub-sample
    of that head's captured population.  Taking the same ``r`` for each head
    preserves the GQA-balanced empirical measure instead of accidentally
    concentrating the online solver on one Q head.
    """

    if budget is None:
        return queries, positions, weights
    if queries.ndim != 3 or positions.shape != queries.shape[:2]:
        raise ValueError("Pile empirical query tensors have incompatible shapes")
    if query_head_ids.shape != queries.shape[:2] or weights.shape != queries.shape[:2]:
        raise ValueError("Pile empirical metadata tensors have incompatible shapes")
    if budget < 1:
        raise ValueError("empirical_query_budget must be positive when set")
    selected_queries: list[Tensor] = []
    selected_positions: list[Tensor] = []
    selected_weights: list[Tensor] = []
    for kv_head in range(queries.shape[0]):
        heads = torch.unique(query_head_ids[kv_head], sorted=True)
        if heads.numel() < 1 or budget % int(heads.numel()):
            raise ValueError(
                "empirical_query_budget must divide the number of Q heads in every KV group"
            )
        per_head = budget // int(heads.numel())
        pieces: list[Tensor] = []
        for q_head in heads.tolist():
            matches = torch.nonzero(query_head_ids[kv_head] == q_head, as_tuple=False).flatten()
            if matches.numel() < per_head:
                raise ValueError(
                    "empirical_query_budget exceeds available captured queries for a Q head"
                )
            pieces.append(matches[:per_head])
        # Interleave heads rather than concatenating head-sized runs.  The
        # active-set solver initializes from a deterministic prefix; this
        # order guarantees every prefix divisible by the GQA group size has
        # equal Q-head representation rather than silently fitting only the
        # first mapped head.
        indices = torch.stack(pieces, dim=0).transpose(0, 1).reshape(-1)
        selected_queries.append(queries[kv_head, indices])
        selected_positions.append(positions[kv_head, indices])
        chosen_weights = weights[kv_head, indices].to(torch.float32)
        selected_weights.append(chosen_weights / chosen_weights.sum().clamp_min(torch.finfo(torch.float32).tiny))
    return (
        torch.stack(selected_queries),
        torch.stack(selected_positions),
        torch.stack(selected_weights),
    )


def calibrate_robust_bias(
    residual: Tensor,
    entropy: Tensor,
    weights: Tensor,
    mode: RobustBiasMode,
) -> Tensor:
    """Fit the scalar in a robust affine summary while retaining its vector.

    ``residual`` is the exact empirical log-partition less ``q @ summary``.
    The tangent intercept is its entropy.  For a fixed minimax tangent vector,
    the midpoint of the empirical residual range minimizes the worst absolute
    affine-score error; the weighted mean minimizes squared scalar error.
    Neither calibrated intercept is a global lower-bound certificate, so the
    attention constructor requires an explicitly exploratory route for them.
    """

    if residual.ndim < 1 or entropy.shape != residual.shape[:-1]:
        raise ValueError("residual and entropy shapes are incompatible")
    if weights.ndim != 1 or weights.numel() != residual.shape[-1]:
        raise ValueError("robust bias weights must align with residual queries")
    if mode == "raw_entropy":
        return entropy
    if not torch.isfinite(residual).all() or not torch.isfinite(weights).all():
        raise ValueError("robust bias inputs must be finite")
    if torch.any(weights < 0) or not bool(weights.sum() > 0):
        raise ValueError("robust bias weights must have positive mass")
    if mode == "mean_residual":
        normalized = weights.to(device=residual.device, dtype=residual.dtype)
        normalized = normalized / normalized.sum()
        return (residual * normalized).sum(-1)
    if mode == "minimax_midpoint":
        return (residual.amin(-1) + residual.amax(-1)) * 0.5
    raise ValueError(f"unsupported robust bias mode {mode!r}")


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


class QueryRobustAttention(AbstractAttention):
    """Online empirical minimax/CVaR affine chunk router.

    The frozen Pile query pool remains explicit during prompt/chunk summary
    construction.  The robust solver chooses one simplex tangent vector per
    chunk; only that vector and its entropy scalar are retained for decode.
    """

    def __init__(
        self,
        token_budget: int,
        chunk_size: int,
        generation_horizon: int,
        sink_chunks: int,
        recent_chunks: int,
        num_layers: int,
        num_q_heads: int,
        num_kv_heads: int,
        tp_size: int,
        block_size: int,
        max_input_tokens: int = 8192,
        max_output_tokens: int = 1024,
        query_pool_path: str | Path | None = None,
        pool_expectations: QueryPoolExpectations | None = None,
        pool: QueryPool | None = None,
        tp_rank: int = 0,
        summary_dtype: str | torch.dtype = "bfloat16",
        score_dtype: str | torch.dtype = "float32",
        share_chunks_across_kv_heads: bool = True,
        shared_chunk_aggregation: SharedChunkAggregation = "sum",
        objective: str = "minimax",
        cvar_alpha: float = 0.95,
        initial_support: int = 64,
        max_support: int | None = None,
        solver_gap_tolerance: float = 1e-3,
        solver_max_iterations: int = 500,
        solver_chunk_batch_size: int = 16,
        solver_violators_per_round: int = 4,
        empirical_query_budget: int | None = None,
        bias_mode: RobustBiasMode = "raw_entropy",
        solver_fail_closed: bool = True,
        collect_solver_audit: bool = False,
        async_prefill_build: bool = False,
        solver_armijo: bool = True,
        solver_backend: str = "active_set",
        triton_fast_iterations: int = 1024,
        triton_retry_iterations: int = 8192,
        triton_armijo_backtracks: int = 12,
        prompt_query_root: str | Path | None = None,
        **unsupported: object,
    ) -> None:
        super().__init__()
        if unsupported:
            raise ValueError(
                f"unsupported Query-Robust options: {sorted(unsupported)}"
            )
        integer_args = {
            "token_budget": token_budget,
            "chunk_size": chunk_size,
            "generation_horizon": generation_horizon,
            "num_layers": num_layers,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "tp_size": tp_size,
            "block_size": block_size,
            "max_input_tokens": max_input_tokens,
            "max_output_tokens": max_output_tokens,
        }
        for name, value in integer_args.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"Query-Robust {name} must be a positive integer")
        if sink_chunks < 0 or recent_chunks < 0:
            raise ValueError("Query-Robust mandatory chunk counts must be nonnegative")
        if token_budget % chunk_size:
            raise ValueError("Query-Robust token budget must be divisible by chunk size")
        if block_size % chunk_size:
            raise ValueError("Query-Robust physical block size must be divisible by chunk size")
        if share_chunks_across_kv_heads and block_size != chunk_size:
            raise ValueError(
                "shared-KV Query-Robust requires physical block size equal to chunk size"
            )
        if token_budget // chunk_size < sink_chunks + recent_chunks + 1:
            raise ValueError(
                "Query-Robust token budget cannot fit mandatory sink/recent/current chunks"
            )
        if num_q_heads % num_kv_heads or num_kv_heads % tp_size:
            raise ValueError(
                "Query-Robust query/KV heads must have divisible GQA and TP geometry"
            )
        if pool is not None and query_pool_path is not None:
            raise ValueError("provide either an in-memory pool or query_pool_path, not both")
        pile_pool: PileQueryPool | None = None
        if pool is None and query_pool_path is not None:
            manifest_path = Path(query_pool_path) / "manifest.json"
            try:
                raw_manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"cannot inspect Query-Robust pool manifest: {exc}") from exc
            if raw_manifest.get("artifact_type") in {"pile_empirical_query_pool", "vllm_empirical_query_pool"}:
                pile_pool = load_pile_query_pool(query_pool_path)
        if pool is None and pile_pool is None:
            if query_pool_path is None or pool_expectations is None:
                raise ValueError(
                    "Query-Robust requires a query pool path and fail-closed expectations"
                )
            pool = load_query_pool(
                query_pool_path,
                expected=pool_expectations,
                tp_rank=tp_rank,
                include_full_queries=True,
            )
        manifest = pool.manifest if pool is not None else pile_pool.manifest
        def manifest_value(name: str):
            return getattr(manifest, name) if pool is not None else manifest[name]
        if pile_pool is not None:
            expected_manifest = pool_expectations
            if expected_manifest is None:
                raise ValueError("Query-Robust Pile pools require model identity expectations")

            def _normalise_model_id(value: str) -> str:
                marker = "models--"
                if marker in value:
                    value = value.split(marker, 1)[1].split("/snapshots", 1)[0]
                    return value.replace("--", "/")
                return value

            identity_checks = {
                "model_id": (
                    _normalise_model_id(str(manifest["model_id"])),
                    _normalise_model_id(str(expected_manifest.model_id)),
                ),
                "model_revision": (str(manifest["model_revision"]), str(expected_manifest.model_revision)),
                "head_dim": (int(manifest["head_dim"]), int(expected_manifest.head_dim)),
                "rope_type": (str(manifest["rope_type"]), str(expected_manifest.rope_type)),
                "rope_parameters": (manifest["rope_parameters"], expected_manifest.rope_parameters),
            }
            mismatches = [name for name, (actual, expected) in identity_checks.items() if actual != expected]
            if abs(float(manifest["attention_scale"]) - float(expected_manifest.attention_scale)) > 1e-7 * float(expected_manifest.attention_scale):
                mismatches.append("attention_scale")
            if str(manifest.get("representation")) not in {
                "post_qk_norm_post_rope_pre_scale",
                REPRESENTATION,
            }:
                mismatches.append("representation")
            if mismatches:
                raise ValueError("Query-Robust Pile pool identity mismatch: " + ", ".join(mismatches))
        expected_geometry = (num_layers, num_q_heads, num_kv_heads, tp_size)
        actual_geometry = (
            manifest_value("num_layers"),
            manifest_value("num_q_heads"),
            manifest_value("num_kv_heads"),
            manifest_value("tp_size"),
        )
        if actual_geometry != expected_geometry:
            raise ValueError(
                "Query-Robust pool/model geometry mismatch: "
                f"artifact={actual_geometry}, model={expected_geometry}"
            )
        max_horizon = (
            manifest.max_decode_offset + 1
            if pool is not None
            else int(manifest.get("sequence_tokens", 2048))
        )
        if not 1 <= generation_horizon <= max_horizon:
            raise ValueError("Query-Robust generation horizon is outside the empirical artifact")
        rope_type = manifest_value("rope_type")
        if rope_type not in {"llama3", "default"}:
            raise ValueError(
                "Query-Robust online v1 supports only Llama-3 or default NeoX RoPE"
            )
        rope_parameters = manifest_value("rope_parameters")
        if "rope_theta" not in rope_parameters:
            raise ValueError("Query-Robust RoPE metadata must include rope_theta")
        if rope_type == "llama3" and not {
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        } <= set(rope_parameters):
            raise ValueError(
                "Query-Robust Llama-3 RoPE metadata is missing frequency-scaling parameters"
            )
        if score_dtype not in {"float32", torch.float32}:
            raise ValueError("Query-Robust online scoring must use float32")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            torch.bfloat16: torch.bfloat16,
            torch.float16: torch.float16,
        }
        if summary_dtype not in dtype_map:
            raise ValueError("Query-Robust summary dtype must be bfloat16 or float16")

        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.generation_horizon = generation_horizon
        self.sink_chunks = sink_chunks
        self.recent_chunks = recent_chunks
        self.num_layers = num_layers
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_q_heads = num_q_heads // tp_size
        self.local_kv_heads = num_kv_heads // tp_size
        self.group_size = num_q_heads // num_kv_heads
        self.block_size = block_size
        self.max_seq_len = max_input_tokens + max_output_tokens
        self.max_chunks = (self.max_seq_len + chunk_size - 1) // chunk_size
        self.pool = pool
        self.pile_pool = pile_pool
        self.robust_pool_available = pile_pool is not None or (
            pool is not None and all(layer.full_queries is not None for layer in pool.layers)
        )
        self.pool_manifest = manifest
        self.prompt_query_root = (
            None if prompt_query_root is None else Path(prompt_query_root)
        )
        if self.prompt_query_root is not None:
            validate_prompt_query_manifest(
                self.prompt_query_root,
                {
                    name: manifest_value(name)
                    for name in (
                        "model_id",
                        "model_revision",
                        "num_layers",
                        "num_q_heads",
                        "num_kv_heads",
                        "head_dim",
                        "tp_size",
                        "rope_type",
                        "rope_parameters",
                        "attention_scale",
                    )
                },
            )
        self.pool_head_dim = int(manifest_value("head_dim"))
        self.scale = float(manifest_value("attention_scale"))
        expected_scale = self.pool_head_dim ** -0.5
        if abs(self.scale - expected_scale) > 1e-7 * expected_scale:
            raise ValueError(
                "Query-Robust requires the artifact attention scale to equal 1/sqrt(head_dim)"
            )
        if self.pool_head_dim < 1:
            raise ValueError("Query-Robust artifact head dimension must be positive")
        self.objective = str(objective).lower()
        if self.objective not in {"minimax", "cvar"}:
            raise ValueError("Query-Robust objective must be minimax or cvar")
        if not 0.0 <= float(cvar_alpha) < 1.0:
            raise ValueError("Query-Robust CVaR alpha must lie in [0,1)")
        if (
            initial_support < 1
            or solver_gap_tolerance <= 0
            or solver_max_iterations < 1
            or solver_chunk_batch_size < 1
            or solver_violators_per_round < 1
            or (empirical_query_budget is not None and empirical_query_budget < 1)
        ):
            raise ValueError("invalid Query-Robust solver configuration")
        self.cvar_alpha = float(cvar_alpha)
        self.initial_support = int(initial_support)
        self.max_support = max_support
        self.solver_gap_tolerance = float(solver_gap_tolerance)
        self.solver_max_iterations = int(solver_max_iterations)
        self.solver_chunk_batch_size = int(solver_chunk_batch_size)
        self.solver_violators_per_round = int(solver_violators_per_round)
        self.empirical_query_budget = (
            None if empirical_query_budget is None else int(empirical_query_budget)
        )
        self.bias_mode: RobustBiasMode = str(bias_mode)  # type: ignore[assignment]
        if self.bias_mode not in {"raw_entropy", "mean_residual", "minimax_midpoint"}:
            raise ValueError("Query-Robust bias_mode must be raw_entropy, mean_residual, or minimax_midpoint")
        if solver_fail_closed and self.bias_mode != "raw_entropy":
            raise ValueError(
                "calibrated Query-Robust bias modes require solver_fail_closed=False; "
                "they are not lower-bound certified"
            )
        self.solver_fail_closed = bool(solver_fail_closed)
        # Exact residual rescans and host-side extrema are useful when
        # diagnosing/calibrating a pool, but raw-entropy routing does not use
        # them to construct a summary.  Keeping them out of the normal decode
        # path avoids several GPU-to-host synchronisations at every newly
        # completed chunk.  Certification remains controlled independently by
        # ``solver_fail_closed`` below.
        self.collect_solver_audit = bool(collect_solver_audit)
        self.solver_armijo = bool(solver_armijo)
        self.solver_backend = str(solver_backend)
        if self.solver_backend not in {
            "active_set",
            "batched_newton",
            "triton_fast_retry",
            "triton_global_armijo",
        }:
            raise ValueError(
                "Query-Robust solver_backend must be active_set, batched_newton, triton_fast_retry, or triton_global_armijo"
            )
        if (
            int(triton_fast_iterations) < 1
            or int(triton_retry_iterations) < int(triton_fast_iterations)
            or int(triton_armijo_backtracks) < 1
        ):
            raise ValueError("invalid Query-Robust Triton retry configuration")
        if self.solver_backend in {"batched_newton", "triton_fast_retry", "triton_global_armijo"} and self.objective != "minimax":
            raise ValueError(f"{self.solver_backend} supports only the minimax objective")
        if self.solver_backend in {"batched_newton", "triton_fast_retry", "triton_global_armijo"} and not self.solver_armijo:
            raise ValueError(f"{self.solver_backend} requires solver_armijo=True")
        self.triton_fast_iterations = int(triton_fast_iterations)
        self.triton_retry_iterations = int(triton_retry_iterations)
        self.triton_armijo_backtracks = int(triton_armijo_backtracks)
        if not self.solver_armijo and (
            self.solver_fail_closed
            or self.collect_solver_audit
            or self.max_support != self.initial_support
        ):
            raise ValueError(
                "unguarded Query-Robust updates require fixed support, audit disabled, and solver_fail_closed=False"
            )
        # Summary construction is independent of the dense prefill result once
        # post-RoPE keys are available.  This opt-in scheduler puts it on a
        # dedicated CUDA stream, but retains the synchronous path everywhere
        # else (including CPU tests and decode-time chunk finalization).
        self.async_prefill_build = bool(async_prefill_build)
        self.storage_dtype = dtype_map[summary_dtype]
        self.share_chunks_across_kv_heads = bool(share_chunks_across_kv_heads)
        self.shared_chunk_aggregation: SharedChunkAggregation = str(  # type: ignore[assignment]
            shared_chunk_aggregation
        )
        if self.shared_chunk_aggregation not in {"sum", "max", "raw_score_sum"}:
            raise ValueError(
                "Query-Robust shared_chunk_aggregation must be 'sum', 'max', or 'raw_score_sum'"
            )
        self.summary_key: Tensor | None = None
        self.summary_bias: Tensor | None = None
        self.summary_ready: Tensor | None = None
        self.summary_valid_tokens: Tensor | None = None
        self.prompt_origin_centroid: Tensor | None = None
        self.request_centroid: Tensor | None = None
        self.request_queries: list[Tensor | None] = [None] * self.num_layers
        self.request_query_weights: list[Tensor | None] = [None] * self.num_layers
        self.request_query_positions: list[Tensor | None] = [None] * self.num_layers
        self.centroid_ready: Tensor | None = None
        self.selected_key_scratch: Tensor | None = None
        self.selected_value_scratch: Tensor | None = None
        self.last_accessed_tokens = 0
        self.summaries_built = 0
        self.fallback_count = 0
        self.summary_build_ms = 0.0
        self.solver_gap_max = 0.0
        self.solver_active_max = 0
        self.solver_nonconverged = 0
        self.solver_retry_count = 0
        self.quantized_error_max = 0.0
        self._dense_fallback = False
        self._centroid_ready_host = [False] * self.num_layers
        self._summary_ready_host = [set() for _ in range(self.num_layers)]
        # The empirical measure is fixed for a loaded artifact.  Determining
        # whether all KV heads share its weights only needs one device sync per
        # layer/request, not one per chunk build.
        self._common_query_weights: list[bool | None] = [None] * self.num_layers
        self._triton_uniform_weights: list[bool | None] = [None] * self.num_layers
        self._prefill_stream: torch.cuda.Stream | None = None
        self._pending_prefill_events: list[
            tuple[torch.cuda.Event, torch.cuda.Event] | None
        ] = [None] * self.num_layers

    def preallocate_memory(self, keys: Tensor) -> None:
        if keys.ndim < 2:
            raise ValueError("Query-Robust profiling key tensor has invalid shape")
        head_dim = keys.shape[-1]
        if head_dim != self.pool_head_dim:
            raise ValueError(
                "Query-Robust key head dimension does not match the empirical pool"
            )
        device = keys.device
        if self.summary_key is None or self.summary_key.device != device:
            self.summary_key = torch.empty(
                self.num_layers,
                self.max_chunks,
                self.local_kv_heads,
                head_dim,
                device=device,
                dtype=self.storage_dtype,
            )
            self.summary_bias = torch.empty(
                self.num_layers,
                self.max_chunks,
                self.local_kv_heads,
                device=device,
                dtype=torch.float32,
            )
            self.summary_ready = torch.zeros(
                self.num_layers,
                self.max_chunks,
                self.local_kv_heads,
                device=device,
                dtype=torch.bool,
            )
            self.summary_valid_tokens = torch.zeros(
                self.num_layers,
                self.max_chunks,
                device=device,
                dtype=torch.int32,
            )
            self.request_centroid = torch.empty(
                self.num_layers,
                self.local_kv_heads,
                head_dim,
                device=device,
                dtype=torch.float32,
            )
            if self.pool is not None:
                self.prompt_origin_centroid = torch.stack(
                    [
                        layer.centroid_by_horizon[:, self.generation_horizon - 1]
                        for layer in self.pool.layers
                    ]
                ).to(device=device, dtype=torch.float32)
            else:
                self.prompt_origin_centroid = None
            self.centroid_ready = torch.zeros(
                self.num_layers, device=device, dtype=torch.bool
            )
            if not self.share_chunks_across_kv_heads:
                self.selected_key_scratch = torch.empty(
                    self.local_kv_heads,
                    self.token_budget,
                    head_dim,
                    device=device,
                    dtype=keys.dtype,
                )
                self.selected_value_scratch = torch.empty_like(
                    self.selected_key_scratch
                )
            if self.async_prefill_build and device.type == "cuda":
                self._prefill_stream = torch.cuda.Stream(device=device)

    def reset(self) -> None:
        self._wait_for_pending_prefill_summaries()
        if self.summary_ready is not None:
            self.summary_ready.zero_()
            self.summary_valid_tokens.zero_()
            self.centroid_ready.zero_()
        self.request_queries = [None] * self.num_layers
        self.request_query_weights = [None] * self.num_layers
        self.request_query_positions = [None] * self.num_layers
        self.last_accessed_tokens = 0
        self.summaries_built = 0
        self.fallback_count = 0
        self.summary_build_ms = 0.0
        self.solver_gap_max = 0.0
        self.solver_active_max = 0
        self.solver_nonconverged = 0
        self.solver_retry_count = 0
        self.quantized_error_max = 0.0
        self._dense_fallback = False
        self._centroid_ready_host = [False] * self.num_layers
        self._summary_ready_host = [set() for _ in range(self.num_layers)]
        self._common_query_weights = [None] * self.num_layers
        self._triton_uniform_weights = [None] * self.num_layers

    def _wait_for_pending_prefill_summaries(self, layer_idx: int | None = None) -> None:
        """Fence asynchronous prefill summaries before their state is reused.

        The end event is recorded after vectors, scalars, and readiness flags
        have all been enqueued on the side stream.  Synchronizing only at the
        first consumer lets those kernels overlap with later prefill work while
        keeping decode selection semantically identical to the synchronous
        implementation.
        """

        indices = range(self.num_layers) if layer_idx is None else (layer_idx,)
        for index in indices:
            pending = self._pending_prefill_events[index]
            if pending is None:
                continue
            started, completed = pending
            completed.synchronize()
            self.summary_build_ms += float(started.elapsed_time(completed))
            self._pending_prefill_events[index] = None
            if layer_idx is not None and index == self.num_layers - 1:
                self._publish_state()

    def _publish_state(self) -> None:
        """Publish bounded router telemetry after all relevant work is timed."""

        from sparse_frontier.utils.sparsity_server import set_query_robust_state

        set_query_robust_state(
            ready=all(self._centroid_ready_host),
            summaries_built=self.summaries_built,
            summary_build_ms=self.summary_build_ms,
            fallback_count=self.fallback_count,
            objective=self.objective,
            solver_gap_max=self.solver_gap_max,
            solver_active_max=self.solver_active_max,
            solver_nonconverged=self.solver_nonconverged,
            solver_retry_count=self.solver_retry_count,
            quantized_error_max=self.quantized_error_max,
        )

    def _require_state(self, reference: Tensor) -> None:
        if self.summary_key is None:
            self.preallocate_memory(reference)
        if self.summary_key is None or self.request_centroid is None:
            raise RuntimeError("Query-Robust summary allocation failed")

    def _rotate_centroid(self, layer_idx: int, prompt_length: int) -> None:
        from sparse_frontier.query_robust_diagnostics import apply_rope

        parameters = dict(self.pool.manifest.rope_parameters)
        if self.prompt_origin_centroid is None:
            raise RuntimeError("Query-Robust origin-centroid allocation failed")
        origin = self.prompt_origin_centroid[layer_idx]
        self.request_centroid[layer_idx].copy_(
            apply_rope(
                origin,
                prompt_length,
                rope_type=str(self.pool.manifest.rope_type),
                parameters=parameters,
            )
        )
        self.centroid_ready[layer_idx] = True
        self._centroid_ready_host[layer_idx] = True

    def _prepare_query_state(self, layer_idx: int, prompt_length: int) -> None:
        """Prepare the frozen empirical queries at this request's target Q position."""
        if self.prompt_query_root is not None:
            import os

            request_id_text = os.getenv("SF_QUERY_ROBUST_REQUEST_ID")
            if request_id_text is None:
                raise RuntimeError(
                    "request-local Query-Robust support is enabled but "
                    "SF_QUERY_ROBUST_REQUEST_ID is missing"
                )
            try:
                request_id = int(request_id_text, 10)
            except ValueError as exc:
                raise RuntimeError(
                    "SF_QUERY_ROBUST_REQUEST_ID must be a decimal nonnegative integer"
                ) from exc
            support = load_prompt_query_support(
                self.prompt_query_root,
                request_id,
                layer_idx,
                tp_rank=self.tp_rank,
                num_q_heads=self.local_q_heads,
                num_kv_heads=self.local_kv_heads,
                head_dim=self.pool_head_dim,
                prompt_length=prompt_length,
            )
            transported = _apply_llama3_rope_at_position(
                support.queries_by_kv_head.to(
                    device=self.summary_key.device, dtype=torch.float32
                ),
                prompt_length + self.generation_horizon - 1,
                dict(self.pool_manifest["rope_parameters"])
                if self.pile_pool is not None
                else dict(self.pool_manifest.rope_parameters),
            )
            self.request_queries[layer_idx] = transported.to(dtype=torch.bfloat16)
            self.request_query_weights[layer_idx] = support.weights_by_kv_head.to(
                device=self.summary_key.device, dtype=torch.float32
            )
            self.request_query_positions[layer_idx] = support.positions_by_kv_head
            self.centroid_ready[layer_idx] = True
            self._centroid_ready_host[layer_idx] = True
            return
        if self.pile_pool is None and self.pool is not None and self.robust_pool_available:
            if self.request_queries[layer_idx] is not None:
                return
            layer = self.pool.layers[layer_idx]
            if layer.full_queries is None or layer.full_weights is None:
                raise RuntimeError("robust empirical query tensors are missing")
            self.request_queries[layer_idx] = layer.full_queries.to(
                device=self.summary_key.device, dtype=torch.float32
            )
            self.request_query_weights[layer_idx] = layer.full_weights.to(
                device=self.summary_key.device, dtype=torch.float64
            )
            self.centroid_ready[layer_idx] = True
            self._centroid_ready_host[layer_idx] = True
            return
        if self.pile_pool is None:
            self._rotate_centroid(layer_idx, prompt_length)
            return
        if self.request_queries[layer_idx] is not None:
            return
        source = self.pile_pool.layers[layer_idx]
        target = prompt_length + self.generation_horizon - 1
        parameters = dict(self.pool_manifest["rope_parameters"])
        source_queries, source_positions, source_weights = select_balanced_pile_queries(
            source.queries_by_kv_head,
            source.positions_by_kv_head,
            source.query_head_ids_by_kv_head,
            source.weights_by_kv_head,
            self.empirical_query_budget,
        )
        transported = _transport_llama3_rope(
            source_queries.to(device=self.summary_key.device, dtype=torch.float32),
            source_positions.to(device=self.summary_key.device),
            target,
            parameters,
        )
        # Keep the frozen pool in BF16 between chunk builds.  The solver casts
        # only the current layer's batch to FP32, cutting persistent request
        # state from roughly 1.6 GB to 0.8 GB for Llama-3.1-8B.
        self.request_queries[layer_idx] = transported.to(dtype=torch.bfloat16)
        self.request_query_weights[layer_idx] = source_weights.to(
            device=self.summary_key.device, dtype=torch.float32
        )
        self.request_query_positions[layer_idx] = source_positions
        self.centroid_ready[layer_idx] = True
        self._centroid_ready_host[layer_idx] = True

    def _summarize(
        self, chunks: Tensor, centroid: Tensor | None = None, layer_idx: int | None = None
    ) -> tuple[Tensor, Tensor]:
        """Summarize [chunks, tokens, kv_heads, dim] in FP32."""
        if self.robust_pool_available:
            if layer_idx is None or self.request_queries[layer_idx] is None:
                raise RuntimeError("robust empirical query state is not ready")
            query_pool = self.request_queries[layer_idx]
            summary = torch.empty(
                chunks.shape[0], self.local_kv_heads, chunks.shape[-1],
                device=chunks.device, dtype=torch.float32
            )
            bias = torch.empty(
                chunks.shape[0], self.local_kv_heads,
                device=chunks.device, dtype=torch.float32
            )
            query_weights = self.request_query_weights[layer_idx]
            if query_weights is None:
                raise RuntimeError("robust empirical query weights are not ready")
            # Pile shards use the same uniform measure for every KV head.  Solve
            # all local heads and chunks in one batched active-set call; this is
            # the dominant online optimization and avoids 8 Python solver loops
            # per layer.  Nonuniform legacy weights retain the exact per-head
            # path below rather than silently changing the empirical objective.
            common_weights = self._common_query_weights[layer_idx]
            if common_weights is None:
                common_weights = bool(
                    torch.allclose(
                        query_weights,
                        query_weights[0].unsqueeze(0).expand_as(query_weights),
                        atol=0.0,
                        rtol=0.0,
                    )
                )
                self._common_query_weights[layer_idx] = common_weights
            if common_weights:
                query_pool = query_pool.float()
                key_heads = chunks.float().permute(2, 0, 1, 3)
                shared_weights = query_weights[0]
                if self.solver_backend in {"triton_fast_retry", "triton_global_armijo"}:
                    uniform_weights = self._triton_uniform_weights[layer_idx]
                    if uniform_weights is None:
                        uniform_weights = bool(
                            torch.allclose(
                                shared_weights,
                                torch.full_like(
                                    shared_weights,
                                    1.0 / shared_weights.numel(),
                                ),
                                atol=0.0,
                                rtol=0.0,
                            )
                        )
                        self._triton_uniform_weights[layer_idx] = uniform_weights
                    if not uniform_weights:
                        raise ValueError(
                            "triton_fast_retry requires the uniform Pile empirical measure"
                        )
                for start in range(0, chunks.shape[0], self.solver_chunk_batch_size):
                    stop = min(chunks.shape[0], start + self.solver_chunk_batch_size)
                    key_batch = key_heads[:, start:stop]
                    logits = _batched_query_chunk_logits(
                        query_pool, key_batch, self.scale
                    )
                    batch_heads, batch_chunks = logits.shape[:2]
                    flat_logits = logits.reshape(batch_heads * batch_chunks, logits.shape[-2], logits.shape[-1])
                    flat_f = torch.logsumexp(flat_logits, dim=-1)
                    # One-shot diagnostics for GPU solver investigations.
                    # The scalar reductions are intentionally gated so normal
                    # serving never pays for host synchronization.
                    import os
                    trace_solver = os.getenv("SF_QR_TRACE_SOLVER") == "1"
                    dump_solver = os.getenv("SF_QR_DUMP_SOLVER")
                    if (trace_solver or dump_solver) and not getattr(
                        self, "_solver_trace_emitted", False
                    ):
                        trace = {
                            "shape": list(flat_logits.shape),
                            "score_min": float(flat_logits.amin().item()),
                            "score_max": float(flat_logits.amax().item()),
                            "score_absmax": float(flat_logits.abs().amax().item()),
                            "score_mean": float(flat_logits.mean().item()),
                            "score_std": float(flat_logits.float().std().item()),
                            "partition_min": float(flat_f.amin().item()),
                            "partition_max": float(flat_f.amax().item()),
                            "query_mean": flat_logits.mean((0, 2)).tolist(),
                            "query_std": flat_logits.float().std((0, 2)).tolist(),
                            "query_absmax": flat_logits.abs().amax((0, 2)).tolist(),
                        }
                        if trace_solver:
                            print(
                                __import__("json").dumps(
                                    {"query_robust_solver_trace": trace},
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                        if dump_solver:
                            torch.save(
                                {
                                    "scores": flat_logits[:1024].detach().cpu(),
                                    "partitions": flat_f[:1024].detach().cpu(),
                                    "trace": trace,
                                },
                                dump_solver,
                            )
                        self._solver_trace_emitted = True
                    if self.solver_backend == "batched_newton":
                        p, lower, upper, gap, converged, _ = (
                            batched_full_support_newton_fit(
                                flat_logits,
                                flat_f,
                                tolerance=self.solver_gap_tolerance,
                                max_iterations=min(self.solver_max_iterations, 8),
                                armijo_backtracks=min(self.triton_armijo_backtracks, 10),
                            )
                        )
                        active_size = torch.full_like(
                            gap, flat_logits.shape[1], dtype=torch.int64
                        )
                    elif self.solver_backend == "triton_fast_retry":
                        if flat_logits.shape[1] > 256:
                            raise ValueError(
                                "triton_fast_retry supports at most 256 empirical queries; "
                                "reduce empirical_query_budget or use active_set"
                            )
                        p, lower, upper, gap, converged, retry_count = (
                            triton_full_support_minimax_retry_fit(
                                flat_logits,
                                flat_f,
                                tolerance=self.solver_gap_tolerance,
                                fast_iterations=self.triton_fast_iterations,
                                retry_iterations=self.triton_retry_iterations,
                                armijo_backtracks=self.triton_armijo_backtracks,
                            )
                        )
                        active_size = torch.full_like(
                            gap, flat_logits.shape[1], dtype=torch.int64
                        )
                        self.solver_retry_count += retry_count
                    elif self.solver_backend == "triton_global_armijo":
                        p, lower, upper, gap, converged = (
                            triton_global_armijo_full_support_fit(
                                flat_logits,
                                flat_f,
                                tolerance=self.solver_gap_tolerance,
                                max_iterations=self.triton_retry_iterations,
                                armijo_backtracks=self.triton_armijo_backtracks,
                            )
                        )
                        active_size = torch.full_like(
                            gap, flat_logits.shape[1], dtype=torch.int64
                        )
                    else:
                        p, lower, upper, gap, active_size, converged, _ = batched_independent_active_fit(
                            flat_logits,
                            flat_f,
                            objective=self.objective,
                            weights=shared_weights,
                            alpha=self.cvar_alpha,
                            initial_support=self.initial_support,
                            max_support=self.max_support,
                            tolerance=self.solver_gap_tolerance,
                            max_iterations=self.solver_max_iterations,
                            active_additions_per_round=self.solver_violators_per_round,
                            compute_certificate=(
                                self.solver_fail_closed
                                or self.collect_solver_audit
                                or self.max_support != self.initial_support
                            ),
                            armijo=self.solver_armijo,
                        )
                    p = p.view(batch_heads, batch_chunks, -1).to(dtype=torch.float32)
                    summary_batch = torch.einsum("hcn,hcnd->hcd", p, key_batch)
                    entropy_batch = -(
                        p * p.clamp_min(torch.finfo(torch.float32).tiny).log()
                    ).sum(-1)
                    summary[start:stop].copy_(summary_batch.permute(1, 0, 2))
                    if self.bias_mode == "raw_entropy":
                        # This is the tangent's certified entropy intercept.
                        # A residual scan cannot alter it, so do not pay for
                        # an extra Q-summary GEMM in online routing.
                        bias_batch = entropy_batch
                    else:
                        affine_score = (
                            torch.einsum("hmd,hcd->hcm", query_pool, summary_batch)
                            * self.scale
                        )
                        residual = flat_f.view(batch_heads, batch_chunks, -1) - affine_score
                        bias_batch = calibrate_robust_bias(
                            residual, entropy_batch, shared_weights, self.bias_mode
                        )
                    bias[start:stop].copy_(bias_batch.permute(1, 0))
                    if self.collect_solver_audit:
                        if self.bias_mode == "raw_entropy":
                            affine_score = (
                                torch.einsum("hmd,hcd->hcm", query_pool, summary_batch)
                                * self.scale
                            )
                        quantized_predicted = affine_score + bias_batch.unsqueeze(-1)
                        quantized_error = (flat_f.view(batch_heads, batch_chunks, -1) - quantized_predicted).abs().max()
                        self.quantized_error_max = max(self.quantized_error_max, float(quantized_error.item()))
                    # Certificate telemetry is mandatory for every backend.
                    # Previously active-set reported no gap/nonconvergence
                    # unless audit mode was enabled, which could make an
                    # uncertified sparse run look healthy.
                    self.solver_gap_max = max(self.solver_gap_max, float(gap.max().item()))
                    self.solver_active_max = max(self.solver_active_max, int(active_size.max().item()))
                    self.solver_nonconverged += int((~converged).sum().item())
                    if self.solver_fail_closed and bool((~converged).any() or (gap > self.solver_gap_tolerance).any()):
                        self._dense_fallback = True
                        self.fallback_count += int((~converged).sum().item()) + int((gap > self.solver_gap_tolerance).sum().item())
            else:
                if self.solver_backend == "triton_fast_retry":
                    raise ValueError(
                        "triton_fast_retry requires common uniform query weights across KV heads"
                    )
                for kv_head in range(self.local_kv_heads):
                    q = query_pool[kv_head]
                    key = chunks[:, :, kv_head, :].float()
                    for start in range(0, chunks.shape[0], self.solver_chunk_batch_size):
                        stop = min(chunks.shape[0], start + self.solver_chunk_batch_size)
                        logits = torch.einsum("md,cnd->cmn", q, key[start:stop]) * self.scale
                        chunk_f = torch.logsumexp(logits, dim=-1)
                        p, lower, upper, gap, active_size, converged, _ = batched_independent_active_fit(
                            logits, chunk_f, objective=self.objective,
                            weights=query_weights[kv_head], alpha=self.cvar_alpha,
                            initial_support=self.initial_support, max_support=self.max_support,
                            tolerance=self.solver_gap_tolerance, max_iterations=self.solver_max_iterations,
                            active_additions_per_round=self.solver_violators_per_round,
                            compute_certificate=(
                                self.solver_fail_closed
                                or self.collect_solver_audit
                                or self.max_support != self.initial_support
                            ),
                            armijo=self.solver_armijo,
                        )
                        p = p.to(device=chunks.device, dtype=torch.float32)
                        summary[start:stop, kv_head] = torch.einsum("cn,cnd->cd", p, key[start:stop])
                        entropy = -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)
                        if self.bias_mode == "raw_entropy":
                            bias[start:stop, kv_head] = entropy
                        else:
                            affine_score = torch.einsum("md,cd->cm", q, summary[start:stop, kv_head].float()) * self.scale
                            residual = chunk_f - affine_score
                            bias[start:stop, kv_head] = calibrate_robust_bias(
                                residual, entropy, query_weights[kv_head], self.bias_mode
                            )
                        if self.collect_solver_audit:
                            if self.bias_mode == "raw_entropy":
                                affine_score = torch.einsum("md,cd->cm", q, summary[start:stop, kv_head].float()) * self.scale
                            quantized_predicted = affine_score + bias[start:stop, kv_head].unsqueeze(-1)
                            quantized_error = (chunk_f - quantized_predicted).abs().max()
                            self.quantized_error_max = max(self.quantized_error_max, float(quantized_error.item()))
                        self.solver_gap_max = max(self.solver_gap_max, float(gap.max().item()))
                        self.solver_active_max = max(self.solver_active_max, int(active_size.max().item()))
                        self.solver_nonconverged += int((~converged).sum().item())
                        if self.solver_fail_closed and bool((~converged).any() or (gap > self.solver_gap_tolerance).any()):
                            self._dense_fallback = True
                            self.fallback_count += int((~converged).sum().item()) + int((gap > self.solver_gap_tolerance).sum().item())
            return summary.to(self.storage_dtype), bias
        if centroid is None:
            raise RuntimeError("centroid is required for legacy summary construction")
        logits = torch.einsum(
            "hd,cnhd->chn", centroid.float(), chunks.float()
        ) * self.scale
        probabilities = torch.softmax(logits, dim=-1)
        summary = torch.einsum(
            "chn,cnhd->chd", probabilities, chunks.float()
        )
        tiny = torch.finfo(torch.float32).tiny
        entropy = -(
            probabilities * probabilities.clamp_min(tiny).log()
        ).sum(dim=-1)
        return summary.to(self.storage_dtype), entropy.float()

    def _store_full_chunks(
        self, keys: Tensor, layer_idx: int, start_chunk: int, stop_chunk: int
    ) -> None:
        if stop_chunk <= start_chunk:
            return
        start = start_chunk * self.chunk_size
        stop = stop_chunk * self.chunk_size
        chunks = keys[start:stop].view(
            stop_chunk - start_chunk,
            self.chunk_size,
            self.local_kv_heads,
            keys.shape[-1],
        )
        summary, bias = self._summarize(
            chunks, self.request_centroid[layer_idx], layer_idx
        )
        self.summary_key[layer_idx, start_chunk:stop_chunk].copy_(summary)
        self.summary_bias[layer_idx, start_chunk:stop_chunk].copy_(bias)
        self.summary_ready[layer_idx, start_chunk:stop_chunk].fill_(True)
        self.summary_valid_tokens[layer_idx, start_chunk:stop_chunk].fill_(
            self.chunk_size
        )
        self._summary_ready_host[layer_idx].update(range(start_chunk, stop_chunk))
        self.summaries_built += stop_chunk - start_chunk

    def _store_prefill_full_chunks(
        self, keys: Tensor, layer_idx: int, stop_chunk: int
    ) -> bool:
        """Schedule full prompt-chunk construction; return whether it is async."""

        if stop_chunk == 0:
            return False
        if self._prefill_stream is None or keys.device.type != "cuda":
            self._store_full_chunks(keys, layer_idx, 0, stop_chunk)
            return False
        if self._pending_prefill_events[layer_idx] is not None:
            raise RuntimeError("Query-Robust layer already has a pending prefill summary")
        source_ready = torch.cuda.Event()
        started = torch.cuda.Event(enable_timing=True)
        completed = torch.cuda.Event(enable_timing=True)
        source_ready.record(torch.cuda.current_stream(device=keys.device))
        self._prefill_stream.wait_event(source_ready)
        with torch.cuda.stream(self._prefill_stream):
            started.record()
            self._store_full_chunks(keys, layer_idx, 0, stop_chunk)
            completed.record()
        self._pending_prefill_events[layer_idx] = (started, completed)
        return True

    @torch.no_grad()
    def prefill_state(self, keys: Tensor, layer_idx: int = 0) -> None:
        """Build router state from post-RoPE keys without recomputing attention.

        ``keys`` may use vLLM's native ``[tokens, kv_heads, dim]`` layout or
        the attention abstraction's ``[1, kv_heads, tokens, dim]`` layout.
        """
        if keys.ndim == 3:
            keys = keys.transpose(0, 1).unsqueeze(0)
        self._require_state(keys)
        if keys.ndim != 4 or keys.shape[0] != 1:
            raise ValueError("Query-Robust prefill expects [1, kv_heads, tokens, dim]")
        prompt_length = keys.shape[2]
        build_start = time.perf_counter()
        self._prepare_query_state(layer_idx, prompt_length)
        # Keep vLLM's head-major prompt K storage as a strided token-major
        # view.  ``_summarize`` already performs the one required FP32
        # conversion; materializing this view first copied the complete prompt
        # a second time (roughly 512 MiB over 32 Llama-3.1-8B layers at 8K).
        token_keys = keys.squeeze(0).transpose(0, 1)
        full_chunks = prompt_length // self.chunk_size
        scheduled_async = self._store_prefill_full_chunks(
            token_keys, layer_idx, full_chunks
        )
        chunks = (prompt_length + self.chunk_size - 1) // self.chunk_size
        if chunks:
            self.summary_valid_tokens[layer_idx, :chunks].fill_(self.chunk_size)
            remainder = prompt_length % self.chunk_size
            if remainder:
                self.summary_valid_tokens[layer_idx, chunks - 1] = remainder
        if not scheduled_async:
            self.summary_build_ms += (time.perf_counter() - build_start) * 1000.0
        if layer_idx == self.num_layers - 1:
            self._publish_state()

    @torch.no_grad()
    def __call__(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        layer_idx: int = 0,
    ) -> Tensor:
        self.prefill_state(keys, layer_idx)
        return AttentionUtils.flash_attention(queries, keys, values)

    def ensure_page_reps_from_cache(
        self, k_cache: Tensor, cache_len: int, layer_idx: int
    ) -> None:
        self._require_state(k_cache)
        if self._centroid_ready_host[layer_idx]:
            return
        if cache_len < 2:
            raise ValueError("Query-Robust cache recovery requires prompt plus decode token")
        self._prepare_query_state(layer_idx, cache_len - 1)
        flat = k_cache.permute(1, 2, 0, 3).reshape(
            -1, self.local_kv_heads, k_cache.shape[-1]
        )[:cache_len]
        full_chunks = cache_len // self.chunk_size
        self._store_full_chunks(flat, layer_idx, 0, full_chunks)
        chunks = (cache_len + self.chunk_size - 1) // self.chunk_size
        self.summary_valid_tokens[layer_idx, :chunks].fill_(self.chunk_size)
        if cache_len % self.chunk_size:
            self.summary_valid_tokens[layer_idx, chunks - 1] = cache_len % self.chunk_size

    def _finalize_new_chunk(self, k_cache: Tensor, total_tokens: int, layer_idx: int) -> None:
        if total_tokens % self.chunk_size:
            return
        chunk = total_tokens // self.chunk_size - 1
        if chunk in self._summary_ready_host[layer_idx]:
            return
        token_keys = k_cache[:, chunk, :, :].permute(1, 0, 2).unsqueeze(0)
        summary, bias = self._summarize(
            token_keys, self.request_centroid[layer_idx], layer_idx
        )
        self.summary_key[layer_idx, chunk].copy_(summary[0])
        self.summary_bias[layer_idx, chunk].copy_(bias[0])
        self.summary_ready[layer_idx, chunk].fill_(True)
        self.summary_valid_tokens[layer_idx, chunk] = self.chunk_size
        self._summary_ready_host[layer_idx].add(chunk)
        self.summaries_built += 1

    def _dense_block_table(
        self, num_blocks: int, active_chunks: int, device: torch.device
    ) -> Tensor:
        logical = torch.arange(active_chunks, device=device, dtype=torch.int32)
        offsets = (
            torch.arange(self.local_kv_heads, device=device, dtype=torch.int32)
            * num_blocks
        )
        table = logical.unsqueeze(0) + offsets.unsqueeze(1)
        return table.repeat_interleave(self.group_size, dim=0)

    def _select_block_table(
        self,
        query: Tensor,
        k_cache: Tensor,
        total_tokens: int,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        num_blocks = k_cache.shape[1]
        partial = None
        if total_tokens % self.chunk_size:
            chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
            partial = k_cache[:, chunks - 1, : total_tokens % self.chunk_size, :].permute(
                1, 0, 2
            )
        logical, cache_seqlens = self._select_logical_chunks(
            query, total_tokens, layer_idx, partial
        )
        offsets = (
            torch.arange(self.local_kv_heads, device=query.device, dtype=torch.int32)
            * num_blocks
        )
        block_table = logical.to(torch.int32) + offsets.unsqueeze(1)
        return (
            block_table.repeat_interleave(self.group_size, dim=0),
            cache_seqlens.repeat_interleave(self.group_size),
        )

    def _select_logical_chunks(
        self,
        query: Tensor,
        total_tokens: int,
        layer_idx: int,
        partial_keys: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        q = query.squeeze(0).float().view(
            self.local_kv_heads, self.group_size, query.shape[-1]
        )
        # The current partial chunk is mandatory exact attention, so rebuilding
        # its *empirical* affine summary on every decode token is pure router
        # overhead.  Score it directly under the current query instead.  This
        # both preserves its contribution to GQA normalization and is strictly
        # more accurate for that chunk; it also avoids cloning all prior
        # summaries merely to substitute the final entry.
        full_count = total_tokens // self.chunk_size
        summary_key = self.summary_key[layer_idx, :full_count]
        summary_bias = self.summary_bias[layer_idx, :full_count]
        log_mass = torch.einsum(
            "hgd,chd->hgc", q, summary_key.float()
        ) * self.scale + summary_bias.permute(1, 0).unsqueeze(1)
        if partial_keys is not None:
            partial_logits = torch.einsum(
                "hgd,thd->hgt", q, partial_keys.float()
            ) * self.scale
            partial_log_mass = torch.logsumexp(partial_logits, dim=-1, keepdim=True)
            log_mass = torch.cat((log_mass, partial_log_mass), dim=-1)
        group_score = torch.softmax(log_mass, dim=-1).sum(dim=1)

        full_chunks = list(range(full_count))
        remainder = total_tokens % self.chunk_size
        partial_chunks = [full_count] if remainder else []
        mandatory = set(full_chunks[: self.sink_chunks])
        if self.recent_chunks:
            mandatory.update(full_chunks[-self.recent_chunks :])
        mandatory.update(partial_chunks)
        mandatory_full_count = sum(index < full_count for index in mandatory)
        mandatory_tokens = mandatory_full_count * self.chunk_size + (
            remainder if partial_chunks else 0
        )
        routed_count = max(0, (self.token_budget - mandatory_tokens) // self.chunk_size)
        candidate_base = torch.tensor(
            [
                index
                for index in full_chunks
                if index not in mandatory
                and index in self._summary_ready_host[layer_idx]
            ],
            device=query.device,
            dtype=torch.long,
        )
        selection_scores = (
            (
                group_score.sum(dim=0, keepdim=True)
                if self.shared_chunk_aggregation == "sum"
                else (
                    group_score.amax(dim=0, keepdim=True)
                    if self.shared_chunk_aggregation == "max"
                    # This deliberately omits per-Q-head normalization.  It
                    # is retained as a routing ablation: the empirical affine
                    # scores and the fixed token budget remain unchanged.
                    else log_mass.sum(dim=(0, 1), keepdim=False).unsqueeze(0)
                )
            )
            if self.share_chunks_across_kv_heads
            else group_score
        )
        selection_heads = selection_scores.shape[0]
        take = min(routed_count, candidate_base.numel())
        if take:
            # Candidates are in ascending logical order. Stable descending sort
            # therefore resolves score ties toward the lower logical chunk.
            order = torch.argsort(
                selection_scores[:, candidate_base],
                dim=-1,
                descending=True,
                stable=True,
            )
            routed = candidate_base[order[:, :take]]
        else:
            routed = candidate_base.new_empty((selection_heads, 0))
        mandatory_tensor = torch.tensor(
            sorted(mandatory), device=query.device, dtype=torch.long
        ).unsqueeze(0).expand(selection_heads, -1)
        block_table = torch.sort(
            torch.cat([mandatory_tensor, routed], dim=1), dim=1
        ).values.to(torch.int32)
        if self.share_chunks_across_kv_heads:
            block_table = block_table.expand(self.local_kv_heads, -1)
        selected_token_count = mandatory_tokens + take * self.chunk_size
        cache_seqlens = torch.full(
            (self.local_kv_heads,),
            selected_token_count,
            device=query.device,
            dtype=torch.int32,
        )
        return block_table, cache_seqlens

    def _gather_paged_tokens(
        self,
        cache: Tensor,
        physical_block_table: Tensor,
        token_indices: Tensor,
        kv_head: int | None = None,
    ) -> Tensor:
        if physical_block_table.ndim != 2 or physical_block_table.shape[0] != 1:
            raise ValueError("Query-Robust paged decode supports one request")
        physical_tokens = cache.shape[1]
        logical_blocks = torch.div(
            token_indices, physical_tokens, rounding_mode="floor"
        )
        offsets = token_indices.remainder(physical_tokens)
        table = physical_block_table[0].to(cache.device, dtype=torch.long)
        physical_blocks = table[logical_blocks]
        if kv_head is None:
            return cache[physical_blocks, offsets]
        return cache[physical_blocks, offsets, kv_head]

    def _gather_selected_by_head(
        self,
        cache: Tensor,
        physical_block_table: Tensor,
        selected_chunks: Tensor,
        total_tokens: int,
    ) -> Tensor:
        """Gather every KV head's selected logical tokens in one GPU launch."""
        within_chunk = torch.arange(
            self.chunk_size, device=cache.device, dtype=torch.long
        )
        token_indices = (
            selected_chunks.long().unsqueeze(-1) * self.chunk_size
            + within_chunk.view(1, 1, -1)
        ).flatten(1)
        valid = token_indices < total_tokens
        safe_indices = torch.where(valid, token_indices, 0)
        physical_tokens = cache.shape[1]
        logical_blocks = torch.div(
            safe_indices, physical_tokens, rounding_mode="floor"
        )
        offsets = safe_indices.remainder(physical_tokens)
        physical_blocks = physical_block_table[0, logical_blocks]
        heads = torch.arange(
            self.local_kv_heads, device=cache.device, dtype=torch.long
        ).unsqueeze(1)
        return cache[physical_blocks, offsets, heads]

    def _store_single_chunk(
        self, token_keys: Tensor, layer_idx: int, chunk: int
    ) -> None:
        summary, bias = self._summarize(
            token_keys.unsqueeze(0), self.request_centroid[layer_idx], layer_idx
        )
        self.summary_key[layer_idx, chunk].copy_(summary[0])
        self.summary_bias[layer_idx, chunk].copy_(bias[0])
        self.summary_ready[layer_idx, chunk].fill_(True)
        self.summary_valid_tokens[layer_idx, chunk] = self.chunk_size
        self._summary_ready_host[layer_idx].add(chunk)
        self.summaries_built += 1

    def _ensure_from_paged_cache(
        self,
        kv_cache: Tensor,
        physical_block_table: Tensor,
        cache_len: int,
        layer_idx: int,
    ) -> None:
        if self._centroid_ready_host[layer_idx]:
            return
        if cache_len < 2:
            raise ValueError("Query-Robust paged recovery requires prompt plus decode token")
        # Schema-2 Pile pools require the full empirical rows to be transported
        # to this request position.  Legacy centroid artifacts retain the
        # older rotation-only path for compatibility with their tests.
        self._prepare_query_state(layer_idx, cache_len - 1)
        token_indices = torch.arange(cache_len, device=kv_cache.device)
        keys = self._gather_paged_tokens(
            kv_cache[0], physical_block_table, token_indices
        )
        full_chunks = cache_len // self.chunk_size
        self._store_full_chunks(keys, layer_idx, 0, full_chunks)
        chunks = (cache_len + self.chunk_size - 1) // self.chunk_size
        self.summary_valid_tokens[layer_idx, :chunks].fill_(self.chunk_size)
        if cache_len % self.chunk_size:
            self.summary_valid_tokens[layer_idx, chunks - 1] = (
                cache_len % self.chunk_size
            )

    @torch.no_grad()
    def decode_paged(
        self,
        query: Tensor,
        kv_cache: Tensor,
        physical_block_table: Tensor,
        tokens_per_head: Tensor | None,
        output: Tensor,
        layer_idx: int = 0,
        total_tokens: int | None = None,
    ) -> Tensor:
        """Route directly from vLLM's canonical cache via selected scratch."""
        self._require_state(kv_cache)
        if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
            raise ValueError("Query-Robust expects canonical [2, blocks, tokens, heads, dim]")
        if total_tokens is None:
            if tokens_per_head is None:
                raise ValueError("Query-Robust decode requires a host or tensor length")
            total_tokens = int(tokens_per_head[0].item())
        self._wait_for_pending_prefill_summaries(layer_idx)
        build_start = time.perf_counter()
        self._ensure_from_paged_cache(
            kv_cache, physical_block_table, total_tokens, layer_idx
        )
        if self._dense_fallback and total_tokens > self.token_budget:
            # A nonconverged robust certificate must never silently become a
            # heuristic sparse route.  Reuse the intact canonical cache and
            # report the explicit dense fallback to telemetry.
            self.last_accessed_tokens = total_tokens
            flash_attn_with_kvcache(
                q=query.unsqueeze(0),
                k_cache=kv_cache[0],
                v_cache=kv_cache[1],
                block_table=physical_block_table.to(device=query.device, dtype=torch.int32),
                cache_seqlens=torch.tensor([total_tokens], device=query.device, dtype=torch.int32),
                causal=True,
                out=output.unsqueeze(0),
            )
            if layer_idx == self.num_layers - 1:
                from sparse_frontier.utils.sparsity_server import mark_query_robust_decode

                mark_query_robust_decode()
            return output
        if total_tokens % self.chunk_size == 0:
            chunk = total_tokens // self.chunk_size - 1
            if chunk not in self._summary_ready_host[layer_idx]:
                token_indices = torch.arange(
                    total_tokens - self.chunk_size,
                    total_tokens,
                    device=kv_cache.device,
                )
                token_keys = self._gather_paged_tokens(
                    kv_cache[0], physical_block_table, token_indices
                )
                self._store_single_chunk(token_keys, layer_idx, chunk)
        self.summary_build_ms += (time.perf_counter() - build_start) * 1000.0

        active_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        if total_tokens <= self.token_budget:
            selected = torch.arange(
                active_chunks, device=query.device, dtype=torch.int32
            ).unsqueeze(0).expand(self.local_kv_heads, -1)
            selected_lengths = torch.full(
                (self.local_kv_heads,),
                total_tokens,
                device=query.device,
                dtype=torch.int32,
            )
        else:
            partial = None
            remainder = total_tokens % self.chunk_size
            if remainder:
                token_indices = torch.arange(
                    total_tokens - remainder, total_tokens, device=kv_cache.device
                )
                partial = self._gather_paged_tokens(
                    kv_cache[0], physical_block_table, token_indices
                )
            selected, selected_lengths = self._select_logical_chunks(
                query,
                total_tokens,
                layer_idx,
                partial,
            )

        selected_blocks = selected.shape[1]
        if selected_blocks * self.chunk_size > self.token_budget:
            raise RuntimeError("Query-Robust selected scratch exceeds the token budget")
        self.last_accessed_tokens = selected_blocks * self.chunk_size
        remainder = total_tokens % self.chunk_size
        if remainder:
            self.last_accessed_tokens -= self.chunk_size - remainder

        if self.share_chunks_across_kv_heads:
            canonical_table = physical_block_table[0, selected[0].long()].to(
                device=query.device, dtype=torch.int32
            ).unsqueeze(0)
            canonical_seqlens = torch.tensor(
                [self.last_accessed_tokens], device=query.device, dtype=torch.int32
            )
            flash_attn_with_kvcache(
                q=query.unsqueeze(0),
                k_cache=kv_cache[0],
                v_cache=kv_cache[1],
                block_table=canonical_table,
                cache_seqlens=canonical_seqlens,
                causal=True,
                out=output.unsqueeze(0),
            )
            if layer_idx == self.num_layers - 1:
                from sparse_frontier.utils.sparsity_server import mark_query_robust_decode

                mark_query_robust_decode()
            return output

        physical_table = physical_block_table.to(
            device=query.device, dtype=torch.long
        )
        gathered_keys = self._gather_selected_by_head(
            kv_cache[0], physical_table, selected, total_tokens
        )
        gathered_values = self._gather_selected_by_head(
            kv_cache[1], physical_table, selected, total_tokens
        )
        selected_capacity = selected_blocks * self.chunk_size
        self.selected_key_scratch[:, :selected_capacity].copy_(gathered_keys)
        self.selected_value_scratch[:, :selected_capacity].copy_(gathered_values)

        scratch_blocks = self.token_budget // self.chunk_size
        offsets = (
            torch.arange(self.local_kv_heads, device=query.device, dtype=torch.int32)
            * scratch_blocks
        )
        scratch_table = (
            torch.arange(selected_blocks, device=query.device, dtype=torch.int32)
            .unsqueeze(0)
            .expand(self.local_kv_heads, -1)
            + offsets.unsqueeze(1)
        ).repeat_interleave(self.group_size, dim=0)
        cache_seqlens = selected_lengths.repeat_interleave(self.group_size)
        flash_attn_with_kvcache(
            q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
            k_cache=self.selected_key_scratch.view(
                self.local_kv_heads * scratch_blocks,
                self.chunk_size,
                1,
                query.shape[-1],
            ),
            v_cache=self.selected_value_scratch.view(
                self.local_kv_heads * scratch_blocks,
                self.chunk_size,
                1,
                query.shape[-1],
            ),
            block_table=scratch_table,
            cache_seqlens=cache_seqlens,
            causal=True,
            out=output.squeeze(0).unsqueeze(1).unsqueeze(1),
        )
        if layer_idx == self.num_layers - 1:
            from sparse_frontier.utils.sparsity_server import mark_query_robust_decode

            mark_query_robust_decode()
        return output

    @torch.no_grad()
    def decode(
        self,
        query: Tensor,
        keys: Tensor,
        values: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        tokens_per_head: Tensor,
        output: Tensor,
        layer_idx: int = 0,
    ) -> Tensor:
        del keys, values
        self._require_state(k_cache)
        total_tokens = int(tokens_per_head[0].item())
        self._wait_for_pending_prefill_summaries(layer_idx)
        build_start = time.perf_counter()
        self._finalize_new_chunk(k_cache, total_tokens, layer_idx)
        self.summary_build_ms += (time.perf_counter() - build_start) * 1000.0
        active_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size
        if total_tokens <= self.token_budget:
            block_table = self._dense_block_table(
                k_cache.shape[1], active_chunks, query.device
            )
            cache_seqlens = torch.full(
                (self.local_q_heads,),
                total_tokens,
                device=query.device,
                dtype=torch.int32,
            )
        else:
            block_table, cache_seqlens = self._select_block_table(
                query, k_cache, total_tokens, layer_idx
            )
        self.last_accessed_tokens = int(cache_seqlens.max().item())
        flash_attn_with_kvcache(
            q=query.squeeze(0).unsqueeze(1).unsqueeze(1),
            k_cache=k_cache.view(
                self.local_kv_heads * k_cache.shape[1],
                self.chunk_size,
                1,
                k_cache.shape[-1],
            ),
            v_cache=v_cache.view(
                self.local_kv_heads * v_cache.shape[1],
                self.chunk_size,
                1,
                v_cache.shape[-1],
            ),
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            causal=True,
            out=output.squeeze(0).unsqueeze(1).unsqueeze(1),
        )
        if layer_idx == self.num_layers - 1:
            from sparse_frontier.utils.sparsity_server import mark_query_robust_decode

            mark_query_robust_decode()
        return output
