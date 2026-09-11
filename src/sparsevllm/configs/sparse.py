"""Sparse-method normalization and layout-dependent validation."""

import math
from pathlib import Path

from sparsevllm.configs.common import (
    _coerce_bool_config,
    _normalize_float_attr,
    _normalize_int_attr,
    _normalize_positive_int,
)
from sparsevllm.method_registry import (
    CANONICAL_PREFILL_SPARSE_METHODS,
    PREFILL_SPARSE_METHOD_COMPATIBILITY,
    SKIPKV_ASSET_MODEL_NAMES,
    SUPPORTED_SPARSE_METHODS,
    normalize_sparse_method,
    resolve_prefill_sparse_method,
    resolve_sparse_prefill_score_mode,
)
from sparsevllm.models.rope import resolve_rope_theta
from sparsevllm.utils.log import logger, log_once


def resolve_shadowkv_outlier_chunks(
    sparse_budget: int,
    configured: int | None = None,
) -> int:
    """Resolve ShadowKV's per-KV-head outlier chunk count.

    Lightweight runtime configs may omit normalized fields.  Keeping the
    derived default here makes the decode payload and cache-manager workspace
    use the same capacity contract.
    """

    if configured is None:
        return max(1, 24 * int(sparse_budget) // 1024)
    value = int(configured)
    if value <= 0:
        raise ValueError(
            "shadowkv_outlier_chunks must be > 0, "
            f"got {value}."
        )
    return value


def resolve_shadowkv_local_token_capacity(
    chunk_size: int,
    local_chunks: int,
    context_capacity: int,
) -> int:
    """Return the direct-tail capacity after ShadowKV chunk alignment."""

    chunk_size = int(chunk_size)
    local_chunks = int(local_chunks)
    context_capacity = int(context_capacity)
    if chunk_size <= 0 or local_chunks < 0 or context_capacity <= 0:
        raise ValueError(
            "ShadowKV tail capacity requires chunk_size > 0, local_chunks >= 0, "
            f"context_capacity > 0; got chunk_size={chunk_size}, "
            f"local_chunks={local_chunks}, context_capacity={context_capacity}."
        )
    # _finalize_entry aligns the sparse chunk count down to a multiple of
    # eight, so the direct tail may contain seven additional full chunks plus
    # one partial chunk.
    aligned_chunk_groups = 8
    return min(
        context_capacity,
        local_chunks * chunk_size
        + (aligned_chunk_groups - 1) * chunk_size
        + chunk_size
        - 1,
    )


def normalize_sparse_method_name(config) -> None:
    config.sparse_method = normalize_sparse_method(config.sparse_method)
    if config.sparse_method not in SUPPORTED_SPARSE_METHODS:
        supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
        raise ValueError(
            f"Unsupported sparse_method={config.sparse_method!r}. "
            f"Supported methods: '', {supported}."
        )
    for name in ("sink_keep_tokens", "decode_keep_tokens", "recent_keep_tokens"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"{name} must be a non-negative integer token count, got {value!r}."
            )


def normalize_prefill_sparse_method(config) -> None:
    sparse_method = normalize_sparse_method(getattr(config, "sparse_method", ""))
    method = resolve_prefill_sparse_method(
        getattr(config, "prefill_sparse_method", ""),
        sparse_method=sparse_method,
    )
    if method not in CANONICAL_PREFILL_SPARSE_METHODS:
        supported = ", ".join(
            repr(name) for name in sorted(CANONICAL_PREFILL_SPARSE_METHODS) if name
        )
        raise ValueError(
            f"Unsupported prefill_sparse_method={method!r}. "
            f"Supported methods: '', {supported}."
        )
    compatible_sparse_methods = PREFILL_SPARSE_METHOD_COMPATIBILITY[method]
    if sparse_method not in compatible_sparse_methods:
        choices = ", ".join(
            "'vanilla'" if name == "" else repr(name)
            for name in sorted(compatible_sparse_methods)
        )
        raise ValueError(
            f"prefill_sparse_method={method!r} is incompatible with "
            f"sparse_method={sparse_method!r}; supported cache/decode methods: "
            f"{choices}."
        )
    config.prefill_sparse_method = method

    for name in (
        "flashprefill_v2_k_block_m",
        "flashprefill_v2_k_block_n",
    ):
        _normalize_positive_int(config, name, fallback=0)
    if config.flashprefill_v2_k_block_m % 16:
        raise ValueError(
            "flashprefill_v2_k_block_m must be a multiple of 16, got "
            f"{config.flashprefill_v2_k_block_m}."
        )
    block_n = int(config.flashprefill_v2_k_block_n)
    if block_n % 64 or block_n & (block_n - 1):
        raise ValueError(
            "flashprefill_v2_k_block_n must be a power-of-two multiple of 64, "
            f"got {block_n}."
        )
    threshold = config.flashprefill_v2_abs_threshold
    if threshold is None:
        if method == "flashprefill_v2":
            raise ValueError(
                "prefill_sparse_method='flashprefill_v2' requires an explicit "
                "flashprefill_v2_abs_threshold calibrated for the model."
            )
    else:
        _normalize_float_attr(config, "flashprefill_v2_abs_threshold")
        if not 0.0 <= config.flashprefill_v2_abs_threshold <= 1.0:
            raise ValueError(
                "flashprefill_v2_abs_threshold must be in [0, 1], got "
                f"{config.flashprefill_v2_abs_threshold}."
            )
    for name in (
        "flashprefill_v2_attention_sink_blocks",
        "flashprefill_v2_window_blocks",
        "flashprefill_v2_last_query_blocks",
        "flashprefill_v2_min_sparse_q_len",
    ):
        _normalize_int_attr(config, name, fallback=0)
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative, got {getattr(config, name)}.")
    config.flashprefill_v2_use_mean_correction = _coerce_bool_config(
        "flashprefill_v2_use_mean_correction",
        config.flashprefill_v2_use_mean_correction,
    )


def _normalize_quest(config) -> None:
    if isinstance(config.full_attention_layers, str):
        layers = config.full_attention_layers.strip()
        if layers.lower() != "auto":
            config.full_attention_layers = (
                [] if not layers else [int(x) for x in layers.split(",")]
            )

    if config.quest_chunk_size <= 0:
        raise ValueError("quest_chunk_size 必须 > 0")
    config.quest_token_budget = 0
    if config.sparse_method == "quest":
        config.quest_token_budget = (
            config.sink_keep_tokens
            + config.decode_keep_tokens
            + config.recent_keep_tokens
        )
        if config.quest_token_budget <= 0:
            raise ValueError(
                "QuEST derived token budget must be > 0: "
                "sink_keep_tokens + decode_keep_tokens + recent_keep_tokens "
                f"= {config.quest_token_budget}."
            )
    if config.quest_skip_layers < 0:
        raise ValueError("quest_skip_layers 不能 < 0")
    if config.sparse_method == "quest":
        config.sparse_page_size = int(config.quest_chunk_size)
        config.sparse_token_budget = int(config.quest_token_budget)
        config.sparse_skip_layers = int(config.quest_skip_layers)


def _normalize_query_robust(config) -> None:
    """Validate QR's asset-backed explicit-KV runtime contract."""

    if config.sparse_method != "query_robust":
        return
    path = config.query_robust_vertices_path
    if not isinstance(path, str) or not path.strip():
        raise ValueError(
            "query_robust requires query_robust_vertices_path pointing to a "
            "calibrated vertex asset."
        )
    path = str(Path(path).expanduser())
    if not Path(path).is_file():
        raise FileNotFoundError(
            "Query-Robust vertex asset does not exist: "
            f"query_robust_vertices_path={path!r}."
        )
    config.query_robust_vertices_path = path

    _normalize_positive_int(config, "query_robust_num_vertices", fallback=0)
    if config.query_robust_num_vertices < 2:
        raise ValueError(
            "query_robust_num_vertices must be at least 2, got "
            f"{config.query_robust_num_vertices}."
        )
    _normalize_positive_int(config, "query_robust_chunk_size", fallback=0)
    _normalize_positive_int(config, "query_robust_solver_iters", fallback=0)
    _normalize_float_attr(config, "query_robust_solver_lr")
    if (
        not math.isfinite(config.query_robust_solver_lr)
        or config.query_robust_solver_lr <= 0
    ):
        raise ValueError(
            "query_robust_solver_lr must be finite and > 0, got "
            f"{config.query_robust_solver_lr}."
        )
    _normalize_float_attr(config, "query_robust_score_alpha")
    if config.query_robust_score_alpha not in (0.0, 0.5, 1.0):
        raise ValueError(
            "query_robust_score_alpha must be one of 0, 0.5, or 1, got "
            f"{config.query_robust_score_alpha}."
        )
    _normalize_int_attr(config, "query_robust_skip_layers", fallback=0)
    if config.query_robust_skip_layers < 0:
        raise ValueError(
            "query_robust_skip_layers must be non-negative, got "
            f"{config.query_robust_skip_layers}."
        )
    config.query_robust_uniform_p = _coerce_bool_config(
        "query_robust_uniform_p",
        config.query_robust_uniform_p,
    )
    if config.query_robust_rope_config is None:
        hf_config = getattr(config, "hf_config", None)
        rope_config = getattr(hf_config, "rope_parameters", None)
        if rope_config is None:
            rope_config = getattr(hf_config, "rope_scaling", None)
        if rope_config is not None:
            config.query_robust_rope_config = dict(rope_config)
            config.query_robust_rope_config["rope_theta"] = resolve_rope_theta(hf_config)
    if config.query_robust_model_fingerprint is not None:
        config.query_robust_model_fingerprint = str(
            config.query_robust_model_fingerprint
        )
    if str(config.attention_cache_layout) != "explicit_kv":
        raise NotImplementedError(
            "Query-Robust currently requires homogeneous explicit KV storage; "
            f"got attention_cache_layout={config.attention_cache_layout!r}."
        )
    runtime_layout = getattr(config, "runtime_layout", None)
    if getattr(runtime_layout, "linear_attention_layer_indices", ()):
        raise NotImplementedError(
            "Query-Robust currently supports transformer models with explicit KV "
            "layers only; mixed recurrent attention is unsupported."
        )

    num_kv_heads = int(getattr(config.hf_config, "num_key_value_heads", 0) or 0)
    parallel_topology = getattr(config, "parallel_topology", None)
    tp_size = int(
        getattr(
            parallel_topology,
            "attention_tp_size",
            getattr(config, "tensor_parallel_size", 1),
        )
    )
    shape_resolver = getattr(runtime_layout, "local_kv_shapes", None)
    local_shapes = shape_resolver(tp_size) if callable(shape_resolver) else ()
    if len(set(local_shapes)) > 1:
        raise NotImplementedError(
            "Query-Robust currently requires one homogeneous explicit-KV shape "
            "across all attention layers."
        )
    if num_kv_heads <= 0 or num_kv_heads % tp_size:
        raise ValueError(
            "Query-Robust requires num_key_value_heads divisible by "
            f"tensor_parallel_size: num_kv_heads={num_kv_heads} "
            f"tensor_parallel_size={tp_size}."
        )
    config.sparse_page_size = int(config.query_robust_chunk_size)
    config.sparse_token_budget = int(
        config.sink_keep_tokens
        + config.decode_keep_tokens
        + config.recent_keep_tokens
    )
    if config.sparse_token_budget <= 0:
        raise ValueError(
            "Query-Robust decode token budget must be positive: "
            "sink_keep_tokens + decode_keep_tokens + recent_keep_tokens = "
            f"{config.sparse_token_budget}."
        )
    config.sparse_skip_layers = int(config.query_robust_skip_layers)


def _normalize_shadowkv(config) -> None:
    for name in (
        "shadowkv_sparse_budget",
        "shadowkv_rank",
        "shadowkv_chunk_size",
        "shadowkv_local_chunks",
        "shadowkv_recent_tokens",
        "shadowkv_svd_batch_size",
        "shadowkv_svd_oversample",
        "shadowkv_svd_niter",
        "shadowkv_decode_page_size",
    ):
        _normalize_positive_int(config, name, fallback=0)
    if config.sparse_method != "shadowkv":
        return
    backend = str(getattr(config, "shadowkv_kernel_backend", "auto")).strip().lower()
    if backend not in {"auto", "torch", "cutlass"}:
        raise ValueError(
            "shadowkv_kernel_backend must be 'auto', 'torch', or 'cutlass', "
            f"got {config.shadowkv_kernel_backend!r}."
        )
    config.shadowkv_kernel_backend = backend
    decode_backend = str(
        getattr(config, "shadowkv_decode_backend", "auto")
    ).strip().lower()
    if decode_backend not in {"auto", "flashinfer", "triton"}:
        raise ValueError(
            "shadowkv_decode_backend must be 'auto', 'flashinfer', or 'triton', "
            f"got {config.shadowkv_decode_backend!r}."
        )
    config.shadowkv_decode_backend = decode_backend
    flashinfer_backend = str(
        getattr(config, "shadowkv_flashinfer_backend", "auto")
    ).strip().lower()
    if flashinfer_backend not in {"auto", "fa2", "fa3", "cute-dsl"}:
        raise ValueError(
            "shadowkv_flashinfer_backend must be one of 'auto', 'fa2', 'fa3', "
            "or 'cute-dsl', "
            f"got {config.shadowkv_flashinfer_backend!r}."
        )
    config.shadowkv_flashinfer_backend = flashinfer_backend
    storage = str(getattr(config, "shadowkv_storage", "gpu_cache")).strip().lower()
    if storage not in {"cpu", "gpu_cache"}:
        raise ValueError(
            "shadowkv_storage must be 'cpu' or 'gpu_cache', "
            f"got {config.shadowkv_storage!r}."
        )
    config.shadowkv_storage = storage
    config.shadowkv_multistream_gather = _coerce_bool_config(
        "shadowkv_multistream_gather",
        getattr(config, "shadowkv_multistream_gather", True),
    )
    config.shadowkv_gather_copy_with_offsets = _coerce_bool_config(
        "shadowkv_gather_copy_with_offsets",
        getattr(config, "shadowkv_gather_copy_with_offsets", True),
    )
    if config.shadowkv_decode_page_size not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError(
            "shadowkv_decode_page_size must be one of 1, 2, 4, 8, 16, 32, or 64, "
            f"got {config.shadowkv_decode_page_size}."
        )
    _normalize_int_attr(config, "shadowkv_gpu_cache_tokens", fallback=0)
    if config.shadowkv_gpu_cache_tokens < 0:
        raise ValueError(
            "shadowkv_gpu_cache_tokens must be non-negative, got "
            f"{config.shadowkv_gpu_cache_tokens}."
        )
    max_model_len = getattr(config, "max_model_len", None)
    if storage == "gpu_cache" and config.shadowkv_gpu_cache_tokens == 0:
        if max_model_len is None or int(max_model_len) <= 0:
            raise ValueError(
                "shadowkv_storage='gpu_cache' needs max_model_len when "
                "shadowkv_gpu_cache_tokens is left at the auto sentinel 0."
            )
        config.shadowkv_gpu_cache_tokens = int(max_model_len)
    if (
        storage == "gpu_cache"
        and max_model_len is not None
        and config.shadowkv_gpu_cache_tokens < int(max_model_len)
    ):
        raise ValueError(
            "shadowkv_gpu_cache_tokens must cover max_model_len in gpu_cache mode: "
            f"capacity={config.shadowkv_gpu_cache_tokens} max_model_len={max_model_len}."
        )
    svd_method = str(getattr(config, "shadowkv_svd_method", "exact")).strip().lower()
    if svd_method not in {"exact", "lowrank"}:
        raise ValueError(
            "shadowkv_svd_method must be 'exact' or 'lowrank', "
            f"got {config.shadowkv_svd_method!r}."
        )
    config.shadowkv_svd_method = svd_method
    if config.shadowkv_sparse_budget % config.shadowkv_chunk_size:
        raise ValueError(
            "shadowkv_sparse_budget must be divisible by shadowkv_chunk_size, got "
            f"budget={config.shadowkv_sparse_budget} chunk_size={config.shadowkv_chunk_size}."
        )
    num_kv_heads = int(getattr(config.hf_config, "num_key_value_heads", 0) or 0)
    head_dim = int(
        getattr(config.hf_config, "head_dim", 0)
        or config.hf_config.hidden_size // config.hf_config.num_attention_heads
    )
    flattened_kv_dim = num_kv_heads * head_dim
    if flattened_kv_dim <= 0:
        raise ValueError("ShadowKV requires a positive flattened KV dimension.")
    if config.shadowkv_rank > flattened_kv_dim:
        raise ValueError(
            "shadowkv_rank cannot exceed the flattened KV dimension: "
            f"rank={config.shadowkv_rank} flattened_kv_dim={flattened_kv_dim}."
        )
    config.shadowkv_outlier_chunks = resolve_shadowkv_outlier_chunks(
        config.shadowkv_sparse_budget,
        config.shadowkv_outlier_chunks,
    )
    if config.tensor_parallel_size != 1:
        raise NotImplementedError(
            "ShadowKV currently supports tensor_parallel_size=1 only."
        )
    if str(config.attention_cache_layout) != "explicit_kv":
        raise NotImplementedError(
            "ShadowKV currently requires homogeneous explicit KV storage."
        )


def _normalize_snapkv(config) -> None:
    _normalize_int_attr(config, "snapkv_num_full_layers")
    if config.snapkv_num_full_layers != 0:
        raise ValueError(
            "snapkv_num_full_layers is unsupported and must be 0, got "
            f"{config.snapkv_num_full_layers}."
        )


def _normalize_h2o(config) -> None:
    _normalize_positive_int(config, "h2o_decode_budget", fallback=0)
    _normalize_positive_int(config, "h2o_decode_eviction_interval", fallback=0)
    _normalize_int_attr(config, "h2o_prefill_budget", fallback=0)
    if config.h2o_prefill_budget < config.h2o_decode_budget:
        raise ValueError(
            "h2o_prefill_budget must be >= h2o_decode_budget, "
            f"got prefill={config.h2o_prefill_budget} decode={config.h2o_decode_budget}."
        )
    _normalize_float_attr(config, "h2o_recent_ratio")
    if not 0.0 < config.h2o_recent_ratio < 1.0:
        raise ValueError(
            f"h2o_recent_ratio must be in (0, 1), got {config.h2o_recent_ratio}."
        )
    _normalize_int_attr(config, "h2o_prefill_score_window", fallback=0)
    score_mode = getattr(config, "sparse_prefill_score_mode", "probability")
    if score_mode == "logits":
        if config.h2o_prefill_score_window < 0:
            raise ValueError(
                "h2o_prefill_score_window must be non-negative in logits "
                f"mode (0 means the full chunk), got {config.h2o_prefill_score_window}."
            )
    elif not 0 <= config.h2o_prefill_score_window <= 128:
        raise ValueError(
            "h2o_prefill_score_window must be in [0, 128] in probability mode "
            "(0 means the full current chunk), got "
            f"{config.h2o_prefill_score_window}."
        )


def _normalize_sparse_prefill_score(config) -> None:
    mode = resolve_sparse_prefill_score_mode(
        config.sparse_method,
        config.sparse_prefill_score_mode,
    )
    allowed = {"probability", "logits"}
    if mode not in allowed:
        raise ValueError(
            "sparse_prefill_score_mode must be one of "
            f"{sorted(allowed)}, got {config.sparse_prefill_score_mode!r}."
        )
    if mode != "probability" and config.sparse_method not in {
        "snapkv",
        "pyramidkv",
        "h2o",
    }:
        raise ValueError(
            "sparse_prefill_score_mode='logits' only applies to "
            f"SnapKV/PyramidKV/H2O, got method={config.sparse_method!r}."
        )
    if mode == "logits" and config.sparse_attn_score_dtype != "float32":
        raise ValueError(
            "sparse_prefill_score_mode='logits' requires "
            "sparse_attn_score_dtype='float32', got "
            f"{config.sparse_attn_score_dtype!r}."
        )
    config.sparse_prefill_score_mode = mode

def _normalize_rkv(config) -> None:
    _normalize_positive_int(config, "rkv_compression_interval", fallback=0)
    _normalize_positive_int(config, "rkv_observation_tokens", fallback=0)
    if config.rkv_observation_tokens > 128:
        raise ValueError(
            "rkv_observation_tokens must be <= 128 because the prefill score kernel "
            f"supports at most 128 query tokens, got {config.rkv_observation_tokens}."
        )
    if config.rkv_observation_tokens > config.rkv_compression_interval:
        raise ValueError(
            "rkv_observation_tokens must be <= rkv_compression_interval so the query cache "
            "can be refreshed between decode evictions, "
            f"got observation={config.rkv_observation_tokens} interval={config.rkv_compression_interval}."
        )
    _normalize_float_attr(config, "rkv_alpha")
    if not 0.0 <= config.rkv_alpha <= 1.0:
        raise ValueError(f"rkv_alpha must be in [0, 1], got {config.rkv_alpha}.")
    _normalize_float_attr(config, "rkv_similarity_threshold")
    if not 0.0 <= config.rkv_similarity_threshold <= 1.0:
        raise ValueError(
            "rkv_similarity_threshold must be in [0, 1], "
            f"got {config.rkv_similarity_threshold}."
        )
    _normalize_int_attr(config, "rkv_recent_similar_keep")
    if config.rkv_recent_similar_keep < 0:
        raise ValueError(
            f"rkv_recent_similar_keep must be >= 0, got {config.rkv_recent_similar_keep}."
        )
    _normalize_positive_int(config, "rkv_max_redundancy_tokens", fallback=0)
    _normalize_int_attr(config, "rkv_redundancy_window", fallback=0)
    if config.rkv_redundancy_window < 0:
        raise ValueError(
            f"rkv_redundancy_window must be >= 0, got {config.rkv_redundancy_window}."
        )
    if 0 < config.rkv_redundancy_window > config.rkv_max_redundancy_tokens:
        raise ValueError(
            "rkv_redundancy_window must be <= rkv_max_redundancy_tokens, "
            f"got window={config.rkv_redundancy_window} max={config.rkv_max_redundancy_tokens}."
        )
    if config.sparse_method == "rkv":
        log_once(
            "R-KV support is an approximation of the official implementation: "
            "Sparse-VLLM uses one shared physical token index set across KV heads, "
            "so official per-KV-head token selection is not fully reproduced. "
            f"rkv_redundancy_window={config.rkv_redundancy_window}; values > 0 score "
            "redundancy only over the trailing candidate tokens.",
            level="WARNING",
        )

def _normalize_skipkv(config) -> None:
    _normalize_positive_int(config, "skipkv_compression_interval", fallback=0)
    _normalize_float_attr(config, "skipkv_alpha")
    if config.skipkv_alpha < 0.0:
        raise ValueError(f"skipkv_alpha must be >= 0, got {config.skipkv_alpha}.")
    _normalize_float_attr(config, "skipkv_similarity_threshold")
    if not 0.0 <= config.skipkv_similarity_threshold <= 1.0:
        raise ValueError(
            "skipkv_similarity_threshold must be in [0, 1], "
            f"got {config.skipkv_similarity_threshold}."
        )
    _normalize_positive_int(config, "skipkv_segment_size", fallback=0)
    _normalize_positive_int(config, "skipkv_max_redundancy_tokens", fallback=0)
    _normalize_positive_int(config, "skipkv_redundancy_window", fallback=0)
    if config.skipkv_redundancy_window > config.skipkv_max_redundancy_tokens:
        raise ValueError(
            "skipkv_redundancy_window must be <= skipkv_max_redundancy_tokens, "
            f"got window={config.skipkv_redundancy_window} max={config.skipkv_max_redundancy_tokens}."
        )
    config.skipkv_enable_sentence_scoring = bool(config.skipkv_enable_sentence_scoring)
    _normalize_float_attr(config, "skipkv_sentence_score_weight")
    if config.skipkv_sentence_score_weight < 0.0:
        raise ValueError(
            "skipkv_sentence_score_weight must be >= 0, "
            f"got {config.skipkv_sentence_score_weight}."
        )
    _normalize_positive_int(config, "skipkv_sentence_min_tokens", fallback=0)
    _normalize_int_attr(config, "skipkv_sentence_max_tokens", fallback=0)
    if config.skipkv_sentence_max_tokens < config.skipkv_sentence_min_tokens:
        raise ValueError(
            "skipkv_sentence_max_tokens must be >= skipkv_sentence_min_tokens, "
            f"got max={config.skipkv_sentence_max_tokens} min={config.skipkv_sentence_min_tokens}."
        )
    _normalize_int_attr(config, "skipkv_sentence_embedding_layer")
    _normalize_positive_int(config, "skipkv_max_tracked_sentences", fallback=0)
    config.skipkv_enable_activation_steering = bool(config.skipkv_enable_activation_steering)
    _normalize_int_attr(config, "skipkv_steering_layer")
    _normalize_float_attr(config, "skipkv_steering_alpha")
    _normalize_float_attr(config, "skipkv_steering_alpha_increment")
    _normalize_float_attr(config, "skipkv_steering_alpha_max")
    if config.skipkv_enable_activation_steering and not config.skipkv_steering_vector_path:
        raise ValueError(
            "skipkv_enable_activation_steering=True requires skipkv_steering_vector_path. "
            "Official SkipKV support is limited to the released steering vectors for "
            f"{', '.join(sorted(SKIPKV_ASSET_MODEL_NAMES))}."
        )


def _validate_prefill_sparse_method_model_compatibility(config) -> None:
    if config.prefill_sparse_method != "flashprefill_v2":
        return
    cache_layout = str(config.attention_cache_layout)
    if cache_layout != "explicit_kv":
        raise NotImplementedError(
            "prefill_sparse_method='flashprefill_v2' requires explicit KV cache "
            f"storage; model {config.model_spec.name!r} uses "
            f"attention_cache_layout={cache_layout!r}."
        )


def normalize_sparse_methods(config) -> None:
    _validate_prefill_sparse_method_model_compatibility(config)
    if (
        getattr(config.hf_config, "model_type", "") == "gemma4_text"
        and int(getattr(config.hf_config, "num_kv_shared_layers", 0) or 0)
        and config.sparse_method == "streamingllm"
    ):
        raise NotImplementedError(
            "Gemma 4 StreamingLLM requires independent per-layer KV caches; "
            "KV-sharing variants support vanilla and OmniKV."
        )
    _normalize_quest(config)
    _normalize_query_robust(config)
    _normalize_snapkv(config)
    # _normalize_sparse_prefill_score must run before _normalize_h2o to validate
    # and canonicalize config.sparse_prefill_score_mode before H2O window checks.
    _normalize_sparse_prefill_score(config)
    _normalize_h2o(config)
    _normalize_rkv(config)
    _normalize_skipkv(config)
    _normalize_shadowkv(config)

def finalize_sparse_layout(config) -> None:
    configured_full_layers = {int(layer) for layer in config.full_attention_layers}
    kv_layers = tuple(int(layer) for layer in config.runtime_layout.kv_idx_to_layer_idx)
    kv_positions = {layer: index for index, layer in enumerate(kv_layers)}
    unknown_full_layers = sorted(configured_full_layers - set(kv_layers))
    if unknown_full_layers and config.sparse_method in {"omnikv", "deltakv"}:
        raise ValueError(
            "full_attention_layers must contain KV/full-attention layer indices for "
            f"{config.sparse_method}; non-KV layers={unknown_full_layers}."
        )
    config.obs_layer_ids = []
    for layer in config.full_attention_layers:
        layer = int(layer)
        kv_position = kv_positions.get(layer)
        if kv_position is None or kv_position + 1 >= len(kv_layers):
            continue
        if kv_layers[kv_position + 1] not in configured_full_layers:
            config.obs_layer_ids.append(layer)

    # PyramidKV 配置验证与智能生成
    if 'pyramidkv' == config.sparse_method:
        num_layers = int(config.runtime_layout.num_layers)
        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if config.pyramid_layer_ratios is None:
            start_l = int(config.pyramidkv_start_layer)
            least_l = (
                int(config.pyramidkv_least_layer)
                if config.pyramidkv_least_layer is not None
                else num_kv_layers - 1
            )
            start_r = float(config.pyramidkv_start_ratio)
            least_r = float(config.pyramidkv_least_ratio)
            if not 0 <= start_l < num_kv_layers:
                raise ValueError(
                    f"pyramidkv_start_layer must be a KV layer position in [0, {num_kv_layers}), "
                    f"got {start_l}."
                )
            if not start_l <= least_l < num_kv_layers:
                raise ValueError(
                    "pyramidkv_least_layer must be a KV layer position between "
                    f"start_layer={start_l} and {num_kv_layers - 1}, got {least_l}."
                )

            ratios = [1.0] * num_kv_layers
            for i in range(start_l, num_kv_layers):
                if i <= least_l:
                    if least_l > start_l:
                        ratio = start_r - (start_r - least_r) * (i - start_l) / (least_l - start_l)
                    else:
                        ratio = least_r
                    ratios[i] = ratio
                else:
                    ratios[i] = least_r
            config.pyramid_layer_ratios = ratios
            logger.info(f"PyramidKV 自动生成 KV layer_ratios = {[f'{r:.3f}' for r in ratios]}")
        else:
            ratios = [float(ratio) for ratio in config.pyramid_layer_ratios]
            if len(ratios) == num_layers and num_layers != num_kv_layers:
                ratios = [ratios[layer_idx] for layer_idx in config.runtime_layout.kv_idx_to_layer_idx]
            config.pyramid_layer_ratios = ratios

    if config.pyramid_layer_ratios is not None:
        # PyramidKV 模式自动启用 SnapKV 逻辑
        if 'pyramidkv' != config.sparse_method:
            raise ValueError('sparse_method 应为 pyramidkv')

        num_kv_layers = int(config.runtime_layout.num_kv_layers)
        if len(config.pyramid_layer_ratios) != num_kv_layers:
            raise ValueError(
                f"pyramid_layer_ratios length ({len(config.pyramid_layer_ratios)}) must equal "
                f"the number of KV/full-attention layers ({num_kv_layers})."
            )

        if any(r <= 0 or r > 1.0 for r in config.pyramid_layer_ratios):
            raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
