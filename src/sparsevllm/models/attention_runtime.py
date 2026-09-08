from __future__ import annotations

import torch
from torch import nn

import sparsevllm.platforms as platforms
from sparsevllm.configs.sparse import resolve_shadowkv_outlier_chunks
from sparsevllm.method_registry import (
    normalize_sparse_method,
    resolve_prefill_sparse_method,
    resolve_sparse_prefill_score_mode,
    sparse_decode_attention_requires_scores,
    sparse_prefill_attention_contract,
)
from sparsevllm.operators.decode_attention import (
    DecodeAttentionOpSpec,
    PreparedDecodeAttentionOp,
    prepare_decode_attention_op,
)
from sparsevllm.operators.full_attention import (
    FullAttentionOpSpec,
    FullAttentionProvider,
    prepare_full_attention_provider,
)
from sparsevllm.operators.moe import model_activation_dtype
from sparsevllm.operators.prefill_attention import (
    FlashPrefillV2Semantics,
    PreparedPrefillAttentionOp,
    PrefillAttentionOpSpec,
    prepare_prefill_attention_op,
)


def resolve_mha_head_dim(config) -> int:
    explicit = getattr(config, "head_dim", None)
    if explicit is not None:
        head_dim = int(explicit)
        if head_dim <= 0:
            raise ValueError(f"MHA head_dim must be positive, got {head_dim}.")
        return head_dim

    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError(
            "Cannot infer MHA head_dim from hidden_size / num_attention_heads: "
            f"hidden_size={hidden_size} num_attention_heads={num_attention_heads}."
        )
    return hidden_size // num_attention_heads


def _resolve_mha_local_shape(
    config,
    *,
    attention_tp_size: int,
) -> tuple[int, int, int, torch.dtype]:
    tp_size = int(attention_tp_size)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if query_heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            "MHA query and KV heads must be divisible by attention TP size: "
            f"query_heads={query_heads} kv_heads={kv_heads} tp_size={tp_size}."
        )
    return (
        query_heads // tp_size,
        kv_heads // tp_size,
        resolve_mha_head_dim(config),
        model_activation_dtype(config),
    )


def _resolve_shadowkv_prefill_page_size(
    normalized_method: str,
    *,
    device_index: int | None,
) -> int:
    """Select the physical prefill page layout for the selected device.

    Blackwell's FlashInfer CuTe-DSL paged prefill requires 16-token pages.
    Other devices keep the token-page layout so the portable Triton provider
    remains a valid fallback.  A missing device index is used by CPU/spec
    tests and deliberately retains the legacy page-1 contract.
    """
    if normalized_method != "shadowkv" or device_index is None:
        return 1
    caps = platforms.current_platform.get_device_caps(int(device_index))
    return 16 if caps.compute_capability == (10, 0) else 1


def _resolve_dense_prefill_page_size(
    normalized_method: str,
    *,
    device_index: int | None,
) -> int:
    """Return the page contract used by the SM100 dense transient view."""

    if normalized_method not in {"", "vanilla"} or device_index is None:
        return 1
    caps = platforms.current_platform.get_device_caps(int(device_index))
    return 16 if caps.compute_capability == (10, 0) else 1


def build_mha_prefill_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    runtime_config=None,
    device_index: int | None = None,
) -> PrefillAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    score_config = config if runtime_config is None else runtime_config
    score_mode = resolve_sparse_prefill_score_mode(
        normalized_method,
        getattr(score_config, "sparse_prefill_score_mode", None),
    )
    prefill_sparse_method = resolve_prefill_sparse_method(
        getattr(score_config, "prefill_sparse_method", ""),
        sparse_method=normalized_method,
    )
    contract = sparse_prefill_attention_contract(
        normalized_method,
        prefill_sparse_method=prefill_sparse_method,
        sparse_prefill_score_mode=score_mode,
        h2o_prefill_score_window=getattr(
            score_config, "h2o_prefill_score_window", 0
        ),
    )
    flashprefill_v2 = None
    if prefill_sparse_method == "flashprefill_v2":
        flashprefill_v2 = FlashPrefillV2Semantics(
            k_block_m=int(score_config.flashprefill_v2_k_block_m),
            k_block_n=int(score_config.flashprefill_v2_k_block_n),
            abs_threshold=float(score_config.flashprefill_v2_abs_threshold),
            attention_sink_blocks=int(
                score_config.flashprefill_v2_attention_sink_blocks
            ),
            window_blocks=int(score_config.flashprefill_v2_window_blocks),
            last_query_blocks=int(
                score_config.flashprefill_v2_last_query_blocks
            ),
            min_sparse_q_len=int(
                score_config.flashprefill_v2_min_sparse_q_len
            ),
            use_mean_correction=bool(
                score_config.flashprefill_v2_use_mean_correction
            ),
        )
    quest_prefill_page_size = getattr(score_config, "quest_chunk_size", None)
    shadowkv_prefill_page_size = _resolve_shadowkv_prefill_page_size(
        normalized_method,
        device_index=device_index,
    )
    dense_prefill_page_size = _resolve_dense_prefill_page_size(
        normalized_method,
        device_index=device_index,
    )
    prefill_page_size = 1
    if normalized_method == "quest" and quest_prefill_page_size is not None:
        prefill_page_size = int(quest_prefill_page_size)
    elif normalized_method == "shadowkv":
        prefill_page_size = shadowkv_prefill_page_size
    elif normalized_method in {"", "vanilla"}:
        prefill_page_size = dense_prefill_page_size
    return PrefillAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        causal=True,
        # QuEST and Blackwell ShadowKV store token slots in contiguous native
        # pages.  Other devices retain token-page semantics for the portable
        # providers.
        page_size=prefill_page_size,
        score_output=contract.main_score_kind,
        layer_varying_page_table=contract.layer_varying_page_table,
        return_softmax_lse=(
            prefill_sparse_method == "h2o_prefill"
            and score_mode == "probability"
        ),
        allow_softmax_lse_fallback=(
            prefill_sparse_method == "h2o_prefill"
            and score_mode == "probability"
        ),
        prefill_sparse_method=prefill_sparse_method,
        flashprefill_v2=flashprefill_v2,
    )


def build_mha_prefill_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    runtime_config=None,
) -> PreparedPrefillAttentionOp:
    return prepare_prefill_attention_op(
        build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
            device_index=int(device.index or 0),
        ),
        device_index=int(device.index or 0),
    )


def build_mha_decode_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> DecodeAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    shadowkv_compact_width = None
    if normalized_method == "shadowkv":
        chunk_size = int(getattr(runtime_config, "shadowkv_chunk_size", 8) or 8)
        sparse_budget = int(
            getattr(runtime_config, "shadowkv_sparse_budget", 2048) or 2048
        )
        outlier_chunks_per_head = resolve_shadowkv_outlier_chunks(
            sparse_budget,
            getattr(runtime_config, "shadowkv_outlier_chunks", None),
        )
        local_chunks = int(
            getattr(runtime_config, "shadowkv_local_chunks", 4) or 4
        )
        recent_tokens = int(
            getattr(runtime_config, "shadowkv_recent_tokens", 512) or 512
        )
        shadowkv_page_size = int(
            getattr(runtime_config, "shadowkv_decode_page_size", 16) or 16
        )
        context_capacity = int(getattr(runtime_config, "max_model_len", 0) or 0)
        # The explicit decode payload is flattened per (request, KV-head), so
        # every head owns a fixed-size outlier region.  Do not multiply this
        # width by ``kv_heads``: that would reserve space for a cross-head
        # union even though the paged payload already separates heads.
        outlier_chunks = outlier_chunks_per_head
        raw_compact_width = (
            outlier_chunks * chunk_size
            + sparse_budget
            + min(context_capacity, local_chunks * chunk_size + chunk_size - 1)
            + min(context_capacity, recent_tokens)
        )
        shadowkv_compact_width = (
            (raw_compact_width + shadowkv_page_size - 1) // shadowkv_page_size
        ) * shadowkv_page_size
    requires_decode_scores = sparse_decode_attention_requires_scores(
        normalized_method
    )
    return DecodeAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        max_batch_size=int(max_batch_size),
        sparse_method=normalized_method,
        causal=True,
        page_size=(
            int(getattr(runtime_config, "quest_chunk_size", 16))
            if normalized_method == "quest"
            else 1
        ),
        # Score demand can change between decode steps for sparse methods.
        # Bind the score-capable implementation up front instead of
        # switching providers in the runtime path.
        may_require_attention_scores=requires_decode_scores,
        layer_varying_page_table=bool(normalized_method),
        cuda_graph=bool(cuda_graph),
        h2o_layerwise_probability_scores=(
            normalized_method == "h2o" and requires_decode_scores
        ),
        context_capacity=int(getattr(runtime_config, "max_model_len", 0) or 0)
        or None,
        sparse_context_budget=(
            int(getattr(runtime_config, "quest_token_budget", 2080))
            if normalized_method == "quest"
            else None
        ),
        may_use_full_layer_kivi_int4=(
            normalized_method == "deltakv"
            and int(
                getattr(runtime_config, "full_layer_kv_quant_bits", 0) or 0
            )
            == 4
            and bool(
                getattr(runtime_config, "enable_full_layer_kivi_quant", True)
            )
        ),
        full_layer_kivi_decode_block_seq=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_seq", 256)
            or 256
        ),
        full_layer_kivi_decode_block_n=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_n", 16) or 16
        ),
        full_layer_kivi_decode_num_warps=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_warps", 2) or 2
        ),
        full_layer_kivi_decode_num_stages=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_stages", 3) or 3
        ),
        shadowkv_compact_width=shadowkv_compact_width,
        shadowkv_page_size=(
            int(getattr(runtime_config, "shadowkv_decode_page_size", 16) or 16)
            if normalized_method == "shadowkv"
            else 16
        ),
        shadowkv_decode_backend=(
            str(getattr(runtime_config, "shadowkv_decode_backend", "auto"))
            if normalized_method == "shadowkv"
            else "auto"
        ),
        shadowkv_flashinfer_backend=(
            str(getattr(runtime_config, "shadowkv_flashinfer_backend", "auto"))
            if normalized_method == "shadowkv"
            else "auto"
        ),
    )


def build_mha_decode_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
) -> PreparedDecodeAttentionOp:
    return prepare_decode_attention_op(
        build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
        ),
        device_index=int(device.index or 0),
    )


def build_mha_full_attention_provider(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> FullAttentionProvider:
    spec = FullAttentionOpSpec(
        prefill=build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
            device_index=int(device.index or 0),
        ),
        decode=build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
            runtime_config=runtime_config,
        ),
    )
    return prepare_full_attention_provider(
        spec,
        device_index=int(device.index or 0),
    )


def bind_mha_full_attention_provider(
    model: nn.Module,
    provider: FullAttentionProvider,
) -> int:
    return provider.bind(model)


__all__ = [
    "bind_mha_full_attention_provider",
    "build_mha_decode_attention_op",
    "build_mha_decode_attention_spec",
    "build_mha_full_attention_provider",
    "build_mha_prefill_attention_op",
    "build_mha_prefill_attention_spec",
    "resolve_mha_head_dim",
]
