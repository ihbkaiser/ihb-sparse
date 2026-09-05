import torch

from sparse_frontier.modelling.attention.query_pool import (
    QueryPool,
    QueryPoolLayer,
    QueryPoolManifest,
    REPRESENTATION,
)
from sparse_frontier.modelling.attention.query_robust import (
    QueryRobustAttention,
    _batched_query_chunk_logits,
    calibrate_robust_bias,
    select_balanced_pile_queries,
)


def test_query_robust_access_flushes_trailing_periodic_counters(monkeypatch):
    from sparse_frontier.modelling.attention.handler import AttentionHandler

    handler = AttentionHandler(
        tp_size=1,
        model_q_heads=2,
        model_kv_heads=1,
        model_layers=2,
        max_input_tokens=8,
        max_output_tokens=4,
        block_size=2,
    )
    handler._decode_access_sum_pending = 12.0
    handler._decode_dense_sum_pending = 96.0
    reported = []
    monkeypatch.setattr(AttentionHandler, "_should_report", staticmethod(lambda: True))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.handler.get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.handler.add_decode_access",
        lambda accessed_sum, dense_sum: reported.append((accessed_sum, dense_sum)),
    )

    handler.flush_decode_access(torch.device("cpu"))

    assert reported == [(12.0, 96.0)]
    assert handler._decode_access_sum_pending == 0.0
    assert handler._decode_dense_sum_pending == 0.0


def _pool(num_layers=2, q_heads=2, kv_heads=1, head_dim=2, horizons=4):
    manifest = QueryPoolManifest(
        schema_version=1,
        model_id="test/model",
        model_revision="a" * 40,
        num_layers=num_layers,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        tp_size=1,
        rope_type="llama3",
        rope_parameters={
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_theta": 500000.0,
        },
        representation=REPRESENTATION,
        attention_scale=head_dim**-0.5,
        max_decode_offset=horizons - 1,
        pool_size_per_kv_group=1,
        coreset_sizes=[],
        capture_split="calibration",
        source_manifest_sha256="b" * 64,
        created_with_git_commit="test",
    )
    layers = tuple(
        QueryPoolLayer(
            centroid_by_horizon=torch.zeros(kv_heads, horizons, head_dim),
            coreset_queries={},
            coreset_weights={},
        )
        for _ in range(num_layers)
    )
    return QueryPool(manifest=manifest, layers=layers, tp_rank=0)


def _attention(**overrides):
    kwargs = dict(
        token_budget=6,
        chunk_size=2,
        generation_horizon=4,
        sink_chunks=1,
        recent_chunks=1,
        num_layers=2,
        num_q_heads=2,
        num_kv_heads=1,
        tp_size=1,
        block_size=2,
        pool=_pool(),
    )
    kwargs.update(overrides)
    return QueryRobustAttention(**kwargs)


def test_query_robust_validates_exact_budget_and_geometry():
    import pytest

    with pytest.raises(ValueError, match="divisible"):
        _attention(token_budget=5)
    with pytest.raises(ValueError, match="mandatory"):
        _attention(token_budget=4, sink_chunks=1, recent_chunks=1)
    with pytest.raises(ValueError, match="block"):
        _attention(block_size=3)
    with pytest.raises(ValueError, match="shared_chunk_aggregation"):
        _attention(shared_chunk_aggregation="median")
    assert _attention(shared_chunk_aggregation="raw_score_sum").shared_chunk_aggregation == "raw_score_sum"


def test_batched_query_chunk_logits_matches_einsum_for_strided_key_view():
    generator = torch.Generator().manual_seed(17)
    # This is the noncontiguous head-major view used after a vLLM prompt's
    # token-major keys are converted to FP32.
    token_major = torch.randn(5, 3, 2, 4, generator=generator)
    keys = token_major.permute(2, 0, 1, 3)
    queries = torch.randn(2, 7, 4, generator=generator)
    scale = 0.5

    actual = _batched_query_chunk_logits(queries, keys, scale)
    expected = torch.einsum("hmd,hcnd->hcmn", queries, keys) * scale

    assert torch.allclose(actual, expected)


def test_balanced_pile_subpool_preserves_each_q_head_and_normalizes_weights():
    # Two KV groups, two Q heads per group, five captured rows per Q head.
    query_ids = torch.tensor([[0] * 5 + [1] * 5, [2] * 5 + [3] * 5], dtype=torch.int16)
    queries = torch.arange(2 * 10 * 2, dtype=torch.float32).view(2, 10, 2)
    positions = torch.arange(10, dtype=torch.int32).repeat(2, 1)
    weights = torch.full((2, 10), 0.1, dtype=torch.float32)
    chosen_q, chosen_pos, chosen_w = select_balanced_pile_queries(
        queries, positions, query_ids, weights, budget=4
    )
    assert chosen_q.shape == (2, 4, 2)
    assert chosen_pos.tolist() == [[0, 5, 1, 6], [0, 5, 1, 6]]
    torch.testing.assert_close(chosen_w.sum(-1), torch.ones(2))
    # Two rows are retained from each mapped Q head, not four from the first.
    assert chosen_q[0, :, 0].tolist() == [0.0, 10.0, 2.0, 12.0]


def test_robust_scalar_calibration_has_expected_empirical_objectives():
    residual = torch.tensor([[1.0, 3.0, 9.0], [2.0, 4.0, 8.0]])
    entropy = torch.tensor([0.25, 0.75])
    weights = torch.tensor([0.5, 0.25, 0.25])
    torch.testing.assert_close(
        calibrate_robust_bias(residual, entropy, weights, "raw_entropy"), entropy
    )
    torch.testing.assert_close(
        calibrate_robust_bias(residual, entropy, weights, "mean_residual"),
        torch.tensor([3.5, 4.0]),
    )
    torch.testing.assert_close(
        calibrate_robust_bias(residual, entropy, weights, "minimax_midpoint"),
        torch.tensor([5.0, 5.0]),
    )


def test_calibrated_bias_requires_explicit_exploratory_routing():
    import pytest

    with pytest.raises(ValueError, match="require solver_fail_closed=False"):
        _attention(bias_mode="minimax_midpoint")
    assert _attention(
        bias_mode="minimax_midpoint", solver_fail_closed=False
    ).bias_mode == "minimax_midpoint"


def test_unguarded_solver_step_requires_fixed_support_exploratory_routing():
    import pytest

    with pytest.raises(ValueError, match="unguarded"):
        _attention(solver_armijo=False)
    assert not _attention(
        solver_armijo=False,
        solver_fail_closed=False,
        initial_support=64,
        max_support=64,
    ).solver_armijo


def test_triton_fast_retry_rejects_a_disabled_armijo_guard():
    import pytest

    with pytest.raises(ValueError, match="requires solver_armijo=True"):
        _attention(
            solver_backend="triton_fast_retry",
            solver_fail_closed=False,
            solver_armijo=False,
            initial_support=64,
            max_support=64,
        )


def test_triton_global_armijo_backend_is_explicitly_supported():
    attention = _attention(
        solver_backend="triton_global_armijo",
        solver_fail_closed=False,
    )
    assert attention.solver_backend == "triton_global_armijo"


def test_batched_newton_backend_is_explicitly_supported():
    attention = _attention(
        solver_backend="batched_newton",
        solver_fail_closed=False,
    )
    assert attention.solver_backend == "batched_newton"


def test_batched_newton_serving_path_uses_newton_solver(monkeypatch):
    attention = _attention(
        solver_backend="batched_newton",
        solver_fail_closed=False,
    )
    attention.robust_pool_available = True
    attention.request_queries[0] = torch.tensor(
        [[[0.5, -0.25], [0.25, 0.5], [-0.5, 0.25], [0.0, 0.5]]]
    )
    attention.request_query_weights[0] = torch.full((1, 4), 0.25)
    attention._common_query_weights[0] = True
    calls = []

    def fake_newton(scores, partitions, **kwargs):
        calls.append((scores.shape, partitions.shape, kwargs))
        chunks = scores.shape[0]
        p = torch.full((chunks, scores.shape[-1]), 1.0 / scores.shape[-1])
        zeros = torch.zeros(chunks)
        return p, zeros, zeros, zeros, torch.ones(chunks, dtype=torch.bool), torch.ones(chunks, dtype=torch.int32)

    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.batched_full_support_newton_fit",
        fake_newton,
    )
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.batched_independent_active_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("active-set used")),
    )

    summary, bias = attention._summarize(
        torch.tensor([[[[1.0, 3.0]], [[5.0, 7.0]]]]), layer_idx=0
    )

    assert calls == [
        ((1, 4, 2), (1, 4), {
            "tolerance": 1e-3,
            "max_iterations": 8,
            "armijo_backtracks": 10,
        })
    ]
    torch.testing.assert_close(summary.float(), torch.tensor([[[3.0, 5.0]]]))
    assert torch.isfinite(bias).all()


def test_triton_fast_retry_serving_path_never_uses_active_set_or_dense_fallback(
    monkeypatch,
):
    """The main-run backend consumes only the fast+retry solver result."""

    attention = _attention(
        solver_backend="triton_fast_retry",
        solver_fail_closed=False,
    )
    attention.robust_pool_available = True
    attention.request_queries[0] = torch.tensor(
        [[[0.5, -0.25], [0.25, 0.5], [-0.5, 0.25], [0.0, 0.5]]]
    )
    attention.request_query_weights[0] = torch.full((1, 4), 0.25)
    attention._common_query_weights[0] = True
    calls = []

    def fake_fast_retry(scores, partitions, **kwargs):
        calls.append((scores.shape, partitions.shape, kwargs))
        chunks, _, tokens = scores.shape
        p = torch.full((chunks, tokens), 1.0 / tokens)
        zeros = torch.zeros(chunks)
        return p, zeros, zeros, zeros, torch.ones(chunks, dtype=torch.bool), 3

    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.triton_full_support_minimax_retry_fit",
        fake_fast_retry,
    )
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.batched_independent_active_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("active-set used")),
    )

    summary, bias = attention._summarize(
        torch.tensor([[[[1.0, 3.0]], [[5.0, 7.0]]]]), layer_idx=0
    )

    assert calls == [
        ((1, 4, 2), (1, 4), {
            "tolerance": 1e-3,
            "fast_iterations": 1024,
            "retry_iterations": 8192,
            "armijo_backtracks": 12,
        })
    ]
    torch.testing.assert_close(summary.float(), torch.tensor([[[3.0, 5.0]]]))
    assert torch.isfinite(bias).all()
    assert attention.solver_retry_count == 3


def test_triton_global_armijo_serving_path_uses_global_solver(monkeypatch):
    attention = _attention(
        solver_backend="triton_global_armijo",
        solver_fail_closed=False,
    )
    attention.robust_pool_available = True
    attention.request_queries[0] = torch.tensor(
        [[[0.5, -0.25], [0.25, 0.5], [-0.5, 0.25], [0.0, 0.5]]]
    )
    attention.request_query_weights[0] = torch.full((1, 4), 0.25)
    attention._common_query_weights[0] = True
    calls = []

    def fake_global(scores, partitions, **kwargs):
        calls.append((scores.shape, partitions.shape, kwargs))
        chunks = scores.shape[0]
        p = torch.full((chunks, scores.shape[-1]), 1.0 / scores.shape[-1])
        zeros = torch.zeros(chunks)
        return p, zeros, zeros, zeros, torch.ones(chunks, dtype=torch.bool)

    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.triton_global_armijo_full_support_fit",
        fake_global,
    )
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.batched_independent_active_fit",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("active-set used")),
    )

    summary, bias = attention._summarize(
        torch.tensor([[[[1.0, 3.0]], [[5.0, 7.0]]]]), layer_idx=0
    )

    assert calls == [
        ((1, 4, 2), (1, 4), {
            "tolerance": 1e-3,
            "max_iterations": 8192,
            "armijo_backtracks": 12,
        })
    ]
    torch.testing.assert_close(summary.float(), torch.tensor([[[3.0, 5.0]]]))
    assert torch.isfinite(bias).all()
    assert attention.fallback_count == 0


def test_prefill_builds_only_full_chunks_and_reset_hides_old_state(monkeypatch):
    attention = _attention()
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    query = torch.zeros(1, 2, 5, 2, dtype=torch.bfloat16)
    keys = torch.arange(10, dtype=torch.bfloat16).view(1, 1, 5, 2)
    values = torch.zeros_like(keys)
    output = attention(query, keys, values, layer_idx=0)

    assert output.shape == query.shape
    assert attention.summary_ready[0, :3, 0].tolist() == [True, True, False]
    assert attention.summary_valid_tokens[0, :3].tolist() == [2, 2, 1]
    attention.reset()
    assert not bool(attention.summary_ready.any())


def test_async_prefill_option_falls_back_to_identical_cpu_state():
    """Scheduling must not change a summary when CUDA streams are unavailable."""

    keys = torch.arange(18, dtype=torch.bfloat16).view(1, 1, 9, 2)
    synchronous = _attention(async_prefill_build=False)
    asynchronous = _attention(async_prefill_build=True)
    synchronous.prefill_state(keys, layer_idx=0)
    asynchronous.prefill_state(keys, layer_idx=0)

    torch.testing.assert_close(
        asynchronous.summary_key[0, :4], synchronous.summary_key[0, :4]
    )
    torch.testing.assert_close(
        asynchronous.summary_bias[0, :4], synchronous.summary_bias[0, :4]
    )
    assert asynchronous._summary_ready_host[0] == synchronous._summary_ready_host[0]


def test_preallocation_caches_prompt_origin_centroids_on_the_runtime_device(monkeypatch):
    attention = _attention()
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    # The artifact remains a CPU/offline object.  Mutating it after allocation
    # must not introduce a request-time host-to-device dependency.
    attention.pool.layers[0].centroid_by_horizon.fill_(7)
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16),
        layer_idx=0,
    )
    torch.testing.assert_close(
        attention.request_centroid[0], torch.zeros_like(attention.request_centroid[0])
    )


def test_decode_uses_shared_deterministic_chunks_and_exact_token_count(monkeypatch):
    attention = _attention()
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        layer_idx=0,
    )

    captured = {}

    def fake_flash(**kwargs):
        captured.update(kwargs)
        kwargs["out"].zero_()

    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        fake_flash,
    )
    k_cache = torch.zeros(1, 5, 2, 2, dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    output = torch.empty(1, 2, 2, dtype=torch.bfloat16)
    attention.decode(
        query=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        keys=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        values=torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        k_cache=k_cache,
        v_cache=v_cache,
        tokens_per_head=torch.tensor([10, 10], dtype=torch.int32),
        output=output,
        layer_idx=0,
    )

    assert captured["cache_seqlens"].tolist() == [6, 6]
    assert captured["block_table"].tolist() == [[0, 1, 4], [0, 1, 4]]
    assert attention.last_accessed_tokens == 6
    assert bool(attention.summary_ready[0, 4, 0])


def test_short_decode_takes_dense_cache_path(monkeypatch):
    attention = _attention(token_budget=10)
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    captured = {}

    def fake_flash(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        fake_flash,
    )
    cache = torch.zeros(1, 5, 2, 2, dtype=torch.bfloat16)
    attention.decode(
        torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        cache,
        cache,
        torch.tensor([5, 5], dtype=torch.int32),
        torch.empty(1, 2, 2, dtype=torch.bfloat16),
        0,
    )
    assert captured["cache_seqlens"].tolist() == [5, 5]
    assert captured["block_table"].shape == (2, 3)
    assert attention.last_accessed_tokens == 5


def test_query_robust_is_registered_with_logical_chunk_size():
    from sparse_frontier.modelling.attention.registry import ATTENTION_REGISTRY

    assert ATTENTION_REGISTRY["query_robust"] is QueryRobustAttention


def test_cache_reshape_includes_partial_last_physical_block():
    from sparse_frontier.modelling.attention.abstract_attention import AttentionUtils

    physical = torch.arange(2 * 8, dtype=torch.float32).view(2, 8, 1, 1)
    cache = torch.stack([physical, physical + 100])
    key, value = AttentionUtils.reshape_kv_cache(
        cache, target_block_size=4, max_blocks=3
    )
    assert key.shape == (1, 3, 4, 1)
    torch.testing.assert_close(key.flatten(), torch.arange(12, dtype=torch.float32))
    torch.testing.assert_close(value.flatten(), torch.arange(100, 112, dtype=torch.float32))


def test_partial_current_chunk_is_mandatory_last_and_counted_exactly(monkeypatch):
    attention = _attention()
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 8, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 8, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 8, 2, dtype=torch.bfloat16),
        0,
    )
    captured = {}
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        lambda **kwargs: captured.update(kwargs),
    )
    cache = torch.zeros(1, 5, 2, 2, dtype=torch.bfloat16)
    attention.decode(
        torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        cache,
        cache,
        torch.tensor([9, 9], dtype=torch.int32),
        torch.empty(1, 2, 2, dtype=torch.bfloat16),
        0,
    )
    assert captured["block_table"].tolist() == [[0, 3, 4], [0, 3, 4]]
    assert captured["cache_seqlens"].tolist() == [5, 5]


def test_partial_chunk_uses_exact_current_query_mass_without_rebuilding_summary(monkeypatch):
    attention = _attention()
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        0,
    )
    # A non-boundary decode step has only a mandatory partial chunk.  It must
    # not invoke the empirical builder just to assign that chunk a router score.
    monkeypatch.setattr(
        attention,
        "_summarize",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected rebuild")),
    )
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        lambda **kwargs: None,
    )
    cache = torch.zeros(1, 5, 2, 2, dtype=torch.bfloat16)
    attention.decode(
        torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 2, dtype=torch.bfloat16),
        cache,
        cache,
        torch.tensor([9, 9], dtype=torch.int32),
        torch.empty(1, 2, 2, dtype=torch.bfloat16),
        0,
    )


def test_cache_reshape_ignores_unallocated_negative_block_table_entries():
    from sparse_frontier.modelling.attention.abstract_attention import AttentionUtils

    physical = torch.arange(3 * 8, dtype=torch.float32).view(3, 8, 1, 1)
    cache = torch.stack([physical, physical + 100])
    key, _ = AttentionUtils.reshape_kv_cache(
        cache,
        target_block_size=4,
        max_blocks=6,
        block_table=torch.tensor([[2, 0, -1, -1]], dtype=torch.int32),
    )
    assert key.shape == (1, 4, 4, 1)
    torch.testing.assert_close(
        key.flatten(), torch.cat([torch.arange(16, 24), torch.arange(0, 8)]).float()
    )


def test_paged_decode_gathers_only_selected_logical_chunks(monkeypatch):
    attention = _attention(share_chunks_across_kv_heads=False)
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        0,
    )
    logical_tokens = torch.arange(12, dtype=torch.bfloat16)
    # Logical physical blocks [2, 0, 1] represent token order 8..11,0..3,4..7.
    physical = logical_tokens.view(3, 4, 1, 1).expand(-1, -1, -1, 2).clone()
    canonical = torch.stack([physical, physical + 100])
    captured = {}
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        lambda **kwargs: captured.update(kwargs),
    )
    attention.decode_paged(
        query=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        kv_cache=canonical,
        physical_block_table=torch.tensor([[2, 0, 1]], dtype=torch.int32),
        tokens_per_head=None,
        total_tokens=10,
        output=torch.empty(1, 2, 2, dtype=torch.bfloat16),
        layer_idx=0,
    )
    gathered = captured["k_cache"][captured["block_table"][0]].flatten()[:12:2]
    torch.testing.assert_close(
        gathered, torch.tensor([8, 9, 10, 11, 4, 5], dtype=torch.bfloat16)
    )
    assert captured["cache_seqlens"].tolist() == [6, 6]


def test_shared_kv_selection_uses_canonical_paged_cache_without_scratch(monkeypatch):
    attention = _attention(share_chunks_across_kv_heads=True)
    attention.preallocate_memory(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.AttentionUtils.flash_attention",
        lambda q, k, v: torch.zeros_like(q),
    )
    attention(
        torch.zeros(1, 2, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        torch.zeros(1, 1, 9, 2, dtype=torch.bfloat16),
        0,
    )
    physical = torch.arange(10, dtype=torch.bfloat16).view(5, 2, 1, 1).expand(
        -1, -1, -1, 2
    ).clone()
    canonical = torch.stack([physical, physical + 100])
    captured = {}
    monkeypatch.setattr(
        "sparse_frontier.modelling.attention.query_robust.flash_attn_with_kvcache",
        lambda **kwargs: captured.update(kwargs),
    )
    attention.decode_paged(
        query=torch.zeros(1, 2, 2, dtype=torch.bfloat16),
        kv_cache=canonical,
        physical_block_table=torch.tensor([[4, 2, 3, 0, 1]], dtype=torch.int32),
        tokens_per_head=None,
        total_tokens=10,
        output=torch.empty(1, 2, 2, dtype=torch.bfloat16),
        layer_idx=0,
    )
    assert captured["k_cache"].data_ptr() == canonical[0].data_ptr()
    assert captured["block_table"].tolist() == [[4, 2, 1]]
    assert captured["cache_seqlens"].tolist() == [6]
