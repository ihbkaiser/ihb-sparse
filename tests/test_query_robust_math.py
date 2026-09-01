import math

import pytest
import torch

from sparse_frontier.modelling.attention.query_robust import (
    build_mean_summaries,
    tangent_gap,
    weighted_centroid,
)


def test_tangent_gap_equals_reverse_kl_and_is_nonnegative():
    torch.manual_seed(43)
    logits = torch.randn(11, 16, dtype=torch.float64) * 4
    p = torch.softmax(torch.randn(16, dtype=torch.float64), dim=-1)

    actual = tangent_gap(logits, p)
    expected = (
        p
        * (
            p.clamp_min(torch.finfo(p.dtype).tiny).log()
            - torch.log_softmax(logits, dim=-1)
        )
    ).sum(-1)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.all(actual >= -1e-12)


def test_tangent_gap_handles_simplex_boundary():
    logits = torch.tensor([[1000.0, -1000.0, 2.0]], dtype=torch.float64)
    p = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    gap = tangent_gap(logits, p)
    assert torch.isfinite(gap).all()
    assert gap.item() >= 0.0


def test_weighted_centroid_normalizes_weights():
    queries = torch.tensor([[1.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    weights = torch.tensor([1.0, 3.0], dtype=torch.float64)
    expected = torch.tensor([0.25, 2.25], dtype=torch.float64)
    torch.testing.assert_close(weighted_centroid(queries, weights), expected)


def _reference_summary(keys, queries, weights):
    weights = weights / weights.sum()
    mean_query = weights @ queries
    scale = keys.shape[-1] ** -0.5
    mean_logits = torch.einsum("d,cnd->cn", mean_query, keys) * scale
    p = torch.softmax(mean_logits, dim=-1)
    summary = torch.einsum("cn,cnd->cd", p, keys)
    entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
    exact = torch.logsumexp(torch.einsum("bd,cnd->bcn", queries, keys) * scale, -1)
    lower = torch.einsum("bd,cd->bc", queries, summary) * scale
    residual = exact - lower - entropy
    bias = entropy + (weights[:, None] * residual).sum(0)
    return summary, entropy, bias


def test_mean_summary_matches_fp64_reference():
    torch.manual_seed(7)
    keys = torch.randn(3, 5, 8, dtype=torch.float64)
    queries = torch.randn(13, 8, dtype=torch.float64)
    weights = torch.arange(1, 14, dtype=torch.float64)
    expected_key, expected_entropy, expected_bias = _reference_summary(
        keys, queries, weights
    )

    result = build_mean_summaries(
        keys,
        queries,
        weights,
        storage_dtype=torch.float64,
        bias_mode="mean_residual",
        query_batch_size=4,
    )

    torch.testing.assert_close(result.summary_key, expected_key)
    torch.testing.assert_close(result.entropy.double(), expected_entropy)
    torch.testing.assert_close(result.summary_bias.double(), expected_bias, rtol=1e-5, atol=1e-6)
    assert result.summary_bias.dtype == torch.float32
    assert result.quantized_audit.num_samples == 39
    assert math.isfinite(result.quantized_audit.weighted_abs_mean)


def test_raw_entropy_mode_skips_fitted_residual():
    torch.manual_seed(11)
    keys = torch.randn(2, 4, 6, dtype=torch.float64)
    queries = torch.randn(5, 6, dtype=torch.float64)
    weights = torch.ones(5, dtype=torch.float64)
    result = build_mean_summaries(
        keys, queries, weights, torch.float32, "raw_entropy", query_batch_size=2
    )
    torch.testing.assert_close(result.summary_bias, result.entropy.float())


def test_quantized_audit_rescans_the_stored_vector():
    keys = torch.tensor(
        [[[1000.0, -1000.0], [999.1, -999.3], [-700.2, 701.1]]],
        dtype=torch.float32,
    )
    queries = torch.tensor(
        [[0.03, -0.02], [-0.01, 0.04], [0.02, 0.01]], dtype=torch.float32
    )
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32)
    result = build_mean_summaries(
        keys, queries, weights, torch.bfloat16, "mean_residual", query_batch_size=1
    )

    scale = keys.shape[-1] ** -0.5
    exact = torch.logsumexp(torch.einsum("bd,cnd->bcn", queries, keys) * scale, -1)
    predicted = (
        torch.einsum("bd,cd->bc", queries, result.summary_key.float()) * scale
        + result.summary_bias
    )
    error = exact - predicted
    norm_weights = weights / weights.sum()
    expected_signed = (norm_weights[:, None] * error).sum(0).mean().item()
    expected_abs = (norm_weights[:, None] * error.abs()).sum(0).mean().item()

    assert result.summary_key.dtype == torch.bfloat16
    assert result.quantized_audit.weighted_signed_mean == pytest.approx(
        expected_signed, abs=1e-6
    )
    assert result.quantized_audit.weighted_abs_mean == pytest.approx(
        expected_abs, abs=1e-6
    )
    assert result.quantized_audit.max_abs == pytest.approx(
        error.abs().max().item(), abs=1e-6
    )


@pytest.mark.parametrize(
    ("queries", "weights", "match"),
    [
        (torch.empty(0, 4), torch.empty(0), "empty"),
        (torch.ones(2, 4), torch.tensor([0.0, 0.0]), "positive"),
        (torch.ones(2, 4), torch.tensor([1.0, -1.0]), "nonnegative"),
        (torch.tensor([[float("nan"), 0.0]]), torch.ones(1), "finite"),
    ],
)
def test_weighted_centroid_rejects_invalid_measure(queries, weights, match):
    with pytest.raises(ValueError, match=match):
        weighted_centroid(queries, weights)


def test_summary_builder_rejects_shape_and_mode_errors():
    keys = torch.randn(2, 4, 8)
    queries = torch.randn(3, 8)
    weights = torch.ones(3)
    with pytest.raises(ValueError, match="bias_mode"):
        build_mean_summaries(keys, queries, weights, torch.float32, "unknown")
    with pytest.raises(ValueError, match="dimension"):
        build_mean_summaries(keys, torch.randn(3, 7), weights, torch.float32, "raw_entropy")
    with pytest.raises(ValueError, match="batch"):
        build_mean_summaries(keys, queries, weights, torch.float32, "raw_entropy", 0)
