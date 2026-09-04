import pytest
import torch

from sparse_frontier.modelling.attention.query_robust_solver import (
    empirical_cvar,
    fit_p,
    batched_independent_active_fit,
    primal_gap,
    project_capped_simplex,
)


def _problem(seed=7, n=24, tokens=6):
    generator = torch.Generator().manual_seed(seed)
    S = torch.randn(n, tokens, generator=generator, dtype=torch.float64)
    f = torch.logsumexp(S, dim=-1)
    return S, f


def test_minimax_active_scan_finds_inactive_violator_and_certificate():
    S, f = _problem()
    result = fit_p(
        S,
        f,
        objective="minimax",
        initial_support=2,
        max_support=S.shape[0],
        tolerance=1e-5,
        max_iterations=1000,
    )
    assert result.objective == "minimax"
    assert result.active_size > 2
    assert result.optimization_gap is not None
    assert result.optimization_gap <= 1e-5
    assert result.lower_bound <= result.upper_bound + 1e-10
    assert torch.all(primal_gap(S, f, result.p) <= result.upper_bound + 1e-8)


def test_cvar_preserves_original_caps_and_tail_upper_bound():
    S, f = _problem(n=16)
    weights = torch.arange(1, 17, dtype=torch.float64)
    alpha = 0.75
    result = fit_p(
        S,
        f,
        weights,
        objective="cvar",
        alpha=alpha,
        initial_support=2,
        max_support=16,
        tolerance=1e-5,
        max_iterations=1000,
    )
    normalized = weights / weights.sum()
    caps = normalized / (1 - alpha)
    assert result.active_size >= 1
    assert result.dual_weights is not None
    assert result.active_indices is not None
    assert torch.all(result.dual_weights <= caps[result.active_indices] + 1e-9)
    assert torch.isclose(result.dual_weights.sum(), torch.ones((), dtype=torch.float64), atol=1e-9)
    assert result.optimization_gap is not None
    assert result.optimization_gap <= 1e-5
    assert result.upper_bound == pytest.approx(
        empirical_cvar(primal_gap(S, f, result.p), normalized, alpha).item(), abs=1e-10
    )


def test_capped_projection_is_feasible_for_unequal_caps():
    y = torch.tensor([2.0, -1.0, 0.3, 4.0], dtype=torch.float64)
    caps = torch.tensor([0.10, 0.55, 0.25, 0.40], dtype=torch.float64)
    result = project_capped_simplex(y, caps)
    assert torch.all(result >= 0)
    assert torch.all(result <= caps + 1e-10)
    assert torch.isclose(result.sum(), torch.ones((), dtype=torch.float64), atol=1e-9)


def test_only_robust_objectives_are_accepted():
    S, f = _problem(n=8)
    with pytest.raises(ValueError, match="minimax.*cvar"):
        fit_p(S, f, objective="mean")


def test_batched_cvar_grows_support_until_unequal_caps_are_feasible():
    S, f = _problem(n=12, tokens=4)
    weights = torch.tensor(
        [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.989],
        dtype=torch.float32,
    )
    result = batched_independent_active_fit(
        S.unsqueeze(0),
        f.unsqueeze(0),
        objective="cvar",
        weights=weights,
        alpha=0.5,
        initial_support=2,
        max_support=12,
        tolerance=1.0,
        max_iterations=4,
    )
    # The large-weight query is not in the first two indices; the seed must be
    # augmented before projecting onto the capped simplex.
    assert int(result[4].item()) >= 3
    assert torch.isfinite(result[0]).all()


def test_batched_solver_clamps_max_support_to_finite_pool():
    S, f = _problem(n=8, tokens=4)
    result = batched_independent_active_fit(
        S.float().unsqueeze(0),
        f.float().unsqueeze(0),
        objective="minimax",
        initial_support=4,
        max_support=1024,
        tolerance=1.0,
        max_iterations=2,
    )
    assert result[0].shape == (1, 4)
    assert torch.isfinite(result[0]).all()


def test_batched_independent_can_admit_several_full_pool_violators_per_round():
    S, f = _problem(n=24, tokens=6)
    batched_S = torch.stack([S.float(), (S + 0.15).float()])
    batched_f = torch.logsumexp(batched_S, dim=-1)
    result = batched_independent_active_fit(
        batched_S,
        batched_f,
        objective="minimax",
        initial_support=2,
        max_support=24,
        tolerance=1e-3,
        max_iterations=200,
        active_additions_per_round=8,
    )
    p, lower, upper, gap, active_size, converged, _ = result
    assert bool(converged.all())
    assert bool((gap <= 1e-3).all())
    assert bool((lower <= upper + 1e-5).all())
    assert bool((active_size >= 2).all())
    assert torch.allclose(p.sum(-1), torch.ones(2), atol=1e-5)


def test_certificate_free_fixed_support_preserves_the_fitted_tangent():
    S, f = _problem(n=12, tokens=5)
    kwargs = dict(
        objective="minimax",
        initial_support=12,
        max_support=12,
        max_iterations=1,
        tolerance=1e-3,
    )
    certified = batched_independent_active_fit(S.float().unsqueeze(0), f.float().unsqueeze(0), **kwargs)
    serving = batched_independent_active_fit(
        S.float().unsqueeze(0),
        f.float().unsqueeze(0),
        compute_certificate=False,
        **kwargs,
    )

    assert torch.allclose(serving[0], certified[0])
    assert torch.isnan(serving[2]).all()
    assert torch.isnan(serving[3]).all()
    assert not bool(serving[5].any())


def test_certificate_free_fixed_support_accepts_an_explicit_unguarded_step():
    """The serving-only schedule remains a finite simplex tangent."""

    S, f = _problem(n=12, tokens=5)
    result = batched_independent_active_fit(
        S.float().unsqueeze(0),
        f.float().unsqueeze(0),
        objective="minimax",
        initial_support=12,
        max_support=12,
        max_iterations=1,
        tolerance=1e-3,
        compute_certificate=False,
        armijo=False,
    )

    assert torch.isfinite(result[0]).all()
    assert torch.allclose(result[0].sum(-1), torch.ones(1), atol=1e-5)
    assert torch.isnan(result[2]).all()
