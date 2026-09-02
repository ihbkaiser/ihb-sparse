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
    limit = n if max_support is None else int(max_support)
    if limit < initial or limit > n:
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
