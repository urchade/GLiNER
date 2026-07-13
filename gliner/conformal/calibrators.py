"""Model-agnostic conformal calibration math.

Pure NumPy/Python, no GLiNER or PyTorch dependency beyond optional tensor inputs
(anything sequence-like works) -- independently testable against synthetic scores
with analytically known coverage, per docs/research/design.md §4. Implements the
three guarantee modes from design.md §1:

- ``split_conformal_quantile``: the ``⌈(n+1)(1-α)⌉``-th order statistic
  (docs/research/theory.md part (i)) underlying "span_filter" mode.
- ``crc_lambda_search``: Conformal Risk Control's finite-sample-conservative
  λ search (theory.md part (iii-b), Eq. 5) underlying "risk_control" mode.
- ``mondrian_calibrate``: per-type application of ``split_conformal_quantile``
  with an explicit floor (theory.md part (v)) underlying "mondrian" mode.

All three raise (never silently degrade) when the finite-sample correction has
no solution -- design.md §6.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple, Mapping, Sequence


def calibration_floor(alpha: float) -> int:
    """Minimum calibration-set size for which the ``⌈(n+1)(1-α)⌉ ≤ n`` correction is solvable.

    Derivation (theory.md part (i)): the correction is solvable iff
    ``n ≥ (1-α)/α``. Returns the smallest integer n satisfying that.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return math.ceil((1 - alpha) / alpha)


def split_conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The ``⌈(n+1)(1-α)⌉``-th smallest of ``scores`` (theory.md part (i), Eq. in §0).

    Args:
        scores: Calibration nonconformity scores (larger = worse agreement).
        alpha: Miscoverage level in (0, 1).

    Returns:
        The conformal quantile ``q̂``; a prediction set ``{y : s(x,y) ≤ q̂}`` then
        satisfies ``P(Y ∈ C(X)) ≥ 1-α`` under exchangeability.

    Raises:
        ValueError: if ``len(scores) < calibration_floor(alpha)`` -- the quantile
            would require a rank beyond the available calibration points
            (undefined, not merely wide; theory.md part (i)).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n = len(scores)
    floor = calibration_floor(alpha)
    if n < floor:
        raise ValueError(
            f"n={n} calibration scores insufficient for alpha={alpha}: need n >= {floor} "
            f"for the ceil((n+1)(1-alpha))/n correction to be defined (docs/research/theory.md part i). "
            "Collect more calibration data or use a larger alpha."
        )
    rank = math.ceil((n + 1) * (1 - alpha))
    return sorted(scores)[rank - 1]


def mondrian_calibrate(
    scores_by_type: Mapping[str, Sequence[float]], alpha: float
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Per-type conformal quantiles, skipping types below the calibration floor.

    Args:
        scores_by_type: gold nonconformity scores, grouped by entity type.
        alpha: Miscoverage level, shared across all types (theory.md part v, Eq. 7).

    Returns:
        ``(thresholds, skipped)``: ``thresholds`` maps qualifying types to their
        per-type quantile; ``skipped`` maps sub-floor types to their observed
        calibration count (design.md §1.3: these fall back to "span_filter"'s
        pooled threshold at predict time, not an error here).
    """
    thresholds: Dict[str, float] = {}
    skipped: Dict[str, int] = {}
    for etype, scores in scores_by_type.items():
        try:
            thresholds[etype] = split_conformal_quantile(scores, alpha)
        except ValueError:
            skipped[etype] = len(scores)
    return thresholds, skipped


def _miss_rate(gold_nc_scores: Sequence[Sequence[float]], lam: float) -> float:
    """Mean per-example miss rate ℓ(Cλ,y) at threshold λ (theory.md Eq. 4)."""
    losses = []
    for example_scores in gold_nc_scores:
        if len(example_scores) == 0:
            losses.append(0.0)
        else:
            covered = sum(1 for s in example_scores if s <= lam)
            losses.append(1.0 - covered / len(example_scores))
    return sum(losses) / len(losses) if losses else 0.0


def crc_lambda_search(
    gold_nc_scores: Sequence[Sequence[float]],
    alpha: float,
    verify_monotone: bool = True,
) -> float:
    """Conformal Risk Control's λ̂ for the missed-entity-rate loss (theory.md Eq. 5, B=1).

    ``λ̂ = inf{λ : R̂ₙ(λ) + (1-α)/n ≤ α}``. Candidate λ breakpoints are exactly the
    observed nonconformity scores (the loss is a finite step function that only
    changes value there -- theory.md part iii-b, right-continuity argument), so a
    grid search over them is exact, not an approximation.

    Args:
        gold_nc_scores: one sublist per calibration example, containing
            ``1 - p_θ(span,t|x)`` for each of that example's gold entities
            (empty sublist for entity-free examples). Use ``float("inf")`` for
            gold entities that are structurally unrepresentable (e.g. wider than
            ``max_width``) -- they can never be covered, which the loss already
            handles correctly without special-casing.
        alpha: target expected-miss-rate bound.
        verify_monotone: if True, assert the empirical risk is non-increasing
            across the candidate grid -- a direct runtime check of the CRC
            precondition proved in theory.md iii-b Claims 1-2. Costs one extra
            pass over the grid; disable only for large-scale/perf-critical calls
            after the property has been established once.

    Returns:
        λ̂ ∈ [0, ∞]. ``float("inf")`` means even admitting every candidate
        (Cλ = full candidate universe) cannot bring the miss rate to target --
        only possible if some gold entities are structurally unrepresentable in
        every example (see the ``float("inf")`` note above).

    Raises:
        ValueError: if ``n`` is too small for any λ (including λ=∞) to satisfy
            the finite-sample correction: solvable iff ``n ≥ (1-α)/α``, exactly
            :func:`calibration_floor` -- the same floor as split conformal,
            re-derived independently here from CRC's own formula as a
            consistency check (theory.md part v).
    """
    n = len(gold_nc_scores)
    floor = calibration_floor(alpha)
    if n < floor:
        raise ValueError(
            f"n={n} calibration examples insufficient for alpha={alpha}: need n >= {floor} "
            "for CRC's finite-sample correction (B-alpha)/n term to be satisfiable even at "
            "lambda=infinity (docs/research/theory.md part v). Collect more calibration data "
            "or use a larger alpha."
        )

    finite_scores = sorted({s for ex in gold_nc_scores for s in ex if math.isfinite(s)})
    candidates = [0.0, *finite_scores, math.inf]

    rhs = alpha - (1 - alpha) / n

    if verify_monotone:
        risks = [_miss_rate(gold_nc_scores, lam) for lam in candidates]
        for a, b in zip(risks, risks[1:]):
            assert a >= b - 1e-12, (
                "CRC monotonicity precondition violated: empirical risk increased as λ grew. "
                "This should be structurally impossible for GLiNER's nested-threshold decode "
                "rule (theory.md iii-b Claims 1-2) -- if this fires, gold_nc_scores was not "
                "built from a genuinely nested family of sets."
            )
    else:
        risks = None

    for i, lam in enumerate(candidates):
        risk = risks[i] if risks is not None else _miss_rate(gold_nc_scores, lam)
        if risk <= rhs:
            return lam

    return math.inf
