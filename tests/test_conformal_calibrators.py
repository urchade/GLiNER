"""Synthetic, network-free tests for gliner/conformal/calibrators.py.

Mirrors tests/test_decoder.py's pattern: hand-built inputs with analytically
known ground truth, no model download. See docs/research/design.md §4 and
docs/research/eval_plan.md for the theory these tests check against.
"""

import random

import pytest

from gliner.conformal.calibrators import (
    calibration_floor,
    crc_lambda_search,
    mondrian_calibrate,
    split_conformal_quantile,
)


class TestCalibrationFloor:
    def test_matches_eval_plan_table(self):
        # docs/research/eval_plan.md §2.1's worked table.
        assert calibration_floor(0.20) == 4
        assert calibration_floor(0.10) == 9
        assert calibration_floor(0.05) == 19

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError):
            calibration_floor(0.0)
        with pytest.raises(ValueError):
            calibration_floor(1.0)
        with pytest.raises(ValueError):
            calibration_floor(-0.1)


class TestSplitConformalQuantile:
    def test_raises_below_floor(self):
        with pytest.raises(ValueError, match="insufficient"):
            split_conformal_quantile([0.1, 0.2, 0.3, 0.4, 0.5], alpha=0.05)

    def test_at_exact_floor_returns_the_max(self):
        floor = calibration_floor(0.2)
        scores = [i / 10 for i in range(floor)]
        assert split_conformal_quantile(scores, alpha=0.2) == max(scores)

    def test_empirical_coverage_matches_theory(self):
        """20000 seeded trials: split-conformal coverage on Uniform(0,1) scores
        should land within a few standard errors of the 1-alpha target
        (theory.md part i, Eq. 1-2)."""
        rng = random.Random(42)
        n, alpha, trials = 500, 0.1, 20000
        hits = 0
        for _ in range(trials):
            calib = [rng.random() for _ in range(n)]
            test = rng.random()
            q = split_conformal_quantile(calib, alpha)
            hits += test <= q
        coverage = hits / trials
        se = (coverage * (1 - coverage) / trials) ** 0.5
        target = 1 - alpha
        assert target - 4 * se <= coverage <= target + 1 / (n + 1) + 4 * se

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError):
            split_conformal_quantile([0.1, 0.2], alpha=1.5)


class TestMondrianCalibrate:
    def test_skips_sub_floor_types_and_calibrates_the_rest(self):
        rng = random.Random(0)
        alpha = 0.1
        floor = calibration_floor(alpha)
        scores_by_type = {
            "common": [rng.random() for _ in range(200)],
            "rare": [rng.random() for _ in range(floor - 1)],
        }
        thresholds, skipped = mondrian_calibrate(scores_by_type, alpha)
        assert "common" in thresholds
        assert "rare" not in thresholds
        assert skipped == {"rare": floor - 1}

    def test_empty_input(self):
        thresholds, skipped = mondrian_calibrate({}, 0.1)
        assert thresholds == {}
        assert skipped == {}


class TestCrcLambdaSearch:
    def test_raises_below_floor(self):
        with pytest.raises(ValueError, match="insufficient"):
            crc_lambda_search([[0.1], [0.2], [0.3]], alpha=0.05)

    def test_boundary_case_all_scores_zero_gives_lambda_zero(self):
        # Every gold entity perfectly scored (nonconformity 0) -> even the
        # tightest threshold (lambda=0) already achieves zero risk.
        gold = [[0.0, 0.0] for _ in range(50)]
        lam = crc_lambda_search(gold, alpha=0.1)
        assert lam == 0.0

    def test_unrepresentable_entities_do_not_block_convergence_when_rare(self):
        # A structurally-unrepresentable gold entity (float("inf")) can never
        # be covered. If only a small fraction of examples have one (each
        # contributing a fixed loss-1 floor), the target is still reachable
        # as long as that floor alone is below alpha.
        rng = random.Random(1)
        easy = [[rng.random() * 0.05] for _ in range(190)]
        unrepresentable = [[float("inf")] for _ in range(10)]
        gold = easy + unrepresentable
        lam = crc_lambda_search(gold, alpha=0.2)
        assert lam < float("inf")

    def test_unrepresentable_entities_correctly_block_convergence_when_common(self):
        # If unrepresentable entities are common enough that even lambda=inf
        # cannot bring the risk under alpha, returning inf (not a finite but
        # invalid lambda) is the mathematically correct answer, not a bug.
        rng = random.Random(1)
        gold = [[rng.random() * 0.05, float("inf")] for _ in range(200)]
        lam = crc_lambda_search(gold, alpha=0.2)
        assert lam == float("inf")

    def test_empirical_risk_control_matches_theory(self):
        """CRC's proved guarantee: E[miss_rate] <= alpha on fresh test data
        (theory.md part iii-b, Eq. 6)."""
        rng = random.Random(7)
        alpha = 0.1
        gold_calib = [[rng.random()] for _ in range(1000)]
        lam = crc_lambda_search(gold_calib, alpha)

        trials, n_test = 200, 500
        miss_rates = []
        for _ in range(trials):
            test = [[rng.random()] for _ in range(n_test)]
            missed = sum(1 for g in test if g[0] > lam) / n_test
            miss_rates.append(missed)
        mean_miss = sum(miss_rates) / len(miss_rates)
        assert mean_miss <= alpha + 0.02  # small slack for Monte Carlo noise

    def test_monotonicity_precondition_is_checked_by_default(self):
        # Sanity: verify_monotone=True must not raise on a genuinely nested
        # (by construction) family -- this is the runtime check tied to
        # theory.md iii-b Claims 1-2.
        rng = random.Random(3)
        gold = [[rng.random() for _ in range(rng.randint(0, 3))] for _ in range(100)]
        crc_lambda_search(gold, alpha=0.2, verify_monotone=True)  # must not raise
