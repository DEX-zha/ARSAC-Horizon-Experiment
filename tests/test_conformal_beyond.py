"""Tests for src/horizon_conformal_beyond.py (weighted conformal, thinning)."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_conformal import conformal_quantile
from src.horizon_conformal_beyond import (
    coverage_gap_bound,
    decay_weights,
    disjoint_indices,
    weighted_conformal_quantile,
)


# ---------------------------------------------------------------------------
# weighted_conformal_quantile
# ---------------------------------------------------------------------------


def test_uniform_weights_match_classical_conformal_quantile():
    rng = np.random.default_rng(0)
    for n in (20, 50, 101, 500):
        scores = rng.normal(size=n)
        for alpha in (0.05, 0.1, 0.2, 0.5):
            expected = conformal_quantile(scores, alpha)
            got = weighted_conformal_quantile(scores, alpha)
            assert got == pytest.approx(expected), (n, alpha)


def test_exact_hand_computed_uniform_case():
    # n=4, alpha=0.5 -> rank ceil(5*0.5)=3 -> third smallest score.
    scores = np.array([4.0, 1.0, 3.0, 2.0])
    assert weighted_conformal_quantile(scores, 0.5) == 3.0


def test_small_n_returns_inf_instead_of_clamping():
    # n=5, alpha=0.1: cumulative calibration mass 5/6 < 0.9 -> +inf atom.
    scores = np.arange(5, dtype=float)
    assert np.isinf(weighted_conformal_quantile(scores, 0.1))
    # The classical helper clamps to the max instead (documented difference).
    assert conformal_quantile(scores, 0.1) == 4.0


def test_boundary_rank_no_floating_point_miss():
    # n=9, alpha=0.1: (n+1)(1-alpha) = 9 exactly -> largest score, not inf.
    scores = np.arange(1.0, 10.0)
    assert weighted_conformal_quantile(scores, 0.1) == 9.0


def test_weighted_hand_computed_case():
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.0, 0.0, 1.0])
    # total = 2; cumulative normalized mass on sorted scores: 0, 0, 0.5.
    assert weighted_conformal_quantile(scores, 0.5, weights) == 3.0
    assert np.isinf(weighted_conformal_quantile(scores, 0.1, weights))


def test_zero_weight_scores_are_ignored():
    scores = np.array([100.0, 1.0, 2.0])
    weights = np.array([0.0, 10.0, 10.0])
    # Mass: 10/21, 20/21 on scores 1, 2 -> target 0.9 needs the +inf atom?
    # cumulative: 1 -> 0.476, 2 -> 0.952 >= 0.9 -> c = 2 (100 never selected).
    assert weighted_conformal_quantile(scores, 0.1, weights) == 2.0


def test_nonfinite_scores_dropped_with_their_weights():
    scores = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
    weights = np.array([1.0, 100.0, 1.0, 100.0, 1.0])
    got = weighted_conformal_quantile(scores, 0.5, weights)
    expected = weighted_conformal_quantile(
        np.array([1.0, 2.0, 3.0]), 0.5, np.ones(3)
    )
    assert got == expected


def test_empty_scores_return_inf():
    assert np.isinf(weighted_conformal_quantile(np.array([]), 0.1))


def test_invalid_weights_raise():
    scores = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, 0.1, np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, 0.1, np.array([1.0, np.nan, 1.0]))
    with pytest.raises(ValueError):
        weighted_conformal_quantile(scores, 0.1, np.array([1.0, 1.0]))


def test_marginal_coverage_iid_at_least_nominal():
    # Finite-sample guarantee under exchangeability: mean coverage >= 1-alpha.
    rng = np.random.default_rng(7)
    alpha = 0.1
    hits = []
    for _ in range(400):
        calib = rng.normal(size=50)
        test = rng.normal()
        c = weighted_conformal_quantile(calib, alpha)
        hits.append(test <= c)
    assert np.mean(hits) >= 1.0 - alpha - 0.02


# ---------------------------------------------------------------------------
# decay_weights
# ---------------------------------------------------------------------------


def test_decay_weights_shape_and_anchor_values():
    w = decay_weights(11, half_life=5)
    assert w.shape == (11,)
    assert w[-1] == 1.0
    assert w[-6] == pytest.approx(0.5)  # exactly one half-life older
    assert np.all(np.diff(w) > 0)  # most recent last, strictly increasing


def test_decay_weights_infinite_half_life_is_uniform():
    np.testing.assert_allclose(decay_weights(8, np.inf), np.ones(8))


def test_decay_weights_edge_cases():
    assert decay_weights(0, 10.0).shape == (0,)
    with pytest.raises(ValueError):
        decay_weights(5, 0.0)
    with pytest.raises(ValueError):
        decay_weights(-1, 1.0)


# ---------------------------------------------------------------------------
# disjoint_indices
# ---------------------------------------------------------------------------


def test_disjoint_indices_values():
    np.testing.assert_array_equal(disjoint_indices(10, 3), [0, 3, 6, 9])
    np.testing.assert_array_equal(disjoint_indices(9, 3), [0, 3, 6])
    np.testing.assert_array_equal(disjoint_indices(5, 1), [0, 1, 2, 3, 4])
    assert disjoint_indices(0, 4).size == 0


def test_disjoint_indices_invalid_gap_raises():
    with pytest.raises(ValueError):
        disjoint_indices(10, 0)


# ---------------------------------------------------------------------------
# coverage_gap_bound
# ---------------------------------------------------------------------------


def test_coverage_gap_bound_uniform_weights():
    n = 9
    assert coverage_gap_bound(np.ones(n)) == pytest.approx(n / (n + 1.0))


def test_coverage_gap_bound_with_dtv():
    w = np.ones(4)
    dtv = np.array([1.0, 1.0, 0.0, 0.0])
    assert coverage_gap_bound(w, dtv) == pytest.approx(2.0 / 5.0)
    assert coverage_gap_bound(w, np.zeros(4)) == 0.0


def test_coverage_gap_bound_decay_smaller_on_old_points():
    # With decaying weights, the loss attributable to the OLD half is smaller
    # than with uniform weights: that is the whole point of the method.
    n = 100
    dtv_old_half = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    uniform_loss = coverage_gap_bound(np.ones(n), dtv_old_half)
    decayed_loss = coverage_gap_bound(decay_weights(n, n / 4), dtv_old_half)
    assert decayed_loss < 0.5 * uniform_loss


def test_coverage_gap_bound_validation():
    assert coverage_gap_bound(np.array([])) == 0.0
    with pytest.raises(ValueError):
        coverage_gap_bound(np.array([1.0, -2.0]))
    with pytest.raises(ValueError):
        coverage_gap_bound(np.ones(3), np.array([0.5, 1.5, 0.0]))
