"""Advanced coverage tests for robustness and reliability."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_calibration import validate_coverage
from src.horizon_experiment_conformal_stats import _coverage_stats


def _make_interval_case(n, include_count, seed, nan_every=0, invert=False):
    rng = np.random.default_rng(seed)
    y_true = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    if include_count > 0:
        idx = rng.choice(n, size=include_count, replace=False)
        mask[idx] = True

    y_lower = y_true - 0.1
    y_upper = y_true + 0.1

    if invert:
        y_lower[~mask] = y_true[~mask] + 0.5
        y_upper[~mask] = y_true[~mask] - 0.5
    else:
        y_lower[~mask] = y_true[~mask] + 0.5
        y_upper[~mask] = y_true[~mask] + 1.0

    if nan_every and nan_every > 0:
        y_true[::nan_every] = np.nan

    expected = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    return y_true, y_lower, y_upper, expected


VALIDATE_CASES = []
case_id = 0
for n in (5, 8, 10, 12, 15, 20, 25, 30, 35, 40):
    for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        include_count = int(round(frac * n))
        nan_every = 0
        if case_id % 5 == 0:
            nan_every = 7 if n >= 7 else 0
        invert = case_id % 4 == 0
        VALIDATE_CASES.append((n, include_count, case_id, nan_every, invert))
        case_id += 1


@pytest.mark.parametrize("n, include_count, seed, nan_every, invert", VALIDATE_CASES)
def test_validate_coverage_cases(n, include_count, seed, nan_every, invert):
    y_true, y_lower, y_upper, expected = _make_interval_case(
        n, include_count, seed, nan_every=nan_every, invert=invert
    )
    cov = validate_coverage(y_true, y_lower, y_upper, 0.9)
    assert cov == pytest.approx(expected)


def _make_lower_bound_case(n, include_count, seed):
    rng = np.random.default_rng(seed)
    y_test = np.arange(1, n + 1, dtype=np.float64)
    mask = np.zeros(n, dtype=bool)
    if include_count > 0:
        idx = rng.choice(n, size=include_count, replace=False)
        mask[idx] = True
    pred = y_test.copy()
    pred[mask] = y_test[mask] - 0.25
    pred[~mask] = y_test[~mask] + 0.25
    expected_cov = float(np.mean(y_test >= pred))
    return y_test, pred, expected_cov


COVERAGE_STATS_CASES = []
case_id = 0
for n in (6, 9, 12, 15, 18, 24):
    for frac in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        include_count = int(round(frac * n))
        COVERAGE_STATS_CASES.append((n, include_count, case_id))
        case_id += 1


@pytest.mark.parametrize("n, include_count, seed", COVERAGE_STATS_CASES)
def test_coverage_stats_lower_bound(n, include_count, seed):
    y_test, pred_test_cal, expected_cov = _make_lower_bound_case(
        n, include_count, seed
    )
    constants = {"slack_quantile": 0.9}
    cov, tightness, slack_med, slack_p90 = _coverage_stats(
        y_test, pred_test_cal, constants
    )

    assert cov == pytest.approx(expected_cov)

    slack = y_test - pred_test_cal
    assert slack_med == pytest.approx(float(np.median(slack)))
    assert slack_p90 == pytest.approx(float(np.quantile(slack, 0.9)))

    median_y = float(np.median(y_test))
    if median_y > 0.0:
        expected_tightness = float(np.median(pred_test_cal) / median_y)
        assert tightness == pytest.approx(expected_tightness)
    else:
        assert tightness is None


def test_coverage_monotonicity_lower_bound():
    rng = np.random.default_rng(123)
    y = rng.normal(size=200)
    pred_base = y - rng.uniform(0.0, 1.0, size=200)
    pred_tighter = pred_base + 0.2
    pred_looser = pred_base - 0.2

    cov_base = float(np.mean(y >= pred_base))
    cov_tighter = float(np.mean(y >= pred_tighter))
    cov_looser = float(np.mean(y >= pred_looser))

    assert cov_looser >= cov_base
    assert cov_base >= cov_tighter
