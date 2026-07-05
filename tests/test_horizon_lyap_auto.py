"""Tests for Lyapunov-time-based horizon_max auto-resolution (audit A3)."""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import DEFAULT_LAMBDA, resolve_horizon_max
from src.horizon_metrics import rolling_rmse


def test_explicit_value_respected():
    resolved, target = resolve_horizon_max("lorenz", 0.01, 60)
    assert resolved == 60
    assert target is None


def test_lorenz_auto_is_three_lyapunov_times():
    resolved, target = resolve_horizon_max("lorenz", 0.01, None, lyap_factor=3.0)
    expected = int(np.ceil(3.0 / (DEFAULT_LAMBDA["lorenz"] * 0.01)))
    assert target == expected  # 331
    assert resolved == expected  # inside [30, 400]


def test_rossler_auto_hits_the_cap():
    resolved, target = resolve_horizon_max("rossler", 0.05, None, lyap_factor=3.0)
    assert target == int(np.ceil(3.0 / (DEFAULT_LAMBDA["rossler"] * 0.05)))  # ~846
    assert resolved == 400  # capped, caller logs the censoring warning


def test_logistic_auto_hits_the_floor():
    resolved, target = resolve_horizon_max("logistic", 1.0, None, lyap_factor=3.0)
    assert target == 5
    assert resolved == 30  # floor: below 30 the labels are too quantized


def test_unknown_dataset_falls_back():
    resolved, target = resolve_horizon_max("unknown", 0.01, None)
    assert resolved == 50
    assert target is None


class _LastValueModel:
    def predict(self, x):
        return float(np.asarray(x, dtype=np.float64)[-1])


def test_rolling_rmse_subsample_matches_full_curve_shape():
    rng = np.random.default_rng(0)
    series = np.sin(np.linspace(0, 40, 1200)) + 0.01 * rng.normal(size=1200)
    model = _LastValueModel()
    full = rolling_rmse(model, series, dim=3, lag=1, horizon_max=15)
    sub = rolling_rmse(model, series, dim=3, lag=1, horizon_max=15, max_windows=300, seed=0)
    assert full.shape == sub.shape
    assert np.all(np.isfinite(sub))
    # Monte-Carlo estimate of the same curve: close in relative terms.
    mask = full > 1e-8
    assert np.max(np.abs(sub[mask] - full[mask]) / full[mask]) < 0.25


def test_rolling_rmse_subsample_deterministic():
    series = np.sin(np.linspace(0, 30, 900))
    model = _LastValueModel()
    a = rolling_rmse(model, series, dim=2, lag=1, horizon_max=10, max_windows=100, seed=7)
    b = rolling_rmse(model, series, dim=2, lag=1, horizon_max=10, max_windows=100, seed=7)
    np.testing.assert_array_equal(a, b)
