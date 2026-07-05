"""Tests for the HorizonEstimator user-facing API (Plan V2 Phase 5 MVP)."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_estimator import HorizonEstimator
from src.horizon_utils import generate_lorenz


def _user_series():
    # A "user" series: raw Lorenz x-component handed over as a plain array.
    return generate_lorenz(3000, dt=0.01, warmup=500)


def test_fit_on_custom_series_produces_calibrated_bounds():
    est = HorizonEstimator(
        model="linear",
        alpha=0.1,
        tolerance=0.4,
        horizon_max=15,
        quantile_ensemble=1,
        mlp_epochs=5,
        output_dir="outputs_estimator_test",
    )
    est.fit(_user_series())

    assert est.lower_bounds_.size > 0
    assert est.test_horizons_.size == est.lower_bounds_.size
    assert np.all(est.lower_bounds_ >= 1.0)
    assert np.all(est.lower_bounds_ <= 15.0)
    # Empirical coverage on the held-out test windows near the 1-alpha target.
    covered = float(np.mean(est.test_horizons_ >= est.lower_bounds_))
    assert covered == pytest.approx(est.coverage_, abs=1e-9)
    assert est.coverage_ >= 0.75

    report = est.report()
    for key in (
        "coverage_test",
        "lower_bound_median",
        "label_identified",
        "horizon_certified",
        "guarantee_level",
    ):
        assert key in report
    assert report["horizon_max"] == 15


def test_report_before_fit_raises():
    with pytest.raises(RuntimeError):
        HorizonEstimator().report()


def test_unknown_option_rejected():
    est = HorizonEstimator(model="linear", not_a_real_option=1)
    with pytest.raises(TypeError):
        est.fit(_user_series())


def test_custom_series_too_short_rejected():
    est = HorizonEstimator(model="linear")
    with pytest.raises(ValueError):
        est.fit(np.arange(50, dtype=float))
