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


def test_bring_your_own_forecaster_with_r_diagnostic():
    # Persistence "user model" via the BYO path: the pipeline must calibrate
    # bounds FOR it and the R diagnostic must flag it as model-limited.
    est = HorizonEstimator(
        model=lambda x: x[-1],
        dim=3,
        lag=1,
        alpha=0.1,
        tolerance=0.4,
        horizon_max=20,
        quantile_ensemble=1,
        mlp_epochs=5,
        output_dir="outputs_estimator_test",
    )
    est.fit(_user_series())
    rep = est.report()
    assert est.lower_bounds_.size > 0
    assert rep["coverage_test"] >= 0.75
    assert rep["R_distance_to_chaos_floor"] is None or (
        rep["R_distance_to_chaos_floor"] > 2.0 and rep["chaos_limited"] is False
    )


def test_byo_requires_dim():
    est = HorizonEstimator(model=lambda x: x[-1])
    with pytest.raises(TypeError):
        est.fit(_user_series())


def test_quasiperiodic_signal_disables_R_diagnostic():
    # Regime guard (added after the BIDMC biosignal test): on a strongly
    # recurrent signal lambda_1 -> 0 and R would blow up mechanically, so it
    # must be suppressed and flagged, while L(x) stays valid.
    t = np.linspace(0, 200 * np.pi, 4000)
    series = np.sin(t) + 0.3 * np.sin(2.3 * t) + 0.02 * np.random.default_rng(0).normal(size=t.size)
    est = HorizonEstimator(
        model="linear", alpha=0.1, tolerance=0.4, horizon_max=20,
        quantile_ensemble=1, mlp_epochs=5, output_dir="outputs_estimator_test",
    )
    est.fit(series)
    rep = est.report()
    assert rep["regime"] in ("quasi-periodic", "non-chaotic")
    assert rep["R_distance_to_chaos_floor"] is None  # suppressed, not a huge artifact
    assert rep["margin_real"] is None
    assert est.lower_bounds_.size > 0  # the calibrated bound is still produced
    assert "does NOT" in rep["R_reading"] or "do not apply" in rep["R_reading"]
    assert "L(x)" in rep["R_reading"]  # the bound stays offered in every regime


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
