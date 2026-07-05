"""Physics validation tests for the chaotic generators and estimators.

Audit F3: previous tests only checked shapes/NaNs. These tests validate the
actual dynamics: chaotic Mackey-Glass (audit A1/A2), Lyapunov exponents
against literature values (audit B1), and the direction of the probabilistic
coverage calibration (audit D1). All tests are deterministic and CPU-only.
"""

import os
import sys
import types

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_experiment_probabilistic import _calibration_scale
from src.horizon_utils import (
    estimate_lyapunov,
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
)


def test_mackey_glass_is_chaotic():
    # tau=17 TIME units at dt=1.0 is the canonical chaotic Mackey-Glass regime.
    s = generate_mackey_glass(3000, tau=17.0, dt=1.0, warmup=300)
    assert s.std() > 0.05
    assert (s.max() - s.min()) > 0.3
    lam, _ = estimate_lyapunov(s, dim=4, lag=2)
    # Literature: lambda_1 ~ 0.006 per unit time (= per sample at dt=1).
    assert 0.0 < lam < 0.05


def test_mackey_glass_delay_in_time_units():
    # Regression guard for audit bug A1: the old generator indexed the delay
    # in steps, so the effective delay was tau*dt = 8.5 time units at dt=0.5
    # (non-chaotic, tiny range). tau must be interpreted in time units,
    # independent of the output sampling interval.
    s = generate_mackey_glass(1500, tau=17.0, dt=0.5, warmup=300)
    assert (s.max() - s.min()) > 0.3


def test_logistic_lyapunov():
    s = generate_logistic_map(4000)
    lam, _ = estimate_lyapunov(s, dim=2, lag=1)
    # Theory: lambda = ln 2 = 0.693 per step for r=4.
    assert 0.4 <= lam <= 0.95


def test_lorenz_lyapunov():
    s = generate_lorenz(6000, dt=0.01, warmup=1000)
    lam, _ = estimate_lyapunov(s, dim=3, lag=3)
    # estimate_lyapunov returns the slope per SAMPLE STEP; divide by dt for
    # the exponent per unit time. Literature: lambda_1 = 0.906.
    assert 0.3 < lam / 0.01 < 1.8


def test_rossler_lyapunov():
    s = generate_rossler(8000, dt=0.05, warmup=1000)
    lam, _ = estimate_lyapunov(s, dim=3, lag=6)
    # Literature: lambda_1 = 0.071 per unit time.
    assert 0.005 < lam / 0.05 < 0.3


def test_coverage_direction():
    # Audit D1: the calibrated bound L = h_model * scale must be a LOWER
    # bound on h_real covering ~ (1 - alpha) of windows. The old code took
    # the (1-alpha)-quantile of h_real/h_model with floor 1.0, producing a
    # bound ABOVE h_real ~90% of the time (true lower-bound coverage ~0.10
    # on this synthetic setup).
    rng = np.random.default_rng(0)
    h_real = rng.integers(5, 31, size=400).astype(np.float64)
    h_model = 0.8 * h_real + rng.normal(0.0, 1.0, size=400)
    assert np.all(h_model > 0)

    ratios = (h_real / h_model).tolist()
    args = types.SimpleNamespace(
        calibrate_coverage=True,
        calibration_alpha=0.1,
        calibration_floor=0.0,
    )
    scale = _calibration_scale(args, ratios)
    # Same hit rule as _coverage_from_ratios: hit when h_real >= h_model*scale.
    coverage = float(np.mean(h_real >= h_model * scale))

    assert coverage >= 0.85
    assert scale <= 1.5


if __name__ == "__main__":
    test_mackey_glass_is_chaotic()
    test_mackey_glass_delay_in_time_units()
    test_logistic_lyapunov()
    test_lorenz_lyapunov()
    test_rossler_lyapunov()
    test_coverage_direction()
    print("all physics tests passed")
