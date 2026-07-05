"""Tests for the predictability profiler (product entry point).

Each canonical case pins the regime classification that the BIDMC episode
showed is required before any chaos diagnostic is reported.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_profile import profile_series
from src.horizon_utils import generate_logistic_map, generate_lorenz


def test_white_noise_is_stochastic():
    x = np.random.default_rng(0).normal(size=6000)
    prof = profile_series(x)
    assert prof.regime == "stochastic"
    assert prof.noise_std_units > 0.7
    assert "invest in better data" in prof.reading


def test_sine_plus_noise_is_quasi_periodic():
    t = np.linspace(0, 300 * np.pi, 6000)
    x = np.sin(t) + 0.05 * np.random.default_rng(1).normal(size=t.size)
    prof = profile_series(x)
    assert prof.regime == "quasi-periodic"
    assert prof.periodicity_index > 0.9
    assert prof.period_samples is not None and 30 <= prof.period_samples <= 50
    assert "R does NOT" in prof.reading


def test_logistic_map_is_chaotic():
    x = generate_logistic_map(6000)
    prof = profile_series(x)
    assert prof.regime == "chaotic"
    assert prof.lambda_resolved
    # lambda close to ln 2 = 0.693 per iteration (theory embedding).
    assert 0.3 < prof.lambda_per_step < 1.0


def test_lorenz_is_chaotic_despite_small_per_step_lambda():
    # Lorenz at dt=0.01 has lambda_step ~ 0.009: an absolute per-step
    # threshold would misclassify it. The growth-product criterion must not.
    x = generate_lorenz(6000, dt=0.01, warmup=1000)
    prof = profile_series(x)
    assert prof.regime == "chaotic"
    assert prof.lambda_resolved
    assert 0.003 < prof.lambda_per_step < 0.03


def test_random_walk_is_not_chaotic_nor_stochastic():
    # A random walk fools Rosenstein (diffusive sqrt(t) divergence fits as a
    # positive slope, so lambda may come out 'resolved') but has NO predictive
    # structure beyond persistence: the structure-ratio gate must reject the
    # chaotic label and fall back to 'regular'.
    x = np.cumsum(np.random.default_rng(2).normal(size=6000))
    prof = profile_series(x)
    assert prof.regime == "regular"
    assert prof.structure_ratio >= 0.5  # no skill beyond persistence
