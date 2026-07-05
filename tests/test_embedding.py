"""Tests for theory-grounded embedding selection (MI lag + FNN dim).

Ground truths used here are analytic: a sine of period T has its first
mutual-information minimum near T/4 and embeds in dimension 2 (a circle);
a strongly autocorrelated AR(1) has no MI local minimum at short lags,
exercising the autocorrelation fallback. All tests are seeded, CPU-only.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_embedding import (
    false_nearest_neighbors,
    mutual_information_lag,
    select_embedding,
)


def _sine(n=2000, period=40.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    return np.sin(2.0 * np.pi * t / period) + noise * rng.standard_normal(n)


def test_mi_lag_sine_quarter_period():
    s = _sine()
    lag, mi = mutual_information_lag(s, max_lag=60)
    # First MI minimum of a sine is near a quarter period (= 10 samples).
    assert 6 <= lag <= 16
    assert len(mi) == 61
    # Lag 0 carries the self-information: the maximum of the curve.
    assert mi[0] >= mi[1:].max()
    assert np.all(np.isfinite(mi))


def test_mi_lag_fallback_acf():
    # AR(1) with rho=0.98: MI decays monotonically (no local minimum
    # within max_lag) and the ACF stays above 1/e (0.98^8 = 0.85), so the
    # fallback must return max_lag itself.
    rng = np.random.default_rng(3)
    n = 3000
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.standard_normal(n)
    for i in range(1, n):
        x[i] = 0.98 * x[i - 1] + 0.2 * eps[i]
    lag, _ = mutual_information_lag(x, max_lag=8)
    assert lag == 8


def test_fnn_henon_selects_dim_two():
    # Henon map, the benchmark of Kennel et al. (1992): the x-series folds
    # in d=1 (many false neighbors) and unfolds completely in d=2.
    n = 3000
    x = np.empty(n)
    y = np.empty(n)
    x[0], y[0] = 0.1, 0.1
    for i in range(n - 1):
        x[i + 1] = 1.0 - 1.4 * x[i] ** 2 + y[i]
        y[i + 1] = 0.3 * x[i]
    dim, fnn = false_nearest_neighbors(x[500:], lag=1, max_dim=5, theiler=10)
    assert dim == 2
    assert fnn[0] > 0.3
    assert fnn[1] < 0.01


def test_fnn_logistic_selects_dim_one():
    from src.horizon_utils import generate_logistic_map

    # A smooth 1-D map has no folds under the FNN criterion: neighbors in
    # x stay neighbors after one application of the map.
    s = generate_logistic_map(3000)
    dim, fnn = false_nearest_neighbors(s, lag=1, max_dim=4, theiler=10)
    assert dim == 1
    assert fnn[0] < 0.01


def test_select_embedding_output_and_determinism():
    s = _sine(seed=1)
    out1 = select_embedding(s, max_dim=5, max_lag=60)
    out2 = select_embedding(s, max_dim=5, max_lag=60)
    assert set(out1) == {"dim", "lag", "mi_curve", "fnn_fractions"}
    assert isinstance(out1["dim"], int) and 1 <= out1["dim"] <= 5
    assert isinstance(out1["lag"], int) and 1 <= out1["lag"] <= 60
    assert out1["dim"] == out2["dim"] and out1["lag"] == out2["lag"]
    assert np.array_equal(out1["mi_curve"], out2["mi_curve"])
    assert np.array_equal(out1["fnn_fractions"], out2["fnn_fractions"])


def test_degenerate_series_do_not_crash():
    const = np.ones(500)
    lag, mi = mutual_information_lag(const, max_lag=20)
    assert lag >= 1 and np.all(mi == 0.0)
    dim, fnn = false_nearest_neighbors(const, lag=1, max_dim=3)
    assert dim >= 1
    short = np.arange(6, dtype=np.float64)
    lag_s, _ = mutual_information_lag(short, max_lag=20)
    assert lag_s >= 1


def test_select_embedding_map_guard_falls_back_to_lag_one():
    from src.horizon_utils import generate_logistic_map

    # Strongly mixing map: the MI curve decays to the noise floor, its
    # spurious "first minimum" gives a noise-like embedding where FNN
    # never reaches 1%. The guard must retry lag=1 (map prescription).
    s = generate_logistic_map(3000)
    out = select_embedding(s, max_dim=6, max_lag=60)
    assert out["lag"] == 1
    assert out["dim"] <= 2
    assert out["fnn_fractions"][out["dim"] - 1] < 0.01


def test_lorenz_smoke():
    from src.horizon_utils import generate_lorenz

    s = generate_lorenz(2500, dt=0.01, warmup=1000)
    out = select_embedding(s, max_dim=6, max_lag=40)
    # Lorenz at dt=0.01: MI minimum around 0.1-0.2 t.u. (10-20 steps),
    # attractor unfolds in a low dimension.
    assert 5 <= out["lag"] <= 40
    assert 2 <= out["dim"] <= 6
    assert out["fnn_fractions"][out["dim"] - 1] < 0.01


if __name__ == "__main__":
    test_mi_lag_sine_quarter_period()
    test_mi_lag_fallback_acf()
    test_fnn_henon_selects_dim_two()
    test_fnn_logistic_selects_dim_one()
    test_select_embedding_output_and_determinism()
    test_degenerate_series_do_not_crash()
    test_select_embedding_map_guard_falls_back_to_lag_one()
    test_lorenz_smoke()
    print("all embedding tests passed")
