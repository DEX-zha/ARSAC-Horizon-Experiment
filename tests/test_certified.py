"""Tests for the certified (Lipschitz-based) horizon bound."""

import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.append(os.getcwd())

from src.horizon_certified import (
    certified_horizon,
    empirical_delta_sup,
    lipschitz_l2,
    lipschitz_linf,
)
from src.horizon_metrics import window_horizons
from src.horizon_models import (
    LinearAR,
    LSTMPredictor,
    MLPPredictor,
    TorchSeqWrapper,
    TorchWrapper,
)
from src.horizon_utils import set_seed


def _manual_linear(weights_with_bias):
    model = LinearAR()
    model.weights = np.asarray(weights_with_bias, dtype=np.float64)
    return model


def _driven_series(n=400, seed=0):
    """Stable AR(2) driven by a bounded term a linear model cannot capture."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float64)
    x[0], x[1] = rng.standard_normal(2)
    for t in range(1, n - 1):
        x[t + 1] = 0.9 * x[t] - 0.2 * x[t - 1] + 0.3 * math.cos(0.7 * t)
    return x


def test_lipschitz_linear_exact():
    model = _manual_linear([0.5, -0.25, 1.0])  # bias = 1.0 is excluded
    assert lipschitz_linf(model) == pytest.approx(0.75)
    assert lipschitz_l2(model) == pytest.approx(math.sqrt(0.25**2 + 0.5**2))
    # input_dim validation
    assert lipschitz_linf(model, input_dim=2) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        lipschitz_linf(model, input_dim=3)


def test_lipschitz_unfitted_linear_raises():
    with pytest.raises(ValueError):
        lipschitz_linf(LinearAR())


def test_lipschitz_mlp_upper_bounds_gradients():
    set_seed(123)
    dim = 3
    mlp = MLPPredictor(input_dim=dim, hidden_dim=8)
    wrapper = TorchWrapper(mlp, "cpu")
    l_inf = lipschitz_linf(wrapper, input_dim=dim)
    l_2 = lipschitz_l2(wrapper, input_dim=dim)
    assert l_inf > 0.0 and l_2 > 0.0

    grad_l1_max = 0.0
    grad_l2_max = 0.0
    for _ in range(50):
        x = torch.randn(1, dim, requires_grad=True)
        out = mlp(x).sum()
        grad = torch.autograd.grad(out, x)[0].reshape(-1)
        grad_l1_max = max(grad_l1_max, float(grad.abs().sum().item()))
        grad_l2_max = max(grad_l2_max, float(grad.norm().item()))

    # Product bounds must dominate any sampled gradient norm.
    assert l_inf >= grad_l1_max - 1e-6
    assert l_2 >= grad_l2_max - 1e-6


def test_lipschitz_lstm_unsupported():
    lstm = TorchSeqWrapper(LSTMPredictor(hidden_dim=4), "cpu")
    with pytest.raises(ValueError):
        lipschitz_linf(lstm)
    with pytest.raises(ValueError):
        lipschitz_l2(lstm)


def test_empirical_delta_sup_exact_and_segments():
    # Model predicts x_{t+1} = x_t + 1; series is exactly that plus one spike.
    model = _manual_linear([1.0, 1.0])  # w=1, b=1
    series = np.arange(10, dtype=np.float64)
    series[7] += 0.5  # residual 0.5 at the pair predicting index 7 (and 8)
    delta = empirical_delta_sup(model, series, dim=1, lag=1)
    assert delta == pytest.approx(0.5)

    # Segments are processed independently and the max is taken.
    seg_a = np.arange(6, dtype=np.float64)  # exact: residual 0
    delta_seg = empirical_delta_sup(model, [seg_a, series], dim=1, lag=1)
    assert delta_seg == pytest.approx(0.5)

    # Too-short segment contributes nothing.
    assert empirical_delta_sup(model, np.array([1.0]), dim=1, lag=1) == 0.0


def test_certified_horizon_perfect_model_is_infinite():
    # Series generated exactly by the model's own recursion => delta = 0.
    # Embedding convention: column 0 is the OLDEST value, so the weight on
    # the newest value x_t is the LAST feature weight.
    model = _manual_linear([0.0, 0.5, 0.0])
    series = np.empty(50, dtype=np.float64)
    series[0], series[1] = 1.0, 0.5
    for t in range(1, 49):
        series[t + 1] = 0.5 * series[t]  # matches w = (0.0, 0.5)
    h_cert, growth, delta = certified_horizon(model, series, 2, 1, tolerance=0.4)
    assert delta == 0.0
    assert math.isinf(h_cert)
    assert growth == 1.0  # max(1, ||w||_1) with ||w||_1 = 0.5


def test_certified_horizon_monotonic_in_tolerance():
    series = _driven_series()
    dim, lag = 2, 1
    from src.horizon_utils import build_supervised

    x, y = build_supervised(series, dim, lag, horizon=1)
    model = LinearAR(reg=1e-6).fit(x, y)
    h_small, _, _ = certified_horizon(model, series, dim, lag, tolerance=0.2)
    h_large, _, _ = certified_horizon(model, series, dim, lag, tolerance=0.8)
    assert h_small <= h_large


def test_certified_horizon_sound_on_same_segment():
    """On the segment delta was measured on, H_w >= h_cert is a theorem."""
    series = _driven_series(n=400)
    dim, lag = 2, 1
    from src.horizon_utils import build_supervised

    x, y = build_supervised(series, dim, lag, horizon=1)
    model = LinearAR(reg=1e-6).fit(x, y)

    delta = empirical_delta_sup(model, series, dim, lag)
    assert delta > 0.0
    tolerance = 6.0 * delta  # keeps h_cert >= 2 so the check is not vacuous
    h_cert, growth, delta_out = certified_horizon(model, series, dim, lag, tolerance)
    assert delta_out == pytest.approx(delta)
    assert growth >= 1.0
    assert 2.0 <= h_cert < 100.0

    horizons, _ = window_horizons(
        model, series, dim, lag, horizon_max=100, tolerance=tolerance, consecutive=1
    )
    assert horizons.size > 0
    violations = int(np.sum(horizons < h_cert))
    assert violations == 0
