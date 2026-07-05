"""Core unit tests for horizon_metrics utilities."""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_metrics import estimate_jacobian_growth, gated_rollout, horizon_from_window_rmse
from src.horizon_models import TorchWrapper


def test_estimate_jacobian_growth_torchwrapper():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight[:] = torch.tensor([[3.0, 4.0]])
    wrapper = TorchWrapper(model, device="cpu")

    rng = np.random.default_rng(0)
    x_samples = rng.normal(size=(5, 2))

    norm_q, norm_mean, norms = estimate_jacobian_growth(
        wrapper, x_samples, quantile=0.5, max_samples=None
    )

    assert np.isfinite(norm_q)
    assert np.isfinite(norm_mean)
    assert norms.size == 5
    assert np.allclose(norms, 5.0)
    assert np.isclose(norm_q, 5.0)
    assert np.isclose(norm_mean, 5.0)


def test_horizon_from_window_rmse_consecutive():
    rmse = np.array([0.1, 0.3, 0.4, 0.2], dtype=np.float64)
    h_one = horizon_from_window_rmse(rmse, tolerance=0.25, consecutive=1)
    h_two = horizon_from_window_rmse(rmse, tolerance=0.25, consecutive=2)

    assert h_one == 2
    assert h_two == 2


def test_gated_rollout_respects_limits():
    class ConstantModel:
        def __init__(self, value):
            self.value = float(value)

        def predict(self, x):
            return self.value

    series = np.arange(10, dtype=np.float64)
    l_values = np.array([0.9, 2.1, 5.0], dtype=np.float64)

    paths, horizons = gated_rollout(
        ConstantModel(1.0),
        series,
        dim=2,
        lag=1,
        l_values=l_values,
        horizon_max=3,
    )

    assert horizons.tolist() == [0, 2, 3]
    assert len(paths) == 3
    assert paths[0].size == 0
    assert np.allclose(paths[1], 1.0)
    assert np.allclose(paths[2], 1.0)
