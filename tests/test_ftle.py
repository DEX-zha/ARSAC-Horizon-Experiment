import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_ftle import (
    companion_matrix,
    ftle_along_series,
    lorenz_ftle_ground_truth,
    lorenz_trajectory,
    model_jacobian_row,
    predict_and_grad,
)
from src.horizon_models import LinearAR, MLPPredictor, TorchWrapper


def _linear_model(weights_with_bias):
    model = LinearAR()
    model.weights = np.asarray(weights_with_bias, dtype=np.float64)
    return model


def test_model_jacobian_row_linear_ar_exact():
    model = _linear_model([0.3, -1.2, 0.7, 0.05])
    x = np.array([1.0, 2.0, -0.5])
    grad = model_jacobian_row(model, x)
    assert grad.shape == (3,)
    np.testing.assert_allclose(grad, [0.3, -1.2, 0.7])
    pred, grad2 = predict_and_grad(model, x)
    assert pred == float(np.dot([0.3, -1.2, 0.7], x) + 0.05)
    np.testing.assert_allclose(grad2, grad)


def test_model_jacobian_row_mlp_matches_finite_difference():
    torch.manual_seed(0)
    net = MLPPredictor(input_dim=4, hidden_dim=8)
    model = TorchWrapper(net, device="cpu")
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4)
    grad = model_jacobian_row(model, x)
    assert grad.shape == (4,)
    eps = 1e-4
    for i in range(4):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        fd = (model.predict(xp) - model.predict(xm)) / (2.0 * eps)
        assert abs(grad[i] - fd) < 5e-3
    # eval mode must be restored to whatever it was.
    assert net.training


def test_predict_and_grad_generic_fallback():
    class Quadratic:
        def predict(self, x):
            x = np.asarray(x, dtype=np.float64)
            return float(x[0] ** 2 + 3.0 * x[1])

    pred, grad = predict_and_grad(Quadratic(), np.array([2.0, -1.0]))
    assert abs(pred - 1.0) < 1e-12
    np.testing.assert_allclose(grad, [4.0, 3.0], atol=1e-5)


def test_companion_matrix_structure_lag1():
    grad = np.array([0.5, -0.25, 2.0])
    mat = companion_matrix(grad, dim=3, lag=1)
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(mat[0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(mat[1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(mat[2], grad)


def test_companion_matrix_structure_lag_gt1():
    grad = np.array([0.5, -0.25])
    mat = companion_matrix(grad, dim=2, lag=3)
    # Buffer length (dim-1)*lag + 1 = 4; grads at columns 0 and 3.
    assert mat.shape == (4, 4)
    shift = np.zeros((3, 4))
    shift[np.arange(3), np.arange(1, 4)] = 1.0
    np.testing.assert_allclose(mat[:3], shift)
    np.testing.assert_allclose(mat[3], [0.5, 0.0, 0.0, -0.25])


def test_ftle_linear_model_matches_spectral_radius():
    # f(x) = -0.3 * x_old + 1.1 * x_new -> companion eigenvalues 0.5, 0.6.
    model = _linear_model([-0.3, 1.1, 0.0])
    mat = companion_matrix(model.weights[:-1], dim=2, lag=1)
    radius = float(np.max(np.abs(np.linalg.eigvals(mat))))
    assert abs(radius - 0.6) < 1e-12
    rng = np.random.default_rng(1)
    series = rng.standard_normal(500)
    ftle, starts = ftle_along_series(
        model, series, dim=2, lag=1, k=300, sample_stride=50, max_windows=3, seed=0
    )
    assert ftle.shape == starts.shape
    assert len(ftle) == 3
    np.testing.assert_allclose(ftle, np.log(radius), atol=0.02)


def test_ftle_along_series_shapes_and_determinism():
    model = _linear_model([0.2, 0.9, 0.0])
    rng = np.random.default_rng(2)
    series = rng.standard_normal(300)
    out1 = ftle_along_series(model, series, 2, 1, k=20, sample_stride=5,
                             max_windows=10, seed=3)
    out2 = ftle_along_series(model, series, 2, 1, k=20, sample_stride=5,
                             max_windows=10, seed=3)
    np.testing.assert_array_equal(out1[1], out2[1])
    np.testing.assert_allclose(out1[0], out2[0])
    assert len(out1[0]) == 10
    assert np.all(np.isfinite(out1[0]))
    # Too-short series returns empty arrays.
    empty_f, empty_s = ftle_along_series(model, series[:10], 2, 1, k=20)
    assert empty_f.size == 0 and empty_s.size == 0


def test_lorenz_trajectory_stays_on_attractor():
    states = lorenz_trajectory(500, dt=0.01, warmup=1000, x0=(1.0, 1.0, 1.0))
    assert states.shape == (500, 3)
    assert np.all(np.isfinite(states))
    assert np.max(np.abs(states[:, 0])) < 25.0
    assert 0.0 < np.min(states[:, 2]) and np.max(states[:, 2]) < 55.0


def test_lorenz_ftle_ground_truth_mean_and_variance_scaling():
    states = lorenz_trajectory(3000, dt=0.01, warmup=1500, x0=(1.0, 1.0, 1.0))
    points = states[::100][:30]
    lam_short = lorenz_ftle_ground_truth(points, T=0.5, dt=0.01)
    lam_long = lorenz_ftle_ground_truth(points, T=3.0, dt=0.01)
    assert lam_short.shape == (30,)
    # Mean approaches lambda_1 ~ 0.906 as T grows (finite-T bias allowed).
    assert 0.6 < float(np.mean(lam_long)) < 1.3
    # Fluctuations shrink with T (~1/T variance scaling).
    assert float(np.std(lam_long)) < float(np.std(lam_short))


def test_lorenz_ftle_ground_truth_single_point():
    states = lorenz_trajectory(10, dt=0.01, warmup=1000)
    lam = lorenz_ftle_ground_truth(states[0], T=1.0, dt=0.01)
    assert np.isscalar(lam) or np.ndim(lam) == 0
    assert np.isfinite(lam)
