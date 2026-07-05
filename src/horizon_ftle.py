"""Finite-time Lyapunov exponents (FTLE) for learned one-step maps.

This module estimates local (finite-time) expansion rates in two ways:

1. ``ftle_along_series``: FTLE of the LEARNED dynamics. The embedded
   rollout buffer ``b_t = (s_{t-w+1}, ..., s_t)`` evolves under the shift
   map ``b_{t+1} = (b_2, ..., b_w, f(x(b)))`` where ``x(b)`` picks the
   ``dim`` lagged coordinates fed to the model. Its exact Jacobian is a
   companion-structured matrix whose last row holds the gradient of the
   scalar one-step map ``f``. Products of these matrices along the
   PREDICTED rollout, re-orthonormalized every step (QR with a single
   column, i.e. vector normalization; the accumulated ``log R_11`` is the
   log of the norm), give the largest finite-time exponent per step.

2. ``lorenz_ftle_ground_truth``: FTLE of the TRUE Lorenz flow, from the
   variational equations (state + 3x3 tangent basis, RK4, QR every step,
   accumulate ``log |R_11|``). Pure numpy, vectorized over sample points.

Caveats (stated for honest use): the model FTLE measures the learned map,
not the true flow — for a linear AR model the Jacobian is constant, so its
FTLE is the same for every window by construction. The delay embedding is
a coordinate change that distorts local norms, which perturbs finite-time
values (not the asymptotic exponent).
"""

import numpy as np
import torch

from src.horizon_models import LinearAR, TorchSeqWrapper, TorchWrapper

LORENZ_SIGMA = 10.0
LORENZ_RHO = 28.0
LORENZ_BETA = 8.0 / 3.0


def _finite_difference_grad(model, x, rel_eps=1e-5):
    """Central-difference gradient of a scalar predict(x) map."""
    x = np.asarray(x, dtype=np.float64)
    grad = np.empty_like(x)
    for i in range(x.size):
        eps = rel_eps * max(1.0, abs(float(x[i])))
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        grad[i] = (float(model.predict(xp)) - float(model.predict(xm))) / (2.0 * eps)
    return grad


def _torch_predict_and_grad(wrapper, x, seq):
    """One forward+backward pass returning (prediction, gradient row)."""
    module = wrapper.model
    was_training = module.training
    module.eval()
    try:
        x_t = torch.tensor(
            x, dtype=torch.float32, device=wrapper.device, requires_grad=True
        )
        shaped = x_t.view(1, -1, 1) if seq else x_t.view(1, -1)
        if seq:
            with torch.backends.cudnn.flags(enabled=False):
                out = module(shaped)
                out = out.squeeze()
                if out.ndim > 0:
                    out = out.sum()
                grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
        else:
            out = module(shaped)
            out = out.squeeze()
            if out.ndim > 0:
                out = out.sum()
            grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
        pred = float(out.detach().item())
        grad_row = grad.detach().view(-1).cpu().double().numpy()
    finally:
        module.train(was_training)
    return pred, grad_row


def predict_and_grad(model, x):
    """Returns (f(x), grad f(x)) for the scalar one-step map of ``model``.

    Exact for LinearAR (weights[:-1]); autograd in eval mode for torch
    wrappers; central finite differences as a generic fallback.
    """
    x = np.asarray(x, dtype=np.float64)
    if isinstance(model, LinearAR) and model.weights is not None:
        weights = np.asarray(model.weights[:-1], dtype=np.float64)
        pred = float(np.dot(weights, x) + float(model.weights[-1]))
        return pred, weights.copy()
    if isinstance(model, TorchSeqWrapper):
        return _torch_predict_and_grad(model, x, seq=True)
    if isinstance(model, TorchWrapper):
        return _torch_predict_and_grad(model, x, seq=False)
    if hasattr(model, "predict"):
        pred = float(model.predict(x))
        return pred, _finite_difference_grad(model, x)
    raise TypeError(f"Unsupported model type for predict_and_grad: {type(model)!r}")


def model_jacobian_row(model, x):
    """Gradient of the scalar one-step map f at x, shape ``(dim,)``.

    Exact ``weights[:-1]`` for LinearAR; ``torch.autograd`` (eval mode) for
    TorchWrapper/TorchSeqWrapper; finite differences otherwise.
    """
    return predict_and_grad(model, x)[1]


def companion_matrix(grad_row, dim, lag=1):
    """Jacobian of the embedded shift dynamics (companion structure).

    The rollout buffer has length ``w = (dim - 1) * lag + 1`` and evolves as
    ``b' = (b_2, ..., b_w, f(b_0, b_lag, ..., b_{(dim-1)lag}))`` (0-based
    coordinates ``i * lag`` feed the model). The exact Jacobian is the
    ``w x w`` matrix with an upper shift block and ``grad_row`` entries at
    columns ``i * lag`` of the last row. For ``lag == 1`` this is the plain
    ``dim x dim`` companion matrix with last row ``grad_row``.
    """
    grad_row = np.asarray(grad_row, dtype=np.float64).reshape(-1)
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    if grad_row.size != dim:
        raise ValueError(f"grad_row has size {grad_row.size}, expected dim={dim}")
    size = (dim - 1) * lag + 1
    mat = np.zeros((size, size), dtype=np.float64)
    if size > 1:
        rows = np.arange(size - 1)
        mat[rows, rows + 1] = 1.0
    mat[size - 1, np.arange(dim) * lag] = grad_row
    return mat


def ftle_along_series(
    model,
    series,
    dim,
    lag,
    k=100,
    sample_stride=1,
    max_windows=None,
    seed=0,
):
    """Largest finite-time Lyapunov exponent of the learned map, per window.

    For each sampled window start, rolls the model out autoregressively for
    ``k`` steps (PREDICTED rollout, as in the label construction) and
    accumulates the product of companion Jacobians with re-orthonormalization
    at every step (single-column QR: ``log R_11`` = log of the propagated
    vector norm). Returns ``(ftle, starts)`` where ``ftle[i]`` is the
    exponent PER STEP (divide by dt for the exponent per unit time) for the
    window starting at ``starts[i]``.
    """
    series = np.asarray(series, dtype=np.float64)
    if k < 1:
        raise ValueError("k must be >= 1")
    window_len = (dim - 1) * lag + 1
    n = len(series) - window_len - k
    if n <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)

    stride = max(1, int(sample_stride))
    starts = np.arange(0, n, stride, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if max_windows is not None and max_windows < len(starts):
        starts = np.sort(rng.choice(starts, size=max_windows, replace=False))

    ftle = np.empty(len(starts), dtype=np.float64)
    for w_idx, start in enumerate(starts):
        history = list(series[start : start + window_len])
        vec = rng.standard_normal(window_len)
        vec /= np.linalg.norm(vec)
        log_sum = 0.0
        valid = True
        for _ in range(k):
            x = np.array([history[i * lag] for i in range(dim)], dtype=np.float64)
            pred, grad_row = predict_and_grad(model, x)
            mat = companion_matrix(grad_row, dim, lag)
            vec = mat @ vec
            norm = float(np.linalg.norm(vec))
            if not np.isfinite(norm) or norm <= 0.0:
                valid = False
                break
            log_sum += np.log(norm)
            vec /= norm
            history.append(pred)
            history.pop(0)
        ftle[w_idx] = log_sum / float(k) if valid else np.nan

    return ftle, starts


def _lorenz_deriv(state, sigma, rho, beta):
    """Lorenz vector field, vectorized over points; state shape (..., 3)."""
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    out = np.empty_like(state)
    out[..., 0] = sigma * (y - x)
    out[..., 1] = x * (rho - z) - y
    out[..., 2] = x * y - beta * z
    return out


def _lorenz_jacobian(state, sigma, rho, beta):
    """Lorenz Jacobian, vectorized over points; returns shape (..., 3, 3)."""
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    jac = np.zeros(state.shape[:-1] + (3, 3), dtype=np.float64)
    jac[..., 0, 0] = -sigma
    jac[..., 0, 1] = sigma
    jac[..., 1, 0] = rho - z
    jac[..., 1, 1] = -1.0
    jac[..., 1, 2] = -x
    jac[..., 2, 0] = y
    jac[..., 2, 1] = x
    jac[..., 2, 2] = -beta
    return jac


def lorenz_trajectory(
    n_samples,
    dt=0.01,
    warmup=2000,
    x0=(1.0, 1.0, 1.0),
    sigma=LORENZ_SIGMA,
    rho=LORENZ_RHO,
    beta=LORENZ_BETA,
):
    """Integrates the Lorenz system with fixed-step RK4 (pure numpy).

    Returns the ``(n_samples, 3)`` post-warmup states; ``warmup`` is in
    integration STEPS. Companion helper for the FTLE studies, where the
    full 3D states (not just the x-component) must stay aligned with the
    scalar series used by the forecaster.
    """
    total = int(n_samples) + int(warmup)
    states = np.empty((total, 3), dtype=np.float64)
    state = np.asarray(x0, dtype=np.float64).copy()
    for i in range(total):
        k1 = _lorenz_deriv(state, sigma, rho, beta)
        k2 = _lorenz_deriv(state + 0.5 * dt * k1, sigma, rho, beta)
        k3 = _lorenz_deriv(state + 0.5 * dt * k2, sigma, rho, beta)
        k4 = _lorenz_deriv(state + dt * k3, sigma, rho, beta)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states[i] = state
    return states[warmup:]


def lorenz_ftle_ground_truth(
    x0_points,
    T,
    dt=0.01,
    sigma=LORENZ_SIGMA,
    rho=LORENZ_RHO,
    beta=LORENZ_BETA,
):
    """Finite-time largest Lyapunov exponent of the true Lorenz flow.

    Integrates the variational equations (state + 3x3 tangent basis) with
    coupled RK4 and QR re-orthonormalization at EVERY step, accumulating
    ``log |R_11|``. Vectorized over points: ``x0_points`` has shape
    ``(n, 3)``; returns ``lambda_T`` samples of shape ``(n,)`` in units of
    inverse time. As T grows the sample mean approaches lambda_1 ~ 0.906
    and the variance shrinks ~ 1/T (large-deviation / CLT scaling).
    """
    x = np.array(x0_points, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("x0_points must have shape (n, 3)")
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be > 0")

    n_steps = max(1, int(round(T / dt)))
    n_pts = x.shape[0]
    tangent = np.tile(np.eye(3, dtype=np.float64), (n_pts, 1, 1))
    log_sum = np.zeros(n_pts, dtype=np.float64)

    for _ in range(n_steps):
        # Coupled RK4 for d/dt x = f(x), d/dt V = J(x) V.
        k1x = _lorenz_deriv(x, sigma, rho, beta)
        k1v = _lorenz_jacobian(x, sigma, rho, beta) @ tangent
        x2 = x + 0.5 * dt * k1x
        k2x = _lorenz_deriv(x2, sigma, rho, beta)
        k2v = _lorenz_jacobian(x2, sigma, rho, beta) @ (tangent + 0.5 * dt * k1v)
        x3 = x + 0.5 * dt * k2x
        k3x = _lorenz_deriv(x3, sigma, rho, beta)
        k3v = _lorenz_jacobian(x3, sigma, rho, beta) @ (tangent + 0.5 * dt * k2v)
        x4 = x + dt * k3x
        k4x = _lorenz_deriv(x4, sigma, rho, beta)
        k4v = _lorenz_jacobian(x4, sigma, rho, beta) @ (tangent + dt * k3v)
        x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        tangent = tangent + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        q, r = np.linalg.qr(tangent)
        log_sum += np.log(np.abs(r[:, 0, 0]))
        tangent = q

    lam = log_sum / (n_steps * dt)
    return lam[0] if single else lam
