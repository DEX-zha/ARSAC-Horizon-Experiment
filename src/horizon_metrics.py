"""Metrics and bounds for horizon experiments."""

import math

import numpy as np
import torch

from src.horizon_models import LinearAR, TorchSeqWrapper, TorchWrapper
from src.horizon_utils import build_supervised


def evaluate_mse(model, x, y, device="cpu"):
    """Computes mean squared error for a model."""
    if hasattr(model, "predict_batch"):
        pred = model.predict_batch(x)
        return float(np.mean((pred - y) ** 2))

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        pred = model(x_t)
        return float(torch.mean((pred - y_t) ** 2).item())


def compute_calibration_residuals(model, series_std, dim, lag):
    """Computes one-step residuals on a calibration series."""
    try:
        x_calib, y_calib = build_supervised(series_std, dim, lag, horizon=1)
    except ValueError:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if x_calib.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    if hasattr(model, "predict_batch"):
        preds = model.predict_batch(x_calib)
    else:
        preds = np.array([model.predict(x) for x in x_calib], dtype=np.float64)

    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    residuals = np.abs(preds - y_calib)
    return x_calib, residuals


def estimate_model_error_from_residuals(
    residuals, mode="quantile", quantile=0.95, scale=3.0
):
    """Estimates a probabilistic model error bound from residuals."""
    if residuals.size == 0:
        return 0.0, "none", 0.0

    mean_resid = float(np.mean(residuals))
    if mode == "max":
        return float(np.max(residuals)), "max", mean_resid
    if mode == "mean_std":
        delta = float(np.mean(residuals) + scale * np.std(residuals))
        return delta, f"mean+{scale:.1f}std", mean_resid
    delta = float(np.quantile(residuals, quantile))
    return delta, f"quantile@{quantile:.2f}", mean_resid


def estimate_model_error(
    model, series_std, dim, lag, mode="quantile", quantile=0.95, scale=3.0
):
    """Estimates a probabilistic model error bound and mean residual."""
    _, residuals = compute_calibration_residuals(model, series_std, dim, lag)
    return estimate_model_error_from_residuals(residuals, mode, quantile, scale)


def estimate_error_growth(
    model,
    series_std,
    dim,
    lag,
    horizon,
    max_windows=500,
    quantile=0.95,
    seed=0,
    eps=1e-8,
):
    """Estimates growth rates from multi-step prediction errors."""
    series_std = np.asarray(series_std, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - horizon
    if n <= 0 or horizon < 2:
        return 1.0, 1.0, np.array([], dtype=np.float64)

    indices = np.arange(n, dtype=np.int64)
    if max_windows is not None and max_windows < len(indices):
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_windows, replace=False)

    lambdas = []
    for start in indices:
        history = list(series_std[start : start + window_len])
        prev_err = None
        log_ratios = []
        for h in range(horizon):
            x = [history[i * lag] for i in range(dim)]
            pred = model.predict(x)
            true = series_std[start + (dim - 1) * lag + h + 1]
            err = abs(pred - true)
            if prev_err is not None:
                log_ratios.append(math.log((err + eps) / (prev_err + eps)))
            prev_err = err
            history.append(pred)
            history.pop(0)
        if log_ratios:
            lambdas.append(float(np.mean(log_ratios)))

    if not lambdas:
        return 1.0, 1.0, np.array([], dtype=np.float64)

    lambdas = np.asarray(lambdas, dtype=np.float64)
    growth_q = float(np.exp(np.quantile(lambdas, quantile)))
    growth_mean = float(np.exp(np.mean(lambdas)))
    growth_q = max(growth_q, 1e-6)
    growth_mean = max(growth_mean, 1e-6)
    return growth_q, growth_mean, lambdas


def estimate_local_delta(
    x_calib,
    residuals,
    k=20,
    quantile=0.95,
    max_samples=500,
    seed=0,
):
    """Estimates a local residual bound using k-nearest neighbors."""
    if residuals.size == 0 or x_calib.size == 0:
        return 0.0, 0.0, np.array([], dtype=np.float64)

    n = residuals.shape[0]
    if n <= 1:
        delta = float(np.quantile(residuals, quantile))
        return delta, float(np.mean(residuals)), np.array([], dtype=np.float64)

    k = max(1, min(k, n - 1))
    indices = np.arange(n, dtype=np.int64)
    if max_samples is not None and max_samples < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    local_deltas = []
    for idx in indices:
        diff = x_calib - x_calib[idx]
        dist = np.linalg.norm(diff, axis=1)
        dist[idx] = np.inf
        neighbor_idx = np.argpartition(dist, k)[:k]
        neighbor_res = residuals[neighbor_idx]
        if neighbor_res.size == 0:
            continue
        local_deltas.append(float(np.quantile(neighbor_res, quantile)))

    if not local_deltas:
        delta = float(np.quantile(residuals, quantile))
        return delta, float(np.mean(residuals)), np.array([], dtype=np.float64)

    local_deltas = np.asarray(local_deltas, dtype=np.float64)
    delta_q = float(np.quantile(local_deltas, quantile))
    delta_mean = float(np.mean(local_deltas))
    return delta_q, delta_mean, local_deltas


def estimate_jacobian_growth(
    model,
    x_samples,
    quantile=0.95,
    max_samples=500,
    seed=0,
):
    """Estimates local Jacobian norms for a model."""
    if x_samples.size == 0:
        return 1.0, 1.0, np.array([], dtype=np.float64)

    n = x_samples.shape[0]
    indices = np.arange(n, dtype=np.int64)
    if max_samples is not None and max_samples < n:
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_samples, replace=False)

    if isinstance(model, LinearAR) and model.weights is not None:
        weights = np.asarray(model.weights[:-1], dtype=np.float64)
        norm = float(np.linalg.norm(weights))
        return norm, norm, np.array([norm], dtype=np.float64)

    if isinstance(model, (TorchWrapper, TorchSeqWrapper)):
        model.model.eval()

    norms = []
    for idx in indices:
        x = x_samples[idx]
        if isinstance(model, TorchSeqWrapper):
            x_t = torch.tensor(
                x, dtype=torch.float32, device=model.device, requires_grad=True
            ).view(1, -1, 1)
            out = model.model(x_t)
            out = out.squeeze()
            if out.ndim > 0:
                out = out.sum()
            grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
            norm = float(torch.norm(grad).item())
        elif isinstance(model, TorchWrapper):
            x_t = torch.tensor(
                x, dtype=torch.float32, device=model.device, requires_grad=True
            ).view(1, -1)
            out = model.model(x_t)
            out = out.squeeze()
            if out.ndim > 0:
                out = out.sum()
            grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
            norm = float(torch.norm(grad).item())
        else:
            return 1.0, 1.0, np.array([], dtype=np.float64)
        norms.append(norm)

    if not norms:
        return 1.0, 1.0, np.array([], dtype=np.float64)


def _predict_one_step(model, x):
    if hasattr(model, "predict"):
        return float(model.predict(x))
    if hasattr(model, "predict_batch"):
        return float(model.predict_batch(np.asarray([x], dtype=np.float64)).reshape(-1)[0])
    raise ValueError("Model does not support prediction API.")


def gated_rollout(model, series_std, dim, lag, l_values, horizon_max=None):
    """Rolls out predictions up to floor(L(x)) for each window; stops afterward."""
    series_std = np.asarray(series_std, dtype=np.float64)
    l_values = np.asarray(l_values, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - 1
    if n <= 0 or l_values.size == 0:
        return [], np.array([], dtype=np.int64)
    n = min(n, l_values.size)
    paths = []
    horizons = []
    for start in range(n):
        max_h = int(np.floor(l_values[start]))
        if horizon_max is not None:
            max_h = min(max_h, int(horizon_max))
        max_h = max(0, max_h)
        history = list(series_std[start : start + window_len])
        preds = []
        for _ in range(max_h):
            x = [history[i * lag] for i in range(dim)]
            pred = _predict_one_step(model, np.asarray(x, dtype=np.float64))
            preds.append(pred)
            history.append(pred)
            history.pop(0)
        paths.append(np.asarray(preds, dtype=np.float64))
        horizons.append(max_h)
    return paths, np.asarray(horizons, dtype=np.int64)

    norms = np.asarray(norms, dtype=np.float64)
    norm_q = float(np.quantile(norms, quantile))
    norm_mean = float(np.mean(norms))
    norm_q = max(norm_q, 1e-6)
    norm_mean = max(norm_mean, 1e-6)
    return norm_q, norm_mean, norms


def jacobian_norm(model, x):
    """Computes a local Jacobian norm for a single input."""
    if isinstance(model, LinearAR) and model.weights is not None:
        weights = np.asarray(model.weights[:-1], dtype=np.float64)
        return float(np.linalg.norm(weights))

    if isinstance(model, TorchSeqWrapper):
        x_t = torch.tensor(
            x, dtype=torch.float32, device=model.device, requires_grad=True
        ).view(1, -1, 1)
        out = model.model(x_t)
        out = out.squeeze()
        if out.ndim > 0:
            out = out.sum()
        grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
        return float(torch.norm(grad).item())
    if isinstance(model, TorchWrapper):
        x_t = torch.tensor(
            x, dtype=torch.float32, device=model.device, requires_grad=True
        ).view(1, -1)
        out = model.model(x_t)
        out = out.squeeze()
        if out.ndim > 0:
            out = out.sum()
        grad = torch.autograd.grad(out, x_t, retain_graph=False)[0]
        return float(torch.norm(grad).item())

    return 0.0


def horizon_from_window_rmse(rmse_by_h, tolerance, consecutive=2):
    """Returns the first horizon where K consecutive RMSE values exceed tolerance."""
    rmse_by_h = np.asarray(rmse_by_h, dtype=np.float64)
    if rmse_by_h.size == 0:
        return 0.0
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        return 1.0

    consecutive = max(1, int(consecutive))
    if consecutive == 1:
        for idx, value in enumerate(rmse_by_h, start=1):
            if value >= tolerance:
                return idx
        return float(len(rmse_by_h))

    limit = rmse_by_h.size - consecutive + 1
    for idx in range(limit):
        window = rmse_by_h[idx : idx + consecutive]
        if np.all(window >= tolerance):
            return float(idx + 1)
    return float(len(rmse_by_h))


def build_horizon_dataset(
    model,
    series_std,
    dim,
    lag,
    horizon_max,
    tolerance,
    max_windows=None,
    seed=0,
    use_jacobian=True,
    error_mode="absolute",
    error_factor=1.0,
    consecutive_k=2,
    stride=1,
    feature_horizon=3,
    eps=1e-8,
):
    """Builds horizon features and per-window horizon labels for conformal training."""
    series_std = np.asarray(series_std, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - horizon_max
    if n <= 0:
        return np.empty((0, dim + 8), dtype=np.float64), np.empty(
            (0,), dtype=np.float64
        )

    stride = max(1, int(stride))
    indices = np.arange(0, n, stride, dtype=np.int64)
    if max_windows is not None and max_windows < len(indices):
        rng = np.random.default_rng(seed)
        indices = rng.choice(indices, size=max_windows, replace=False)
    indices = np.sort(indices)

    features = []
    targets = []
    for start in indices:
        history = list(series_std[start : start + window_len])
        x0 = np.array([history[i * lag] for i in range(dim)], dtype=np.float64)
        pred0 = float(model.predict(x0))
        delta_pred = abs(pred0 - history[-1])
        true0 = series_std[start + (dim - 1) * lag + 1]
        resid1 = abs(pred0 - true0)
        jac_norm0 = jacobian_norm(model, x0) if use_jacobian else 0.0

        hist = history.copy()
        errors = np.zeros(horizon_max, dtype=np.float64)
        jac_values = []
        preds = []
        for h in range(horizon_max):
            x = [hist[i * lag] for i in range(dim)]
            pred = model.predict(x)
            true = series_std[start + (dim - 1) * lag + h + 1]
            err = pred - true
            errors[h] = err * err
            if use_jacobian and h < feature_horizon:
                jac_values.append(jacobian_norm(model, np.asarray(x, dtype=np.float64)))
            if h < feature_horizon:
                preds.append(pred)
            hist.append(pred)
            hist.pop(0)

        rmse_by_h = np.sqrt(np.cumsum(errors) / np.arange(1, horizon_max + 1))
        if error_mode == "relative":
            tolerance_local = rmse_by_h[0] * error_factor
        else:
            tolerance_local = tolerance

        horizon = horizon_from_window_rmse(
            rmse_by_h, tolerance_local, consecutive=consecutive_k
        )

        jac_mean = jac_norm0
        jac_log_mean = math.log(max(jac_norm0, eps)) if jac_norm0 > 0.0 else 0.0
        if use_jacobian and jac_values:
            jac_arr = np.asarray(jac_values, dtype=np.float64)
            jac_mean = float(np.mean(jac_arr))
            jac_log_mean = float(np.mean(np.log(np.maximum(jac_arr, eps))))
        err_window = min(feature_horizon, horizon_max)
        err_var = float(np.var(np.sqrt(errors[:err_window]))) if err_window > 1 else 0.0
        pred_var = float(np.var(np.asarray(preds, dtype=np.float64))) if len(preds) > 1 else 0.0

        feat = np.concatenate(
            [
                x0,
                [
                    pred0,
                    delta_pred,
                    jac_norm0,
                    resid1,
                    jac_mean,
                    jac_log_mean,
                    err_var,
                    pred_var,
                ],
            ]
        )

        features.append(feat)
        targets.append(horizon)

    return np.asarray(features, dtype=np.float64), np.asarray(
        targets, dtype=np.float64
    )


def rolling_rmse(model, series_std, dim, lag, horizon_max):
    """Computes multi-step RMSE by rolling autoregression."""
    series_std = np.asarray(series_std, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - horizon_max
    if n <= 0:
        return np.full(horizon_max, np.nan, dtype=np.float64)

    errors = np.zeros(horizon_max, dtype=np.float64)
    count = np.zeros(horizon_max, dtype=np.float64)
    for start in range(n):
        history = list(series_std[start : start + window_len])
        for h in range(horizon_max):
            x = [history[i * lag] for i in range(dim)]
            pred = model.predict(x)
            true = series_std[start + (dim - 1) * lag + h + 1]
            errors[h] += (pred - true) ** 2
            count[h] += 1.0
            history.append(pred)
            history.pop(0)

    rmse = np.sqrt(errors / np.maximum(count, 1.0))
    return rmse


def window_horizons(model, series_std, dim, lag, horizon_max, tolerance):
    """Computes per-window horizons using absolute error threshold."""
    series_std = np.asarray(series_std, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - horizon_max
    if n <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    horizons = []
    init_errs = []
    for start in range(n):
        history = list(series_std[start : start + window_len])
        horizon = horizon_max
        init_err = None
        for h in range(horizon_max):
            x = [history[i * lag] for i in range(dim)]
            pred = model.predict(x)
            true = series_std[start + (dim - 1) * lag + h + 1]
            err = abs(pred - true)
            if h == 0:
                init_err = err
            if err >= tolerance:
                horizon = h + 1
                break
            history.append(pred)
            history.pop(0)
        if init_err is None:
            continue
        horizons.append(horizon)
        init_errs.append(init_err)

    return np.array(horizons, dtype=np.float64), np.array(init_errs, dtype=np.float64)


def horizon_from_rmse(rmse, tolerance):
    """Returns the first horizon where RMSE exceeds tolerance."""
    for idx, value in enumerate(rmse, start=1):
        if value >= tolerance:
            return idx
    return len(rmse)


def horizon_from_lyapunov(lyapunov, init_err, tolerance):
    """Estimates theoretical horizon from Lyapunov growth."""
    if lyapunov <= 0 or init_err <= 0:
        return float("inf")
    if tolerance <= init_err:
        return 0.0
    return math.log(tolerance / init_err) / lyapunov
