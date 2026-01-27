"""Utilities for chaotic time series generation and embedding."""

import math
import random

import numpy as np
import torch


def set_seed(seed):
    """Sets random seeds for Python, NumPy, and Torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def generate_logistic_map(length, r=4.0, x0=0.2, warmup=100):
    """Generates a logistic map time series."""
    total = length + warmup
    x = np.empty(total, dtype=np.float64)
    x[0] = x0
    for i in range(total - 1):
        x[i + 1] = r * x[i] * (1.0 - x[i])
    return x[warmup:]


def generate_lorenz(
    length,
    dt=0.01,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    warmup=1000,
    x0=1.0,
    y0=1.0,
    z0=1.0,
    integrator="rk45",
):
    """Generates the x-component of the Lorenz system using Scipy."""
    from scipy.integrate import solve_ivp

    def lorenz_deriv(t, state):
        x, y, z = state
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    total_steps = length + warmup
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, dt)
    
    # solve_ivp guarantees accuracy but might not match exact steps if we don't be careful.
    # We use t_eval to get the exact time points we want.
    
    sol = solve_ivp(
        lorenz_deriv,
        t_span,
        [x0, y0, z0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9, 
        atol=1e-9
    )
    
    # sol.y has shape (3, n_points)
    # We want the x-component (index 0)
    # And we discard the warmup
    if sol.y.shape[1] < total_steps:
         # Fallback if solver fails to reach end (unlikely with RK45 on this system)
         raise RuntimeError("ODE solver failed to generate sufficient points.")
         
    return sol.y[0, warmup:]


def generate_rossler(
    length,
    dt=0.05,
    a=0.2,
    b=0.2,
    c=5.7,
    warmup=1000,
    x0=1.0,
    y0=0.0,
    z0=0.0,
    integrator="rk45",
):
    """Generates the x-component of the Rossler system using Scipy."""
    from scipy.integrate import solve_ivp

    def rossler_deriv(t, state):
        x, y, z = state
        return [-y - z, x + a * y, b + z * (x - c)]

    total_steps = length + warmup
    t_span = (0, total_steps * dt)
    t_eval = np.arange(0, total_steps * dt, dt)

    sol = solve_ivp(
        rossler_deriv,
        t_span,
        [x0, y0, z0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9, 
        atol=1e-9
    )

    if sol.y.shape[1] < total_steps:
         raise RuntimeError("ODE solver failed to generate sufficient points.")

    return sol.y[0, warmup:]


def generate_mackey_glass(
    length,
    tau=17,
    beta=0.2,
    gamma=0.1,
    n=10,
    dt=1.0,
    warmup=200,
    integrator="euler",
):
    """Generates a Mackey-Glass delay differential series."""
    total = length + warmup + tau + 1
    x = np.zeros(total, dtype=np.float64)
    x[: tau + 1] = 1.2
    integrator = integrator.lower()
    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        if integrator == "rk4":
            def mg_rhs(x_val, x_delayed):
                return beta * x_delayed / (1.0 + x_delayed**n) - gamma * x_val

            k1 = mg_rhs(x[t], x_tau)
            k2 = mg_rhs(x[t] + 0.5 * dt * k1, x_tau)
            k3 = mg_rhs(x[t] + 0.5 * dt * k2, x_tau)
            k4 = mg_rhs(x[t] + dt * k3, x_tau)
            x[t + 1] = x[t] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif integrator == "euler":
            dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x[t]
            x[t + 1] = x[t] + dx * dt
        else:
            raise ValueError(f"Unknown integrator: {integrator}")
    return x[tau + 1 + warmup :]


def standardize_series(series, mean=None, std=None):
    """Standardizes a series and returns standardized data with mean/std."""
    series = np.asarray(series, dtype=np.float64)
    if mean is None:
        mean = series.mean()
    if std is None:
        std = series.std()
    std = std if std > 0 else 1.0
    return (series - mean) / std, mean, std


def split_series(series, train_ratio=0.7, val_ratio=0.15, calib_ratio=0.0):
    """Splits a series into train, validation, calibration, and test parts."""
    if train_ratio + val_ratio + calib_ratio >= 1.0:
        raise ValueError("train+val+calib ratios must sum to < 1.0")
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    calib_end = int(n * (train_ratio + val_ratio + calib_ratio))
    train = series[:train_end]
    val = series[train_end:val_end]
    calib = series[val_end:calib_end]
    test = series[calib_end:]
    return train, val, calib, test


def horizon_from_model_bound_by_growth(growth, init_err, delta, tolerance):
    """Estimates horizon from a model-aware error bound.

    Uses e_{t+1} <= L * e_t + delta with L = growth (scalar > 0).
    """
    if tolerance <= 0:
        return 0.0
    init_err = max(0.0, float(init_err))
    delta = max(0.0, float(delta))
    if init_err >= tolerance:
        return 0.0

    if growth <= 0.0:
        return 0.0
    if growth <= 1.0 + 1e-12:
        if delta <= 0.0:
            return float("inf")
        return math.ceil((tolerance - init_err) / delta)

    offset = delta / (growth - 1.0)
    denom = init_err + offset
    if denom <= 0.0:
        return 0.0
    ratio = (tolerance + offset) / denom
    if ratio <= 1.0:
        return 0.0
    return math.ceil(math.log(ratio) / math.log(growth))


def horizon_from_model_bound(lyap_step, init_err, delta, tolerance):
    """Backward-compatible wrapper using lyap_step as log-growth."""
    growth = math.exp(lyap_step)
    return horizon_from_model_bound_by_growth(growth, init_err, delta, tolerance)


def estimate_expansion_quantile(
    series,
    dim,
    lag,
    quantile=0.95,
    theiler=10,
    max_pairs=500,
    seed=0,
    horizon=1,
):
    """Estimates a quantile of local expansion factors in embedded space."""
    series = np.asarray(series, dtype=np.float64)
    embedded = embed_series(series, dim, lag)
    n = embedded.shape[0]
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n < 3:
        return 1.0, np.array([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    max_index = n - horizon - 1
    if max_index <= 0:
        return 1.0, np.array([], dtype=np.float64)
    indices = np.arange(max_index, dtype=np.int64)
    if max_pairs is None or max_pairs >= len(indices):
        sample = indices
    else:
        sample = rng.choice(indices, size=max_pairs, replace=False)

    ratios = []
    for i in sample:
        diff = embedded - embedded[i]
        dist = np.linalg.norm(diff, axis=1)
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dist[lo:hi] = np.inf
        j = int(np.argmin(dist))
        d0 = dist[j]
        if not np.isfinite(d0) or d0 <= 1e-12:
            continue
        if i + horizon >= n or j + horizon >= n:
            continue
        d1 = np.linalg.norm(embedded[i + horizon] - embedded[j + horizon])
        if d1 <= 1e-12:
            continue
        ratios.append((d1 / d0) ** (1.0 / horizon))

    if not ratios:
        return 1.0, np.array([], dtype=np.float64)

    ratios = np.asarray(ratios, dtype=np.float64)
    q = float(np.quantile(ratios, quantile))
    q = max(q, 1e-6)
    return q, ratios


def embed_series(series, dim, lag):
    """Builds a delay embedding of a time series."""
    series = np.asarray(series, dtype=np.float64)
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if lag < 1:
        raise ValueError("lag must be >= 1")
    n = len(series) - (dim - 1) * lag
    if n <= 0:
        raise ValueError("series too short for given dim/lag")
    out = np.empty((n, dim), dtype=np.float64)
    for i in range(dim):
        start = i * lag
        out[:, i] = series[start : start + n]
    return out


def build_supervised(series, dim, lag, horizon=1):
    """Builds supervised (x, y) pairs from a delay-embedded series."""
    series = np.asarray(series, dtype=np.float64)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(series) - (dim - 1) * lag - horizon
    if n <= 0:
        raise ValueError("series too short for given dim/lag/horizon")
    x = np.empty((n, dim), dtype=np.float64)
    for i in range(dim):
        start = i * lag
        x[:, i] = series[start : start + n]
    y_start = (dim - 1) * lag + horizon
    y = series[y_start : y_start + n]
    return x, y


def estimate_lyapunov(
    series,
    dim,
    lag,
    max_t=25,
    theiler=10,
    fit_start=1,
    fit_end=10,
    dt=1.0,
):
    """Estimates the largest Lyapunov exponent (Rosenstein method)."""
    series = np.asarray(series, dtype=np.float64)
    x = embed_series(series, dim, lag)
    n = len(x)
    if n <= max_t + 1:
        return 0.0, np.zeros(max_t, dtype=np.float64)

    neighbors = []
    for i in range(0, n - max_t):
        diff = x - x[i]
        dist = np.linalg.norm(diff, axis=1)
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        dist[lo:hi] = np.inf
        j = int(np.argmin(dist))
        if np.isfinite(dist[j]):
            neighbors.append((i, j))

    if not neighbors:
        return 0.0, np.zeros(max_t, dtype=np.float64)

    sum_log = np.zeros(max_t, dtype=np.float64)
    count = np.zeros(max_t, dtype=np.float64)
    for i, j in neighbors:
        max_k = min(max_t, n - max(i, j))
        for k in range(max_k):
            d = np.linalg.norm(x[i + k] - x[j + k])
            if d <= 1e-12:
                continue
            sum_log[k] += math.log(d)
            count[k] += 1.0

    valid = count > 0
    avg_log = np.zeros(max_t, dtype=np.float64)
    avg_log[valid] = sum_log[valid] / count[valid]

    fit_end = min(fit_end, max_t)
    fit_start = min(fit_start, fit_end - 1)
    if fit_end - fit_start < 2:
        return 0.0, avg_log

    idx = np.arange(max_t, dtype=np.float64)
    use = valid & (idx >= fit_start) & (idx < fit_end)
    if use.sum() < 2:
        return 0.0, avg_log

    slope, _ = np.polyfit(idx[use], avg_log[use], 1)
    # Slope is per-step (sample index). Convert to per-time externally if needed.
    _ = dt
    return slope, avg_log
