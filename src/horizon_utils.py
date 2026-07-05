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
    if torch.cuda.is_available():
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
    t_span = (0.0, (total_steps - 1) * dt)
    t_eval = np.linspace(0.0, (total_steps - 1) * dt, total_steps)

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
    if sol.y.shape[1] != total_steps:
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
    t_span = (0.0, (total_steps - 1) * dt)
    t_eval = np.linspace(0.0, (total_steps - 1) * dt, total_steps)

    sol = solve_ivp(
        rossler_deriv,
        t_span,
        [x0, y0, z0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9, 
        atol=1e-9
    )

    if sol.y.shape[1] != total_steps:
         raise RuntimeError("ODE solver failed to generate sufficient points.")

    return sol.y[0, warmup:]


def generate_mackey_glass(
    length,
    tau=17.0,
    beta=0.2,
    gamma=0.1,
    n=10,
    dt=1.0,
    warmup=200,
    integrator="rk4",
    dt_int=0.1,
):
    """Generates a Mackey-Glass delay differential series (method of steps).

    Units: ``tau`` is the delay in TIME units (not steps), ``dt`` is the
    output sampling interval in time units, ``warmup`` is the number of
    output SAMPLES discarded, and ``dt_int`` is the maximum internal
    integration step. The DDE is integrated with the method of steps on
    a constant history x(t) = 1.2 over [-tau, 0]. RK4 uses linearly
    interpolated delayed values at the half-step, so the formal order is
    limited by the interpolation; this is adequate at h <= 0.1.
    Chaos requires tau >~ 16.8 time units for the standard parameters.
    """
    integrator = integrator.lower()
    substeps = max(1, int(round(dt / dt_int)))
    h = dt / substeps
    n_delay = max(1, int(round(tau / h)))  # delay in internal steps
    total_samples = length + warmup
    total_steps = total_samples * substeps

    def mg_rhs(x_val, x_del):
        return beta * x_del / (1.0 + x_del**n) - gamma * x_val

    buf = np.empty(n_delay + total_steps + 1, dtype=np.float64)
    buf[: n_delay + 1] = 1.2  # constant history on [-tau, 0]
    for k in range(total_steps):
        i = n_delay + k
        xd0 = buf[i - n_delay]
        xd1 = buf[i - n_delay + 1]
        if integrator == "rk4":
            xd_half = 0.5 * (xd0 + xd1)  # linear interp at t + h/2 - tau
            k1 = mg_rhs(buf[i], xd0)
            k2 = mg_rhs(buf[i] + 0.5 * h * k1, xd_half)
            k3 = mg_rhs(buf[i] + 0.5 * h * k2, xd_half)
            k4 = mg_rhs(buf[i] + h * k3, xd1)
            buf[i + 1] = buf[i] + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        elif integrator == "euler":
            buf[i + 1] = buf[i] + h * mg_rhs(buf[i], xd0)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    sample_idx = n_delay + np.arange(total_samples, dtype=np.int64) * substeps
    series = buf[sample_idx]
    return series[warmup:]


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
    # NaN guard: return 0 if any input is NaN
    if not math.isfinite(growth) or not math.isfinite(init_err) or not math.isfinite(delta) or not math.isfinite(tolerance):
        return 0.0
    
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
    
    log_growth = math.log(growth)
    if log_growth <= 0.0:
        return 0.0
    return math.ceil(math.log(ratio) / log_growth)


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
    """Estimates a quantile of local expansion factors in embedded space.

    Pairs whose evolved distance d1 exceeds half the attractor diameter
    are skipped: such divergence is saturated (capped at the attractor
    size), and including it would bias the growth quantile downward.
    """
    series = np.asarray(series, dtype=np.float64)
    embedded = embed_series(series, dim, lag)
    n = embedded.shape[0]
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if n < 3:
        return 1.0, np.array([], dtype=np.float64)

    diameter = float(np.linalg.norm(embedded.max(axis=0) - embedded.min(axis=0)))

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
        if d1 > 0.5 * diameter:
            # Saturated divergence: skip to avoid biasing the quantile down.
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


def adaptive_horizon(lower_bound, horizon_max):
    """Clamp a lower-bound horizon to an integer in [1, horizon_max]."""
    if horizon_max is None:
        raise ValueError("horizon_max must be provided")
    if np.isscalar(lower_bound):
        return int(np.clip(np.floor(lower_bound), 1, horizon_max))
    lower_bound = np.asarray(lower_bound, dtype=np.float64)
    return np.clip(np.floor(lower_bound), 1, horizon_max).astype(np.int64)


def estimate_lyapunov(
    series,
    dim,
    lag,
    max_t=None,
    theiler=None,
    fit_start=None,
    fit_end=None,
    dt=1.0,
):
    """Estimates the largest Lyapunov exponent (Rosenstein method).

    Returns (slope, avg_log) where slope is per SAMPLE STEP; divide by
    ``dt`` to obtain the exponent per unit time. When left as None,
    parameters are auto-selected: ``theiler`` from the first zero
    crossing of the series autocorrelation, ``max_t`` from the number of
    embedded points, and ``fit_start``/``fit_end`` by scanning for the
    most linear region (highest R^2) of the divergence curve. Explicit
    values bypass the auto-selection and behave as before.
    """
    series = np.asarray(series, dtype=np.float64)
    x = embed_series(series, dim, lag)
    n = len(x)

    if theiler is None:
        centered = series - series.mean()
        denom = float(np.dot(centered, centered))
        max_lag = min(1000, len(series) // 2)
        theiler = max_lag
        if denom > 0:
            for lag_ac in range(1, max_lag + 1):
                ac = float(np.dot(centered[:-lag_ac], centered[lag_ac:])) / denom
                if ac <= 0:
                    theiler = lag_ac
                    break
        theiler = int(np.clip(theiler, 10, max(10, n // 10)))

    if max_t is None:
        max_t = int(np.clip(n // 4, 20, 400))

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
        d = np.linalg.norm(x[i : i + max_k] - x[j : j + max_k], axis=1)
        ks = np.nonzero(d > 1e-12)[0]
        sum_log[ks] += np.log(d[ks])
        count[ks] += 1.0

    valid = count > 0
    avg_log = np.zeros(max_t, dtype=np.float64)
    avg_log[valid] = sum_log[valid] / count[valid]

    idx = np.arange(max_t, dtype=np.float64)

    if fit_start is None or fit_end is None:
        # Auto-detect the linear region of avg_log: slide a window and
        # keep the one whose linear fit has the highest R^2. If no window
        # of width max_t // 4 is acceptably linear (fast systems saturate
        # early), retry with halved widths down to 8 before falling back.
        w = max(8, max_t // 4)
        best_r2 = -np.inf
        best = None
        while True:
            for s in range(1, max_t - w + 1):
                sel = valid & (idx >= s) & (idx < s + w)
                if sel.sum() < 2:
                    continue
                xs = idx[sel]
                ys = avg_log[sel]
                coeffs = np.polyfit(xs, ys, 1)
                ss_res = float(np.sum((ys - np.polyval(coeffs, xs)) ** 2))
                ss_tot = float(np.sum((ys - ys.mean()) ** 2))
                if ss_tot <= 0:
                    continue
                r2 = 1.0 - ss_res / ss_tot
                if r2 > best_r2:
                    best_r2 = r2
                    best = (s, s + w)
            if (best is not None and best_r2 >= 0.85) or w <= 8:
                break
            w = max(8, w // 2)
            best_r2 = -np.inf
            best = None
        if best is None or best_r2 < 0.85:
            best = (1, max(10, max_t // 3))
        if fit_start is None:
            fit_start = best[0]
        if fit_end is None:
            fit_end = best[1]

    fit_end = min(fit_end, max_t)
    fit_start = min(fit_start, fit_end - 1)
    if fit_end - fit_start < 2:
        return 0.0, avg_log

    use = valid & (idx >= fit_start) & (idx < fit_end)
    if use.sum() < 2:
        return 0.0, avg_log

    slope, _ = np.polyfit(idx[use], avg_log[use], 1)
    # Slope is per SAMPLE STEP; divide by dt for the exponent per unit time.
    return slope, avg_log
