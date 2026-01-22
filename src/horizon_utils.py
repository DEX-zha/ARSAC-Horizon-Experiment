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
):
    """Generates the x-component of the Lorenz system."""
    total = length + warmup
    x = x0
    y = y0
    z = z0
    out = np.empty(total, dtype=np.float64)
    for i in range(total):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        out[i] = x
    return out[warmup:]


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
):
    """Generates the x-component of the Rossler system."""
    total = length + warmup
    x = x0
    y = y0
    z = z0
    out = np.empty(total, dtype=np.float64)
    for i in range(total):
        dx = -y - z
        dy = x + a * y
        dz = b + z * (x - c)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        out[i] = x
    return out[warmup:]


def generate_mackey_glass(
    length,
    tau=17,
    beta=0.2,
    gamma=0.1,
    n=10,
    dt=1.0,
    warmup=200,
):
    """Generates a Mackey-Glass delay differential series."""
    total = length + warmup + tau + 1
    x = np.zeros(total, dtype=np.float64)
    x[: tau + 1] = 1.2
    for t in range(tau, total - 1):
        x_tau = x[t - tau]
        dx = beta * x_tau / (1.0 + x_tau**n) - gamma * x[t]
        x[t + 1] = x[t] + dx * dt
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


def split_series(series, train_ratio=0.7, val_ratio=0.15):
    """Splits a series into train, validation, and test parts."""
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]
    return train, val, test


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
