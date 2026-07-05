"""Observation-noise estimation for the noise-aware predictability floor.

On real data the forecaster's residual conflates model deficit with
irreducible observation noise. The reachable floor under observation noise
sigma is the ONE-SHOT twin at eps = sigma (initial-state uncertainty growing
at the Lyapunov rate; validated on Lorenz, docs/theory/chaos_floor.md):

    H_reachable ~ ln(tau / sigma) / (lambda_1 * dt)   [steps]

so sigma must be estimated from the data. Method: local-linear residuals in
delay-embedding space — for each sampled point, fit a ridge-regularized local
linear model of the next value on its k nearest neighbors (Theiler-excluded)
and predict the held-out point; as the neighborhood scale shrinks, the
dynamics is locally linear and the residual converges to the noise level
(plus a curvature bias, making sigma_hat an UPPER bound on clean data).
Validated on Lorenz with known synthetic noise in studies/study_noisy_floor.py
(pre-registered criterion: sigma_hat/sigma in [0.5, 2]).
"""

from __future__ import annotations

import numpy as np

from src.horizon_utils import embed_series


def estimate_observation_noise(
    series,
    dim=6,
    lag=1,
    k=12,
    n_samples=400,
    theiler=None,
    ridge=1e-6,
    seed=0,
):
    """Returns (sigma_hat, residuals) in the units of ``series``."""
    series = np.asarray(series, dtype=np.float64)
    x = embed_series(series, dim, lag)
    horizon_shift = (dim - 1) * lag + 1
    n = len(x) - 1
    if n < 5 * k:
        return float("nan"), np.array([], dtype=np.float64)
    if theiler is None:
        centered = series - series.mean()
        denom = float(np.dot(centered, centered))
        theiler = 10
        if denom > 0:
            for lag_ac in range(1, min(500, len(series) // 2)):
                ac = float(np.dot(centered[:-lag_ac], centered[lag_ac:])) / denom
                if ac <= 0:
                    theiler = lag_ac
                    break
        theiler = int(np.clip(theiler, 5, n // 10))

    rng = np.random.default_rng(seed)
    idx = rng.choice(n - 1, size=min(n_samples, n - 1), replace=False)
    residuals = []
    for i in idx:
        target_i = series[i + horizon_shift] if i + horizon_shift < len(series) else None
        if target_i is None:
            continue
        diff = x[:n] - x[i]
        dist = np.linalg.norm(diff, axis=1)
        lo, hi = max(0, i - theiler), min(n, i + theiler + 1)
        dist[lo:hi] = np.inf
        # Drop neighbors whose own target would be out of range.
        dist[max(0, len(series) - horizon_shift):] = np.inf
        neigh = np.argpartition(dist, k)[:k]
        neigh = neigh[np.isfinite(dist[neigh])]
        if neigh.size < max(4, dim // 2 + 2):
            continue
        A = np.column_stack([np.ones(neigh.size), x[neigh] - x[i]])
        y = series[neigh + horizon_shift]
        G = A.T @ A + ridge * np.eye(A.shape[1])
        try:
            w = np.linalg.solve(G, A.T @ y)
        except np.linalg.LinAlgError:
            continue
        pred_i = w[0]  # local-linear prediction at the held-out point
        residuals.append(float(target_i - pred_i))

    residuals = np.asarray(residuals, dtype=np.float64)
    if residuals.size < 30:
        return float("nan"), residuals
    # Robust scale (median absolute deviation) against heavy-tailed curvature.
    sigma_hat = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
    # Correct the held-out prediction variance inflation ~ (1 + 1/k).
    sigma_hat /= float(np.sqrt(1.0 + 1.0 / k))
    return sigma_hat, residuals


def reachable_horizon_steps(tolerance, sigma_hat, lam_per_step):
    """One-shot-law reachable horizon (steps) at initial uncertainty sigma_hat.

    Validated bridge (docs/theory/chaos_floor.md): the physical floor at
    initial error eps follows H ~ ln(tau/eps)/(lambda_1*dt); with observation
    noise, the best reachable initial-state uncertainty is ~ sigma_obs.
    Returns inf when sigma_hat is not informative (>= tolerance impossible,
    <= 0 or lambda <= 0 -> no chaotic bound).
    """
    if not np.isfinite(sigma_hat) or sigma_hat <= 0 or lam_per_step is None or lam_per_step <= 0:
        return float("inf")
    if sigma_hat >= tolerance:
        return 1.0
    return float(np.log(tolerance / sigma_hat) / lam_per_step)
