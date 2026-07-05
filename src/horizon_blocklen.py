"""Automatic block-length selection for block bootstrap (Politis-White).

Implements the automatic block-length rule of Politis & White (2004),
"Automatic block-length selection for the dependent bootstrap",
Econometric Reviews 23(1), with the correction of Patton, Politis & White
(2009), Econometric Reviews 28(4).

The estimator targets the optimal (MSE-minimising) block length for the
bootstrap of the sample mean of a stationary series:

    b_opt = ceil((2 * G_hat^2 / D_hat)^(1/3) * n^(1/3))

where G_hat and D_hat are flat-top lag-window estimates built from the
sample autocovariances (see ``politis_white_block_length``).

All functions are pure numpy, deterministic, and dependency-free.
"""

from __future__ import annotations

import numpy as np

__all__ = ["flat_top_lambda", "sample_autocovariances", "politis_white_block_length"]

_MIN_BLOCK_LEN = 10


def flat_top_lambda(t):
    """Trapezoidal flat-top lag-window kernel of Politis & Romano (1995).

    lambda(t) = 1            for |t| <= 1/2
              = 2 * (1-|t|)  for 1/2 < |t| <= 1
              = 0            otherwise

    Accepts scalars or arrays; returns a float64 array (0-d for scalars).
    """
    t = np.abs(np.asarray(t, dtype=np.float64))
    return np.where(t <= 0.5, 1.0, np.where(t <= 1.0, 2.0 * (1.0 - t), 0.0))


def sample_autocovariances(x, max_lag):
    """Biased (1/n) sample autocovariances gamma_0..gamma_max_lag.

    gamma_k = (1/n) * sum_{t=0}^{n-k-1} (x_t - xbar) * (x_{t+k} - xbar)

    The 1/n normalisation (rather than 1/(n-k)) is the standard choice for
    spectral estimation and is what Politis-White assumes.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n == 0:
        raise ValueError("empty series")
    max_lag = int(max_lag)
    if max_lag < 0 or max_lag > n - 1:
        raise ValueError(f"max_lag must be in [0, {n - 1}], got {max_lag}")
    xc = x - x.mean()
    gamma = np.empty(max_lag + 1, dtype=np.float64)
    for k in range(max_lag + 1):
        gamma[k] = np.dot(xc[: n - k], xc[k:]) / n
    return gamma


def politis_white_block_length(x, c=2.0):
    """Automatic block length for the moving-block/stationary bootstrap.

    Politis & White (2004) rule with the Patton-Politis-White (2009)
    correction, as specified for this project:

    1. rho_k = gamma_k / gamma_0 (sample autocorrelations).
    2. m_hat = smallest lag m >= 0 such that |rho_k| < c*sqrt(log10(n)/n)
       for K_n consecutive lags k = m+1..m+K_n, with c = 2 and
       K_n = max(5, ceil(sqrt(log10(n)))). If no such m exists within the
       search range m <= ceil(sqrt(n)) + K_n, fall back to the largest lag
       with a significant autocorrelation.
    3. M = 2 * m_hat (bandwidth of the flat-top window, capped at n-1).
    4. G_hat = sum_{|k| <= M} lambda(k/M) * |k| * gamma_k
       D_hat = 2 * (sum_{|k| <= M} lambda(k/M) * gamma_k)^2
       with lambda the trapezoidal flat-top kernel (``flat_top_lambda``).
    5. b_opt = ceil((2 * G_hat^2 / D_hat)^(1/3) * n^(1/3)).
    6. Clamp to [10, n // 3].

    Degenerate inputs (constant series, n < 2, zero variance, M = 0,
    non-finite intermediate values) return the minimum block length 10.

    Parameters
    ----------
    x : array-like
        Stationary series (e.g. a binary coverage-hit sequence).
    c : float
        Significance constant of the m_hat rule (default 2.0).

    Returns
    -------
    int
        Selected block length, in [10, max(10, n // 3)].
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2 or not np.all(np.isfinite(x)):
        return _MIN_BLOCK_LEN

    xc = x - x.mean()
    gamma0 = float(np.dot(xc, xc)) / n
    # Constant (or numerically constant) series: compare the variance to the
    # squared scale of the data so float rounding residuals do not register
    # as spuriously perfectly-correlated noise.
    scale2 = max(1.0, float(np.mean(x * x)))
    if not np.isfinite(gamma0) or gamma0 <= 1e-14 * scale2:
        return _MIN_BLOCK_LEN

    log10_n = np.log10(n)
    k_n = int(max(5, np.ceil(np.sqrt(log10_n))))
    threshold = float(c) * np.sqrt(log10_n / n)
    m_max = int(np.ceil(np.sqrt(n))) + k_n

    # Autocovariances up to the largest lag we may need: the m_hat search
    # inspects lags up to m_max + k_n and the window uses lags up to
    # M = 2*m_hat <= 2*m_max. Everything is capped at n-1.
    lag_cap = min(n - 1, max(2 * m_max, m_max + k_n))
    gamma = sample_autocovariances(x, lag_cap)
    rho = gamma[1:] / gamma[0]
    insignificant = np.abs(rho) < threshold  # index j -> lag j+1

    m_hat = None
    for m in range(0, m_max + 1):
        window = insignificant[m : m + k_n]
        if window.size == k_n and bool(np.all(window)):
            m_hat = m
            break
    if m_hat is None:
        significant = np.nonzero(~insignificant[:m_max])[0]
        m_hat = int(significant[-1]) + 1 if significant.size else 1

    big_m = min(2 * m_hat, n - 1)
    if big_m < 1:
        # No detectable correlation: any short block is fine.
        return _MIN_BLOCK_LEN

    lags = np.arange(1, big_m + 1)
    lam = flat_top_lambda(lags / big_m)
    gamma_pos = gamma[1 : big_m + 1]
    # Symmetric sums over |k| <= M; the k = 0 term of G_hat vanishes.
    g_hat = 2.0 * float(np.sum(lam * lags * gamma_pos))
    spectral0 = gamma[0] + 2.0 * float(np.sum(lam * gamma_pos))
    d_hat = 2.0 * spectral0 * spectral0

    if not np.isfinite(g_hat) or not np.isfinite(d_hat) or d_hat <= 0.0:
        return _MIN_BLOCK_LEN

    b_opt = float(np.ceil((2.0 * g_hat * g_hat / d_hat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)))
    lower = _MIN_BLOCK_LEN
    upper = max(lower, n // 3)
    return int(min(max(b_opt, lower), upper))
