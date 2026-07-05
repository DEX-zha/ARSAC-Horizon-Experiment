"""Conformal prediction beyond exchangeability (Barber et al. 2023) and
mixing-based thinning utilities for temporally dependent calibration scores.

Context
-------
The split-conformal guarantee P(score_test <= c) >= 1 - alpha requires the
calibration and test scores to be exchangeable (Vovk et al. 2005; Lei et al.
2018).  Horizon labels computed on overlapping windows of a single trajectory
are serially correlated, which violates exchangeability, so the guarantee of
the current pipeline is empirical only (audit item E1).

This module provides pure numpy functions implementing two theoretically
grounded repairs:

1. Weighted conformal quantiles with FIXED (data-independent) weights
   (Barber, Candes, Ramdas, Tibshirani, "Conformal prediction beyond
   exchangeability", Annals of Statistics 2023).  With normalized weights
   w_tilde_i = w_i / (sum_j w_j + 1), coverage satisfies
       P(score_test <= c) >= 1 - alpha - sum_i w_tilde_i * d_TV(Z_i, Z_test),
   so decaying weights bound the coverage loss under distribution drift.

2. Thinning to (near-)disjoint calibration windows: scores separated by a
   gap of at least one decorrelation length are approximately exchangeable
   (approximate-validity arguments in the style of Chernozhukov, Wuthrich,
   Zhu 2018), restoring near-nominal seed-level coverage at the cost of a
   smaller effective n.

All functions are deterministic (no RNG) and depend only on numpy.
"""

import numpy as np

__all__ = [
    "weighted_conformal_quantile",
    "decay_weights",
    "disjoint_indices",
    "coverage_gap_bound",
]


def weighted_conformal_quantile(scores, alpha, weights=None):
    """Weighted conformal quantile of Barber et al. (2023).

    Computes the level-(1 - alpha) quantile of the discrete distribution

        sum_i w_tilde_i * delta_{s_i}  +  w_tilde_{n+1} * delta_{+inf},

    where the test point carries weight w_{n+1} = 1 placed at +infinity
    (the standard conservative convention: the unseen test score is treated
    as the worst case) and w_tilde_i = w_i / (sum_j w_j + 1).

    The returned value is the smallest sorted score s_(k) whose cumulative
    normalized weight reaches 1 - alpha.  If even the total calibration mass
    sum_i w_tilde_i is below 1 - alpha, the quantile falls on the +infinity
    atom and the function returns numpy.inf (an uninformative bound - the
    honest answer when n is too small for the requested alpha).

    With uniform weights this reduces exactly to the classical split-conformal
    quantile at rank ceil((n + 1) * (1 - alpha)), except that the classical
    helper in ``horizon_conformal.conformal_quantile`` clamps the rank to the
    maximum score instead of returning +inf.

    Parameters
    ----------
    scores : array-like of float
        Calibration nonconformity scores. Non-finite entries are dropped
        together with their weights.
    alpha : float
        Miscoverage level in (0, 1).
    weights : array-like of float, optional
        Non-negative fixed weights, same length as ``scores``, most recent
        score LAST by project convention.  Defaults to uniform weights.

    Returns
    -------
    float
        The weighted conformal quantile (possibly numpy.inf).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if weights is None:
        weights = np.ones_like(scores)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.shape != scores.shape:
        raise ValueError(
            f"weights shape {weights.shape} != scores shape {scores.shape}"
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    mask = np.isfinite(scores)
    scores = scores[mask]
    weights = weights[mask]
    if scores.size == 0:
        # Only the +infinity test mass remains.
        return float(np.inf)
    total = float(weights.sum()) + 1.0  # +1.0 = test-point mass at +infinity
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    cum = np.cumsum(weights[order]) / total
    target = 1.0 - float(alpha)
    if target <= 0.0:
        return float(sorted_scores[0])
    # Tiny tolerance so that exact-rational targets (e.g. 0.9 with n = 9
    # uniform weights) are not missed through floating-point rounding.
    idx = int(np.searchsorted(cum, target - 1e-12, side="left"))
    if idx >= sorted_scores.size:
        return float(np.inf)
    return float(sorted_scores[idx])


def decay_weights(n, half_life):
    """Exponentially decaying fixed weights, most recent LAST.

    w_i = 0.5 ** ((n - 1 - i) / half_life) for i = 0..n-1, so the most recent
    calibration score (index n - 1) has weight 1 and a score ``half_life``
    steps older has weight 0.5.  These are valid FIXED weights in the sense of
    Barber et al. (2023): they do not depend on the data values.

    Parameters
    ----------
    n : int
        Number of calibration scores (>= 0).
    half_life : float
        Positive half-life in number of scores. ``numpy.inf`` yields uniform
        weights.

    Returns
    -------
    numpy.ndarray of shape (n,)
    """
    n = int(n)
    if n < 0:
        raise ValueError("n must be >= 0")
    half_life = float(half_life)
    if not half_life > 0.0:
        raise ValueError("half_life must be > 0")
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)  # age of each index i
    return np.power(0.5, ages / half_life)


def disjoint_indices(n_windows, gap):
    """Indices 0, gap, 2*gap, ... < n_windows for disjoint calibration windows.

    Thinning overlapping horizon windows with a gap of at least the window
    footprint (or the score decorrelation length) makes the retained scores
    approximately exchangeable, at the cost of fewer calibration points.

    Parameters
    ----------
    n_windows : int
        Total number of available (overlapping) windows.
    gap : int
        Stride between retained windows (>= 1).

    Returns
    -------
    numpy.ndarray of int
    """
    n_windows = int(n_windows)
    gap = int(gap)
    if n_windows < 0:
        raise ValueError("n_windows must be >= 0")
    if gap < 1:
        raise ValueError("gap must be >= 1")
    return np.arange(0, n_windows, gap, dtype=np.int64)


def coverage_gap_bound(weights, dtv=None):
    """Coverage-loss term of the Barber et al. (2023) bound, for reporting.

    The beyond-exchangeability guarantee is

        coverage >= 1 - alpha - sum_i w_tilde_i * d_TV(Z_i, Z_test),

    with w_tilde_i = w_i / (sum_j w_j + 1).  This function returns the loss
    term.  If per-point total-variation distances ``dtv`` are unknown, it
    returns sum_i w_tilde_i, i.e. the multiplier of a uniform d_TV bound
    (worst case d_TV = 1); reported coverage loss is then
    ``coverage_gap_bound(w) * max_i d_TV``.

    Parameters
    ----------
    weights : array-like of float
        Non-negative fixed calibration weights.
    dtv : array-like of float, optional
        Per-point total-variation distances d_TV(Z_i, Z_test) in [0, 1],
        same length as ``weights``.

    Returns
    -------
    float
        sum_i w_tilde_i * dtv_i (with dtv_i = 1 when ``dtv`` is None).
    """
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if weights.size == 0:
        return 0.0
    total = float(weights.sum()) + 1.0
    w_tilde = weights / total
    if dtv is None:
        return float(w_tilde.sum())
    dtv = np.asarray(dtv, dtype=np.float64).reshape(-1)
    if dtv.shape != weights.shape:
        raise ValueError(f"dtv shape {dtv.shape} != weights shape {weights.shape}")
    if np.any(dtv < 0.0) or np.any(dtv > 1.0):
        raise ValueError("dtv entries must lie in [0, 1]")
    return float(np.sum(w_tilde * dtv))
