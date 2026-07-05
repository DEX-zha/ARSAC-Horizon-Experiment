"""Censoring-aware quantile losses and gates for capped horizon labels.

Horizon labels are right-censored at ``horizon_max``: a recorded value
``H_w == Hmax`` means ``H_true >= Hmax``, not ``H_true == Hmax`` (audit
point C3). This module provides pure functions to handle that censoring:

- ``censored_pinball_loss`` (torch, differentiable) and
  ``censored_pinball_np`` (numpy, evaluation): Powell (1986) censored
  quantile regression loss ``pinball(y_rec, min(pred, cap))``. With
  ``cap=None`` both reduce exactly to the plain pinball loss, so the
  functions are drop-in replacements for ``pinball_loss`` in
  ``src/horizon_training.py``.
- ``saturation_gate``: identification check for the target alpha-quantile
  under right-censoring. The quantile ``Q_alpha(H)`` is identified from
  censored data iff ``Q_alpha(H) < Hmax``, which for a continuous label
  distribution is equivalent to ``p_sat <= 1 - alpha``.

Why Powell's transform: quantiles are equivariant under the non-decreasing
map ``t -> min(t, C)``, hence ``Q_tau(min(Y, C) | x) = min(Q_tau(Y|x), C)``.
Applying the same transform to the prediction inside the pinball loss makes
the loss a consistent estimator of ``Q_tau(Y|x)`` wherever it is identified
(``Q_tau(Y|x) < C``), and makes the loss flat (zero gradient) in the region
``pred >= cap`` where the data carry no information — instead of actively
dragging the prediction down to the cap as the naive pinball loss does.

Reference: J. L. Powell, "Censored regression quantiles", Journal of
Econometrics 32 (1986) 143-155 (left-censored form; the right-censored
form used here is the mirror image).
"""

import numpy as np
import torch

# Tolerance used to flag a label as saturated, consistent with
# ``_saturation_rate`` in src/horizon_experiment_conformal_stats.py.
SATURATION_EPS = 1e-9


def censored_pinball_loss(pred, target, quantile, cap):
    """Powell (1986) censored pinball loss (torch, differentiable).

    Args:
        pred: predictions ``q_theta(x)``, torch tensor.
        target: recorded labels ``y_rec = min(y_true, cap)``, torch tensor
            broadcastable to ``pred``.
        quantile: target quantile level ``tau`` in (0, 1).
        cap: right-censoring point ``C`` (float, or tensor broadcastable to
            ``pred`` for per-sample caps). ``None`` disables censoring and
            reduces to the plain pinball loss.

    Returns:
        Scalar torch tensor (mean loss). ``torch.minimum(pred, cap)`` has
        zero gradient w.r.t. ``pred`` wherever ``pred > cap``: censored
        regions stop pulling the prediction down, which is exactly the
        Powell estimator's flat region.
    """
    if cap is not None:
        cap_t = torch.as_tensor(cap, dtype=pred.dtype, device=pred.device)
        pred = torch.minimum(pred, cap_t)
    diff = target - pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1.0) * diff))


def censored_pinball_np(pred, target, quantile, cap):
    """Numpy twin of ``censored_pinball_loss`` for evaluation code paths."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if cap is not None:
        pred = np.minimum(pred, cap)
    diff = target - pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def saturation_gate(y, horizon_max, alpha, eps=SATURATION_EPS):
    """Checks whether the alpha-quantile is identified under censoring.

    A label ``y >= horizon_max - eps`` is counted as saturated (censored).
    For right-censored data the marginal quantile ``Q_alpha`` is identified
    iff it lies strictly below the cap, i.e. iff ``p_sat <= 1 - alpha``.

    Args:
        y: recorded labels (array-like); saturated entries equal
            ``horizon_max``.
        horizon_max: the censoring cap ``C`` (Hmax).
        alpha: miscoverage level of the target lower quantile, in (0, 1).

    Returns:
        dict with keys ``p_sat`` (float or None when ``y`` is empty),
        ``identified`` (bool) and ``message`` (str).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0:
        return {
            "p_sat": None,
            "identified": False,
            "message": "no labels: saturation rate undefined, gate closed",
        }
    p_sat = float(np.mean(y >= float(horizon_max) - eps))
    identified = bool(p_sat <= 1.0 - alpha)
    if identified:
        message = (
            f"p_sat={p_sat:.3f} <= 1-alpha={1.0 - alpha:.3f}: "
            f"Q_{alpha:g}(H) lies below Hmax={horizon_max:g}, "
            "the target quantile is identified from censored labels"
        )
    else:
        message = (
            f"p_sat={p_sat:.3f} > 1-alpha={1.0 - alpha:.3f}: "
            f"Q_{alpha:g}(H) sits in the censored region at Hmax={horizon_max:g}; "
            "increase horizon_max or lower the target quantile"
        )
    return {"p_sat": p_sat, "identified": identified, "message": message}
