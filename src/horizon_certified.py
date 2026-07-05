"""Certified (non-statistical) horizon bounds from model Lipschitz constants.

Setting: the forecaster is a scalar one-step map ``f`` applied
autoregressively in a delay embedding. One rollout step is the shift map

    F(x) = (x_2, ..., x_d, f(x)).

With the sup norm, ``||F(x) - F(y)||_inf <= max(1, L_inf(f)) ||x - y||_inf``
where ``L_inf(f)`` is the Lipschitz constant of ``f`` with respect to the
sup norm (equal to ``sup_x ||grad f(x)||_1``). If the one-step residual of
the model on TRUE states is bounded by ``delta``, the embedded rollout
error obeys the recursion

    e_{h+1} <= G * e_h + delta,   G = max(1, L_inf(f)),   e_0 = 0,

hence the closed form ``e_h <= delta * (G^h - 1) / (G - 1)`` (``h * delta``
when G == 1). The certified horizon ``h_cert`` is the first 1-indexed step
where this bound can reach the tolerance: every per-window horizon label
H_w (first step whose absolute error reaches the tolerance) satisfies
``H_w >= h_cert`` for EVERY window -- no distributional assumption, no
1-alpha exception set.

Honest caveat (the weak link): ``delta`` is an EMPIRICAL sup of one-step
residuals on train+val+calib data. The certificate is therefore "certified
modulo the residual bound holding on the attractor": if a test state has a
one-step residual above ``delta``, the guarantee can break there. This is
still qualitatively different from a quantile/conformal bound (which by
construction fails on an alpha-fraction of windows even in-distribution).

Lipschitz constants used here are UPPER bounds computed as products of
per-layer operator norms (Szegedy et al. 2014; Virmaux & Scaman 2018).
They are exact for LinearAR (constant gradient) and may be loose for deep
networks; our MLP has 2 hidden layers with Tanh activations (1-Lipschitz).
"""

import math

import numpy as np
import torch
import torch.nn as nn

from src.horizon_models import LinearAR, TorchWrapper
from src.horizon_utils import build_supervised, horizon_from_model_bound_by_growth

# Element-wise activations with Lipschitz constant <= 1 (exactly 1 for
# Tanh/ReLU). They cannot increase either the l_inf or the l_2 bound.
_UNIT_LIPSCHITZ = (nn.Tanh, nn.ReLU, nn.Identity, nn.Flatten)


def _unwrap_torch(model):
    """Returns the underlying nn.Module of a wrapper, or None."""
    if isinstance(model, TorchWrapper):
        return model.model
    if isinstance(model, nn.Module):
        return model
    return None


def _linear_layers(module):
    """Collects nn.Linear layers of a feed-forward chain, in order.

    Only plain sequential compositions are supported: nn.Linear layers,
    1-Lipschitz element-wise activations, and nn.Sequential containers.
    Anything else (LSTM, residual blocks, ...) raises ValueError because
    the layer-product bound would not be valid for them.
    """
    layers = []
    for child in module.children():
        if isinstance(child, nn.Linear):
            layers.append(child)
        elif isinstance(child, _UNIT_LIPSCHITZ):
            continue
        elif isinstance(child, nn.Sequential):
            layers.extend(_linear_layers(child))
        else:
            raise ValueError(
                "Unsupported layer for Lipschitz product bound: "
                f"{type(child).__name__}"
            )
    return layers


def _feedforward_layers(model):
    """Extracts and validates the Linear layers of a wrapped feed-forward net."""
    module = _unwrap_torch(model)
    if module is None:
        raise ValueError(
            f"Unsupported model type for Lipschitz bound: {type(model).__name__}"
        )
    layers = _linear_layers(module)
    if not layers:
        raise ValueError("No nn.Linear layer found in model")
    return layers


def lipschitz_linf(model, input_dim=None):
    """Upper bound on the Lipschitz constant of the scalar map wrt sup norm.

    LinearAR: exact global value ``||w||_1`` (constant gradient), where
    ``w`` excludes the bias term. Feed-forward torch nets: product over
    layers of the matrix inf-operator norm (max absolute row sum);
    activations must be 1-Lipschitz (Tanh here). The product is an upper
    bound that can be loose for deep nets.
    """
    if isinstance(model, LinearAR):
        if model.weights is None:
            raise ValueError("LinearAR model is not fitted")
        w = np.asarray(model.weights[:-1], dtype=np.float64)
        if input_dim is not None and w.shape[0] != int(input_dim):
            raise ValueError(
                f"input_dim mismatch: weights have {w.shape[0]}, got {input_dim}"
            )
        return float(np.sum(np.abs(w)))

    layers = _feedforward_layers(model)
    if input_dim is not None and layers[0].in_features != int(input_dim):
        raise ValueError(
            f"input_dim mismatch: first layer expects {layers[0].in_features}, "
            f"got {input_dim}"
        )
    bound = 1.0
    with torch.no_grad():
        for layer in layers:
            # inf-operator norm: max over rows of the l1 norm of the row.
            bound *= float(layer.weight.abs().sum(dim=1).max().item())
    return bound


def lipschitz_l2(model, input_dim=None):
    """Upper bound on the Lipschitz constant wrt the Euclidean norm.

    LinearAR: exact ``||w||_2``. Feed-forward torch nets: product of the
    spectral norms (largest singular values) of the weight matrices.
    Provided for comparison with :func:`lipschitz_linf`; the certified
    recursion uses the sup-norm constant.
    """
    if isinstance(model, LinearAR):
        if model.weights is None:
            raise ValueError("LinearAR model is not fitted")
        w = np.asarray(model.weights[:-1], dtype=np.float64)
        if input_dim is not None and w.shape[0] != int(input_dim):
            raise ValueError(
                f"input_dim mismatch: weights have {w.shape[0]}, got {input_dim}"
            )
        return float(np.linalg.norm(w))

    layers = _feedforward_layers(model)
    if input_dim is not None and layers[0].in_features != int(input_dim):
        raise ValueError(
            f"input_dim mismatch: first layer expects {layers[0].in_features}, "
            f"got {input_dim}"
        )
    bound = 1.0
    with torch.no_grad():
        for layer in layers:
            svals = torch.linalg.svdvals(layer.weight.double())
            bound *= float(svals[0].item())
    return bound


def _predict_all(model, x):
    """Batch one-step predictions as a flat float64 array."""
    if hasattr(model, "predict_batch"):
        preds = model.predict_batch(x)
    else:
        preds = np.array([model.predict(v) for v in x], dtype=np.float64)
    return np.asarray(preds, dtype=np.float64).reshape(-1)


def empirical_delta_sup(model, series, dim, lag):
    """Max absolute one-step residual of the model on true states.

    ``series`` is a 1-D array or a sequence of disjoint 1-D segments; each
    segment is processed separately so no fake window straddles a segment
    junction (audit E2). Returns 0.0 when no supervised pair can be built.
    """
    if isinstance(series, (list, tuple)):
        segments = [np.asarray(s, dtype=np.float64) for s in series]
    else:
        segments = [np.asarray(series, dtype=np.float64)]

    delta = 0.0
    found = False
    for segment in segments:
        try:
            x, y = build_supervised(segment, dim, lag, horizon=1)
        except ValueError:
            continue
        if x.size == 0:
            continue
        preds = _predict_all(model, x)
        delta = max(delta, float(np.max(np.abs(preds - y))))
        found = True
    return delta if found else 0.0


def certified_horizon(model, series_std, dim, lag, tolerance):
    """Assembles the certified horizon; returns ``(h_cert, G, delta)``.

    ``h_cert`` is the first 1-indexed rollout step at which the closed-form
    bound ``delta * (G^h - 1) / (G - 1)`` can reach ``tolerance``. As long
    as ``delta`` bounds the one-step residual on the states visited,
    every per-window horizon label satisfies ``H_w >= h_cert``.

    The algebra is delegated to ``horizon_from_model_bound_by_growth``
    with ``init_err = delta`` (bound on the FIRST-step error e_1); the
    function counts steps after that first one, hence the ``+ 1`` to
    convert to the 1-indexed step convention of the H_w labels.
    Returns ``h_cert = inf`` when the bound never crosses (delta == 0).
    """
    growth = max(1.0, float(lipschitz_linf(model, input_dim=dim)))
    delta = empirical_delta_sup(model, series_std, dim, lag)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        return 0.0, growth, delta
    if delta <= 0.0:
        return float("inf"), growth, delta
    if delta >= tolerance:
        return 1.0, growth, delta
    steps_after_first = horizon_from_model_bound_by_growth(
        growth, delta, delta, tolerance
    )
    if math.isinf(steps_after_first):
        return float("inf"), growth, delta
    return float(steps_after_first) + 1.0, growth, delta
