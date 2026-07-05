"""Core helpers and data structures for horizon_experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging
import math
import os

import numpy as np
import torch
import yaml

from src.horizon_certified import certified_horizon
from src.horizon_data import DataManager
from src.horizon_embedding import select_embedding
from src.horizon_forecast import Forecaster
from src.horizon_metrics import (
    compute_calibration_residuals,
    evaluate_mse,
    horizon_from_lyapunov,
    horizon_from_rmse,
    rolling_rmse,
)
from src.horizon_utils import build_supervised, estimate_lyapunov, set_seed

DEFAULT_CONSTANTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "experiment_constants.yaml")
)

DEFAULT_STATS: Dict[str, Any] = {
    "horizon_window_median": None,
    "horizon_window_mean": None,
    "model_error": 0.0,
    "model_error_mode": "none",
    "model_error_mean": 0.0,
    "delta_local_used": False,
    "expansion_q": 1.0,
    "expansion_mean": 1.0,
    "growth_q": 1.0,
    "growth_mean": 1.0,
    "growth_source": None,
    "growth_horizon": None,
    "scale": 1.0,
    "coverage": None,
    "calibration_samples": 0,
    "horizon_model_steps": 0.0,
    "horizon_model_time": 0.0,
    "horizon_est_steps": 0.0,
    "horizon_est_time": 0.0,
    "horizon_model_cal": 0.0,
    "horizon_model_cal_time": 0.0,
    "coverage_test": None,
    "coverage_hits": None,
    "test_samples": None,
    "tightness_ratio": None,
    "slack_median": None,
    "slack_p90": None,
    "p_sat_calib": None,
    "p_sat_test": None,
    "leaf_coverage_stats": None,
    "jac_quantile_coverages": None,
    "score_pos_frac": None,
    "score_neg_frac": None,
    "score_zero_frac": None,
    "score_p10": None,
    "score_p50": None,
    "score_p90": None,
    "score_mean": None,
    "signed_med": None,
    "sigma_med": None,
    "sigma_p90": None,
    "sigma_max": None,
    "pred_calib_med": None,
    "y_calib_med": None,
    "l_calib_med": None,
    "bin_count": None,
    "bin_min_count": None,
    "bin_med_count": None,
    "bin_c_min": None,
    "bin_c_med": None,
    "bin_c_max": None,
    "c_global": 0.0,
    "coverage_guard": 0.0,
    "debias_delta": 0.0,
    "debias_quantile": None,
    "predictability_corr_jac": None,
    "predictability_corr_resid": None,
    "label_identified": None,
    "horizon_certified": 0.0,
    "lipschitz_G": 0.0,
    "delta_sup": 0.0,
}

@dataclass
class DataSplits:
    train_std: np.ndarray
    val_std: np.ndarray
    calib_std: np.ndarray
    test_std: np.ndarray
    train_raw: np.ndarray
    val_raw: np.ndarray
    calib_raw: np.ndarray
    test_raw: np.ndarray

    def calib_series(self):
        return self.calib_std if self.calib_std.size else self.val_std

@dataclass
class BaseMetrics:
    test_mse: float
    rmse_by_h: np.ndarray
    base_err: float
    tolerance: float
    horizon_real: float
    horizon_real_time: float

@dataclass
class LyapunovMetrics:
    step: float
    time: float
    dim: int
    lag: int
    horizon_theory: float
    horizon_theory_time: float

@dataclass
class HorizonSets:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_calib: np.ndarray
    y_calib: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_train_raw: np.ndarray
    x_calib_raw: np.ndarray
    x_test_raw: np.ndarray

@dataclass
class Predictions:
    pred_calib: np.ndarray
    pred_test: np.ndarray
    sigma_calib: np.ndarray
    sigma_test: np.ndarray

@dataclass
class ExperimentContext:
    args: Any
    constants: Dict[str, Any]
    device: torch.device
    data: DataSplits
    model: Any
    best: Dict[str, Any]

@dataclass
class ConformalModel:
    mode: str
    c_global: float
    tree: Optional[Any] = None
    bin_model: Optional[Dict[str, Any]] = None

def _constants_path(args):
    override = getattr(args, "constants_config", None)
    return os.path.abspath(override or DEFAULT_CONSTANTS_PATH)

def _load_constants(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing constants config: {path}")
    with open(path, "r") as f:
        constants = yaml.safe_load(f) or {}
    return constants

def _const(constants, *keys):
    value = constants
    for key in keys:
        value = value[key]
    return value

def _seed_offset(seed, constants, key):
    return seed + int(_const(constants, "seed_offsets", key))

def _seed_with_base(seed, constants, base_key, fold_idx=0):
    base = int(_const(constants, "seed_bases", base_key))
    stride = int(_const(constants, "seed_strides", "cv_fold"))
    return seed + base + fold_idx * stride

def _resolve_dt(args):
    from src.horizon_cli import resolve_dt

    return resolve_dt(args.dataset, args.dt)

def _horizon_time(steps, dt):
    return steps * dt if math.isfinite(steps) else float("inf")

def _tolerance_from_mode(base_err, args):
    return base_err * args.error_factor if args.error_mode == "relative" else args.error_tolerance

def _clamp01(value):
    return float(min(max(value, 0.0), 1.0))

def _clip_horizon(values, args, constants):
    if not values.size:
        return values
    lo = float(_const(constants, "horizon_min"))
    return np.clip(values, lo, float(args.horizon_max))

def _get_device(args):
    return torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")

def _load_data(args):
    data_manager = DataManager(args)
    train_std, val_std, calib_std, test_std = data_manager.prepare_data()
    train_raw, val_raw, calib_raw, test_raw = data_manager.get_raw_splits()
    return DataSplits(train_std, val_std, calib_std, test_std, train_raw, val_raw, calib_raw, test_raw)

def _train_forecaster(args, device, data):
    forecaster = Forecaster(args, device)
    best = forecaster.select_embedding(data.train_std, data.val_std)
    model = forecaster.train_final_model(data.train_std, data.val_std)
    return model, best

def _test_mse(model, test_std, best, device):
    x_test, y_test = build_supervised(test_std, best["dim"], best["lag"], horizon=1)
    return evaluate_mse(model, x_test, y_test, device=device)

def _base_metrics(model, data, best, args, device, dt):
    test_mse = _test_mse(model, data.test_std, best, device)
    rmse_by_h = rolling_rmse(
        model, data.test_std, best["dim"], best["lag"], args.horizon_max,
        max_windows=800, seed=args.seed,
    )
    base_err = rmse_by_h[0] if rmse_by_h.size else 0.0
    tolerance = _tolerance_from_mode(base_err, args)
    horizon_real = horizon_from_rmse(rmse_by_h, tolerance)
    horizon_real_time = horizon_real * dt
    return BaseMetrics(test_mse, rmse_by_h, base_err, tolerance, horizon_real, horizon_real_time)

def _lyapunov_metrics(data, best, args, base_err, tolerance, dt):
    series = np.concatenate([data.train_raw, data.val_raw])
    lyap_dim = args.lyap_dim
    lyap_lag = args.lyap_lag
    if lyap_dim is None and lyap_lag is None:
        # Theory-grounded embedding for the chaos estimators (Plan V2 Phase 1):
        # lag from the first mutual-information minimum, dim from false
        # nearest neighbors. The forecaster keeps its val-MSE embedding;
        # explicit --lyap-dim/--lyap-lag still win (handled above).
        try:
            emb = select_embedding(series)
            lyap_dim = int(emb["dim"])
            lyap_lag = int(emb["lag"])
        except Exception as exc:  # never crash a run on a diagnostic
            logging.warning("select_embedding failed (%s); falling back to model embedding", exc)
    if lyap_dim is None:
        lyap_dim = best["dim"]
    if lyap_lag is None:
        lyap_lag = best["lag"]
    lyap_step, _ = estimate_lyapunov(
        series,
        dim=lyap_dim,
        lag=lyap_lag,
        max_t=args.lyap_max_t,
        theiler=args.lyap_theiler,
        fit_start=args.lyap_fit_start,
        fit_end=args.lyap_fit_end,
        dt=dt,
    )
    lyap_time = lyap_step / dt if dt > 0 else 0.0
    horizon_theory = horizon_from_lyapunov(lyap_step, base_err, tolerance)
    horizon_theory_time = _horizon_time(horizon_theory, dt)
    return LyapunovMetrics(lyap_step, lyap_time, lyap_dim, lyap_lag, horizon_theory, horizon_theory_time)

def _init_stats(args):
    stats = DEFAULT_STATS.copy()
    stats["growth_source"] = args.growth_source
    stats["growth_horizon"] = args.expansion_horizon
    return stats

def _certified_stats(model, data, best, tolerance, stats):
    """Exports the certified (non-statistical) horizon diagnostic (study P4).

    h_cert = first 1-indexed step where delta*(G^h-1)/(G-1) can reach the
    tolerance; every window label satisfies H_w >= h_cert as long as delta
    bounds the one-step residual. Diagnostic only: the conformal L(x) stays
    the operational bound. Exceptions (e.g. LSTM, where the layer-product
    Lipschitz bound is invalid) fall back to 0.0 with a warning.
    """
    try:
        h_cert, growth, delta = certified_horizon(
            model, data.calib_series(), best["dim"], best["lag"], tolerance
        )
        stats["horizon_certified"] = float(h_cert)
        stats["lipschitz_G"] = float(growth)
        stats["delta_sup"] = float(delta)
    except Exception as exc:
        logging.warning("certified horizon unavailable (%s); exporting 0.0", exc)
        stats["horizon_certified"] = 0.0
        stats["lipschitz_G"] = 0.0
        stats["delta_sup"] = 0.0
    return stats

def _delta_local_quantile(args):
    return args.delta_local_quantile if args.delta_local_quantile is not None else args.delta_quantile

def _resolve_horizon_max_for_run(args, data, best, dt, model):
    """Resolves horizon_max in Lyapunov times (audit A3) with a data budget.

    Auto mode (args.horizon_max is None) targets horizon_lyap_factor
    Lyapunov times, then clamps so that the test and calibration splits
    still contain enough rollout windows. Explicit values are respected
    unchanged (legacy behavior).
    """
    from src.horizon_cli import DEFAULT_LAMBDA, resolve_horizon_max

    resolved, target = resolve_horizon_max(
        args.dataset, dt, args.horizon_max, getattr(args, "horizon_lyap_factor", 3.0)
    )
    if args.horizon_max is not None:
        return int(args.horizon_max)
    # Refine the Lyapunov-times target with the model's actual one-step error:
    # a good model can be trusted beyond 3 T_lambda before reaching tolerance
    # (H_theory ~ ln(tol/e0)/lambda), so take max(3 T_lambda, 1.2 * H_theory),
    # still capped at 400 steps for rollout cost.
    lam = DEFAULT_LAMBDA.get(args.dataset, 0.0)
    if lam > 0.0 and dt > 0.0:
        _, residuals = compute_calibration_residuals(
            model, data.calib_series(), best["dim"], best["lag"]
        )
        if residuals.size:
            e0 = float(np.sqrt(np.mean(residuals**2)))
            tol_est = (
                e0 * args.error_factor
                if args.error_mode == "relative"
                else float(args.error_tolerance)
            )
            if e0 > 0.0 and tol_est > e0:
                h_theory = math.log(tol_est / e0) / (lam * dt)
                refined = int(math.ceil(1.2 * h_theory))
                if refined > resolved:
                    target = max(target or 0, refined)
                    resolved = min(refined, 400)
    window_len = (best["dim"] - 1) * best["lag"] + 1
    budget = min(len(data.test_std), len(data.calib_series())) - window_len - 10
    clamped = int(max(10, min(resolved, max(10, budget // 2))))
    if target is not None and clamped < target:
        logging.warning(
            "horizon_max auto: %d steps < target %d (%.1f Lyapunov times wanted); "
            "horizons beyond %d are right-censored — increase series_len or "
            "test/calib ratios for a chaos-limited horizon study.",
            clamped, target, getattr(args, "horizon_lyap_factor", 3.0), clamped,
        )
    else:
        logging.info("horizon_max auto-resolved to %d steps (~%.1f Lyapunov times)",
                     clamped, clamped * dt * DEFAULT_LAMBDA.get(args.dataset, 0.0))
    return clamped


def _experiment_setup(args):
    constants = _load_constants(_constants_path(args))
    set_seed(args.seed)
    device = _get_device(args)
    data = _load_data(args)
    model, best = _train_forecaster(args, device, data)
    dt = _resolve_dt(args)
    args.horizon_max = _resolve_horizon_max_for_run(args, data, best, dt, model)
    base = _base_metrics(model, data, best, args, device, dt)
    lyap = _lyapunov_metrics(data, best, args, base.base_err, base.tolerance, dt)
    stats = _init_stats(args)
    stats = _certified_stats(model, data, best, base.tolerance, stats)
    ctx = ExperimentContext(args=args, constants=constants, device=device, data=data, model=model, best=best)
    exp_dim = args.expansion_dim if args.expansion_dim is not None else lyap.dim
    exp_lag = args.expansion_lag if args.expansion_lag is not None else lyap.lag
    return ctx, base, lyap, stats, exp_dim, exp_lag, dt

