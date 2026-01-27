"""Core helpers and data structures for horizon_experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import math
import os

import numpy as np
import torch
import yaml

from src.horizon_data import DataManager
from src.horizon_forecast import Forecaster
from src.horizon_metrics import evaluate_mse, horizon_from_lyapunov, horizon_from_rmse, rolling_rmse
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
    "tightness_ratio": None,
    "slack_median": None,
    "slack_p90": None,
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
    return args.dt if args.dataset != "logistic" else 1.0

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
    rmse_by_h = rolling_rmse(model, data.test_std, best["dim"], best["lag"], args.horizon_max)
    base_err = rmse_by_h[0] if rmse_by_h.size else 0.0
    tolerance = _tolerance_from_mode(base_err, args)
    horizon_real = horizon_from_rmse(rmse_by_h, tolerance)
    horizon_real_time = horizon_real * dt
    return BaseMetrics(test_mse, rmse_by_h, base_err, tolerance, horizon_real, horizon_real_time)

def _lyapunov_metrics(data, best, args, base_err, tolerance, dt):
    lyap_dim = args.lyap_dim if args.lyap_dim is not None else best["dim"]
    lyap_lag = args.lyap_lag if args.lyap_lag is not None else best["lag"]
    series = np.concatenate([data.train_raw, data.val_raw])
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

def _delta_local_quantile(args):
    return args.delta_local_quantile if args.delta_local_quantile is not None else args.delta_quantile

def _experiment_setup(args):
    constants = _load_constants(_constants_path(args))
    set_seed(args.seed)
    device = _get_device(args)
    data = _load_data(args)
    model, best = _train_forecaster(args, device, data)
    dt = _resolve_dt(args)
    base = _base_metrics(model, data, best, args, device, dt)
    lyap = _lyapunov_metrics(data, best, args, base.base_err, base.tolerance, dt)
    stats = _init_stats(args)
    ctx = ExperimentContext(args=args, constants=constants, device=device, data=data, model=model, best=best)
    exp_dim = args.expansion_dim if args.expansion_dim is not None else lyap.dim
    exp_lag = args.expansion_lag if args.expansion_lag is not None else lyap.lag
    return ctx, base, lyap, stats, exp_dim, exp_lag, dt

