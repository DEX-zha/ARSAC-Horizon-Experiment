"""Probabilistic horizon estimation helpers."""

from __future__ import annotations

import numpy as np

from src.horizon_metrics import (
    compute_calibration_residuals,
    estimate_error_growth,
    estimate_jacobian_growth,
    estimate_local_delta,
    estimate_model_error_from_residuals,
    rolling_rmse,
    window_horizons,
)
from src.horizon_utils import estimate_expansion_quantile, horizon_from_model_bound_by_growth
from src.horizon_experiment_core import _delta_local_quantile, _horizon_time, _tolerance_from_mode


def _apply_local_delta(args, x_calib, calib_residuals, delta_local_quantile):
    if not args.delta_local:
        return None
    delta_local_q, delta_local_mean, _ = estimate_local_delta(
        x_calib,
        calib_residuals,
        k=args.delta_local_k,
        quantile=delta_local_quantile,
        max_samples=args.delta_local_samples,
        seed=args.seed,
    )
    if delta_local_q <= 0.0:
        return None
    return delta_local_q, delta_local_mean, f"local@{delta_local_quantile:.2f}", True



def _estimate_model_error(args, model, calib_std, best, delta_local_quantile):
    x_calib, calib_residuals = compute_calibration_residuals(model, calib_std, best["dim"], best["lag"])
    model_error, model_error_mode, model_error_mean = estimate_model_error_from_residuals(
        calib_residuals,
        mode=args.delta_mode,
        quantile=args.delta_quantile,
        scale=args.delta_scale,
    )
    delta_local_used = False
    override = _apply_local_delta(args, x_calib, calib_residuals, delta_local_quantile)
    if override is not None:
        model_error, model_error_mean, model_error_mode, delta_local_used = override
    return x_calib, model_error, model_error_mode, model_error_mean, delta_local_used



def _expansion_mean(ratios):
    if ratios.size:
        positive = ratios[ratios > 0]
        if positive.size:
            return float(np.exp(np.mean(np.log(positive))))
    return 1.0



def _estimate_expansion(args, exp_series, exp_dim, exp_lag):
    expansion_q, ratios = estimate_expansion_quantile(
        exp_series,
        dim=exp_dim,
        lag=exp_lag,
        quantile=args.expansion_quantile,
        theiler=args.expansion_theiler,
        max_pairs=args.expansion_samples,
        seed=args.seed,
        horizon=args.expansion_horizon,
    )
    return expansion_q, _expansion_mean(ratios)



def _growth_from_error(args, model, calib_series, best):
    growth_q, growth_mean, _ = estimate_error_growth(
        model,
        calib_series,
        best["dim"],
        best["lag"],
        horizon=args.expansion_horizon,
        max_windows=args.expansion_samples,
        quantile=args.expansion_quantile,
        seed=args.seed,
    )
    return growth_q, growth_mean



def _growth_from_jacobian(args, model, x_calib, expansion_q, expansion_mean):
    growth_q, growth_mean, _ = estimate_jacobian_growth(
        model,
        x_calib,
        quantile=args.expansion_quantile,
        max_samples=args.expansion_samples,
        seed=args.seed,
    )
    if x_calib.size == 0:
        return expansion_q, expansion_mean
    return growth_q, growth_mean



def _estimate_growth(args, model, calib_series, x_calib, best, growth_source, expansion_q, expansion_mean):
    if growth_source == "error":
        return _growth_from_error(args, model, calib_series, best)
    if growth_source == "jacobian":
        return _growth_from_jacobian(args, model, x_calib, expansion_q, expansion_mean)
    return expansion_q, expansion_mean



def _probabilistic_horizons(base_err, tolerance, model_error, model_error_mean, growth_q, growth_mean):
    h_model = horizon_from_model_bound_by_growth(growth_q, base_err, model_error, tolerance)
    h_est = horizon_from_model_bound_by_growth(growth_mean, base_err, model_error_mean, tolerance)
    return h_model, h_est



def _horizon_ratio_list(calib_horizons, calib_init_errs, growth_q, model_error, tolerance_calib):
    ratios = []
    for h_real, init_err in zip(calib_horizons, calib_init_errs):
        if init_err is None:
            continue
        h_model = horizon_from_model_bound_by_growth(growth_q, init_err, model_error, tolerance_calib)
        if h_model > 0:
            ratios.append(h_real / h_model)
    return ratios



def _calib_horizon_ratios(model, calib_series, best, args, growth_q, model_error):
    rmse_calib = rolling_rmse(model, calib_series, best["dim"], best["lag"], args.horizon_max)
    base_err_calib = rmse_calib[0] if rmse_calib.size else 0.0
    tolerance_calib = _tolerance_from_mode(base_err_calib, args)
    calib_horizons, calib_init_errs = window_horizons(
        model,
        calib_series,
        best["dim"],
        best["lag"],
        args.horizon_max,
        tolerance_calib,
    )
    ratios = _horizon_ratio_list(calib_horizons, calib_init_errs, growth_q, model_error, tolerance_calib)
    return ratios, calib_horizons, calib_init_errs, tolerance_calib



def _calibration_scale(args, ratios):
    if args.calibrate_coverage and ratios:
        scale = float(np.quantile(ratios, 1.0 - args.calibration_alpha))
        return max(scale, args.calibration_floor)
    return 1.0



def _coverage_from_ratios(ratios, calib_horizons, calib_init_errs, growth_q, model_error, tolerance_calib, scale):
    if not ratios:
        return None, 0
    hits = 0
    total = 0
    for h_real, init_err in zip(calib_horizons, calib_init_errs):
        if init_err is None:
            continue
        h_model = horizon_from_model_bound_by_growth(growth_q, init_err, model_error, tolerance_calib)
        if h_model <= 0:
            continue
        total += 1
        if h_model * scale >= h_real:
            hits += 1
    return hits / total if total > 0 else None, total



def _finalize_probabilistic_stats(stats, dt):
    stats["horizon_model_time"] = _horizon_time(stats["horizon_model_steps"], dt)
    stats["horizon_est_time"] = _horizon_time(stats["horizon_est_steps"], dt)
    stats["horizon_model_cal_time"] = _horizon_time(stats["horizon_model_cal"], dt)
    return stats



def _set_prob_stats(stats, model_error, model_error_mode, model_error_mean, delta_local_used, expansion_q, expansion_mean, growth_q, growth_mean, h_model, h_est, scale, coverage, samples, growth_source, growth_horizon):
    stats.update(
        {
            "model_error": model_error,
            "model_error_mode": model_error_mode,
            "model_error_mean": model_error_mean,
            "delta_local_used": delta_local_used,
            "expansion_q": expansion_q,
            "expansion_mean": expansion_mean,
            "growth_q": growth_q,
            "growth_mean": growth_mean,
            "growth_source": growth_source,
            "growth_horizon": growth_horizon,
            "horizon_model_steps": h_model,
            "horizon_est_steps": h_est,
            "scale": scale,
            "coverage": coverage,
            "calibration_samples": samples,
        }
    )
    stats["horizon_model_cal"] = h_model * scale



def _probabilistic_inputs(ctx, lyap):
    calib_series = ctx.data.calib_series()
    exp_dim = ctx.args.expansion_dim if ctx.args.expansion_dim is not None else lyap.dim
    exp_lag = ctx.args.expansion_lag if ctx.args.expansion_lag is not None else lyap.lag
    exp_series = np.concatenate([ctx.data.train_std, ctx.data.val_std], axis=0)
    return calib_series, exp_dim, exp_lag, exp_series



def _probabilistic_growth(ctx, calib_series, x_calib, exp_series, exp_dim, exp_lag):
    expansion_q, expansion_mean = _estimate_expansion(ctx.args, exp_series, exp_dim, exp_lag)
    growth_q, growth_mean = _estimate_growth(ctx.args, ctx.model, calib_series, x_calib, ctx.best, ctx.args.growth_source, expansion_q, expansion_mean)
    return expansion_q, expansion_mean, growth_q, growth_mean



def _probabilistic_calibration(ctx, calib_series, growth_q, model_error):
    ratios, calib_horizons, calib_init_errs, tol_calib = _calib_horizon_ratios(ctx.model, calib_series, ctx.best, ctx.args, growth_q, model_error)
    scale = _calibration_scale(ctx.args, ratios)
    coverage, samples = _coverage_from_ratios(ratios, calib_horizons, calib_init_errs, growth_q, model_error, tol_calib, scale)
    return scale, coverage, samples



def _run_probabilistic(ctx, base, lyap, stats, dt):
    calib_series, exp_dim, exp_lag, exp_series = _probabilistic_inputs(ctx, lyap)
    delta_local_quantile = _delta_local_quantile(ctx.args)
    x_calib, model_error, model_error_mode, model_error_mean, delta_local_used = _estimate_model_error(ctx.args, ctx.model, ctx.data.calib_std, ctx.best, delta_local_quantile)
    expansion_q, expansion_mean, growth_q, growth_mean = _probabilistic_growth(ctx, calib_series, x_calib, exp_series, exp_dim, exp_lag)
    h_model, h_est = _probabilistic_horizons(base.base_err, base.tolerance, model_error, model_error_mean, growth_q, growth_mean)
    scale, coverage, samples = _probabilistic_calibration(ctx, calib_series, growth_q, model_error)
    _set_prob_stats(stats, model_error, model_error_mode, model_error_mean, delta_local_used, expansion_q, expansion_mean, growth_q, growth_mean, h_model, h_est, scale, coverage, samples, ctx.args.growth_source, ctx.args.expansion_horizon)
    return _finalize_probabilistic_stats(stats, dt)


