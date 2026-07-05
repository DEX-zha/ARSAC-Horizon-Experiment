"""Conformal data prep and prediction helpers for horizon_experiment."""

from __future__ import annotations

import numpy as np

from src.horizon_conformal import (
    make_contiguous_folds,
    predict_quantile_ensemble,
    predict_sigma_mlp,
    predict_sigma_quantile_ensemble,
)
from src.horizon_metrics import build_horizon_dataset
from src.horizon_training import quantile_cap
from src.horizon_experiment_core import (
    HorizonSets,
    Predictions,
    _clamp01,
    _clip_horizon,
    _seed_offset,
    _seed_with_base,
)


def _censoring_cap(args):
    """Right-censoring cap for quantile training: horizon_max when enabled."""
    if getattr(args, "censored_quantile", False):
        return float(args.horizon_max)
    return None


def _horizon_dataset(args, model, series, best, seed, stride):
    return build_horizon_dataset(
        model,
        series,
        best["dim"],
        best["lag"],
        args.horizon_max,
        args.error_tolerance,
        max_windows=args.horizon_samples,
        seed=seed,
        use_jacobian=args.horizon_use_jacobian,
        error_mode=args.error_mode,
        error_factor=args.error_factor,
        consecutive_k=args.horizon_consecutive_k,
        stride=stride,
        feature_horizon=args.horizon_feature_horizon,
    )



def _dataset_with_offset(ctx, series, offset_key, stride):
    seed = _seed_offset(ctx.args.seed, ctx.constants, offset_key)
    return _horizon_dataset(ctx.args, ctx.model, series, ctx.best, seed, stride)



def _horizon_sets(ctx, calib_series):
    args = ctx.args
    x_train, y_train = _dataset_with_offset(ctx, ctx.data.train_std, "train", args.horizon_thin)
    x_val, y_val = _dataset_with_offset(ctx, ctx.data.val_std, "val", args.horizon_thin)
    x_calib, y_calib = _dataset_with_offset(ctx, calib_series, "calib", args.horizon_calib_thin)
    x_test, y_test = _dataset_with_offset(ctx, ctx.data.test_std, "test", args.horizon_thin)
    if x_train.size == 0:
        raise RuntimeError("Not enough data for conformal horizon training.")
    return HorizonSets(x_train, y_train, x_val, y_val, x_calib, y_calib, x_test, y_test, x_train, x_calib, x_test)



def _standardize_optional(x, mean, std):
    return (x - mean) / std if x.size else x



def _normalize_horizon_sets(sets):
    feat_mean = np.mean(sets.x_train, axis=0)
    feat_std = np.std(sets.x_train, axis=0)
    feat_std[feat_std == 0.0] = 1.0
    sets.x_train = (sets.x_train - feat_mean) / feat_std
    sets.x_val = _standardize_optional(sets.x_val, feat_mean, feat_std)
    sets.x_calib = _standardize_optional(sets.x_calib, feat_mean, feat_std)
    sets.x_test = _standardize_optional(sets.x_test, feat_mean, feat_std)
    return feat_mean, feat_std



def _use_sigma(args):
    return args.conformal_mode in ("normalized", "tree", "bins") and not args.conformal_no_sigma



def _resolve_quantile(args):
    return args.horizon_quantile if args.horizon_quantile is not None else args.calibration_alpha



def _cv_folds(args):
    return max(1, int(args.conformal_cv_folds))



def _empty_features(ref):
    return np.empty((0, ref.shape[1]), dtype=np.float64)



def _predict_quantile(x_train, y_train, x_val, y_val, x_calib, x_test, quantile, args, device, seed):
    return predict_quantile_ensemble(x_train, y_train, x_val, y_val, x_calib, x_test, quantile, args, device, seed_base=seed)



def _predict_sigma(x_train, y_train, x_val, y_val, x_target, args, device, seed):
    if args.scale_from_quantiles:
        return predict_sigma_quantile_ensemble(x_train, y_train, x_val, y_val, x_target, args, device, seed_base=seed)
    return predict_sigma_mlp(x_train, y_train, x_val, y_val, x_target, args, device, seed_base=seed)



def _cv_dataset(ctx, calib_series):
    """Build the CV pool from train and calib segments separately.

    The val segment sits between train and calib in real time, so concatenating
    the raw series would create a temporal discontinuity: windows spanning the
    junction would be fictitious trajectories. Instead the horizon dataset is
    built on each segment and the results are concatenated. Limitation: the
    train-segment windows are in-sample for the forecaster (trained on
    train+val), so their labels are optimistic; a full fix is deferred to V2.
    """
    seed = _seed_offset(ctx.args.seed, ctx.constants, "cv")
    x_a, y_a = _horizon_dataset(ctx.args, ctx.model, ctx.data.train_std, ctx.best, seed, ctx.args.horizon_calib_thin)
    x_b, y_b = _horizon_dataset(ctx.args, ctx.model, calib_series, ctx.best, seed + 1, ctx.args.horizon_calib_thin)
    if x_a.size == 0:
        return x_b, y_b
    if x_b.size == 0:
        return x_a, y_a
    return np.concatenate([x_a, x_b], axis=0), np.concatenate([y_a, y_b], axis=0)



def _override_calib_for_cv(sets, x_cv_raw, y_cv, feat_mean, feat_std):
    sets.x_calib_raw = x_cv_raw
    sets.y_calib = y_cv
    sets.x_calib = _standardize_optional(x_cv_raw, feat_mean, feat_std)



def _fold_indices(n, start, end):
    train_idx = np.concatenate([np.arange(0, start, dtype=np.int64), np.arange(end, n, dtype=np.int64)])
    val_idx = np.arange(start, end, dtype=np.int64)
    return train_idx, val_idx



def _fill_cv_predictions(pred_calib, sigma_calib, x_fit, y_fit, x_val, y_val, use_sigma, quantile, args, device, constants, cv_folds, seed):
    folds = make_contiguous_folds(y_fit.shape[0], cv_folds)
    empty = _empty_features(x_fit)
    for fold_idx, (start, end) in enumerate(folds):
        if end <= start:
            continue
        train_idx, val_idx = _fold_indices(y_fit.shape[0], start, end)
        pred_seed = _seed_with_base(seed, constants, "quantile_cv", fold_idx)
        pred_fold, _ = _predict_quantile(x_fit[train_idx], y_fit[train_idx], x_val, y_val, x_fit[val_idx], empty, quantile, args, device, pred_seed)
        pred_calib[val_idx] = pred_fold
        if use_sigma:
            sigma_seed = _seed_with_base(seed, constants, "sigma_cv", fold_idx)
            sigma_calib[val_idx] = _predict_sigma(x_fit[train_idx], y_fit[train_idx], x_val, y_val, x_fit[val_idx], args, device, sigma_seed)



def _predict_test_from_fit(x_fit, y_fit, x_val, y_val, x_test, quantile, use_sigma, args, device, constants, seed):
    empty = _empty_features(x_fit)
    test_seed = _seed_with_base(seed, constants, "quantile_test")
    _, pred_test = _predict_quantile(x_fit, y_fit, x_val, y_val, empty, x_test, quantile, args, device, test_seed)
    sigma_test = np.ones_like(pred_test)
    if use_sigma and x_fit.size and x_test.size:
        sigma_seed = _seed_with_base(seed, constants, "sigma_test")
        sigma_test = _predict_sigma(x_fit, y_fit, x_val, y_val, x_test, args, device, sigma_seed)
    return pred_test, sigma_test



def _cv_predictions_from_fit(sets, quantile, use_sigma, args, device, constants, cv_folds, seed):
    pred_calib = np.zeros(sets.y_calib.shape[0], dtype=np.float64)
    sigma_calib = np.ones_like(pred_calib)
    _fill_cv_predictions(pred_calib, sigma_calib, sets.x_calib, sets.y_calib, sets.x_val, sets.y_val, use_sigma, quantile, args, device, constants, cv_folds, seed)
    pred_calib = _clip_horizon(pred_calib, args, constants)
    pred_test, sigma_test = _predict_test_from_fit(sets.x_calib, sets.y_calib, sets.x_val, sets.y_val, sets.x_test, quantile, use_sigma, args, device, constants, seed)
    pred_test = _clip_horizon(pred_test, args, constants)
    return Predictions(pred_calib, pred_test, sigma_calib, sigma_test)



def _try_cv_predictions(ctx, sets, calib_series, quantile, use_sigma, constants, feat_mean, feat_std):
    cv_folds = _cv_folds(ctx.args)
    if cv_folds <= 1:
        return None, False
    x_cv_raw, y_cv = _cv_dataset(ctx, calib_series)
    if x_cv_raw.size == 0 or y_cv.size < cv_folds:
        return None, False
    _override_calib_for_cv(sets, x_cv_raw, y_cv, feat_mean, feat_std)
    preds = _cv_predictions_from_fit(sets, quantile, use_sigma, ctx.args, ctx.device, constants, cv_folds, ctx.args.seed)
    return preds, True



def _direct_predictions(sets, quantile, use_sigma, args, device, constants, seed):
    pred_seed = _seed_with_base(seed, constants, "quantile_no_cv")
    pred_calib, pred_test = _predict_quantile(sets.x_train, sets.y_train, sets.x_val, sets.y_val, sets.x_calib, sets.x_test, quantile, args, device, pred_seed)
    pred_calib = _clip_horizon(pred_calib, args, constants)
    pred_test = _clip_horizon(pred_test, args, constants)
    sigma_calib = np.ones_like(pred_calib)
    sigma_test = np.ones_like(pred_test)
    if use_sigma and sets.x_train.size:
        if sets.x_calib.size:
            sigma_seed = _seed_with_base(seed, constants, "sigma_no_cv_calib")
            sigma_calib = _predict_sigma(sets.x_train, sets.y_train, sets.x_val, sets.y_val, sets.x_calib, args, device, sigma_seed)
        if sets.x_test.size:
            sigma_seed = _seed_with_base(seed, constants, "sigma_no_cv_test")
            sigma_test = _predict_sigma(sets.x_train, sets.y_train, sets.x_val, sets.y_val, sets.x_test, args, device, sigma_seed)
    return Predictions(pred_calib, pred_test, sigma_calib, sigma_test)



def _apply_sigma_bounds(sigma, args, ref=None):
    if sigma.size == 0:
        return sigma
    ref = sigma if ref is None else ref
    sigma = np.maximum(sigma, args.scale_floor)
    if args.scale_floor_quantile is not None and ref.size:
        floor_q = float(np.quantile(ref, _clamp01(args.scale_floor_quantile)))
        if np.isfinite(floor_q) and floor_q > 0.0:
            sigma = np.maximum(sigma, floor_q)
    if args.scale_cap is not None:
        return np.minimum(sigma, args.scale_cap)
    if args.scale_cap_quantile is None or not ref.size:
        return sigma
    cap = float(np.quantile(ref, _clamp01(args.scale_cap_quantile)))
    if np.isfinite(cap) and cap > 0.0:
        sigma = np.minimum(sigma, cap)
    return sigma



def _apply_sigma_bounds_to_predictions(preds, args):
    preds.sigma_calib = _apply_sigma_bounds(preds.sigma_calib, args)
    preds.sigma_test = _apply_sigma_bounds(preds.sigma_test, args, preds.sigma_calib)
    return preds



def _apply_offset_calibration(preds, y_calib, use_sigma, args):
    if not args.offset_calibration or not preds.pred_calib.size:
        return preds
    offset_q = args.offset_quantile if args.offset_quantile is not None else args.calibration_alpha
    offset_q = _clamp01(offset_q)
    if use_sigma:
        resid_scaled = (y_calib - preds.pred_calib) / np.maximum(preds.sigma_calib, args.scale_floor)
        offset_scale = max(float(np.quantile(resid_scaled, offset_q)), 0.0)
        preds.pred_calib = preds.pred_calib + offset_scale * preds.sigma_calib
        if preds.pred_test.size:
            preds.pred_test = preds.pred_test + offset_scale * preds.sigma_test
        return preds
    resid = y_calib - preds.pred_calib
    offset = max(float(np.quantile(resid, offset_q)), 0.0)
    preds.pred_calib = preds.pred_calib + offset
    if preds.pred_test.size:
        preds.pred_test = preds.pred_test + offset
    return preds



def _conformal_predictions(ctx, sets, calib_series, quantile, use_sigma, constants, feat_mean, feat_std):
    # Powell censored pinball (audit C3): thread cap=horizon_max into every
    # quantile-model training below when --censored-quantile is on.
    with quantile_cap(_censoring_cap(ctx.args)):
        preds, cv_used = _try_cv_predictions(ctx, sets, calib_series, quantile, use_sigma, constants, feat_mean, feat_std)
        if not cv_used:
            preds = _direct_predictions(sets, quantile, use_sigma, ctx.args, ctx.device, constants, ctx.args.seed)
    preds = _apply_sigma_bounds_to_predictions(preds, ctx.args)
    preds = _apply_offset_calibration(preds, sets.y_calib, use_sigma, ctx.args)
    return preds


