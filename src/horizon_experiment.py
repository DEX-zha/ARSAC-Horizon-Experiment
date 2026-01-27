"""AI-driven prediction horizon experiment for chaotic time series."""

import argparse
import csv
import math
import os
import sys
import time

import yaml
import logging
import numpy as np
import torch

from src.horizon_metrics import (
    build_horizon_dataset,
    compute_calibration_residuals,
    estimate_error_growth,
    estimate_jacobian_growth,
    estimate_local_delta,
    estimate_model_error_from_residuals,
    evaluate_mse,
    horizon_from_lyapunov,
    horizon_from_rmse,
    rolling_rmse,
    window_horizons,
)
from src.horizon_models import LinearAR, TorchSeqWrapper, TorchWrapper
from src.horizon_plots import plot_log_divergence, plot_rmse
# ProgressBar removed - now handled by Forecaster
from src.horizon_training import (
    build_multistep_supervised,
    train_lstm,
    train_lstm_multistep,
    train_mlp,
    train_mlp_multistep,
    train_quantile_mlp,
)
from src.horizon_utils import (
    set_seed, 
    build_supervised,
    estimate_lyapunov,
    estimate_expansion_quantile,
    horizon_from_model_bound_by_growth
)
from src.horizon_data import DataManager
from src.horizon_forecast import Forecaster
from src.horizon_conformal import (
    conformal_quantile,
    block_conformal_margin,
    compute_bin_edges,
    assign_bin_ids,
    ConformalTreeEstimator,
    fit_mondrian_bins,
    extract_bin_features,
    predict_quantile_ensemble,
    predict_sigma_quantile_ensemble,
    predict_sigma_mlp,
    make_contiguous_folds,
)




def run_experiment(args):
    """Runs a full horizon experiment and writes summary CSV output."""
    set_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    # --- DATA LOADING AND PREPARATION ---
    data_manager = DataManager(args)
    train_std, val_std, calib_std, test_std = data_manager.prepare_data()
    train_raw, val_raw, calib_raw, test_raw = data_manager.get_raw_splits()

    t0 = time.time()
    
    # --- MODEL TRAINING ---
    forecaster = Forecaster(args, device)
    best = forecaster.select_embedding(train_std, val_std)
    model = forecaster.train_final_model(train_std, val_std)

    x_test, y_test = build_supervised(test_std, best["dim"], best["lag"], horizon=1)
    test_mse = evaluate_mse(model, x_test, y_test, device=device)

    rmse_by_h = rolling_rmse(
        model, test_std, best["dim"], best["lag"], args.horizon_max
    )
    base_err = rmse_by_h[0] if rmse_by_h.size > 0 else 0.0
    if args.error_mode == "relative":
        tolerance = base_err * args.error_factor
    else:
        tolerance = args.error_tolerance
    horizon_real = horizon_from_rmse(rmse_by_h, tolerance)
    horizon_window_median = None
    horizon_window_mean = None

    lyap_dim = args.lyap_dim if args.lyap_dim is not None else best["dim"]
    lyap_lag = args.lyap_lag if args.lyap_lag is not None else best["lag"]
    lyap_step, _ = estimate_lyapunov(
        np.concatenate([train_raw, val_raw]),
        dim=lyap_dim,
        lag=lyap_lag,
        max_t=args.lyap_max_t,
        theiler=args.lyap_theiler,
        fit_start=args.lyap_fit_start,
        fit_end=args.lyap_fit_end,
        dt=args.dt if args.dataset != "logistic" else 1.0,
    )
    init_err = rmse_by_h[0] if rmse_by_h.size > 0 else 0.0
    dt = args.dt if args.dataset != "logistic" else 1.0
    lyap_time = lyap_step / dt if dt > 0 else 0.0
    horizon_theory_steps = horizon_from_lyapunov(lyap_step, base_err, tolerance)
    horizon_theory_time = (
        horizon_theory_steps * dt if math.isfinite(horizon_theory_steps) else float("inf")
    )
    horizon_real_time = horizon_real * dt

    calib_series = calib_std if calib_std.size else val_std
    delta_local_quantile = args.delta_local_quantile
    if delta_local_quantile is None:
        delta_local_quantile = args.delta_quantile

    bound_mode = args.bound_mode
    model_error = 0.0
    model_error_mode = "none"
    model_error_mean = 0.0
    delta_local_used = False
    growth_source = args.growth_source
    growth_horizon = args.expansion_horizon
    growth_q = 1.0
    growth_mean = 1.0
    exp_dim = args.expansion_dim if args.expansion_dim is not None else lyap_dim
    exp_lag = args.expansion_lag if args.expansion_lag is not None else lyap_lag
    expansion_q = 1.0
    expansion_mean = 1.0
    scale = 1.0
    coverage = None
    calibration_samples = 0
    horizon_model_steps = 0.0
    horizon_model_time = 0.0
    horizon_est_steps = 0.0
    horizon_est_time = 0.0
    horizon_model_cal = 0.0
    horizon_model_cal_time = 0.0
    coverage_test = None
    tightness_ratio = None
    slack_median = None
    slack_p90 = None
    leaf_coverage_stats = None
    jac_quantile_coverages = None
    score_pos_frac = None
    score_neg_frac = None
    score_zero_frac = None
    score_p10 = None
    score_p50 = None
    score_p90 = None
    score_mean = None
    signed_med = None
    sigma_med = None
    sigma_p90 = None
    sigma_max = None
    pred_calib_med = None
    y_calib_med = None
    l_calib_med = None
    bin_count = None
    bin_min_count = None
    bin_med_count = None
    bin_c_min = None
    bin_c_med = None
    bin_c_max = None

    c_global = 0.0
    bin_count = None
    bin_min_count = None
    bin_med_count = None
    bin_c_min = None
    bin_c_med = None
    bin_c_max = None

    if bound_mode == "horizon_conformal":
        x_train, y_train = build_horizon_dataset(
            model,
            train_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            args.error_tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed,
            use_jacobian=args.horizon_use_jacobian,
            error_mode=args.error_mode,
            error_factor=args.error_factor,
            consecutive_k=args.horizon_consecutive_k,
            stride=args.horizon_thin,
            feature_horizon=args.horizon_feature_horizon,
        )
        x_val, y_val = build_horizon_dataset(
            model,
            val_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            args.error_tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 1,
            use_jacobian=args.horizon_use_jacobian,
            error_mode=args.error_mode,
            error_factor=args.error_factor,
            consecutive_k=args.horizon_consecutive_k,
            stride=args.horizon_thin,
            feature_horizon=args.horizon_feature_horizon,
        )
        x_calib_h, y_calib_h = build_horizon_dataset(
            model,
            calib_series,
            best["dim"],
            best["lag"],
            args.horizon_max,
            args.error_tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 2,
            use_jacobian=args.horizon_use_jacobian,
            error_mode=args.error_mode,
            error_factor=args.error_factor,
            consecutive_k=args.horizon_consecutive_k,
            stride=args.horizon_calib_thin,
            feature_horizon=args.horizon_feature_horizon,
        )
        x_test_h, y_test_h = build_horizon_dataset(
            model,
            test_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            args.error_tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 3,
            use_jacobian=args.horizon_use_jacobian,
            error_mode=args.error_mode,
            error_factor=args.error_factor,
            consecutive_k=args.horizon_consecutive_k,
            stride=args.horizon_thin,
            feature_horizon=args.horizon_feature_horizon,
        )

        if x_train.size == 0:
            raise RuntimeError("Not enough data for conformal horizon training.")

        x_train_raw = x_train
        x_calib_raw = x_calib_h
        x_test_raw = x_test_h
        feat_mean = np.mean(x_train, axis=0)
        feat_std = np.std(x_train, axis=0)
        feat_std[feat_std == 0.0] = 1.0
        x_train = (x_train - feat_mean) / feat_std
        x_val = (x_val - feat_mean) / feat_std if x_val.size else x_val
        x_calib_h = (x_calib_h - feat_mean) / feat_std if x_calib_h.size else x_calib_h
        x_test_h = (x_test_h - feat_mean) / feat_std if x_test_h.size else x_test_h

        use_sigma = (
            args.conformal_mode in ("normalized", "tree", "bins")
            and not args.conformal_no_sigma
        )
        quantile = args.horizon_quantile
        if quantile is None:
            quantile = args.calibration_alpha

        pred_calib = np.array([], dtype=np.float64)
        pred_test = np.array([], dtype=np.float64)
        sigma_calib = np.ones(0, dtype=np.float64)
        sigma_test = np.ones(0, dtype=np.float64)

        cv_folds = max(1, int(args.conformal_cv_folds))
        cv_ready = cv_folds > 1
        if cv_ready:
            cv_series = np.concatenate([train_std, calib_series], axis=0)
            x_cv_raw, y_cv = build_horizon_dataset(
                model,
                cv_series,
                best["dim"],
                best["lag"],
                args.horizon_max,
                args.error_tolerance,
                max_windows=args.horizon_samples,
                seed=args.seed + 4,
                use_jacobian=args.horizon_use_jacobian,
                error_mode=args.error_mode,
                error_factor=args.error_factor,
                consecutive_k=args.horizon_consecutive_k,
                stride=args.horizon_calib_thin,
                feature_horizon=args.horizon_feature_horizon,
            )
            if x_cv_raw.size == 0 or y_cv.size < cv_folds:
                cv_ready = False
            else:
                x_calib_raw = x_cv_raw
                y_calib_h = y_cv
                x_calib_h = (x_cv_raw - feat_mean) / feat_std
                x_fit = x_calib_h
                y_fit = y_calib_h

                pred_calib = np.zeros(y_calib_h.shape[0], dtype=np.float64)
                sigma_calib = np.ones_like(pred_calib)
                folds = make_contiguous_folds(y_calib_h.shape[0], cv_folds)
                for fold_idx, (start, end) in enumerate(folds):
                    if end <= start:
                        continue
                    train_idx = np.concatenate(
                        [
                            np.arange(0, start, dtype=np.int64),
                            np.arange(end, y_calib_h.shape[0], dtype=np.int64),
                        ]
                    )
                    val_idx = np.arange(start, end, dtype=np.int64)
                    pred_fold, _ = predict_quantile_ensemble(
                        x_fit[train_idx],
                        y_fit[train_idx],
                        x_val,
                        y_val,
                        x_fit[val_idx],
                        np.empty((0, x_fit.shape[1]), dtype=np.float64),
                        quantile,
                        args,
                        device,
                        seed_base=args.seed + 1000 + fold_idx * 10000,
                    )
                    pred_calib[val_idx] = pred_fold
                    if use_sigma:
                        if args.scale_from_quantiles:
                            sigma_fold = predict_sigma_quantile_ensemble(
                                x_fit[train_idx],
                                y_fit[train_idx],
                                x_val,
                                y_val,
                                x_fit[val_idx],
                                args,
                                device,
                                seed_base=args.seed + 2000 + fold_idx * 10000,
                            )
                        else:
                            sigma_fold = predict_sigma_mlp(
                                x_fit[train_idx],
                                y_fit[train_idx],
                                x_val,
                                y_val,
                                x_fit[val_idx],
                                args,
                                device,
                                seed_base=args.seed + 2000 + fold_idx * 10000,
                            )
                        sigma_calib[val_idx] = sigma_fold

                if pred_calib.size:
                    pred_calib = np.clip(pred_calib, 1.0, float(args.horizon_max))

                _, pred_test = predict_quantile_ensemble(
                    x_fit,
                    y_fit,
                    x_val,
                    y_val,
                    np.empty((0, x_fit.shape[1]), dtype=np.float64),
                    x_test_h,
                    quantile,
                    args,
                    device,
                    seed_base=args.seed + 5000,
                )
                if pred_test.size:
                    pred_test = np.clip(pred_test, 1.0, float(args.horizon_max))
                if use_sigma:
                    if args.scale_from_quantiles:
                        sigma_test = predict_sigma_quantile_ensemble(
                            x_fit,
                            y_fit,
                            x_val,
                            y_val,
                            x_test_h,
                            args,
                            device,
                            seed_base=args.seed + 6000,
                        )
                    else:
                        sigma_test = predict_sigma_mlp(
                            x_fit,
                            y_fit,
                            x_val,
                            y_val,
                            x_test_h,
                            args,
                            device,
                            seed_base=args.seed + 6000,
                        )

        if not cv_ready:
            pred_calib, pred_test = predict_quantile_ensemble(
                x_train,
                y_train,
                x_val,
                y_val,
                x_calib_h,
                x_test_h,
                quantile,
                args,
                device,
                seed_base=args.seed + 1000,
            )
            if pred_calib.size:
                pred_calib = np.clip(pred_calib, 1.0, float(args.horizon_max))
            if pred_test.size:
                pred_test = np.clip(pred_test, 1.0, float(args.horizon_max))
            sigma_calib = np.ones_like(pred_calib)
            sigma_test = np.ones_like(pred_test)
            if use_sigma and x_train.size:
                if args.scale_from_quantiles:
                    if x_calib_h.size:
                        sigma_calib = predict_sigma_quantile_ensemble(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_calib_h,
                            args,
                            device,
                            seed_base=args.seed + 2000,
                        )
                    if x_test_h.size:
                        sigma_test = predict_sigma_quantile_ensemble(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_test_h,
                            args,
                            device,
                            seed_base=args.seed + 3000,
                        )
                else:
                    if x_calib_h.size:
                        sigma_calib = predict_sigma_mlp(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_calib_h,
                            args,
                            device,
                            seed_base=args.seed + 2000,
                        )
                    if x_test_h.size:
                        sigma_test = predict_sigma_mlp(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            x_test_h,
                            args,
                            device,
                            seed_base=args.seed + 3000,
                        )

        if use_sigma and sigma_calib.size:
            sigma_calib = np.maximum(sigma_calib, args.scale_floor)
            if args.scale_floor_quantile is not None:
                floor_q = float(
                    np.quantile(
                        sigma_calib,
                        min(max(args.scale_floor_quantile, 0.0), 1.0),
                    )
                )
                if np.isfinite(floor_q) and floor_q > 0.0:
                    sigma_calib = np.maximum(sigma_calib, floor_q)
            if args.scale_cap is not None:
                sigma_calib = np.minimum(sigma_calib, args.scale_cap)
            elif args.scale_cap_quantile is not None:
                cap = float(
                    np.quantile(
                        sigma_calib,
                        min(max(args.scale_cap_quantile, 0.0), 1.0),
                    )
                )
                if np.isfinite(cap) and cap > 0.0:
                    sigma_calib = np.minimum(sigma_calib, cap)
        if use_sigma and sigma_test.size:
            sigma_test = np.maximum(sigma_test, args.scale_floor)
            if args.scale_floor_quantile is not None and sigma_calib.size:
                floor_q = float(
                    np.quantile(
                        sigma_calib,
                        min(max(args.scale_floor_quantile, 0.0), 1.0),
                    )
                )
                if np.isfinite(floor_q) and floor_q > 0.0:
                    sigma_test = np.maximum(sigma_test, floor_q)
            if args.scale_cap is not None:
                sigma_test = np.minimum(sigma_test, args.scale_cap)
            elif args.scale_cap_quantile is not None and sigma_calib.size:
                cap = float(
                    np.quantile(
                        sigma_calib,
                        min(max(args.scale_cap_quantile, 0.0), 1.0),
                    )
                )
                if np.isfinite(cap) and cap > 0.0:
                    sigma_test = np.minimum(sigma_test, cap)

        if args.offset_calibration and pred_calib.size:
            offset_q = (
                args.offset_quantile
                if args.offset_quantile is not None
                else args.calibration_alpha
            )
            offset_q = float(min(max(offset_q, 0.0), 1.0))
            if use_sigma:
                resid_scaled = (y_calib_h - pred_calib) / np.maximum(
                    sigma_calib, args.scale_floor
                )
                offset_scale = float(np.quantile(resid_scaled, offset_q))
                offset_scale = max(offset_scale, 0.0)
                pred_calib = pred_calib + offset_scale * sigma_calib
                if pred_test.size:
                    pred_test = pred_test + offset_scale * sigma_test
            else:
                resid = y_calib_h - pred_calib
                offset = float(np.quantile(resid, offset_q))
                offset = max(offset, 0.0)
                pred_calib = pred_calib + offset
                if pred_test.size:
                    pred_test = pred_test + offset

        c_global = 0.0
        pred_test_cal = np.array([], dtype=np.float64)
        tree = None
        bin_model = None
        leaf_ids = None
        tie_rng = np.random.default_rng(args.seed + 12345)
        if pred_calib.size:
            signed = pred_calib - y_calib_h
            if use_sigma:
                scores = signed / np.maximum(sigma_calib, args.scale_floor)
            else:
                scores = signed
            split_scores = np.abs(pred_calib - y_calib_h)
            if use_sigma:
                split_scores = split_scores / np.maximum(sigma_calib, args.scale_floor)
            if scores.size:
                score_pos_frac = float(np.mean(scores > 0.0))
                score_neg_frac = float(np.mean(scores < 0.0))
                score_zero_frac = float(np.mean(scores == 0.0))
                score_p10 = float(np.quantile(scores, 0.1))
                score_p50 = float(np.median(scores))
                score_p90 = float(np.quantile(scores, 0.9))
                score_mean = float(np.mean(scores))
            if signed.size:
                signed_med = float(np.median(signed))
            if pred_calib.size:
                pred_calib_med = float(np.median(pred_calib))
            if y_calib_h.size:
                y_calib_med = float(np.median(y_calib_h))
            if use_sigma and sigma_calib.size:
                sigma_med = float(np.median(sigma_calib))
                sigma_p90 = float(np.quantile(sigma_calib, 0.9))
                sigma_max = float(np.max(sigma_calib))
            c_global = block_conformal_margin(
                scores,
                args.calibration_alpha,
                args.block_count,
                block_quantile=args.block_quantile,
                rng=tie_rng,
                tie_jitter=args.conformal_tie_jitter,
            )

            if args.conformal_mode == "tree":
                jac_idx = best["dim"] + 2
                jac_log_idx = best["dim"] + 5
                resid_idx = best["dim"] + 3
                err_var_idx = best["dim"] + 6
                pred_var_idx = best["dim"] + 7
                jac_calib = (
                    x_calib_raw[:, jac_idx]
                    if x_calib_raw.size
                    else np.zeros_like(pred_calib)
                )
                jac_log_calib = (
                    x_calib_raw[:, jac_log_idx]
                    if x_calib_raw.size
                    else np.zeros_like(pred_calib)
                )
                resid_calib = (
                    x_calib_raw[:, resid_idx]
                    if x_calib_raw.size
                    else np.zeros_like(pred_calib)
                )
                err_var_calib = (
                    x_calib_raw[:, err_var_idx]
                    if x_calib_raw.size
                    else np.zeros_like(pred_calib)
                )
                pred_var_calib = (
                    x_calib_raw[:, pred_var_idx]
                    if x_calib_raw.size
                    else np.zeros_like(pred_calib)
                )
                min_leaf_eff = min(
                    args.conformal_min_leaf,
                    max(30, int(scores.size // 4))
                    if scores.size
                    else args.conformal_min_leaf,
                )
                tree_features_calib = np.column_stack(
                    [
                        pred_calib,
                        sigma_calib,
                        jac_calib,
                        jac_log_calib,
                        resid_calib,
                        err_var_calib,
                        pred_var_calib,
                    ]
                )
                tree = ConformalTreeEstimator(
                    min_samples_leaf=min_leaf_eff,
                    max_depth=args.conformal_tree_depth,
                    min_gain=args.conformal_tree_min_gain
                )
                tree.fit(
                    tree_features_calib,
                    scores,
                    args.calibration_alpha,
                    rng=tie_rng,
                    tie_jitter=args.conformal_tie_jitter
                )
                c_calib = tree.predict(tree_features_calib)
                
                # Apply to Test Set
                if x_test_raw.size and pred_test.size:
                     jac_test = x_test_raw[:, jac_idx]
                     jac_log_test = x_test_raw[:, jac_log_idx]
                     resid_test = x_test_raw[:, resid_idx]
                     err_var_test = x_test_raw[:, err_var_idx]
                     pred_var_test = x_test_raw[:, pred_var_idx]
                     
                     tree_features_test = np.column_stack([
                        pred_test, sigma_test, jac_test, jac_log_test, 
                        resid_test, err_var_test, pred_var_test
                     ])
                     
                     c_test = tree.predict(tree_features_test)
                     leaf_ids = tree.apply(tree_features_test)
                     
                     if use_sigma:
                         pred_test_cal = pred_test - c_test * sigma_test
                     else:
                         pred_test_cal = pred_test - c_test
            elif args.conformal_mode == "bins":
                bin_features_train = extract_bin_features(
                    x_train_raw, best["dim"], args.conformal_bin_feature
                )
                bin_features_calib = extract_bin_features(
                    x_calib_raw, best["dim"], args.conformal_bin_feature
                )
                if bin_features_train.size and bin_features_calib.size:
                    bin_pool = np.vstack([bin_features_train, bin_features_calib])
                elif bin_features_calib.size:
                    bin_pool = bin_features_calib
                else:
                    bin_pool = bin_features_train
                bin_dim = bin_pool.shape[1] if bin_pool.ndim == 2 else 1
                edges_list = [
                    compute_bin_edges(bin_pool[:, col], args.conformal_bins)
                    for col in range(bin_dim)
                ]
                c_groups, bin_ids_calib, bin_counts = fit_mondrian_bins(
                    bin_features_calib,
                    scores,
                    args.calibration_alpha,
                    edges_list,
                    args.conformal_min_bin,
                    args.conformal_bin_shrinkage,
                    c_global,
                    rng=tie_rng,
                    tie_jitter=args.conformal_tie_jitter,
                )
                bin_model = {
                    "edges": edges_list,
                    "c_groups": c_groups,
                    "counts": bin_counts,
                }
                c_calib = (
                    c_groups[bin_ids_calib]
                    if bin_ids_calib.size
                    else np.full_like(pred_calib, c_global)
                )
                if bin_counts.size:
                    nonzero_counts = bin_counts[bin_counts > 0]
                    if nonzero_counts.size:
                        bin_count = int(nonzero_counts.size)
                        bin_min_count = int(np.min(nonzero_counts))
                        bin_med_count = float(np.median(nonzero_counts))
                        c_used = c_groups[bin_counts > 0]
                        if c_used.size:
                            bin_c_min = float(np.min(c_used))
                            bin_c_med = float(np.median(c_used))
                            bin_c_max = float(np.max(c_used))
            else:
                c_calib = np.full_like(pred_calib, c_global)

            sigma_term = sigma_calib if use_sigma else np.ones_like(c_calib)
            l_calib = np.clip(
                pred_calib - c_calib * sigma_term, 1.0, float(args.horizon_max)
            )
            if l_calib.size:
                l_calib_med = float(np.median(l_calib))
            calibration_samples = int(len(y_calib_h))
            coverage = float(np.mean(y_calib_h >= l_calib)) if y_calib_h.size else None

        if pred_test.size:
            if args.conformal_mode == "tree" and tree is not None:
                jac_idx = best["dim"] + 2
                jac_log_idx = best["dim"] + 5
                resid_idx = best["dim"] + 3
                err_var_idx = best["dim"] + 6
                pred_var_idx = best["dim"] + 7
                jac_test = (
                    x_test_raw[:, jac_idx]
                    if x_test_raw.size
                    else np.zeros_like(pred_test)
                )
                jac_log_test = (
                    x_test_raw[:, jac_log_idx]
                    if x_test_raw.size
                    else np.zeros_like(pred_test)
                )
                resid_test = (
                    x_test_raw[:, resid_idx]
                    if x_test_raw.size
                    else np.zeros_like(pred_test)
                )
                err_var_test = (
                    x_test_raw[:, err_var_idx]
                    if x_test_raw.size
                    else np.zeros_like(pred_test)
                )
                pred_var_test = (
                    x_test_raw[:, pred_var_idx]
                    if x_test_raw.size
                    else np.zeros_like(pred_test)
                )
                tree_features_test = np.column_stack(
                    [
                        pred_test,
                        sigma_test,
                        jac_test,
                        jac_log_test,
                        resid_test,
                        err_var_test,
                        pred_var_test,
                    ]
                )
                c_test = tree.predict(tree_features_test)
                leaf_ids = tree.apply(tree_features_test)
            elif args.conformal_mode == "bins" and bin_model is not None:
                bin_features_test = extract_bin_features(
                    x_test_raw, best["dim"], args.conformal_bin_feature
                )
                bin_ids_test, _ = assign_bin_ids(
                    bin_features_test, bin_model["edges"]
                )
                c_test = (
                    bin_model["c_groups"][bin_ids_test]
                    if bin_ids_test.size
                    else np.full_like(pred_test, c_global)
                )
                leaf_ids = bin_ids_test
            else:
                c_test = np.full_like(pred_test, c_global)

            sigma_term_test = sigma_test if use_sigma else np.ones_like(pred_test)
            pred_test_cal = np.clip(
                pred_test - c_test * sigma_term_test, 1.0, float(args.horizon_max)
            )
            horizon_model_steps = float(np.median(pred_test))
            horizon_est_steps = float(np.mean(pred_test))
            horizon_model_cal = float(np.median(pred_test_cal))

        if y_test_h.size:
            horizon_window_median = float(np.median(y_test_h))
            horizon_window_mean = float(np.mean(y_test_h))
            if pred_test_cal.size:
                coverage_test = float(np.mean(y_test_h >= pred_test_cal))
                slack = y_test_h - pred_test_cal
                tightness_ratio = (
                    float(np.median(pred_test_cal) / np.median(y_test_h))
                    if np.median(y_test_h) > 0.0
                    else None
                )
                slack_median = float(np.median(slack))
                slack_p90 = float(np.quantile(slack, 0.9))

        if pred_test_cal.size and x_test_raw.size and y_test_h.size:
            jac_idx = best["dim"] + 2
            jac_values = x_test_raw[:, jac_idx]
            edges = np.quantile(jac_values, np.linspace(0.0, 1.0, 5))
            edges[0] = -np.inf
            edges[-1] = np.inf
            jac_bins = np.digitize(jac_values, edges[1:-1], right=False)
            jac_quantile_coverages = {}
            for b in range(4):
                mask = jac_bins == b
                if np.any(mask):
                    jac_quantile_coverages[f"jac_q{b + 1}"] = float(
                        np.mean(y_test_h[mask] >= pred_test_cal[mask])
                    )
                else:
                    jac_quantile_coverages[f"jac_q{b + 1}"] = None

        if pred_test_cal.size and leaf_ids is not None and y_test_h.size:
            leaf_coverages = []
            for leaf_id in np.unique(leaf_ids):
                mask = leaf_ids == leaf_id
                if np.any(mask):
                    leaf_coverages.append(float(np.mean(y_test_h[mask] >= pred_test_cal[mask])))
            if leaf_coverages:
                leaf_coverages = np.asarray(leaf_coverages, dtype=np.float64)
                leaf_coverage_stats = {
                    "leaf_count": int(leaf_coverages.size),
                    "leaf_min": float(np.min(leaf_coverages)),
                    "leaf_p10": float(np.quantile(leaf_coverages, 0.1)),
                    "leaf_med": float(np.median(leaf_coverages)),
                    "leaf_mean": float(np.mean(leaf_coverages)),
                }

        horizon_model_time = (
            horizon_model_steps * dt
            if math.isfinite(horizon_model_steps)
            else float("inf")
        )
        horizon_est_time = (
            horizon_est_steps * dt if math.isfinite(horizon_est_steps) else float("inf")
        )
        horizon_model_cal_time = (
            horizon_model_cal * dt
            if math.isfinite(horizon_model_cal)
            else float("inf")
        )
        model_error_mode = f"conformal_{args.conformal_mode}"
        scale = c_global
        growth_source = "conformal"
    else:
        x_calib, calib_residuals = compute_calibration_residuals(
            model, calib_std, best["dim"], best["lag"]
        )
        model_error, model_error_mode, model_error_mean = (
            estimate_model_error_from_residuals(
                calib_residuals,
                mode=args.delta_mode,
                quantile=args.delta_quantile,
                scale=args.delta_scale,
            )
        )
        if args.delta_local:
            delta_local_q, delta_local_mean, _ = estimate_local_delta(
                x_calib,
                calib_residuals,
                k=args.delta_local_k,
                quantile=delta_local_quantile,
                max_samples=args.delta_local_samples,
                seed=args.seed,
            )
            if delta_local_q > 0.0:
                model_error = delta_local_q
                model_error_mean = delta_local_mean
                model_error_mode = f"local@{delta_local_quantile:.2f}"
                delta_local_used = True
        exp_series = np.concatenate([train_std, val_std], axis=0)
        expansion_q, expansion_ratios = estimate_expansion_quantile(
            exp_series,
            dim=exp_dim,
            lag=exp_lag,
            quantile=args.expansion_quantile,
            theiler=args.expansion_theiler,
            max_pairs=args.expansion_samples,
            seed=args.seed,
            horizon=args.expansion_horizon,
        )
        expansion_mean = 1.0
        if expansion_ratios.size:
            positive = expansion_ratios[expansion_ratios > 0]
            if positive.size:
                expansion_mean = float(np.exp(np.mean(np.log(positive))))
        growth_q = expansion_q
        growth_mean = expansion_mean
        if growth_source == "error":
            growth_q, growth_mean, _ = estimate_error_growth(
                model,
                calib_series,
                best["dim"],
                best["lag"],
                horizon=growth_horizon,
                max_windows=args.expansion_samples,
                quantile=args.expansion_quantile,
                seed=args.seed,
            )
        elif growth_source == "jacobian":
            growth_q, growth_mean, _ = estimate_jacobian_growth(
                model,
                x_calib,
                quantile=args.expansion_quantile,
                max_samples=args.expansion_samples,
                seed=args.seed,
            )
            if x_calib.size == 0:
                growth_q = expansion_q
                growth_mean = expansion_mean

        horizon_model_steps = horizon_from_model_bound_by_growth(
            growth_q, base_err, model_error, tolerance
        )
        horizon_model_time = (
            horizon_model_steps * dt
            if math.isfinite(horizon_model_steps)
            else float("inf")
        )
        horizon_est_steps = horizon_from_model_bound_by_growth(
            growth_mean, base_err, model_error_mean, tolerance
        )
        horizon_est_time = (
            horizon_est_steps * dt if math.isfinite(horizon_est_steps) else float("inf")
        )
        rmse_calib = rolling_rmse(
            model, calib_series, best["dim"], best["lag"], args.horizon_max
        )
        base_err_calib = rmse_calib[0] if rmse_calib.size > 0 else 0.0
        if args.error_mode == "relative":
            tolerance_calib = base_err_calib * args.error_factor
        else:
            tolerance_calib = args.error_tolerance

        calib_horizons, calib_init_errs = window_horizons(
            model,
            calib_series,
            best["dim"],
            best["lag"],
            args.horizon_max,
            tolerance_calib,
        )
        ratios = []
        for h_real, init_err in zip(calib_horizons, calib_init_errs):
            if init_err is None:
                continue
            h_model = horizon_from_model_bound_by_growth(
                growth_q, init_err, model_error, tolerance_calib
            )
            if h_model <= 0:
                continue
            ratios.append(h_real / h_model)

        if args.calibrate_coverage and ratios:
            scale = float(np.quantile(ratios, 1.0 - args.calibration_alpha))
            if scale < args.calibration_floor:
                scale = args.calibration_floor
        else:
            scale = 1.0

        horizon_model_cal = horizon_model_steps * scale
        horizon_model_cal_time = (
            horizon_model_cal * dt
            if math.isfinite(horizon_model_cal)
            else float("inf")
        )
        if ratios:
            hits = 0
            total = 0
            for h_real, init_err in zip(calib_horizons, calib_init_errs):
                h_model = horizon_from_model_bound_by_growth(
                    growth_q, init_err, model_error, tolerance_calib
                )
                if h_model <= 0:
                    continue
                total += 1
                if h_model * scale >= h_real:
                    hits += 1
            coverage = hits / total if total > 0 else None
            calibration_samples = total

    os.makedirs(args.output_dir, exist_ok=True)
    csv_name = "horizon_results.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    header = [
        "dataset",
        "model",
        "dim",
        "lag",
        "val_mse",
        "selection_metric",
        "selection_horizon",
        "test_mse",
        "lyapunov_step",
        "lyapunov_time",
        "lyapunov_dim",
        "lyapunov_lag",
        "horizon_real",
        "horizon_real_time",
        "horizon_real_window_median",
        "horizon_real_window_mean",
        "horizon_theory",
        "horizon_theory_time",
        "error_mode",
        "error_factor",
        "error_tolerance",
        "error_tolerance_used",
        "calib_ratio",
        "model_error",
        "model_error_mode",
        "model_error_mean",
        "delta_local",
        "delta_local_k",
        "delta_local_quantile",
        "delta_local_samples",
        "horizon_model",
        "horizon_model_time",
        "horizon_est",
        "horizon_est_time",
        "expansion_quantile",
        "expansion_samples",
        "expansion_theiler",
        "expansion_dim",
        "expansion_lag",
        "expansion_horizon",
        "expansion_Lq",
        "expansion_mean",
        "growth_source",
        "growth_horizon",
        "growth_Lq",
        "growth_Lmean",
        "bound_mode",
        "calibration_alpha",
        "calibration_floor",
        "calibration_scale",
        "calibration_samples",
        "calibration_coverage",
        "horizon_model_cal",
        "horizon_model_cal_time",
    ]

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != header:
            base, ext = os.path.splitext(csv_name)
            suffix = 2
            while True:
                alt_name = f"{base}_v{suffix}{ext}"
                alt_path = os.path.join(args.output_dir, alt_name)
                if not os.path.exists(alt_path):
                    csv_path = alt_path
                    break
                suffix += 1
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                args.dataset,
                args.model,
                best["dim"],
                best["lag"],
                f"{best['val_loss']:.6f}",
                args.selection_metric,
                best["selection"]["horizon"]
                if best.get("selection") and best["selection"]["horizon"] is not None
                else "",
                f"{test_mse:.6f}",
                f"{lyap_step:.6f}",
                f"{lyap_time:.6f}",
                lyap_dim,
                lyap_lag,
                horizon_real,
                f"{horizon_real_time:.3f}",
                f"{horizon_window_median:.3f}"
                if horizon_window_median is not None
                else "",
                f"{horizon_window_mean:.3f}"
                if horizon_window_mean is not None
                else "",
                f"{horizon_theory_steps:.3f}"
                if math.isfinite(horizon_theory_steps)
                else "inf",
                f"{horizon_theory_time:.3f}"
                if math.isfinite(horizon_theory_time)
                else "inf",
                args.error_mode,
                args.error_factor,
                args.error_tolerance,
                f"{tolerance:.6f}",
                f"{args.calib_ratio:.3f}",
                f"{model_error:.6f}",
                model_error_mode,
                f"{model_error_mean:.6f}",
                str(delta_local_used),
                args.delta_local_k,
                f"{delta_local_quantile:.3f}",
                args.delta_local_samples,
                f"{horizon_model_steps:.3f}"
                if math.isfinite(horizon_model_steps)
                else "inf",
                f"{horizon_model_time:.3f}"
                if math.isfinite(horizon_model_time)
                else "inf",
                f"{horizon_est_steps:.3f}"
                if math.isfinite(horizon_est_steps)
                else "inf",
                f"{horizon_est_time:.3f}"
                if math.isfinite(horizon_est_time)
                else "inf",
                f"{args.expansion_quantile:.3f}",
                args.expansion_samples,
                args.expansion_theiler,
                exp_dim,
                exp_lag,
                args.expansion_horizon,
                f"{expansion_q:.6f}",
                f"{expansion_mean:.6f}",
                growth_source,
                growth_horizon,
                f"{growth_q:.6f}",
                f"{growth_mean:.6f}",
                bound_mode,
                f"{args.calibration_alpha:.3f}",
                f"{args.calibration_floor:.3f}",
                f"{scale:.6f}",
                calibration_samples,
                f"{coverage:.3f}" if coverage is not None else "",
                f"{horizon_model_cal:.3f}"
                if math.isfinite(horizon_model_cal)
                else "inf",
                f"{horizon_model_cal_time:.3f}"
                if math.isfinite(horizon_model_cal_time)
                else "inf",
            ]
        )

    if args.plot:
        rmse_path = os.path.join(args.output_dir, f"{args.plot_prefix}_rmse.png")
        log_path = os.path.join(args.output_dir, f"{args.plot_prefix}_log.png")
        plot_rmse(
            rmse_by_h,
            horizon_real,
            horizon_theory_steps,
            horizon_model_steps,
            horizon_model_cal,
            horizon_est_steps,
            rmse_path,
        )
        plot_log_divergence(rmse_by_h, lyap_step, log_path)
        plot_log_divergence(rmse_by_h, lyap_step, log_path)
        logging.info(f"Plots saved to {rmse_path} and {log_path}")

    elapsed = time.time() - t0
    selection_note = ""
    if best.get("selection") and best["selection"]["horizon"] is not None:
        selection_note = f" sel_h={best['selection']['horizon']}"
    window_note = ""
    if horizon_window_median is not None and horizon_window_mean is not None:
        window_note = (
            f" h_win_med={horizon_window_median:.2f}"
            f" h_win_mean={horizon_window_mean:.2f}"
        )
    logging.info(
        f"Best dim={best['dim']} lag={best['lag']} val={best['val_loss']:.6f} "
        f"test={test_mse:.6f} lyap_step={lyap_step:.4f} lyap_time={lyap_time:.4f} "
        f"horizon_real={horizon_real} horizon_real_time={horizon_real_time:.2f} "
        f"horizon_theory={horizon_theory_steps:.2f} horizon_theory_time={horizon_theory_time:.2f} "
        f"horizon_model={horizon_model_steps:.2f} horizon_model_time={horizon_model_time:.2f} "
        f"horizon_cal={horizon_model_cal:.2f} horizon_cal_time={horizon_model_cal_time:.2f} "
        f"horizon_est={horizon_est_steps:.2f} horizon_est_time={horizon_est_time:.2f} "
        f"delta={model_error:.4f} delta_mean={model_error_mean:.4f} "
        f"Lq={growth_q:.4f} Lmean={growth_mean:.4f} k={growth_horizon} "
        f"growth={growth_source} bound={bound_mode} "
        f"{'calib_c' if bound_mode == 'horizon_conformal' else 'scale'}={scale:.3f} "
        f"tol={tolerance:.6f}"
        f"{window_note}"
        f"{selection_note} elapsed={elapsed:.1f}s"
    )
    return {
        "dim": best["dim"],
        "lag": best["lag"],
        "val_loss": best["val_loss"],
        "test_mse": test_mse,
        "lyapunov_step": lyap_step,
        "lyapunov_time": lyap_time,
        "lyapunov_dim": lyap_dim,
        "lyapunov_lag": lyap_lag,
        "horizon_real": horizon_real,
        "horizon_real_window_median": horizon_window_median,
        "horizon_real_window_mean": horizon_window_mean,
        "horizon_theory": horizon_theory_steps,
        "horizon_real_time": horizon_real_time,
        "horizon_theory_time": horizon_theory_time,
        "horizon_model": horizon_model_steps,
        "horizon_model_time": horizon_model_time,
        "horizon_est": horizon_est_steps,
        "horizon_est_time": horizon_est_time,
        "model_error": model_error,
        "model_error_mode": model_error_mode,
        "model_error_mean": model_error_mean,
        "delta_local": delta_local_used,
        "delta_local_k": args.delta_local_k,
        "delta_local_quantile": delta_local_quantile,
        "delta_local_samples": args.delta_local_samples,
        "calib_ratio": args.calib_ratio,
        "expansion_quantile": args.expansion_quantile,
        "expansion_samples": args.expansion_samples,
        "expansion_theiler": args.expansion_theiler,
        "expansion_dim": exp_dim,
        "expansion_lag": exp_lag,
        "expansion_horizon": args.expansion_horizon,
        "expansion_Lq": expansion_q,
        "expansion_mean": expansion_mean,
        "growth_source": growth_source,
        "growth_horizon": growth_horizon,
        "growth_Lq": growth_q,
        "growth_Lmean": growth_mean,
        "bound_mode": bound_mode,
        "calibration_alpha": args.calibration_alpha,
        "calibration_floor": args.calibration_floor,
        "calibration_scale": scale,
        "calibration_samples": calibration_samples,
        "calibration_coverage": coverage,
        "horizon_model_cal": horizon_model_cal,
        "horizon_model_cal_time": horizon_model_cal_time,
        "coverage_test": coverage_test,
        "tightness_ratio": tightness_ratio,
        "slack_median": slack_median,
        "slack_p90": slack_p90,
        "leaf_coverage_stats": leaf_coverage_stats,
        "jac_quantile_coverages": jac_quantile_coverages,
        "score_pos_frac": score_pos_frac,
        "score_neg_frac": score_neg_frac,
        "score_zero_frac": score_zero_frac,
        "score_p10": score_p10,
        "score_p50": score_p50,
        "score_p90": score_p90,
        "score_mean": score_mean,
        "signed_med": signed_med,
        "c_global": c_global,
        "sigma_med": sigma_med,
        "sigma_p90": sigma_p90,
        "sigma_max": sigma_max,
        "pred_calib_med": pred_calib_med,
        "y_calib_med": y_calib_med,
        "l_calib_med": l_calib_med,
        "bin_count": bin_count,
        "bin_min_count": bin_min_count,
        "bin_med_count": bin_med_count,
        "bin_c_min": bin_c_min,
        "bin_c_med": bin_c_med,
        "bin_c_max": bin_c_max,
        "error_tolerance_used": tolerance,
        "selection_metric": args.selection_metric,
        "selection_horizon": best["selection"]["horizon"]
        if best.get("selection") and best["selection"]["horizon"] is not None
        else None,
    }


# CLI functions moved to horizon_cli.py
from src.horizon_cli import build_parser, load_config, setup_logging, main


if __name__ == "__main__":
    main()
