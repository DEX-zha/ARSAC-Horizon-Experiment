"""AI-driven prediction horizon experiment for chaotic time series."""

import argparse
import csv
import math
import os
import time

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
from src.horizon_progress import ProgressBar
from src.horizon_training import (
    build_multistep_supervised,
    train_lstm,
    train_lstm_multistep,
    train_mlp,
    train_mlp_multistep,
    train_quantile_mlp,
)
from src.horizon_utils import (
    build_supervised,
    estimate_expansion_quantile,
    estimate_lyapunov,
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
    horizon_from_model_bound_by_growth,
    set_seed,
    split_series,
    standardize_series,
)


def get_series(args):
    """Generates the selected chaotic time series."""
    if args.dataset == "logistic":
        return generate_logistic_map(
            args.series_len, r=args.r, x0=args.x0, warmup=args.warmup
        )
    if args.dataset == "lorenz":
        return generate_lorenz(
            args.series_len,
            dt=args.dt,
            sigma=args.sigma,
            rho=args.rho,
            beta=args.beta,
            warmup=args.warmup,
        )
    if args.dataset == "rossler":
        return generate_rossler(
            args.series_len,
            dt=args.dt,
            a=args.a,
            b=args.b,
            c=args.c,
            warmup=args.warmup,
        )
    if args.dataset == "mackey_glass":
        return generate_mackey_glass(
            args.series_len,
            tau=args.tau,
            beta=args.mg_beta,
            gamma=args.gamma,
            n=args.n,
            dt=args.dt,
            warmup=args.warmup,
        )
    raise ValueError("Unknown dataset")


def select_embedding(args, train_series, val_series, dim_values, lag_values, device):
    """Selects the best (dim, lag) embedding based on validation criteria."""
    best = None
    progress = None
    if args.progress:
        progress = ProgressBar(len(dim_values) * len(lag_values), label="embed-search")
    for dim in dim_values:
        for lag in lag_values:
            try:
                if args.train_multistep and args.model in ("mlp", "lstm"):
                    x_train, y_train = build_multistep_supervised(
                        train_series, dim, lag, horizon=args.train_horizon
                    )
                    x_val, y_val = build_multistep_supervised(
                        val_series, dim, lag, horizon=args.train_horizon
                    )
                else:
                    x_train, y_train = build_supervised(
                        train_series, dim, lag, horizon=1
                    )
                    x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
            except ValueError:
                if progress:
                    progress.update(1, extra=f"dim={dim} lag={lag}")
                continue

            if args.model == "linear":
                model = LinearAR(reg=args.linear_reg).fit(x_train, y_train)
                val_loss = evaluate_mse(model, x_val, y_val)
                wrapped = model
            elif args.model == "mlp":
                if args.train_multistep:
                    model, val_loss = train_mlp_multistep(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        input_dim=dim,
                        hidden_dim=args.mlp_hidden,
                        epochs=args.mlp_epochs,
                        lr=args.mlp_lr,
                        batch_size=args.mlp_batch,
                        patience=args.mlp_patience,
                        tf_start=args.tf_start,
                        tf_end=args.tf_end,
                        tf_val=args.tf_val,
                        device=device,
                        show_progress=False,
                    )
                else:
                    model, val_loss = train_mlp(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        input_dim=dim,
                        hidden_dim=args.mlp_hidden,
                        epochs=args.mlp_epochs,
                        lr=args.mlp_lr,
                        batch_size=args.mlp_batch,
                        patience=args.mlp_patience,
                        device=device,
                        show_progress=False,
                    )
                wrapped = TorchWrapper(model, device)
            else:
                if args.train_multistep:
                    model, val_loss = train_lstm_multistep(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        hidden_dim=args.lstm_hidden,
                        num_layers=args.lstm_layers,
                        epochs=args.lstm_epochs,
                        lr=args.lstm_lr,
                        batch_size=args.lstm_batch,
                        patience=args.lstm_patience,
                        tf_start=args.tf_start,
                        tf_end=args.tf_end,
                        tf_val=args.tf_val,
                        device=device,
                        show_progress=False,
                    )
                else:
                    model, val_loss = train_lstm(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        hidden_dim=args.lstm_hidden,
                        num_layers=args.lstm_layers,
                        epochs=args.lstm_epochs,
                        lr=args.lstm_lr,
                        batch_size=args.lstm_batch,
                        patience=args.lstm_patience,
                        device=device,
                        show_progress=False,
                    )
                wrapped = TorchSeqWrapper(model, device)

            selection = {
                "metric": args.selection_metric,
                "score": -val_loss,
                "horizon": None,
            }
            if args.selection_metric == "horizon":
                rmse_val = rolling_rmse(
                    wrapped, val_series, dim, lag, args.selection_horizon_max
                )
                base_err = rmse_val[0] if rmse_val.size > 0 else 0.0
                if args.error_mode == "relative":
                    tolerance = base_err * args.error_factor
                else:
                    tolerance = args.error_tolerance
                if not np.isfinite(tolerance) or tolerance <= 0:
                    horizon_val = 0
                else:
                    horizon_val = horizon_from_rmse(rmse_val, tolerance)
                selection["score"] = horizon_val
                selection["horizon"] = horizon_val

            if best is None:
                best = {
                    "dim": dim,
                    "lag": lag,
                    "val_loss": val_loss,
                    "model": wrapped,
                    "selection": selection,
                }
                if progress:
                    progress.update(
                        1,
                        extra=f"dim={dim} lag={lag} val={val_loss:.4f}",
                    )
                continue

            if args.selection_metric == "horizon":
                if selection["score"] > best["selection"]["score"]:
                    best = {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": val_loss,
                        "model": wrapped,
                        "selection": selection,
                    }
                elif selection["score"] == best["selection"]["score"]:
                    if val_loss < best["val_loss"]:
                        best = {
                            "dim": dim,
                            "lag": lag,
                            "val_loss": val_loss,
                            "model": wrapped,
                            "selection": selection,
                        }
            else:
                if val_loss < best["val_loss"]:
                    best = {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": val_loss,
                        "model": wrapped,
                        "selection": selection,
                    }
            if progress:
                extra = f"dim={dim} lag={lag} val={val_loss:.4f}"
                if selection["horizon"] is not None:
                    extra += f" h={selection['horizon']}"
                progress.update(1, extra=extra)
    if best is None:
        raise RuntimeError("No valid embedding configuration found.")
    if progress:
        progress.close()
    return best


def train_final_model(
    args,
    train_series,
    val_series,
    dim,
    lag,
    device,
    show_progress=False,
):
    """Trains the final model on train+val with the chosen embedding."""
    merged = np.concatenate([train_series, val_series], axis=0)
    if args.train_multistep and args.model in ("mlp", "lstm"):
        x_train, y_train = build_multistep_supervised(
            merged, dim, lag, horizon=args.train_horizon
        )
        x_val, y_val = build_multistep_supervised(
            val_series, dim, lag, horizon=args.train_horizon
        )
    else:
        x_train, y_train = build_supervised(merged, dim, lag, horizon=1)
        x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
    if args.model == "linear":
        model = LinearAR(reg=args.linear_reg).fit(x_train, y_train)
        return model
    if args.model == "mlp":
        if args.train_multistep:
            model, _ = train_mlp_multistep(
                x_train,
                y_train,
                x_val,
                y_val,
                input_dim=dim,
                hidden_dim=args.mlp_hidden,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                batch_size=args.mlp_batch,
                patience=args.mlp_patience,
                tf_start=args.tf_start,
                tf_end=args.tf_end,
                tf_val=args.tf_val,
                device=device,
                show_progress=show_progress,
            )
        else:
            model, _ = train_mlp(
                x_train,
                y_train,
                x_val,
                y_val,
                input_dim=dim,
                hidden_dim=args.mlp_hidden,
                epochs=args.mlp_epochs,
                lr=args.mlp_lr,
                batch_size=args.mlp_batch,
                patience=args.mlp_patience,
                device=device,
                show_progress=show_progress,
            )
        return TorchWrapper(model, device)
    if args.train_multistep:
        model, _ = train_lstm_multistep(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            epochs=args.lstm_epochs,
            lr=args.lstm_lr,
            batch_size=args.lstm_batch,
            patience=args.lstm_patience,
            tf_start=args.tf_start,
            tf_end=args.tf_end,
            tf_val=args.tf_val,
            device=device,
            show_progress=show_progress,
        )
    else:
        model, _ = train_lstm(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dim=args.lstm_hidden,
            num_layers=args.lstm_layers,
            epochs=args.lstm_epochs,
            lr=args.lstm_lr,
            batch_size=args.lstm_batch,
            patience=args.lstm_patience,
            device=device,
            show_progress=show_progress,
        )
    return TorchSeqWrapper(model, device)


def run_experiment(args):
    """Runs a full horizon experiment and writes summary CSV output."""
    def conformal_margin(scores, alpha):
        scores = np.asarray(scores, dtype=np.float64)
        scores = scores[np.isfinite(scores)]
        if scores.size == 0:
            return 0.0
        n = scores.size
        rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
        rank = max(0, min(rank, n - 1))
        sorted_scores = np.sort(scores)
        return float(sorted_scores[rank])

    def assign_conformal_bins(preds, jac_norms, bins, feature):
        preds = np.asarray(preds, dtype=np.float64)
        jac_norms = np.asarray(jac_norms, dtype=np.float64)
        if bins <= 1:
            return np.zeros(preds.shape[0], dtype=np.int64), None
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        pred_edges = np.quantile(preds, quantiles)
        pred_edges[0] = -np.inf
        pred_edges[-1] = np.inf
        pred_bins = np.digitize(preds, pred_edges[1:-1], right=False)
        if feature != "pred_jacobian":
            return pred_bins, (pred_edges,)
        jac_edges = np.quantile(jac_norms, quantiles)
        jac_edges[0] = -np.inf
        jac_edges[-1] = np.inf
        jac_bins = np.digitize(jac_norms, jac_edges[1:-1], right=False)
        return pred_bins * bins + jac_bins, (pred_edges, jac_edges)

    set_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    series = get_series(args)

    train_raw, val_raw, calib_raw, test_raw = split_series(
        series,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        calib_ratio=args.calib_ratio,
    )
    train_std, mean, std = standardize_series(train_raw)
    val_std = (val_raw - mean) / std
    calib_std = (calib_raw - mean) / std if calib_raw.size else val_std
    test_std = (test_raw - mean) / std

    dim_values = list(range(args.dim_min, args.dim_max + 1))
    lag_values = list(range(args.lag_min, args.lag_max + 1))

    t0 = time.time()
    best = select_embedding(
        args,
        train_std,
        val_std,
        dim_values,
        lag_values,
        device,
    )
    model = train_final_model(
        args,
        train_std,
        val_std,
        best["dim"],
        best["lag"],
        device,
        show_progress=args.progress,
    )

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

    if bound_mode == "horizon_conformal":
        x_train, y_train = build_horizon_dataset(
            model,
            train_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed,
            use_jacobian=args.horizon_use_jacobian,
        )
        x_val, y_val = build_horizon_dataset(
            model,
            val_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 1,
            use_jacobian=args.horizon_use_jacobian,
        )
        x_calib_h, y_calib_h = build_horizon_dataset(
            model,
            calib_series,
            best["dim"],
            best["lag"],
            args.horizon_max,
            tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 2,
            use_jacobian=args.horizon_use_jacobian,
        )
        x_test_h, y_test_h = build_horizon_dataset(
            model,
            test_std,
            best["dim"],
            best["lag"],
            args.horizon_max,
            tolerance,
            max_windows=args.horizon_samples,
            seed=args.seed + 3,
            use_jacobian=args.horizon_use_jacobian,
        )

        if x_train.size == 0:
            raise RuntimeError("Not enough data for conformal horizon training.")

        x_calib_raw = x_calib_h
        x_test_raw = x_test_h
        feat_mean = np.mean(x_train, axis=0)
        feat_std = np.std(x_train, axis=0)
        feat_std[feat_std == 0.0] = 1.0
        x_train = (x_train - feat_mean) / feat_std
        x_val = (x_val - feat_mean) / feat_std if x_val.size else x_val
        x_calib_h = (x_calib_h - feat_mean) / feat_std if x_calib_h.size else x_calib_h
        x_test_h = (x_test_h - feat_mean) / feat_std if x_test_h.size else x_test_h

        quantile = args.horizon_quantile
        if quantile is None:
            quantile = 1.0 - args.calibration_alpha
        horizon_model, _ = train_quantile_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=x_train.shape[1],
            quantile=quantile,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch,
            patience=args.mlp_patience,
            device=device,
            show_progress=args.progress,
        )
        horizon_wrapper = TorchWrapper(horizon_model, device)

        margin = 0.0
        margin_global = 0.0
        margin_by_bin = {}
        if x_calib_h.size:
            pred_calib = horizon_wrapper.predict_batch(x_calib_h).reshape(-1)
            pred_calib = np.clip(pred_calib, 1.0, float(args.horizon_max))
            jac_calib = (
                x_calib_raw[:, -1]
                if x_calib_raw.size and args.horizon_use_jacobian
                else np.zeros_like(pred_calib)
            )
            scores = y_calib_h - pred_calib
            margin_global = conformal_margin(scores, args.calibration_alpha)
            bin_ids, _ = assign_conformal_bins(
                pred_calib,
                jac_calib,
                args.conformal_bins,
                args.conformal_feature,
            )
            for bin_id in np.unique(bin_ids):
                bin_scores = scores[bin_ids == bin_id]
                if bin_scores.size >= args.conformal_min_bin:
                    margin_by_bin[int(bin_id)] = conformal_margin(
                        bin_scores, args.calibration_alpha
                    )
            margins = np.array(
                [margin_by_bin.get(int(b), margin_global) for b in bin_ids],
                dtype=np.float64,
            )
            margin = float(np.median(margins)) if margins.size else margin_global
            calibration_samples = int(len(y_calib_h))
            coverage = float(np.mean(pred_calib + margins >= y_calib_h))

        if x_test_h.size:
            pred_test = horizon_wrapper.predict_batch(x_test_h).reshape(-1)
            pred_test = np.clip(pred_test, 1.0, float(args.horizon_max))
            jac_test = (
                x_test_raw[:, -1]
                if x_test_raw.size and args.horizon_use_jacobian
                else np.zeros_like(pred_test)
            )
            test_bin_ids, _ = assign_conformal_bins(
                pred_test,
                jac_test,
                args.conformal_bins,
                args.conformal_feature,
            )
            test_margins = np.array(
                [margin_by_bin.get(int(b), margin_global) for b in test_bin_ids],
                dtype=np.float64,
            )
            pred_test_cal = np.clip(
                pred_test + test_margins, 1.0, float(args.horizon_max)
            )
            horizon_model_steps = float(np.median(pred_test))
            horizon_est_steps = float(np.mean(pred_test))
            horizon_model_cal = float(np.median(pred_test_cal))
        if y_test_h.size:
            horizon_window_median = float(np.median(y_test_h))
            horizon_window_mean = float(np.mean(y_test_h))

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
        model_error_mode = "conformal"
        scale = margin
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
        print(f"Plots saved to {rmse_path} and {log_path}")

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
    print(
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
        f"{'margin' if bound_mode == 'horizon_conformal' else 'scale'}={scale:.3f} "
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
        "error_tolerance_used": tolerance,
        "selection_metric": args.selection_metric,
        "selection_horizon": best["selection"]["horizon"]
        if best.get("selection") and best["selection"]["horizon"] is not None
        else None,
    }


def build_parser(add_help=True):
    """Builds the argument parser for the experiment CLI."""
    parser = argparse.ArgumentParser(
        description="AI-driven prediction horizon experiment for chaotic series",
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["logistic", "lorenz", "rossler", "mackey_glass"],
        default="lorenz",
    )
    parser.add_argument("--series-len", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--calib-ratio", type=float, default=0.05)

    parser.add_argument("--dim-min", type=int, default=2)
    parser.add_argument("--dim-max", type=int, default=8)
    parser.add_argument("--lag-min", type=int, default=1)
    parser.add_argument("--lag-max", type=int, default=8)
    parser.add_argument(
        "--model", type=str, choices=["linear", "mlp", "lstm"], default="mlp"
    )
    parser.add_argument("--linear-reg", type=float, default=1e-4)

    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch", type=int, default=64)
    parser.add_argument("--mlp-patience", type=int, default=12)
    parser.add_argument("--train-multistep", action="store_true")
    parser.add_argument("--train-horizon", type=int, default=5)
    parser.add_argument("--tf-start", type=float, default=1.0)
    parser.add_argument("--tf-end", type=float, default=0.2)
    parser.add_argument("--tf-val", type=float, default=0.0)

    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-epochs", type=int, default=80)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch", type=int, default=64)
    parser.add_argument("--lstm-patience", type=int, default=12)

    parser.add_argument("--horizon-max", type=int, default=50)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["val_mse", "horizon"],
        default="val_mse",
    )
    parser.add_argument("--selection-horizon-max", type=int, default=20)
    parser.add_argument(
        "--error-mode",
        type=str,
        choices=["absolute", "relative"],
        default="relative",
    )
    parser.add_argument("--error-factor", type=float, default=10.0)
    parser.add_argument("--error-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--bound-mode",
        type=str,
        choices=["probabilistic", "horizon_conformal"],
        default="probabilistic",
    )
    parser.add_argument("--horizon-quantile", type=float, default=None)
    parser.add_argument("--conformal-bins", type=int, default=1)
    parser.add_argument("--conformal-min-bin", type=int, default=50)
    parser.add_argument(
        "--conformal-feature",
        type=str,
        choices=["pred", "pred_jacobian"],
        default="pred",
    )
    parser.add_argument(
        "--delta-mode",
        type=str,
        choices=["quantile", "max", "mean_std"],
        default="quantile",
    )
    parser.add_argument("--delta-quantile", type=float, default=0.95)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--delta-local", action="store_true")
    parser.add_argument("--delta-local-k", type=int, default=20)
    parser.add_argument("--delta-local-quantile", type=float, default=None)
    parser.add_argument("--delta-local-samples", type=int, default=500)
    parser.add_argument("--horizon-samples", type=int, default=None)
    parser.add_argument(
        "--horizon-use-jacobian",
        dest="horizon_use_jacobian",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-horizon-jacobian",
        dest="horizon_use_jacobian",
        action="store_false",
    )
    parser.add_argument(
        "--growth-source",
        type=str,
        choices=["state", "error", "jacobian"],
        default="error",
    )
    parser.add_argument("--expansion-quantile", type=float, default=0.95)
    parser.add_argument("--expansion-samples", type=int, default=500)
    parser.add_argument("--expansion-theiler", type=int, default=10)
    parser.add_argument("--expansion-dim", type=int, default=None)
    parser.add_argument("--expansion-lag", type=int, default=None)
    parser.add_argument("--expansion-horizon", type=int, default=10)
    parser.add_argument("--calibrate-coverage", action="store_true")
    parser.add_argument("--calibration-alpha", type=float, default=0.1)
    parser.add_argument("--calibration-floor", type=float, default=1.0)

    parser.add_argument("--lyap-max-t", type=int, default=25)
    parser.add_argument("--lyap-theiler", type=int, default=10)
    parser.add_argument("--lyap-fit-start", type=int, default=1)
    parser.add_argument("--lyap-fit-end", type=int, default=10)
    parser.add_argument("--lyap-dim", type=int, default=None)
    parser.add_argument("--lyap-lag", type=int, default=None)

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", type=str, default="horizon")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")

    parser.add_argument("--r", type=float, default=4.0)
    parser.add_argument("--x0", type=float, default=0.2)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)

    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=5.7)

    parser.add_argument("--tau", type=int, default=17)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=10)
    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
