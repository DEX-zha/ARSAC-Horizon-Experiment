"""CSV/output/logging helpers for horizon_experiment."""

from __future__ import annotations

import csv
import logging
import math
import os

from src.horizon_plots import plot_log_divergence, plot_rmse
from src.horizon_experiment_core import _delta_local_quantile

CSV_NAME = "horizon_results.csv"
CSV_HEADER = [
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
    "horizon_certified",
    "lipschitz_G",
    "delta_sup",
]


def _fmt_optional(value, decimals):
    if value is None:
        return ""
    return f"{value:.{decimals}f}"



def _fmt_inf(value, decimals):
    return f"{value:.{decimals}f}" if math.isfinite(value) else "inf"



def _selection_horizon(best):
    if best.get("selection") and best["selection"].get("horizon") is not None:
        return best["selection"]["horizon"]
    return None



def _csv_row_part_a(args, best, base, lyap, stats):
    selection = _selection_horizon(best)
    return [
        args.dataset, args.model, best["dim"], best["lag"], f"{best['val_loss']:.6f}", args.selection_metric,
        selection if selection is not None else "", f"{base.test_mse:.6f}", f"{lyap.step:.6f}", f"{lyap.time:.6f}",
        lyap.dim, lyap.lag, base.horizon_real, f"{base.horizon_real_time:.3f}",
        _fmt_optional(stats["horizon_window_median"], 3), _fmt_optional(stats["horizon_window_mean"], 3),
        _fmt_inf(lyap.horizon_theory, 3), _fmt_inf(lyap.horizon_theory_time, 3),
    ]



def _csv_row_part_b(args, base, stats):
    delta_q = _delta_local_quantile(args)
    return [
        args.error_mode, args.error_factor, args.error_tolerance, f"{base.tolerance:.6f}", f"{args.calib_ratio:.3f}",
        f"{stats['model_error']:.6f}", stats["model_error_mode"], f"{stats['model_error_mean']:.6f}",
        str(stats["delta_local_used"]), args.delta_local_k, f"{delta_q:.3f}", args.delta_local_samples,
        _fmt_inf(stats["horizon_model_steps"], 3), _fmt_inf(stats["horizon_model_time"], 3),
        _fmt_inf(stats["horizon_est_steps"], 3), _fmt_inf(stats["horizon_est_time"], 3),
    ]



def _csv_row_part_c(args, exp_dim, exp_lag, stats):
    return [
        f"{args.expansion_quantile:.3f}", args.expansion_samples, args.expansion_theiler, exp_dim, exp_lag,
        args.expansion_horizon, f"{stats['expansion_q']:.6f}", f"{stats['expansion_mean']:.6f}",
        stats["growth_source"], stats["growth_horizon"], f"{stats['growth_q']:.6f}", f"{stats['growth_mean']:.6f}",
        args.bound_mode, f"{args.calibration_alpha:.3f}", f"{args.calibration_floor:.3f}", f"{stats['scale']:.6f}",
        stats["calibration_samples"], _fmt_optional(stats["coverage"], 3),
        _fmt_inf(stats["horizon_model_cal"], 3), _fmt_inf(stats["horizon_model_cal_time"], 3),
        _fmt_inf(stats.get("horizon_certified", 0.0), 3), f"{stats.get('lipschitz_G', 0.0):.6f}",
        f"{stats.get('delta_sup', 0.0):.6f}",
    ]



def _csv_row(args, best, base, lyap, stats, exp_dim, exp_lag):
    return _csv_row_part_a(args, best, base, lyap, stats) + _csv_row_part_b(args, base, stats) + _csv_row_part_c(args, exp_dim, exp_lag, stats)



def _resolve_csv_path(output_dir, header):
    csv_path = os.path.join(output_dir, CSV_NAME)
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != header:
            base, ext = os.path.splitext(CSV_NAME)
            suffix = 2
            while True:
                alt_name = f"{base}_v{suffix}{ext}"
                alt_path = os.path.join(output_dir, alt_name)
                if not os.path.exists(alt_path):
                    return alt_path
                suffix += 1
    return csv_path



def _write_csv(args, best, base, lyap, stats, exp_dim, exp_lag):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = _resolve_csv_path(args.output_dir, CSV_HEADER)
    write_header = not os.path.exists(csv_path)
    row = _csv_row(args, best, base, lyap, stats, exp_dim, exp_lag)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)



def _window_note(stats):
    if stats["horizon_window_median"] is None or stats["horizon_window_mean"] is None:
        return ""
    return f"h_win_med={stats['horizon_window_median']:.2f} h_win_mean={stats['horizon_window_mean']:.2f}"



def _selection_note(best):
    selection = _selection_horizon(best)
    return f"sel_h={selection}" if selection is not None else ""



def _summary_parts(args, best, base, lyap, stats):
    parts = [
        f"Best dim={best['dim']} lag={best['lag']} val={best['val_loss']:.6f}", f"test={base.test_mse:.6f}",
        f"lyap_step={lyap.step:.4f} lyap_time={lyap.time:.4f}", f"horizon_real={base.horizon_real} horizon_real_time={base.horizon_real_time:.2f}",
        f"horizon_theory={lyap.horizon_theory:.2f} horizon_theory_time={lyap.horizon_theory_time:.2f}",
        f"horizon_model={stats['horizon_model_steps']:.2f} horizon_model_time={stats['horizon_model_time']:.2f}",
        f"horizon_cal={stats['horizon_model_cal']:.2f} horizon_cal_time={stats['horizon_model_cal_time']:.2f}", f"horizon_est={stats['horizon_est_steps']:.2f} horizon_est_time={stats['horizon_est_time']:.2f}",
        f"delta={stats['model_error']:.4f} delta_mean={stats['model_error_mean']:.4f}", f"Lq={stats['growth_q']:.4f} Lmean={stats['growth_mean']:.4f} k={stats['growth_horizon']}",
        f"growth={stats['growth_source']} bound={args.bound_mode}", f"{'calib_c' if args.bound_mode == 'horizon_conformal' else 'scale'}={stats['scale']:.3f}", f"tol={base.tolerance:.6f}",
    ]
    window = _window_note(stats)
    if window:
        parts.append(window)
    selection = _selection_note(best)
    if selection:
        parts.append(selection)
    return parts



def _log_summary(args, best, base, lyap, stats, elapsed):
    parts = _summary_parts(args, best, base, lyap, stats)
    parts.append(f"elapsed={elapsed:.1f}s")
    logging.info(" ".join([p for p in parts if p]))



def _maybe_plot(args, base, lyap, stats):
    if not args.plot:
        return
    rmse_path = os.path.join(args.output_dir, f"{args.plot_prefix}_rmse.png")
    log_path = os.path.join(args.output_dir, f"{args.plot_prefix}_log.png")
    plot_rmse(
        base.rmse_by_h,
        base.horizon_real,
        lyap.horizon_theory,
        stats["horizon_model_steps"],
        stats["horizon_model_cal"],
        stats["horizon_est_steps"],
        rmse_path,
    )
    plot_log_divergence(base.rmse_by_h, lyap.step, log_path)
    logging.info(f"Plots saved to {rmse_path} and {log_path}")



def _return_from_stats(stats, base):
    result = stats.copy()
    result.update(
        {
            "horizon_real_window_median": result.pop("horizon_window_median"),
            "horizon_real_window_mean": result.pop("horizon_window_mean"),
            "horizon_model": result.pop("horizon_model_steps"),
            "horizon_est": result.pop("horizon_est_steps"),
            "expansion_Lq": result.pop("expansion_q"),
            "growth_Lq": result.pop("growth_q"),
            "growth_Lmean": result.pop("growth_mean"),
            "calibration_scale": result.pop("scale"),
            "calibration_coverage": result.pop("coverage"),
            "delta_local": result.pop("delta_local_used"),
            "error_tolerance_used": base.tolerance,
        }
    )
    return result



def _return_metrics(best, base, lyap):
    return {
        "dim": best["dim"],
        "lag": best["lag"],
        "val_loss": best["val_loss"],
        "test_mse": base.test_mse,
        "lyapunov_step": lyap.step,
        "lyapunov_time": lyap.time,
        "lyapunov_dim": lyap.dim,
        "lyapunov_lag": lyap.lag,
        "horizon_real": base.horizon_real,
        "horizon_real_time": base.horizon_real_time,
        "horizon_theory": lyap.horizon_theory,
        "horizon_theory_time": lyap.horizon_theory_time,
    }



def _return_args(args, exp_dim, exp_lag, selection_horizon):
    delta_local_quantile = _delta_local_quantile(args)
    return {
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
        "growth_source": args.growth_source,
        "growth_horizon": args.expansion_horizon,
        "bound_mode": args.bound_mode,
        "calibration_alpha": args.calibration_alpha,
        "calibration_floor": args.calibration_floor,
        "selection_metric": args.selection_metric,
        "selection_horizon": selection_horizon,
    }



def _build_return(best, base, lyap, stats, args, exp_dim, exp_lag):
    selection_horizon = _selection_horizon(best)
    result = _return_metrics(best, base, lyap)
    result.update(_return_args(args, exp_dim, exp_lag, selection_horizon))
    result.update(_return_from_stats(stats, base))
    if getattr(args, "return_embed_search", False):
        result["embed_search"] = best.get("search_history")
    return result


