"""Helpers for horizon comparison tables."""

import numpy as np


def parse_row(row):
    """Parses a CSV row into a normalized dictionary."""
    n = len(row)
    if n == 52:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
            "error_tolerance": row[18],
            "error_tolerance_used": row[19],
            "calib_ratio": row[20],
            "model_error": row[21],
            "model_error_mode": row[22],
            "model_error_mean": row[23],
            "delta_local": row[24],
            "delta_local_k": row[25],
            "delta_local_quantile": row[26],
            "delta_local_samples": row[27],
            "horizon_model": row[28],
            "horizon_model_time": row[29],
            "horizon_est": row[30],
            "horizon_est_time": row[31],
            "expansion_quantile": row[32],
            "expansion_samples": row[33],
            "expansion_theiler": row[34],
            "expansion_dim": row[35],
            "expansion_lag": row[36],
            "expansion_horizon": row[37],
            "expansion_Lq": row[38],
            "expansion_mean": row[39],
            "growth_source": row[40],
            "growth_horizon": row[41],
            "growth_Lq": row[42],
            "growth_Lmean": row[43],
            "bound_mode": row[44],
            "calibration_alpha": row[45],
            "calibration_floor": row[46],
            "calibration_scale": row[47],
            "calibration_samples": row[48],
            "calibration_coverage": row[49],
            "horizon_model_cal": row[50],
            "horizon_model_cal_time": row[51],
        }
    if n == 48:
        parsed = {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
            "error_tolerance": row[18],
            "error_tolerance_used": row[19],
            "calib_ratio": row[20],
            "model_error": row[21],
            "model_error_mode": row[22],
            "model_error_mean": row[23],
            "horizon_model": row[24],
            "horizon_model_time": row[25],
            "horizon_est": row[26],
            "horizon_est_time": row[27],
            "expansion_quantile": row[28],
            "expansion_samples": row[29],
            "expansion_theiler": row[30],
            "expansion_dim": row[31],
            "expansion_lag": row[32],
            "expansion_horizon": row[33],
            "expansion_Lq": row[34],
            "expansion_mean": row[35],
            "growth_source": row[36],
            "growth_horizon": row[37],
            "growth_Lq": row[38],
            "growth_Lmean": row[39],
            "bound_mode": row[40],
            "calibration_alpha": row[41],
            "calibration_floor": row[42],
            "calibration_scale": row[43],
            "calibration_samples": row[44],
            "calibration_coverage": row[45],
            "horizon_model_cal": row[46],
            "horizon_model_cal_time": row[47],
        }
        parsed["delta_local"] = ""
        parsed["delta_local_k"] = ""
        parsed["delta_local_quantile"] = ""
        parsed["delta_local_samples"] = ""
        return parsed
    if n == 44:
        parsed = {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
            "error_tolerance": row[18],
            "error_tolerance_used": row[19],
            "calib_ratio": row[20],
            "model_error": row[21],
            "model_error_mode": row[22],
            "model_error_mean": row[23],
            "horizon_model": row[24],
            "horizon_model_time": row[25],
            "horizon_est": row[26],
            "horizon_est_time": row[27],
            "expansion_quantile": row[28],
            "expansion_samples": row[29],
            "expansion_theiler": row[30],
            "expansion_dim": row[31],
            "expansion_lag": row[32],
            "expansion_horizon": row[33],
            "expansion_Lq": row[34],
            "expansion_mean": row[35],
            "bound_mode": row[36],
            "calibration_alpha": row[37],
            "calibration_floor": row[38],
            "calibration_scale": row[39],
            "calibration_samples": row[40],
            "calibration_coverage": row[41],
            "horizon_model_cal": row[42],
            "horizon_model_cal_time": row[43],
        }
        parsed["growth_source"] = "state"
        parsed["growth_horizon"] = parsed.get("expansion_horizon")
        parsed["growth_Lq"] = parsed.get("expansion_Lq")
        parsed["growth_Lmean"] = parsed.get("expansion_mean")
        parsed["delta_local"] = ""
        parsed["delta_local_k"] = ""
        parsed["delta_local_quantile"] = ""
        parsed["delta_local_samples"] = ""
        return parsed
    if n == 25:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
            "error_tolerance": row[18],
            "error_tolerance_used": row[19],
            "calib_ratio": row[20],
            "model_error": row[21],
            "model_error_mode": row[22],
            "horizon_model": row[23],
            "horizon_model_time": row[24],
        }
    if n == 20:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
            "error_tolerance": row[18],
            "error_tolerance_used": row[19],
        }
    if n >= 18:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": row[5],
            "selection_horizon": row[6],
            "test_mse": row[7],
            "lyapunov_step": row[8],
            "lyapunov_time": row[9],
            "lyapunov_dim": row[10],
            "lyapunov_lag": row[11],
            "horizon_real": row[12],
            "horizon_real_time": row[13],
            "horizon_theory": row[14],
            "horizon_theory_time": row[15],
            "error_mode": row[16],
            "error_factor": row[17],
        }
    if n == 12:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": "",
            "selection_horizon": "",
            "test_mse": row[5],
            "lyapunov_step": row[6],
            "lyapunov_time": row[7],
            "horizon_real": row[8],
            "horizon_real_time": row[9],
            "horizon_theory": row[10],
            "horizon_theory_time": row[11],
            "error_mode": "absolute",
            "error_factor": "",
        }
    if n == 10:
        return {
            "dataset": row[0],
            "model": row[1],
            "dim": row[2],
            "lag": row[3],
            "val_mse": row[4],
            "selection_metric": "",
            "selection_horizon": "",
            "test_mse": row[5],
            "lyapunov_step": row[6],
            "lyapunov_time": "",
            "horizon_real": row[7],
            "horizon_real_time": "",
            "horizon_theory": row[8],
            "horizon_theory_time": "",
            "error_mode": "absolute",
            "error_factor": "",
            "error_tolerance": row[9],
            "error_tolerance_used": row[9],
        }
    return None


def format_row(row):
    """Formats a parsed row for table output."""
    return [
        row.get("model", ""),
        row.get("dim", ""),
        row.get("lag", ""),
        row.get("val_mse", ""),
        row.get("test_mse", ""),
        row.get("lyapunov_step", ""),
        row.get("lyapunov_dim", ""),
        row.get("lyapunov_lag", ""),
        row.get("horizon_real", ""),
        row.get("horizon_theory", ""),
        row.get("horizon_model", ""),
        row.get("horizon_real_time", ""),
        row.get("horizon_theory_time", ""),
        row.get("horizon_model_time", ""),
        row.get("horizon_est", ""),
        row.get("horizon_est_time", ""),
        row.get("model_error", ""),
        row.get("model_error_mode", ""),
        row.get("model_error_mean", ""),
        row.get("delta_local", ""),
        row.get("delta_local_k", ""),
        row.get("delta_local_quantile", ""),
        row.get("delta_local_samples", ""),
        row.get("growth_horizon", row.get("expansion_horizon", "")),
        row.get("growth_Lq", row.get("expansion_Lq", "")),
        row.get("growth_Lmean", row.get("expansion_mean", "")),
        row.get("growth_source", ""),
        row.get("bound_mode", ""),
        row.get("calibration_scale", ""),
        row.get("horizon_model_cal", ""),
        row.get("selection_metric", ""),
        row.get("selection_horizon", ""),
        row.get("error_mode", ""),
        row.get("error_tolerance_used", ""),
        row.get("calib_ratio", ""),
    ]


def format_result_row(result, model, args):
    """Formats a live run result for table output."""
    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            if not value and value != 0.0:
                return ""
            if value == float("inf"):
                return "inf"
            return f"{value:.6f}"
        return str(value)

    return [
        model,
        str(result.get("dim", "")),
        str(result.get("lag", "")),
        fmt(result.get("val_loss")),
        fmt(result.get("test_mse")),
        fmt(result.get("lyapunov_step")),
        fmt(result.get("lyapunov_dim")),
        fmt(result.get("lyapunov_lag")),
        fmt(result.get("horizon_real")),
        fmt(result.get("horizon_theory")),
        fmt(result.get("horizon_model")),
        fmt(result.get("horizon_real_time")),
        fmt(result.get("horizon_theory_time")),
        fmt(result.get("horizon_model_time")),
        fmt(result.get("horizon_est")),
        fmt(result.get("horizon_est_time")),
        fmt(result.get("model_error")),
        result.get("model_error_mode", ""),
        fmt(result.get("model_error_mean")),
        str(result.get("delta_local", "")),
        fmt(result.get("delta_local_k")),
        fmt(result.get("delta_local_quantile")),
        fmt(result.get("delta_local_samples")),
        fmt(result.get("growth_horizon", result.get("expansion_horizon"))),
        fmt(result.get("growth_Lq", result.get("expansion_Lq"))),
        fmt(result.get("growth_Lmean", result.get("expansion_mean"))),
        result.get("growth_source", ""),
        result.get("bound_mode", ""),
        fmt(result.get("calibration_scale")),
        fmt(result.get("horizon_model_cal")),
        result.get("selection_metric", ""),
        fmt(result.get("selection_horizon")),
        args.error_mode,
        fmt(result.get("error_tolerance_used")),
        fmt(result.get("calib_ratio")),
    ]


def parse_seeds(seed_str):
    """Parses a comma-separated list of seeds."""
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    return seeds


def median_value(values):
    """Returns the median of non-None values."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.median(np.array(values, dtype=np.float64)))


def format_median_row(result, model, seed_count, args):
    """Formats the median row for rigorous comparisons."""
    def fmt(value):
        if value is None:
            return ""
        if value == float("inf"):
            return "inf"
        return f"{value:.6f}"

    return [
        model,
        str(seed_count),
        fmt(result.get("dim")),
        fmt(result.get("lag")),
        fmt(result.get("val_mse")),
        fmt(result.get("test_mse")),
        fmt(result.get("lyapunov_step")),
        fmt(result.get("lyapunov_dim")),
        fmt(result.get("lyapunov_lag")),
        fmt(result.get("horizon_real")),
        fmt(result.get("horizon_theory")),
        fmt(result.get("horizon_model")),
        fmt(result.get("horizon_real_time")),
        fmt(result.get("horizon_theory_time")),
        fmt(result.get("horizon_model_time")),
        fmt(result.get("horizon_est")),
        fmt(result.get("horizon_est_time")),
        fmt(result.get("model_error")),
        fmt(result.get("model_error_mean")),
        str(result.get("delta_local", "")),
        fmt(result.get("delta_local_k")),
        fmt(result.get("delta_local_quantile")),
        fmt(result.get("delta_local_samples")),
        fmt(result.get("growth_horizon", result.get("expansion_horizon"))),
        fmt(result.get("growth_Lq", result.get("expansion_Lq"))),
        fmt(result.get("growth_Lmean", result.get("expansion_mean"))),
        result.get("growth_source", ""),
        fmt(result.get("calibration_scale")),
        fmt(result.get("horizon_model_cal")),
        args.selection_metric,
        args.error_mode,
        fmt(result.get("error_tolerance_used")),
    ]
