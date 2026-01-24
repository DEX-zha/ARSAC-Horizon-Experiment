"""Helpers for horizon benchmark reporting."""

import csv
import os

import numpy as np


def parse_list(value, cast=str):
    """Parses a comma-separated list."""
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(cast(part))
    return items


def median_value(values):
    """Returns the median of non-None values."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.median(np.array(values, dtype=np.float64)))


def format_value(value, decimals=6):
    """Formats numeric values for tables and CSV output."""
    if value is None:
        return ""
    if value == float("inf"):
        return "inf"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    fmt = f"{{:.{decimals}f}}"
    return fmt.format(float(value))


def summarize_results(results):
    """Summarizes run results with median statistics."""
    return {
        "seed_count": len(results),
        "dim": median_value([r.get("dim") for r in results]),
        "lag": median_value([r.get("lag") for r in results]),
        "val_mse": median_value([r.get("val_mse") for r in results]),
        "test_mse": median_value([r.get("test_mse") for r in results]),
        "lyapunov_step": median_value([r.get("lyapunov_step") for r in results]),
        "lyapunov_dim": median_value([r.get("lyapunov_dim") for r in results]),
        "lyapunov_lag": median_value([r.get("lyapunov_lag") for r in results]),
        "horizon_real": median_value([r.get("horizon_real") for r in results]),
        "horizon_theory": median_value([r.get("horizon_theory") for r in results]),
        "horizon_model": median_value([r.get("horizon_model") for r in results]),
        "horizon_real_time": median_value([r.get("horizon_real_time") for r in results]),
        "horizon_theory_time": median_value(
            [r.get("horizon_theory_time") for r in results]
        ),
        "horizon_model_time": median_value(
            [r.get("horizon_model_time") for r in results]
        ),
        "horizon_est": median_value([r.get("horizon_est") for r in results]),
        "horizon_est_time": median_value(
            [r.get("horizon_est_time") for r in results]
        ),
        "model_error": median_value([r.get("model_error") for r in results]),
        "model_error_mean": median_value(
            [r.get("model_error_mean") for r in results]
        ),
        "delta_local": results[0].get("delta_local") if results else "",
        "delta_local_k": results[0].get("delta_local_k") if results else "",
        "delta_local_quantile": results[0].get("delta_local_quantile") if results else "",
        "delta_local_samples": results[0].get("delta_local_samples") if results else "",
        "expansion_Lq": median_value([r.get("expansion_Lq") for r in results]),
        "expansion_mean": median_value([r.get("expansion_mean") for r in results]),
        "expansion_horizon": median_value(
            [r.get("expansion_horizon") for r in results]
        ),
        "growth_source": results[0].get("growth_source") if results else "",
        "growth_horizon": median_value([r.get("growth_horizon") for r in results]),
        "growth_Lq": median_value([r.get("growth_Lq") for r in results]),
        "growth_Lmean": median_value([r.get("growth_Lmean") for r in results]),
        "calibration_scale": median_value(
            [r.get("calibration_scale") for r in results]
        ),
        "horizon_model_cal": median_value(
            [r.get("horizon_model_cal") for r in results]
        ),
        "error_tolerance_used": median_value(
            [r.get("error_tolerance_used") for r in results]
        ),
    }


def write_markdown_table(rows, output_path, selection_metric, error_mode):
    """Writes a Markdown summary table."""
    header = [
        "dataset",
        "model",
        "seeds",
        "dim_med",
        "lag_med",
        "val_mse_med",
        "test_mse_med",
        "lyap_step_med",
        "lyap_dim_med",
        "lyap_lag_med",
        "h_real_med",
        "h_theory_med",
        "h_model_med",
        "h_real_time_med",
        "h_theory_time_med",
        "h_model_time_med",
        "h_est_med",
        "h_est_time_med",
        "delta_med",
        "delta_mean_med",
        "delta_local",
        "delta_local_k",
        "delta_local_quantile",
        "delta_local_samples",
        "growth_horizon_med",
        "Lq_med",
        "L_mean_med",
        "growth_source",
        "calibration_scale_med",
        "horizon_model_cal_med",
        "selection_metric",
        "error_mode",
        "error_tol_used_med",
    ]
    lines = ["# Horizon benchmark (rigorous)", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        values = [
            row.get("dataset", ""),
            row.get("model", ""),
            str(row.get("seed_count", "")),
            format_value(row.get("dim")),
            format_value(row.get("lag")),
            format_value(row.get("val_mse")),
            format_value(row.get("test_mse")),
            format_value(row.get("lyapunov_step")),
            format_value(row.get("lyapunov_dim")),
            format_value(row.get("lyapunov_lag")),
            format_value(row.get("horizon_real")),
            format_value(row.get("horizon_theory")),
            format_value(row.get("horizon_model")),
            format_value(row.get("horizon_real_time")),
            format_value(row.get("horizon_theory_time")),
            format_value(row.get("horizon_model_time")),
            format_value(row.get("horizon_est")),
            format_value(row.get("horizon_est_time")),
            format_value(row.get("model_error")),
            format_value(row.get("model_error_mean")),
            str(row.get("delta_local", "")),
            str(row.get("delta_local_k", "")),
            format_value(row.get("delta_local_quantile")),
            str(row.get("delta_local_samples", "")),
            format_value(row.get("growth_horizon")),
            format_value(row.get("growth_Lq")),
            format_value(row.get("growth_Lmean")),
            row.get("growth_source", ""),
            format_value(row.get("calibration_scale")),
            format_value(row.get("horizon_model_cal")),
            selection_metric,
            error_mode,
            format_value(row.get("error_tolerance_used")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex_table(rows, output_path, selection_metric, error_mode):
    """Writes a LaTeX table for the benchmark summary."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{l l r r r r r r r r r r r r r r r r r}")
    lines.append(r"\hline")
    lines.append(
        r"Dataset & Model & Seeds & dim & lag & valMSE & testMSE & $H_{real}$ & $H_{theory}$ & $H_{model}$ & $H_{est}$ & $\lambda$ & $\delta_q$ & $\delta_{mean}$ & $k$ & $L_q$ & $L_{mean}$ & $s$ & $H_{cal}$ \\"
    )
    lines.append(r"\hline")
    for row in rows:
        values = [
            row.get("dataset", ""),
            row.get("model", ""),
            str(row.get("seed_count", "")),
            format_value(row.get("dim"), decimals=2),
            format_value(row.get("lag"), decimals=2),
            format_value(row.get("val_mse"), decimals=6),
            format_value(row.get("test_mse"), decimals=6),
            format_value(row.get("horizon_real"), decimals=2),
            format_value(row.get("horizon_theory"), decimals=2),
            format_value(row.get("horizon_model"), decimals=2),
            format_value(row.get("horizon_est"), decimals=2),
            format_value(row.get("lyapunov_step"), decimals=4),
            format_value(row.get("model_error"), decimals=4),
            format_value(row.get("model_error_mean"), decimals=4),
            format_value(row.get("growth_horizon"), decimals=2),
            format_value(row.get("growth_Lq"), decimals=4),
            format_value(row.get("growth_Lmean"), decimals=4),
            format_value(row.get("calibration_scale"), decimals=3),
            format_value(row.get("horizon_model_cal"), decimals=2),
        ]
        lines.append(" & ".join(values) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(
        rf"\caption{{Rigorous horizon benchmark (selection={selection_metric}, error={error_mode}).}}"
    )
    lines.append(r"\end{table}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_csv_rows(rows, output_path, header):
    """Writes rows to CSV with a header."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
