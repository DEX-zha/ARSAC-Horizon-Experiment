"""Rigorous multi-seed benchmark for chaotic horizon experiments."""

import argparse
import copy
import csv
import os

import numpy as np

from src.horizon_experiment import ProgressBar, build_parser, run_experiment


def parse_list(value, cast=str):
    """Parses a comma-separated list.

    Args:
        value: Comma-separated string.
        cast: Callable to convert each token.

    Returns:
        List of converted items.
    """
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
    """Summarizes run results with median statistics.

    Args:
        results: List of per-seed result dictionaries.

    Returns:
        Dictionary with median metrics.
    """
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
        "horizon_real_time": median_value([r.get("horizon_real_time") for r in results]),
        "horizon_theory_time": median_value([r.get("horizon_theory_time") for r in results]),
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
        "h_real_time_med",
        "h_theory_time_med",
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
            format_value(row.get("horizon_real_time")),
            format_value(row.get("horizon_theory_time")),
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
    lines.append(r"\begin{tabular}{l l r r r r r r r r}")
    lines.append(r"\hline")
    lines.append(
        r"Dataset & Model & Seeds & dim & lag & valMSE & testMSE & $H_{real}$ & $H_{theory}$ & $\lambda$ \\"
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
            format_value(row.get("lyapunov_step"), decimals=4),
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


def main():
    """Runs the benchmark across datasets, models, and seeds."""
    exp_parser = build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Rigorous benchmark for Lorenz/Rossler horizons",
        parents=[exp_parser],
    )
    parser.add_argument("--datasets", type=str, default="lorenz,rossler")
    parser.add_argument("--models", type=str, default="mlp,lstm")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--output-md", type=str, default="outputs/horizon_benchmark.md")
    parser.add_argument(
        "--output-runs", type=str, default="outputs/horizon_benchmark_runs.csv"
    )
    parser.add_argument(
        "--output-summary", type=str, default="outputs/horizon_benchmark_summary.csv"
    )
    parser.add_argument(
        "--output-tex", type=str, default="outputs/horizon_benchmark_table.tex"
    )
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")

    parser.set_defaults(
        series_len=8000,
        warmup=2000,
        dim_max=10,
        lag_max=10,
        horizon_max=120,
        selection_metric="horizon",
        selection_horizon_max=50,
        error_mode="absolute",
        error_tolerance=0.2,
        mlp_epochs=150,
        mlp_patience=20,
        lstm_epochs=150,
        lstm_patience=20,
        progress=False,
    )

    args = parser.parse_args()

    datasets = parse_list(args.datasets, str)
    models = parse_list(args.models, str)
    seeds = parse_list(args.seeds, int)

    raw_rows = []
    summary_rows = []

    total_runs = len(datasets) * len(models) * len(seeds)
    progress = ProgressBar(total_runs, label="benchmark") if args.progress else None

    for dataset in datasets:
        for model in models:
            results = []
            for seed in seeds:
                run_args = copy.deepcopy(args)
                run_args.dataset = dataset
                run_args.model = model
                run_args.seed = seed
                run_args.progress = False
                if run_args.plot:
                    run_args.plot_prefix = f"{run_args.plot_prefix}_{dataset}_{model}_s{seed}"
                result = run_experiment(run_args)
                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "dim": result.get("dim"),
                    "lag": result.get("lag"),
                    "val_mse": result.get("val_loss"),
                    "test_mse": result.get("test_mse"),
                    "lyapunov_step": result.get("lyapunov_step"),
                    "lyapunov_time": result.get("lyapunov_time"),
                    "lyapunov_dim": result.get("lyapunov_dim"),
                    "lyapunov_lag": result.get("lyapunov_lag"),
                    "horizon_real": result.get("horizon_real"),
                    "horizon_real_time": result.get("horizon_real_time"),
                    "horizon_theory": result.get("horizon_theory"),
                    "horizon_theory_time": result.get("horizon_theory_time"),
                    "error_tolerance_used": result.get("error_tolerance_used"),
                    "selection_metric": result.get("selection_metric"),
                    "selection_horizon": result.get("selection_horizon"),
                }
                results.append(record)
                raw_rows.append(
                    [
                        dataset,
                        model,
                        seed,
                        record["dim"],
                        record["lag"],
                        format_value(record["val_mse"]),
                        format_value(record["test_mse"]),
                        format_value(record["lyapunov_step"]),
                        format_value(record["lyapunov_time"]),
                        record["lyapunov_dim"],
                        record["lyapunov_lag"],
                        record["horizon_real"],
                        format_value(record["horizon_real_time"], decimals=3),
                        format_value(record["horizon_theory"]),
                        format_value(record["horizon_theory_time"], decimals=3),
                        args.selection_metric,
                        record.get("selection_horizon", ""),
                        args.error_mode,
                        format_value(record["error_tolerance_used"]),
                    ]
                )
                if progress:
                    extra = f"{dataset} {model} seed={seed}"
                    progress.update(1, extra=extra)

            summary = summarize_results(results)
            summary["dataset"] = dataset
            summary["model"] = model
            summary_rows.append(summary)

    summary_rows_sorted = sorted(
        summary_rows, key=lambda r: (r.get("dataset", ""), r.get("model", ""))
    )

    write_markdown_table(
        summary_rows_sorted, args.output_md, args.selection_metric, args.error_mode
    )
    write_latex_table(
        summary_rows_sorted, args.output_tex, args.selection_metric, args.error_mode
    )

    raw_header = [
        "dataset",
        "model",
        "seed",
        "dim",
        "lag",
        "val_mse",
        "test_mse",
        "lyapunov_step",
        "lyapunov_time",
        "lyapunov_dim",
        "lyapunov_lag",
        "horizon_real",
        "horizon_real_time",
        "horizon_theory",
        "horizon_theory_time",
        "selection_metric",
        "selection_horizon",
        "error_mode",
        "error_tolerance_used",
    ]
    write_csv_rows(raw_rows, args.output_runs, raw_header)

    summary_header = [
        "dataset",
        "model",
        "seed_count",
        "dim_med",
        "lag_med",
        "val_mse_med",
        "test_mse_med",
        "lyap_step_med",
        "lyap_dim_med",
        "lyap_lag_med",
        "horizon_real_med",
        "horizon_theory_med",
        "horizon_real_time_med",
        "horizon_theory_time_med",
        "error_tol_used_med",
    ]
    summary_csv_rows = []
    for row in summary_rows_sorted:
        summary_csv_rows.append(
            [
                row.get("dataset", ""),
                row.get("model", ""),
                row.get("seed_count", ""),
                format_value(row.get("dim")),
                format_value(row.get("lag")),
                format_value(row.get("val_mse")),
                format_value(row.get("test_mse")),
                format_value(row.get("lyapunov_step")),
                format_value(row.get("lyapunov_dim")),
                format_value(row.get("lyapunov_lag")),
                format_value(row.get("horizon_real")),
                format_value(row.get("horizon_theory")),
                format_value(row.get("horizon_real_time")),
                format_value(row.get("horizon_theory_time")),
                format_value(row.get("error_tolerance_used")),
            ]
        )
    write_csv_rows(summary_csv_rows, args.output_summary, summary_header)

    if progress:
        progress.close()

    print(f"Benchmark summary saved to {args.output_md}")


if __name__ == "__main__":
    main()
