"""Rigorous multi-seed benchmark for chaotic horizon experiments."""

import argparse
import copy
import sys

import torch

from src.horizon_benchmark_utils import (
    format_value,
    parse_list,
    summarize_results,
    write_csv_rows,
    write_latex_table,
    write_markdown_table,
)
from src.horizon_experiment import ProgressBar, build_parser, run_experiment


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
    )

    args = parser.parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available. Exiting.")
        sys.exit(1)

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
                    "horizon_real_window_median": result.get(
                        "horizon_real_window_median"
                    ),
                    "horizon_real_window_mean": result.get(
                        "horizon_real_window_mean"
                    ),
                    "horizon_theory": result.get("horizon_theory"),
                    "horizon_theory_time": result.get("horizon_theory_time"),
                    "horizon_model": result.get("horizon_model"),
                    "horizon_model_time": result.get("horizon_model_time"),
                    "horizon_est": result.get("horizon_est"),
                    "horizon_est_time": result.get("horizon_est_time"),
                    "model_error": result.get("model_error"),
                    "model_error_mode": result.get("model_error_mode"),
                    "model_error_mean": result.get("model_error_mean"),
                    "delta_local": result.get("delta_local"),
                    "delta_local_k": result.get("delta_local_k"),
                    "delta_local_quantile": result.get("delta_local_quantile"),
                    "delta_local_samples": result.get("delta_local_samples"),
                    "calib_ratio": result.get("calib_ratio"),
                    "expansion_Lq": result.get("expansion_Lq"),
                    "expansion_horizon": result.get("expansion_horizon"),
                    "expansion_mean": result.get("expansion_mean"),
                    "growth_source": result.get("growth_source"),
                    "growth_horizon": result.get("growth_horizon"),
                    "growth_Lq": result.get("growth_Lq"),
                    "growth_Lmean": result.get("growth_Lmean"),
                    "bound_mode": result.get("bound_mode"),
                    "calibration_scale": result.get("calibration_scale"),
                    "horizon_model_cal": result.get("horizon_model_cal"),
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
                        format_value(record.get("horizon_real_window_median")),
                        format_value(record.get("horizon_real_window_mean")),
                        format_value(record["horizon_theory"]),
                        format_value(record["horizon_theory_time"], decimals=3),
                        format_value(record["horizon_model"]),
                        format_value(record["horizon_model_time"], decimals=3),
                        format_value(record["horizon_est"]),
                        format_value(record["horizon_est_time"], decimals=3),
                        format_value(record["model_error"]),
                        record["model_error_mode"],
                        format_value(record.get("model_error_mean")),
                        record.get("delta_local"),
                        record.get("delta_local_k"),
                        format_value(record.get("delta_local_quantile")),
                        record.get("delta_local_samples"),
                        format_value(record.get("calib_ratio")),
                        format_value(record.get("expansion_Lq")),
                        format_value(record.get("expansion_horizon")),
                        format_value(record.get("expansion_mean")),
                        record.get("growth_source", ""),
                        format_value(record.get("growth_horizon")),
                        format_value(record.get("growth_Lq")),
                        format_value(record.get("growth_Lmean")),
                        record.get("bound_mode"),
                        format_value(record.get("calibration_scale")),
                        format_value(record.get("horizon_model_cal")),
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
        "horizon_real_window_median",
        "horizon_real_window_mean",
        "horizon_theory",
        "horizon_theory_time",
        "horizon_model",
        "horizon_model_time",
        "horizon_est",
        "horizon_est_time",
        "model_error",
        "model_error_mode",
        "model_error_mean",
        "delta_local",
        "delta_local_k",
        "delta_local_quantile",
        "delta_local_samples",
        "calib_ratio",
        "expansion_Lq",
        "expansion_horizon",
        "expansion_mean",
        "growth_source",
        "growth_horizon",
        "growth_Lq",
        "growth_Lmean",
        "bound_mode",
        "calibration_scale",
        "horizon_model_cal",
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
        "horizon_model_med",
        "horizon_real_time_med",
        "horizon_theory_time_med",
        "horizon_model_time_med",
        "horizon_est_med",
        "horizon_est_time_med",
        "delta_med",
        "delta_mean_med",
        "growth_horizon_med",
        "Lq_med",
        "L_mean_med",
        "growth_source",
        "calibration_scale_med",
        "horizon_model_cal_med",
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
                format_value(row.get("horizon_model")),
                format_value(row.get("horizon_real_time")),
                format_value(row.get("horizon_theory_time")),
                format_value(row.get("horizon_model_time")),
                format_value(row.get("horizon_est")),
                format_value(row.get("horizon_est_time")),
                format_value(row.get("model_error")),
                format_value(row.get("model_error_mean")),
                row.get("delta_local", ""),
                row.get("delta_local_k", ""),
                format_value(row.get("delta_local_quantile")),
                row.get("delta_local_samples", ""),
                format_value(row.get("growth_horizon")),
                format_value(row.get("growth_Lq")),
                format_value(row.get("growth_Lmean")),
                row.get("growth_source", ""),
                format_value(row.get("calibration_scale")),
                format_value(row.get("horizon_model_cal")),
                format_value(row.get("error_tolerance_used")),
            ]
        )
    write_csv_rows(summary_csv_rows, args.output_summary, summary_header)

    if progress:
        progress.close()

    print(f"Benchmark summary saved to {args.output_md}")


if __name__ == "__main__":
    main()
