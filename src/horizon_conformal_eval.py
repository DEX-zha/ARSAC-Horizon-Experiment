"""Multi-seed evaluation for conformal horizon lower bounds."""

import argparse
import copy
import sys

import torch

from src.horizon_benchmark_utils import (
    format_value,
    median_value,
    parse_list,
    write_csv_rows,
)
from src.horizon_experiment import ProgressBar, build_parser, load_config, run_experiment


def summarize_records(records, fields):
    """Summarizes numeric fields with medians across records."""
    summary = {}
    for field in fields:
        values = [r.get(field) for r in records]
        summary[field] = median_value(values)
    return summary


def main():
    """Runs multi-seed evaluation for conformal horizon bounds."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    known_args, _ = pre_parser.parse_known_args()
    config = load_config(known_args.config)

    exp_parser = build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Multi-seed evaluation for conformal horizon lower bounds",
        parents=[exp_parser],
    )
    parser.add_argument("--datasets", type=str, default="lorenz,rossler")
    parser.add_argument("--models", type=str, default="mlp,lstm")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument(
        "--output-runs", type=str, default="outputs/horizon_conformal_runs.csv"
    )
    parser.add_argument(
        "--output-summary", type=str, default="outputs/horizon_conformal_summary.csv"
    )
    if config:
        parser.set_defaults(**config)
    # Force conformal-eval defaults after config so bound_mode isn't overridden.
    parser.set_defaults(
        bound_mode="horizon_conformal",
        conformal_mode="bins",
        series_len=10000,
        warmup=500,
        calib_ratio=0.1,
        horizon_max=60,
        horizon_samples=800,
        conformal_cv_folds=5,
        horizon_quantile=0.15,
        conformal_bins=2,
    )

    args = parser.parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available. Exiting.")
        sys.exit(1)

    datasets = parse_list(args.datasets, str)
    models = parse_list(args.models, str)
    seeds = parse_list(args.seeds, int)

    run_rows = []
    summary_rows = []

    total_runs = len(datasets) * len(models) * len(seeds)
    progress = ProgressBar(total_runs, label="conformal-eval") if args.progress else None

    for dataset in datasets:
        for model in models:
            records = []
            for seed in seeds:
                run_args = copy.deepcopy(args)
                run_args.dataset = dataset
                run_args.model = model
                run_args.seed = seed
                run_args.progress = False
                result = run_experiment(run_args)

                leaf_stats = result.get("leaf_coverage_stats") or {}
                jac_stats = result.get("jac_quantile_coverages") or {}

                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "coverage_test": result.get("coverage_test"),
                    "tightness_ratio": result.get("tightness_ratio"),
                    "slack_median": result.get("slack_median"),
                    "slack_p90": result.get("slack_p90"),
                    "leaf_count": leaf_stats.get("leaf_count"),
                    "leaf_min": leaf_stats.get("leaf_min"),
                    "leaf_p10": leaf_stats.get("leaf_p10"),
                    "leaf_med": leaf_stats.get("leaf_med"),
                    "leaf_mean": leaf_stats.get("leaf_mean"),
                    "jac_q1": jac_stats.get("jac_q1"),
                    "jac_q2": jac_stats.get("jac_q2"),
                    "jac_q3": jac_stats.get("jac_q3"),
                    "jac_q4": jac_stats.get("jac_q4"),
                    "calibration_coverage": result.get("calibration_coverage"),
                    "calibration_alpha": args.calibration_alpha,
                    "conformal_mode": args.conformal_mode,
                    "horizon_consecutive_k": args.horizon_consecutive_k,
                    "horizon_calib_thin": args.horizon_calib_thin,
                    "c_global": result.get("c_global"),
                    "score_pos_frac": result.get("score_pos_frac"),
                    "score_neg_frac": result.get("score_neg_frac"),
                    "score_zero_frac": result.get("score_zero_frac"),
                    "score_p10": result.get("score_p10"),
                    "score_p50": result.get("score_p50"),
                    "score_p90": result.get("score_p90"),
                    "score_mean": result.get("score_mean"),
                    "signed_med": result.get("signed_med"),
                    "sigma_med": result.get("sigma_med"),
                    "sigma_p90": result.get("sigma_p90"),
                    "sigma_max": result.get("sigma_max"),
                    "pred_calib_med": result.get("pred_calib_med"),
                    "y_calib_med": result.get("y_calib_med"),
                    "l_calib_med": result.get("l_calib_med"),
                    "bin_count": result.get("bin_count"),
                    "bin_min_count": result.get("bin_min_count"),
                    "bin_med_count": result.get("bin_med_count"),
                    "bin_c_min": result.get("bin_c_min"),
                    "bin_c_med": result.get("bin_c_med"),
                    "bin_c_max": result.get("bin_c_max"),
                }
                records.append(record)

                run_rows.append(
                    [
                        record["dataset"],
                        record["model"],
                        record["seed"],
                        format_value(record["coverage_test"]),
                        format_value(record["tightness_ratio"]),
                        format_value(record["slack_median"]),
                        format_value(record["slack_p90"]),
                        record.get("leaf_count", ""),
                        format_value(record.get("leaf_min")),
                        format_value(record.get("leaf_p10")),
                        format_value(record.get("leaf_med")),
                        format_value(record.get("leaf_mean")),
                        format_value(record.get("jac_q1")),
                        format_value(record.get("jac_q2")),
                        format_value(record.get("jac_q3")),
                        format_value(record.get("jac_q4")),
                    format_value(record.get("calibration_coverage")),
                    format_value(record.get("calibration_alpha")),
                    record.get("conformal_mode", ""),
                    record.get("horizon_consecutive_k", ""),
                    record.get("horizon_calib_thin", ""),
                    format_value(record.get("c_global")),
                    format_value(record.get("score_pos_frac")),
                    format_value(record.get("score_neg_frac")),
                    format_value(record.get("score_zero_frac")),
                    format_value(record.get("score_p10")),
                    format_value(record.get("score_p50")),
                    format_value(record.get("score_p90")),
                    format_value(record.get("score_mean")),
                    format_value(record.get("signed_med")),
                    format_value(record.get("sigma_med")),
                    format_value(record.get("sigma_p90")),
                    format_value(record.get("sigma_max")),
                    format_value(record.get("pred_calib_med")),
                    format_value(record.get("y_calib_med")),
                    format_value(record.get("l_calib_med")),
                    record.get("bin_count", ""),
                    format_value(record.get("bin_min_count")),
                    format_value(record.get("bin_med_count")),
                    format_value(record.get("bin_c_min")),
                    format_value(record.get("bin_c_med")),
                    format_value(record.get("bin_c_max")),
                ]
            )

                if progress:
                    progress.update(1, extra=f"{dataset} {model} seed={seed}")

            summary_fields = [
                "coverage_test",
                "tightness_ratio",
                "slack_median",
                "slack_p90",
                "leaf_min",
                "leaf_p10",
                "leaf_med",
                "leaf_mean",
                "jac_q1",
                "jac_q2",
                "jac_q3",
                "jac_q4",
            ]
            summary = summarize_records(records, summary_fields)
            summary["dataset"] = dataset
            summary["model"] = model
            summary["seed_count"] = len(records)
            summary["calibration_alpha"] = args.calibration_alpha
            summary["conformal_mode"] = args.conformal_mode
            summary_rows.append(summary)

    runs_header = [
        "dataset",
        "model",
        "seed",
        "coverage_test",
        "tightness_ratio",
        "slack_median",
        "slack_p90",
        "leaf_count",
        "leaf_min",
        "leaf_p10",
        "leaf_med",
        "leaf_mean",
        "jac_q1",
        "jac_q2",
        "jac_q3",
        "jac_q4",
        "calibration_coverage",
        "calibration_alpha",
        "conformal_mode",
        "horizon_consecutive_k",
        "horizon_calib_thin",
        "c_global",
        "score_pos_frac",
        "score_neg_frac",
        "score_zero_frac",
        "score_p10",
        "score_p50",
        "score_p90",
        "score_mean",
        "signed_med",
        "sigma_med",
        "sigma_p90",
        "sigma_max",
        "pred_calib_med",
        "y_calib_med",
        "l_calib_med",
        "bin_count",
        "bin_min_count",
        "bin_med_count",
        "bin_c_min",
        "bin_c_med",
        "bin_c_max",
    ]
    write_csv_rows(run_rows, args.output_runs, runs_header)

    summary_header = [
        "dataset",
        "model",
        "seed_count",
        "coverage_test_med",
        "tightness_ratio_med",
        "slack_median_med",
        "slack_p90_med",
        "leaf_min_med",
        "leaf_p10_med",
        "leaf_med_med",
        "leaf_mean_med",
        "jac_q1_med",
        "jac_q2_med",
        "jac_q3_med",
        "jac_q4_med",
        "calibration_alpha",
        "conformal_mode",
    ]
    summary_rows_sorted = sorted(
        summary_rows, key=lambda r: (r.get("dataset", ""), r.get("model", ""))
    )
    summary_csv_rows = []
    for row in summary_rows_sorted:
        summary_csv_rows.append(
            [
                row.get("dataset", ""),
                row.get("model", ""),
                row.get("seed_count", ""),
                format_value(row.get("coverage_test")),
                format_value(row.get("tightness_ratio")),
                format_value(row.get("slack_median")),
                format_value(row.get("slack_p90")),
                format_value(row.get("leaf_min")),
                format_value(row.get("leaf_p10")),
                format_value(row.get("leaf_med")),
                format_value(row.get("leaf_mean")),
                format_value(row.get("jac_q1")),
                format_value(row.get("jac_q2")),
                format_value(row.get("jac_q3")),
                format_value(row.get("jac_q4")),
                format_value(row.get("calibration_alpha")),
                row.get("conformal_mode", ""),
            ]
        )
    write_csv_rows(summary_csv_rows, args.output_summary, summary_header)

    if progress:
        progress.close()

    print(f"Conformal evaluation CSVs saved to {args.output_runs} and {args.output_summary}")


if __name__ == "__main__":
    main()
