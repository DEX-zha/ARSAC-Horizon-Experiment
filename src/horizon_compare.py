"""Compare MLP vs LSTM horizon results and render tables."""

import argparse
import copy
import csv
import os

import numpy as np

from src.horizon_experiment import build_parser, run_experiment


def parse_row(row):
    """Parses a CSV row into a normalized dictionary."""
    n = len(row)
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
            "horizon_real": row[10],
            "horizon_real_time": row[11],
            "horizon_theory": row[12],
            "horizon_theory_time": row[13],
            "error_mode": row[14],
            "error_factor": row[15],
            "error_tolerance": row[16],
            "error_tolerance_used": row[17],
        }
    if n == 13:
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
            "error_tolerance": row[12],
            "error_tolerance_used": row[12],
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
        row.get("model_error", ""),
        row.get("model_error_mode", ""),
        row.get("selection_metric", ""),
        row.get("selection_horizon", ""),
        row.get("error_mode", ""),
        row.get("error_tolerance_used", ""),
        row.get("calib_ratio", ""),
    ]


def format_result_row(result, dataset, model, args):
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
        fmt(result.get("model_error")),
        result.get("model_error_mode", ""),
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
        fmt(result.get("model_error")),
        args.selection_metric,
        args.error_mode,
        fmt(result.get("error_tolerance_used")),
    ]


def main():
    """CLI entry point."""
    exp_parser = build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Compare MLP vs LSTM results",
        parents=[exp_parser],
    )
    parser.add_argument("--mode", type=str, choices=["run", "csv", "rigorous"], default="run")
    parser.add_argument("--input", type=str, default="outputs/horizon_results.csv")
    parser.add_argument("--models", type=str, default="mlp,lstm")
    parser.add_argument("--output", type=str, default="outputs/horizon_compare.md")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    args = parser.parse_args()

    target_models = {m.strip() for m in args.models.split(",") if m.strip()}
    latest = {}

    if args.mode == "csv":
        if not os.path.exists(args.input):
            raise FileNotFoundError(args.input)
        with open(args.input, "r", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if row[0].lower() == "dataset":
                    continue
                parsed = parse_row(row)
                if not parsed:
                    continue
                if parsed["dataset"] != args.dataset:
                    continue
                if parsed["model"] not in target_models:
                    continue
                latest[parsed["model"]] = parsed
    elif args.mode == "run":
        for model in sorted(target_models):
            run_args = copy.deepcopy(args)
            run_args.model = model
            run_args.progress = False
            if run_args.plot:
                run_args.plot_prefix = f"{run_args.plot_prefix}_{model}"
            result = run_experiment(run_args)
            latest[model] = {
                "model": model,
                "dim": result.get("dim"),
                "lag": result.get("lag"),
                "val_mse": result.get("val_loss"),
                "selection_metric": result.get("selection_metric"),
                "selection_horizon": result.get("selection_horizon"),
                "test_mse": result.get("test_mse"),
                "lyapunov_step": result.get("lyapunov_step"),
                "lyapunov_time": result.get("lyapunov_time"),
                "lyapunov_dim": result.get("lyapunov_dim"),
                "lyapunov_lag": result.get("lyapunov_lag"),
                "horizon_real": result.get("horizon_real"),
                "horizon_real_time": result.get("horizon_real_time"),
                "horizon_theory": result.get("horizon_theory"),
                "horizon_theory_time": result.get("horizon_theory_time"),
                "horizon_model": result.get("horizon_model"),
                "horizon_model_time": result.get("horizon_model_time"),
                "model_error": result.get("model_error"),
                "model_error_mode": result.get("model_error_mode"),
                "calib_ratio": result.get("calib_ratio"),
                "error_mode": args.error_mode,
                "error_tolerance_used": result.get("error_tolerance_used"),
            }
    else:
        args.error_mode = "absolute"
        seeds = parse_seeds(args.seeds)
        for model in sorted(target_models):
            results = []
            for seed in seeds:
                run_args = copy.deepcopy(args)
                run_args.model = model
                run_args.seed = seed
                run_args.progress = False
                if run_args.plot:
                    run_args.plot_prefix = f"{run_args.plot_prefix}_{model}_s{seed}"
                result = run_experiment(run_args)
                results.append(result)

            latest[model] = {
                "model": model,
                "seed_count": len(results),
                "dim": median_value([r.get("dim") for r in results]),
                "lag": median_value([r.get("lag") for r in results]),
                "val_mse": median_value([r.get("val_loss") for r in results]),
                "test_mse": median_value([r.get("test_mse") for r in results]),
                "lyapunov_step": median_value([r.get("lyapunov_step") for r in results]),
                "lyapunov_dim": median_value([r.get("lyapunov_dim") for r in results]),
                "lyapunov_lag": median_value([r.get("lyapunov_lag") for r in results]),
                "horizon_real": median_value([r.get("horizon_real") for r in results]),
                "horizon_theory": median_value([r.get("horizon_theory") for r in results]),
                "horizon_model": median_value([r.get("horizon_model") for r in results]),
                "horizon_real_time": median_value(
                    [r.get("horizon_real_time") for r in results]
                ),
                "horizon_theory_time": median_value(
                    [r.get("horizon_theory_time") for r in results]
                ),
                "horizon_model_time": median_value(
                    [r.get("horizon_model_time") for r in results]
                ),
                "model_error": median_value([r.get("model_error") for r in results]),
                "error_tolerance_used": median_value(
                    [r.get("error_tolerance_used") for r in results]
                ),
            }

    if args.mode == "rigorous":
        header = [
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
            "model_error_med",
            "selection_metric",
            "error_mode",
            "error_tol_used_med",
        ]
    else:
        header = [
            "model",
            "dim",
            "lag",
            "val_mse",
            "test_mse",
            "lyap_step",
            "lyap_dim",
            "lyap_lag",
            "horizon_real",
            "horizon_theory",
            "horizon_model",
            "horizon_real_time",
            "horizon_theory_time",
            "horizon_model_time",
            "model_error",
            "model_error_mode",
            "selection_metric",
            "selection_horizon",
            "error_mode",
            "error_tol_used",
            "calib_ratio",
        ]

    lines = []
    lines.append(f"# Horizon comparison ({args.dataset})")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for model in sorted(target_models):
        row = latest.get(model)
        if not row:
            continue
        if args.mode == "csv":
            values = format_row(row)
        elif args.mode == "run":
            values = format_result_row(row, args.dataset, model, args)
        else:
            values = format_median_row(row, model, row.get("seed_count", 0), args)
        lines.append("| " + " | ".join(values) + " |")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Comparison saved to {args.output}")


if __name__ == "__main__":
    main()
