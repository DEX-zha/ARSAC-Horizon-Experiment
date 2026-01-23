"""Compare MLP vs LSTM horizon results and render tables."""

import argparse
import copy
import csv
import os

from src.horizon_compare_utils import (
    format_median_row,
    format_result_row,
    format_row,
    median_value,
    parse_row,
    parse_seeds,
)
from src.horizon_experiment import build_parser, run_experiment


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
                "delta_local_quantile": results[0].get("delta_local_quantile")
                if results
                else "",
                "delta_local_samples": results[0].get("delta_local_samples")
                if results
                else "",
                "expansion_Lq": median_value([r.get("expansion_Lq") for r in results]),
                "expansion_mean": median_value(
                    [r.get("expansion_mean") for r in results]
                ),
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
            "horizon_est_med",
            "horizon_est_time_med",
            "model_error_med",
            "model_error_mean_med",
            "delta_local",
            "delta_local_k",
            "delta_local_quantile",
            "delta_local_samples",
            "growth_horizon_med",
            "growth_Lq_med",
            "growth_Lmean_med",
            "growth_source",
            "calibration_scale_med",
            "horizon_model_cal_med",
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
            "horizon_est",
            "horizon_est_time",
            "model_error",
            "model_error_mode",
            "model_error_mean",
            "delta_local",
            "delta_local_k",
            "delta_local_quantile",
            "delta_local_samples",
            "growth_horizon",
            "growth_Lq",
            "growth_Lmean",
            "growth_source",
            "bound_mode",
            "calibration_scale",
            "horizon_model_cal",
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
