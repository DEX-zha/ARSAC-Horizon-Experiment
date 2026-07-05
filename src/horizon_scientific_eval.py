"""Scientific-quality evaluation runner with optional quality checks."""

from __future__ import annotations

import argparse
import copy
import sys
import time

import numpy as np
import torch
from statistics import NormalDist

try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

from src.horizon_benchmark_utils import format_value, median_value, parse_list, write_csv_rows
from src.horizon_blocklen import politis_white_block_length
from src.horizon_cli import build_parser, load_config
from src.horizon_experiment import run_experiment
from src.horizon_progress import ProgressBar


def _min_value(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(min(values))


def _percentile(values, q):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _quality_thresholds(profile, alpha, horizon_max):
    target = 1.0 - float(alpha)
    if profile == "strict":
        return {
            "coverage_med": target - 0.01,
            "coverage_p10": target - 0.03,
            "coverage_min": target - 0.05,
            "tightness_med": 0.25,
            "slack_p90_med": float(horizon_max) * 0.60,
            "p_sat_test_med": 0.25,
        }
    if profile == "relaxed":
        return {
            "coverage_med": target - 0.03,
            "coverage_p10": target - 0.06,
            "coverage_min": target - 0.10,
            "tightness_med": 0.15,
            "slack_p90_med": float(horizon_max) * 0.85,
            "p_sat_test_med": 0.40,
        }
    return {}


def _summarize_records(records):
    coverage = [r.get("coverage_test") for r in records]
    coverage_lb = [r.get("coverage_lb") for r in records]
    tightness = [r.get("tightness_ratio") for r in records]
    slack_p90 = [r.get("slack_p90") for r in records]
    p_sat_test = [r.get("p_sat_test") for r in records]
    p_sat_calib = [r.get("p_sat_calib") for r in records]
    return {
        "coverage_med": median_value(coverage),
        "coverage_p10": _percentile(coverage, 0.10),
        "coverage_min": _min_value(coverage),
        "coverage_lb_med": median_value(coverage_lb),
        "coverage_lb_min": _min_value(coverage_lb),
        "tightness_med": median_value(tightness),
        "slack_p90_med": median_value(slack_p90),
        "p_sat_test_med": median_value(p_sat_test),
        "p_sat_calib_med": median_value(p_sat_calib),
    }


def _quality_checks(summary, thresholds):
    results = {}
    for key, threshold in thresholds.items():
        value = summary.get(key)
        if value is None:
            results[key] = None
        else:
            results[key] = bool(value >= threshold) if "slack" not in key and "p_sat" not in key else bool(value <= threshold)
    return results


def _wilson_lower_bound(k, n, alpha):
    """Wilson score lower bound, valid for i.i.d. Bernoulli hits only.

    Coverage hits from overlapping windows are serially dependent, so this
    bound is too optimistic there; it is kept only as a small-sample fallback
    for _block_bootstrap_lower_bound.
    """
    if n <= 0:
        return None
    z = NormalDist().inv_cdf(1.0 - alpha)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2.0 * n)
    radius = z * np.sqrt((phat * (1.0 - phat) / n) + (z * z) / (4.0 * n * n))
    lower = (center - radius) / denom
    return float(max(0.0, min(1.0, lower)))


def _block_bootstrap_lower_bound(hits, alpha, n_boot=1000, block_len=None, seed=0):
    """Lower confidence bound on coverage via circular moving-block bootstrap.

    Coverage hits come from overlapping test windows of the same trajectory,
    so they are strongly serially correlated (audit E4): the effective sample
    size is much smaller than n and the i.i.d. Wilson bound is far too
    optimistic. The moving-block bootstrap resamples wrapped blocks, which
    preserves the local dependence structure and widens the interval
    accordingly. Falls back to Wilson when n < 30 (too short for blocks).
    """
    hits = np.asarray(hits, dtype=int)
    n = len(hits)
    if n < 30:
        return _wilson_lower_bound(int(hits.sum()), n, alpha)
    if block_len is None:
        # Politis-White (2004, with Patton-Politis-White 2009 correction)
        # automatic block length; fall back to the old n^(1/3)*4 heuristic
        # when the estimator fails or returns a degenerate value.
        try:
            block_len = int(politis_white_block_length(hits))
        except Exception:
            block_len = None
        if block_len is None or block_len < 10:
            block_len = max(10, int(round(n ** (1.0 / 3.0) * 4)))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_len))
    offsets = np.arange(block_len)
    means = np.empty(int(n_boot), dtype=np.float64)
    for b in range(int(n_boot)):
        starts = rng.integers(0, n, size=n_blocks)
        sample = np.take(hits, (starts[:, None] + offsets[None, :]) % n)
        means[b] = float(np.mean(sample.reshape(-1)[:n]))
    lb = float(np.quantile(means, alpha))
    return float(max(0.0, min(1.0, lb)))


def _grouped_median(rows, value_key):
    grouped = {}
    for row in rows:
        key = (row.get("dim"), row.get("lag"))
        value = row.get(value_key)
        if value is None:
            continue
        grouped.setdefault(key, []).append(float(value))
    output = []
    for (dim, lag), values in grouped.items():
        if values:
            output.append({"dim": dim, "lag": lag, "value": float(np.median(values))})
    return output


def _write_report(rows, output_path, thresholds):
    lines = ["# Scientific quality report", ""]
    lines.append("Threshold profile:")
    if thresholds:
        for key, value in thresholds.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- `none`")
    lines.append("")
    header = [
        "dataset",
        "model",
        "seeds",
        "coverage_med",
        "coverage_p10",
        "coverage_min",
        "tightness_med",
        "slack_p90_med",
        "p_sat_test_med",
        "p_sat_calib_med",
        "checks",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        checks = row.get("checks") or {}
        if checks:
            parts = []
            for key, ok in checks.items():
                if ok is None:
                    parts.append(f"{key}=NA")
                else:
                    parts.append(f"{key}={'OK' if ok else 'FAIL'}")
            checks_text = "; ".join(parts)
        else:
            checks_text = ""
        values = [
            row.get("dataset", ""),
            row.get("model", ""),
            str(row.get("seed_count", "")),
            format_value(row.get("coverage_med")),
            format_value(row.get("coverage_p10")),
            format_value(row.get("coverage_min")),
            format_value(row.get("tightness_med")),
            format_value(row.get("slack_p90_med")),
            format_value(row.get("p_sat_test_med")),
            format_value(row.get("p_sat_calib_med")),
            checks_text,
        ]
        lines.append("| " + " | ".join(values) + " |")
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    known_args, _ = pre_parser.parse_known_args()
    config = load_config(known_args.config)

    exp_parser = build_parser(add_help=False)
    parser = argparse.ArgumentParser(
        description="Scientific-quality evaluation for horizon conformal bounds",
        parents=[exp_parser],
    )
    parser.add_argument("--datasets", type=str, default="lorenz,rossler,mackey_glass,logistic")
    parser.add_argument("--models", type=str, default="mlp,lstm")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--output-runs", type=str, default="outputs/scientific_runs.csv")
    parser.add_argument("--output-summary", type=str, default="outputs/scientific_summary.csv")
    parser.add_argument("--output-report", type=str, default="outputs/scientific_report.md")
    parser.add_argument("--max-hours", type=float, default=None)
    parser.add_argument("--quality-profile", type=str, choices=["off", "relaxed", "strict"], default="relaxed")
    parser.add_argument("--quality-fail", action="store_true", default=False)
    parser.add_argument("--coverage-ci-alpha", type=float, default=0.01)
    parser.add_argument("--coverage-lb-min-plus", type=float, default=0.0)
    parser.add_argument("--coverage-lb-med-plus", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb-project", type=str, default="arsac-horizon")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-advanced", action="store_true", default=False)
    parser.add_argument("--wandb-artifacts", action="store_true", default=False)

    if config:
        parser.set_defaults(**config)

    args = parser.parse_args()
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA requested but not available. Exiting.")
        sys.exit(1)

    datasets = parse_list(args.datasets, str)
    models = parse_list(args.models, str)
    seeds = parse_list(args.seeds, int)

    wandb_run = None
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("wandb not installed; disabling wandb logging.")
            args.wandb = False
        elif args.wandb_mode == "disabled":
            args.wandb = False
        else:
            tags = parse_list(args.wandb_tags, str) if args.wandb_tags else None
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.wandb_name,
                tags=tags,
                mode=args.wandb_mode,
                config=vars(args),
            )

    deadline = None
    if args.max_hours is not None and args.max_hours > 0:
        deadline = time.time() + float(args.max_hours) * 3600.0

    run_rows = []
    summary_rows = []
    embed_rows = []
    chaos_rows = []

    total_runs = len(datasets) * len(models) * len(seeds)
    progress = ProgressBar(total_runs, label="scientific-eval") if args.progress else None

    run_idx = 0
    start_time = time.time()
    stop_early = False
    for dataset in datasets:
        for model in models:
            records = []
            embed_records = []
            for seed in seeds:
                if deadline is not None and time.time() >= deadline:
                    stop_early = True
                    break
                run_args = copy.deepcopy(args)
                run_args.dataset = dataset
                run_args.model = model
                run_args.seed = seed
                run_args.progress = False
                if args.wandb and args.wandb_advanced:
                    run_args.return_embed_search = True
                result = run_experiment(run_args)

                coverage_hits = result.get("coverage_hits")
                test_samples = result.get("test_samples")
                coverage_hit_series = result.get("coverage_hit_series")
                coverage_lb = None
                if coverage_hit_series:
                    # Overlapping windows -> serially dependent hits: use the
                    # block bootstrap LB instead of the i.i.d. Wilson bound.
                    coverage_lb = _block_bootstrap_lower_bound(
                        coverage_hit_series, float(args.coverage_ci_alpha)
                    )
                elif coverage_hits is not None and test_samples:
                    coverage_lb = _wilson_lower_bound(
                        int(coverage_hits), int(test_samples), float(args.coverage_ci_alpha)
                    )
                record = {
                    "dataset": dataset,
                    "model": model,
                    "seed": seed,
                    "coverage_test": result.get("coverage_test"),
                    "coverage_lb": coverage_lb,
                    "coverage_hits": coverage_hits,
                    "test_samples": test_samples,
                    "tightness_ratio": result.get("tightness_ratio"),
                    "slack_median": result.get("slack_median"),
                    "slack_p90": result.get("slack_p90"),
                    "p_sat_calib": result.get("p_sat_calib"),
                    "p_sat_test": result.get("p_sat_test"),
                    "coverage_guard": result.get("coverage_guard"),
                    "debias_delta": result.get("debias_delta"),
                    "predictability_corr_jac": result.get("predictability_corr_jac"),
                    "predictability_corr_resid": result.get("predictability_corr_resid"),
                    "c_global": result.get("c_global"),
                }
                records.append(record)
                chaos_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "lyapunov_step": result.get("lyapunov_step"),
                        "expansion_Lq": result.get("expansion_Lq"),
                        "expansion_mean": result.get("expansion_mean"),
                        "growth_Lq": result.get("growth_Lq"),
                        "growth_Lmean": result.get("growth_Lmean"),
                        "horizon_real": result.get("horizon_real"),
                        "horizon_model": result.get("horizon_model"),
                        "horizon_model_cal": result.get("horizon_model_cal"),
                    }
                )
                run_rows.append(
                    [
                        record["dataset"],
                        record["model"],
                        record["seed"],
                        format_value(record.get("coverage_test")),
                        format_value(record.get("coverage_lb")),
                        record.get("coverage_hits") if record.get("coverage_hits") is not None else "",
                        record.get("test_samples") if record.get("test_samples") is not None else "",
                        format_value(record.get("tightness_ratio")),
                        format_value(record.get("slack_median")),
                        format_value(record.get("slack_p90")),
                        format_value(record.get("p_sat_calib")),
                        format_value(record.get("p_sat_test")),
                        format_value(record.get("coverage_guard")),
                        format_value(record.get("debias_delta")),
                        format_value(record.get("predictability_corr_jac")),
                        format_value(record.get("predictability_corr_resid")),
                        format_value(record.get("c_global")),
                    ]
                )
                if progress:
                    progress.update(1, extra=f"{dataset} {model} seed={seed}")
                run_idx += 1
                if args.wandb and wandb_run is not None:
                    elapsed = time.time() - start_time
                    avg = elapsed / float(run_idx)
                    eta = avg * float(max(0, total_runs - run_idx))
                    log_payload = {
                        "run_step": run_idx,
                        "progress": run_idx / float(total_runs),
                        "elapsed_s": elapsed,
                        "eta_s": eta,
                        "dataset": dataset,
                        "model": model,
                        "seed": seed,
                        "coverage_test": record.get("coverage_test"),
                        "coverage_lb": record.get("coverage_lb"),
                        "tightness_ratio": record.get("tightness_ratio"),
                        "slack_median": record.get("slack_median"),
                        "slack_p90": record.get("slack_p90"),
                        "p_sat_test": record.get("p_sat_test"),
                        "p_sat_calib": record.get("p_sat_calib"),
                        "coverage_guard": record.get("coverage_guard"),
                        "debias_delta": record.get("debias_delta"),
                        "predictability_corr_jac": record.get("predictability_corr_jac"),
                        "predictability_corr_resid": record.get("predictability_corr_resid"),
                        "lyapunov_step": result.get("lyapunov_step"),
                        "expansion_Lq": result.get("expansion_Lq"),
                        "expansion_mean": result.get("expansion_mean"),
                        "growth_Lq": result.get("growth_Lq"),
                        "growth_Lmean": result.get("growth_Lmean"),
                        "horizon_real": result.get("horizon_real"),
                        "horizon_theory": result.get("horizon_theory"),
                        "horizon_model": result.get("horizon_model"),
                        "horizon_model_cal": result.get("horizon_model_cal"),
                        "dim": result.get("dim"),
                        "lag": result.get("lag"),
                        "val_loss": result.get("val_loss"),
                        "test_mse": result.get("test_mse"),
                    }
                    if args.wandb_advanced:
                        embed_search = result.get("embed_search")
                        if embed_search:
                            for row in embed_search:
                                embed_rows.append(
                                    {
                                        "dataset": dataset,
                                        "model": model,
                                        "seed": seed,
                                        "dim": row.get("dim"),
                                        "lag": row.get("lag"),
                                        "val_loss": row.get("val_loss"),
                                        "selection_score": row.get("selection_score"),
                                        "selection_horizon": row.get("selection_horizon"),
                                    }
                                )
                                embed_records.append(
                                    {
                                        "dim": row.get("dim"),
                                        "lag": row.get("lag"),
                                        "val_loss": row.get("val_loss"),
                                        "selection_score": row.get("selection_score"),
                                        "selection_horizon": row.get("selection_horizon"),
                                    }
                                )
                            table = wandb.Table(columns=["dim", "lag", "val_loss", "selection_score", "selection_horizon"])
                            for row in embed_search:
                                table.add_data(
                                    row.get("dim"),
                                    row.get("lag"),
                                    row.get("val_loss"),
                                    row.get("selection_score"),
                                    row.get("selection_horizon"),
                                )
                            log_payload["embed_search"] = table
                    wandb.log(log_payload, step=run_idx)
            summary = _summarize_records(records)
            summary["dataset"] = dataset
            summary["model"] = model
            summary["seed_count"] = len(records)
            summary["calibration_alpha"] = args.calibration_alpha
            summary_rows.append(summary)
            if args.wandb and wandb_run is not None and args.wandb_advanced and embed_records:
                val_heat = _grouped_median(embed_records, "val_loss")
                score_heat = _grouped_median(embed_records, "selection_score")
                if val_heat:
                    table = wandb.Table(columns=["dim", "lag", "value"])
                    for row in val_heat:
                        table.add_data(row["dim"], row["lag"], row["value"])
                    wandb.log(
                        {
                            f"heatmap/{dataset}/{model}/val_loss": wandb.plot_table(
                                "wandb/heatmap/v1",
                                table,
                                {"x": "dim", "y": "lag", "value": "value"},
                                {"title": f"{dataset}-{model} val_loss"},
                            )
                        }
                    )
                if score_heat:
                    table = wandb.Table(columns=["dim", "lag", "value"])
                    for row in score_heat:
                        table.add_data(row["dim"], row["lag"], row["value"])
                    wandb.log(
                        {
                            f"heatmap/{dataset}/{model}/selection_score": wandb.plot_table(
                                "wandb/heatmap/v1",
                                table,
                                {"x": "dim", "y": "lag", "value": "value"},
                                {"title": f"{dataset}-{model} selection_score"},
                            )
                        }
                    )
            if stop_early:
                break
        if stop_early:
            break

    runs_header = [
        "dataset",
        "model",
        "seed",
        "coverage_test",
        "coverage_lb",
        "coverage_hits",
        "test_samples",
        "tightness_ratio",
        "slack_median",
        "slack_p90",
        "p_sat_calib",
        "p_sat_test",
        "coverage_guard",
        "debias_delta",
        "predictability_corr_jac",
        "predictability_corr_resid",
        "c_global",
    ]
    write_csv_rows(run_rows, args.output_runs, runs_header)

    summary_header = [
        "dataset",
        "model",
        "seed_count",
        "coverage_med",
        "coverage_p10",
        "coverage_min",
        "coverage_lb_med",
        "coverage_lb_min",
        "tightness_med",
        "slack_p90_med",
        "p_sat_test_med",
        "p_sat_calib_med",
        "calibration_alpha",
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
                format_value(row.get("coverage_med")),
                format_value(row.get("coverage_p10")),
                format_value(row.get("coverage_min")),
                format_value(row.get("coverage_lb_med")),
                format_value(row.get("coverage_lb_min")),
                format_value(row.get("tightness_med")),
                format_value(row.get("slack_p90_med")),
                format_value(row.get("p_sat_test_med")),
                format_value(row.get("p_sat_calib_med")),
                format_value(row.get("calibration_alpha")),
            ]
        )
    write_csv_rows(summary_csv_rows, args.output_summary, summary_header)

    embed_csv = None
    if embed_rows:
        embed_csv = "outputs/scientific_embed_search.csv"
        embed_header = [
            "dataset",
            "model",
            "seed",
            "dim",
            "lag",
            "val_loss",
            "selection_score",
            "selection_horizon",
        ]
        embed_csv_rows = []
        for row in embed_rows:
            embed_csv_rows.append(
                [
                    row.get("dataset", ""),
                    row.get("model", ""),
                    row.get("seed", ""),
                    row.get("dim", ""),
                    row.get("lag", ""),
                    format_value(row.get("val_loss")),
                    format_value(row.get("selection_score")),
                    row.get("selection_horizon", ""),
                ]
            )
        write_csv_rows(embed_csv_rows, embed_csv, embed_header)

    thresholds = _quality_thresholds(args.quality_profile, args.calibration_alpha, args.horizon_max)
    if thresholds:
        target = 1.0 - float(args.calibration_alpha)
        thresholds["coverage_lb_med"] = min(1.0, (target - 0.01) + float(args.coverage_lb_med_plus))
        thresholds["coverage_lb_min"] = min(1.0, (target - 0.03) + float(args.coverage_lb_min_plus))
    report_rows = []
    any_fail = False
    for row in summary_rows_sorted:
        checks = _quality_checks(row, thresholds) if thresholds else {}
        if checks:
            for ok in checks.values():
                if ok is False:
                    any_fail = True
                    break
        row_with_checks = row.copy()
        row_with_checks["checks"] = checks
        report_rows.append(row_with_checks)
    _write_report(report_rows, args.output_report, thresholds)

    if progress:
        progress.close()

    if args.wandb and wandb_run is not None:
        for row in summary_rows_sorted:
            wandb.log(
                {
                    "summary/dataset": row.get("dataset"),
                    "summary/model": row.get("model"),
                    "summary/seed_count": row.get("seed_count"),
                    "summary/coverage_med": row.get("coverage_med"),
                    "summary/coverage_p10": row.get("coverage_p10"),
                    "summary/coverage_min": row.get("coverage_min"),
                    "summary/coverage_lb_med": row.get("coverage_lb_med"),
                    "summary/coverage_lb_min": row.get("coverage_lb_min"),
                    "summary/tightness_med": row.get("tightness_med"),
                    "summary/slack_p90_med": row.get("slack_p90_med"),
                    "summary/p_sat_test_med": row.get("p_sat_test_med"),
                    "summary/p_sat_calib_med": row.get("p_sat_calib_med"),
                }
            )
        if args.wandb_advanced and chaos_rows:
            lyap = [r.get("lyapunov_step") for r in chaos_rows if r.get("lyapunov_step") is not None]
            lq = [r.get("expansion_Lq") for r in chaos_rows if r.get("expansion_Lq") is not None]
            growth = [r.get("growth_Lq") for r in chaos_rows if r.get("growth_Lq") is not None]
            if lyap:
                wandb.log({"dist/lyapunov_step": wandb.Histogram(lyap)})
            if lq:
                wandb.log({"dist/expansion_Lq": wandb.Histogram(lq)})
            if growth:
                wandb.log({"dist/growth_Lq": wandb.Histogram(growth)})
        if args.wandb_artifacts:
            artifact = wandb.Artifact("scientific_eval", type="dataset")
            artifact.add_file(args.output_runs)
            artifact.add_file(args.output_summary)
            artifact.add_file(args.output_report)
            if embed_csv:
                artifact.add_file(embed_csv)
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    if stop_early:
        print("Time budget reached. Partial results saved.")

    print(f"Runs saved to {args.output_runs}")
    print(f"Summary saved to {args.output_summary}")
    print(f"Report saved to {args.output_report}")

    if args.quality_fail and any_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
