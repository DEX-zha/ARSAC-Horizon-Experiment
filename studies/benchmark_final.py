"""Definitive benchmark (Plan V2 Phase 4): real numbers for README and paper.

4 systems x 5 seeds x 2 models (linear, mlp) with production-grade settings,
alpha=0.05, attractor-scale tolerance 0.4 std, auto horizon_max (Lyapunov
times). Resumable: results are appended to outputs/benchmark_final.csv and
already-recorded (dataset, model, seed) triples are skipped, so the script can
be re-invoked in time-budgeted chunks:

    python studies/benchmark_final.py --budget-seconds 540

Exits 0 with "BENCHMARK COMPLETE" when all runs are recorded, else prints
"CHUNK DONE remaining=N".
"""

import argparse
import csv
import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import build_parser, load_config
from src.horizon_experiment import run_experiment

CSV_PATH = os.path.join("outputs", "benchmark_final.csv")
FIELDS = [
    "dataset", "model", "seed", "horizon_max", "dim", "lag",
    "coverage_test", "tightness_ratio", "slack_median", "slack_p90",
    "p_sat_test", "horizon_window_median", "horizon_model_cal",
    "horizon_certified", "lyapunov_step", "elapsed_s",
]
DATASETS = {
    "lorenz": {"series_len": 12000, "warmup": 1000},
    "rossler": {"series_len": 12000, "warmup": 1000},
    "mackey_glass": {"series_len": 6000, "warmup": 400},
    "logistic": {"series_len": 8000, "warmup": 200},
}
MODELS = ["linear", "mlp"]
SEEDS = [0, 1, 2, 3, 4]


def _done_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    with open(CSV_PATH, newline="") as f:
        return {(r["dataset"], r["model"], r["seed"]) for r in csv.DictReader(f)}


def _append_row(row):
    write_header = not os.path.exists(CSV_PATH)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-seconds", type=float, default=540.0)
    opts = ap.parse_args()
    deadline = time.time() + opts.budget_seconds

    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser = build_parser()
    parser.set_defaults(**config)

    done = _done_keys()
    todo = [
        (d, m, s)
        for d in DATASETS
        for m in MODELS
        for s in SEEDS
        if (d, m, str(s)) not in done
    ]
    if not todo:
        print("BENCHMARK COMPLETE")
        return

    for dataset, model, seed in todo:
        if time.time() >= deadline:
            break
        args = parser.parse_args([])
        args.dataset = dataset
        args.model = model
        args.seed = seed
        args.series_len = DATASETS[dataset]["series_len"]
        args.warmup = DATASETS[dataset]["warmup"]
        args.train_ratio = 0.6
        args.val_ratio = 0.15
        args.calib_ratio = 0.15
        # Production-grade quantile stack, trimmed grid for CPU cost on MLP.
        args.quantile_ensemble = 2
        args.mlp_epochs = 50
        args.dim_max = 6
        args.lag_max = 6
        args.progress = False
        args.use_cuda = False
        args.output_dir = "outputs_benchmark_final"
        t0 = time.time()
        result = run_experiment(args)
        _append_row(
            {
                "dataset": dataset,
                "model": model,
                "seed": seed,
                "horizon_max": args.horizon_max,
                "dim": result.get("dim"),
                "lag": result.get("lag"),
                "coverage_test": result.get("coverage_test"),
                "tightness_ratio": result.get("tightness_ratio"),
                "slack_median": result.get("slack_median"),
                "slack_p90": result.get("slack_p90"),
                "p_sat_test": result.get("p_sat_test"),
                "horizon_window_median": result.get("horizon_real_window_median"),
                "horizon_model_cal": result.get("horizon_model_cal"),
                "horizon_certified": result.get("horizon_certified"),
                "lyapunov_step": result.get("lyapunov_step"),
                "elapsed_s": round(time.time() - t0, 1),
            }
        )
        print(
            f"done {dataset}/{model}/seed{seed} "
            f"cov={result.get('coverage_test'):.3f} ({time.time() - t0:.0f}s)",
            flush=True,
        )

    remaining = len(_done_keys())
    total = len(DATASETS) * len(MODELS) * len(SEEDS)
    if remaining >= total:
        print("BENCHMARK COMPLETE")
    else:
        print(f"CHUNK DONE remaining={total - remaining}")


if __name__ == "__main__":
    main()
