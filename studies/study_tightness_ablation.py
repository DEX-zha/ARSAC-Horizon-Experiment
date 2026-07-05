"""Ablation: do guard/debias/bins defaults cost tightness without coverage benefit?

The pipeline stacks several conservative mechanisms (coverage guard, debias,
Mondrian bins, block-quantile margin). Now that the labels are attractor-scale
and validated, this ablation measures each mechanism's actual contribution on
two contrasting systems (Lorenz: fast growth, mild calib->test shift;
Mackey-Glass: slow growth), 5 seeds, alpha=0.05, linear forecaster.
Goal: evidence-based defaults — drop what costs tightness with no coverage
gain, keep what protects worst-seed coverage.
Reproducible: python studies/study_tightness_ablation.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import build_parser, load_config
from src.horizon_experiment import run_experiment

ARMS = {
    "baseline": {},
    "no_guard": {"coverage_guard_quantile": None, "coverage_guard_min_scale": 0.0},
    "no_debias": {"debias_scale": 0.0},
    "no_guard_no_debias": {
        "coverage_guard_quantile": None,
        "coverage_guard_min_scale": 0.0,
        "debias_scale": 0.0,
    },
    "bins1": {"conformal_mode": "global"},
    "bins4": {"conformal_bins": 4},
    "block_q50": {"block_quantile": 0.5},
}
DATASETS = {"lorenz": 12000, "mackey_glass": 6000}
SEEDS = [0, 1, 2, 3, 4]


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser = build_parser()
    parser.set_defaults(**config)

    rows = []
    for dataset, series_len in DATASETS.items():
        for arm, overrides in ARMS.items():
            for seed in SEEDS:
                args = parser.parse_args([])
                args.dataset = dataset
                args.seed = seed
                args.model = "linear"
                args.series_len = series_len
                args.warmup = 1000 if dataset == "lorenz" else 400
                args.train_ratio = 0.6
                args.val_ratio = 0.15
                args.calib_ratio = 0.15
                args.quantile_ensemble = 1
                args.mlp_epochs = 30
                args.progress = False
                args.use_cuda = False
                args.output_dir = "outputs_ablation"
                for key, value in overrides.items():
                    setattr(args, key, value)
                result = run_experiment(args)
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "coverage": result.get("coverage_test"),
                        "tightness": result.get("tightness_ratio"),
                        "slack_p90": result.get("slack_p90"),
                    }
                )

    print("\n=== ABLATION (median [min] coverage | median tightness), alpha=0.05 ===")
    for dataset in DATASETS:
        print(f"--- {dataset}")
        for arm in ARMS:
            sub = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm]
            cov = np.array([r["coverage"] for r in sub], dtype=float)
            tight = np.array([r["tightness"] for r in sub], dtype=float)
            slack = np.array([r["slack_p90"] for r in sub], dtype=float)
            print(
                f"{arm:20s} cov {np.median(cov):.3f} [{cov.min():.3f}] "
                f"tight {np.median(tight):.3f} slack_p90 {np.median(slack):.1f}"
            )


if __name__ == "__main__":
    main()
