"""Validation study: Lyapunov-time auto horizon_max + absolute tolerance.

Runs the production pipeline (config.yaml defaults) with the new auto
horizon_max (max(3 T_lambda, 1.2 * H_theory), capped [30, 400] and by the
data budget) and the attractor-scale tolerance (absolute 0.4 std), on
3 systems x 5 seeds, linear forecaster. Reports coverage / tightness /
slack / saturation per system to validate that horizons are chaos-limited
and coverage holds. Reproducible: python studies/study_lyap_hmax.py
"""

import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import build_parser, load_config
from src.horizon_experiment import run_experiment

CONFIGS = {
    "lorenz": {"series_len": 12000, "warmup": 1000},
    "rossler": {"series_len": 12000, "warmup": 1000},
    "mackey_glass": {"series_len": 6000, "warmup": 400},
}
SEEDS = [0, 1, 2, 3, 4]


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser = build_parser()
    parser.set_defaults(**config)

    rows = []
    for dataset, overrides in CONFIGS.items():
        for seed in SEEDS:
            args = parser.parse_args([])
            args.dataset = dataset
            args.seed = seed
            args.model = "linear"
            args.series_len = overrides["series_len"]
            args.warmup = overrides["warmup"]
            args.train_ratio = 0.6
            args.val_ratio = 0.15
            args.calib_ratio = 0.15  # test = 0.10
            args.quantile_ensemble = 1
            args.mlp_epochs = 30
            args.progress = False
            args.use_cuda = False
            args.output_dir = "outputs_lyap_study"
            t0 = time.time()
            result = run_experiment(args)
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "horizon_max": args.horizon_max,
                    "coverage": result.get("coverage_test"),
                    "tightness": result.get("tightness_ratio"),
                    "slack_p90": result.get("slack_p90"),
                    "p_sat_test": result.get("p_sat_test"),
                    "h_win_med": result.get("horizon_real_window_median"),
                    "h_cal": result.get("horizon_model_cal"),
                    "h_cert": result.get("horizon_certified"),
                    "lyap": result.get("lyapunov_time"),
                    "elapsed": time.time() - t0,
                }
            )
            r = rows[-1]
            print(
                f"{dataset} seed={seed} Hmax={r['horizon_max']} cov={r['coverage']:.3f} "
                f"tight={r['tightness']:.3f} p_sat={r['p_sat_test']:.3f} "
                f"h_med={r['h_win_med']:.1f} L_med={r['h_cal']:.1f} "
                f"h_cert={r['h_cert']:.0f} ({r['elapsed']:.0f}s)",
                flush=True,
            )

    print("\n=== SUMMARY (median [min] over seeds) ===")
    for dataset in CONFIGS:
        sub = [r for r in rows if r["dataset"] == dataset]
        cov = np.array([r["coverage"] for r in sub], dtype=float)
        tight = np.array([r["tightness"] for r in sub], dtype=float)
        psat = np.array([r["p_sat_test"] for r in sub], dtype=float)
        hmed = np.array([r["h_win_med"] for r in sub], dtype=float)
        print(
            f"{dataset:14s} Hmax={sub[0]['horizon_max']:4d} "
            f"cov {np.median(cov):.3f} [{cov.min():.3f}] "
            f"tight {np.median(tight):.3f} p_sat {np.median(psat):.3f} "
            f"h_win_med {np.median(hmed):.1f}"
        )


if __name__ == "__main__":
    main()
