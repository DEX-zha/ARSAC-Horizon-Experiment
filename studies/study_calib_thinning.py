"""Study: does decorrelating calibration windows fix the Lorenz undercoverage?

Follow-up to study_lyap_hmax.py, which found a mild but systematic Lorenz
undercoverage (0.937-0.943 on 5/5 seeds vs target 0.95 at alpha=0.05) with
overlapping calibration windows (stride 1). Theory P1 (conformal under
dependence, docs/theory/conformal_dependence.md) predicts that thinning the
calibration windows restores approximate exchangeability under mixing at the
cost of calibration size. Full disjointness (stride >= window + Hmax ~ 405)
is unaffordable (n ~ 4), but the label only depends on the next ~H_w steps
(median 24), so strides of 1-2x the median horizon should remove most of the
label correlation. Sweep: stride in {1, 12, 24, 48}, 5 seeds.
Reproducible: python studies/study_calib_thinning.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import build_parser, load_config
from src.horizon_experiment import run_experiment

STRIDES = [1, 12, 24, 48]
SEEDS = [0, 1, 2, 3, 4]


def main():
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser = build_parser()
    parser.set_defaults(**config)

    rows = []
    for stride in STRIDES:
        for seed in SEEDS:
            args = parser.parse_args([])
            args.dataset = "lorenz"
            args.seed = seed
            args.model = "linear"
            args.series_len = 12000
            args.warmup = 1000
            args.train_ratio = 0.6
            args.val_ratio = 0.15
            args.calib_ratio = 0.15
            args.horizon_calib_thin = stride
            args.quantile_ensemble = 1
            args.mlp_epochs = 30
            args.progress = False
            args.use_cuda = False
            args.output_dir = "outputs_thin_study"
            result = run_experiment(args)
            rows.append(
                {
                    "stride": stride,
                    "seed": seed,
                    "coverage": result.get("coverage_test"),
                    "tightness": result.get("tightness_ratio"),
                    "slack_p90": result.get("slack_p90"),
                    "n_calib": result.get("calibration_samples"),
                }
            )
            r = rows[-1]
            print(
                f"stride={stride:3d} seed={seed} cov={r['coverage']:.3f} "
                f"tight={r['tightness']:.3f} slack_p90={r['slack_p90']:.1f} "
                f"n_calib={r['n_calib']}",
                flush=True,
            )

    print("\n=== SUMMARY (median [min] over seeds, alpha=0.05, target 0.95) ===")
    for stride in STRIDES:
        sub = [r for r in rows if r["stride"] == stride]
        cov = np.array([r["coverage"] for r in sub], dtype=float)
        tight = np.array([r["tightness"] for r in sub], dtype=float)
        ncal = int(np.median([r["n_calib"] for r in sub]))
        print(
            f"stride={stride:3d} cov {np.median(cov):.3f} [{cov.min():.3f}] "
            f"tight {np.median(tight):.3f} n_calib~{ncal}"
        )


if __name__ == "__main__":
    main()
