"""Study: Politis-White automatic block length vs the n^(1/3)*4 heuristic.

Question: does the automatic block length improve the calibration of the
circular moving-block bootstrap lower bound used for `coverage_lb` in
`src/horizon_scientific_eval.py`?

Protocol (fully seeded):
- Hit series: hit_t = 1{z_t <= q} where z is a Gaussian AR(1) with
  phi in {0.0, 0.5, 0.9} and q is set so the marginal hit rate is ~0.90
  (mimics coverage hits at alpha = 0.1).
- True mean known by simulation: one long AR(1) run (n = 2,000,000) per phi.
- n = 500 observations, 200 replications per phi.
- For each replication, compute the bootstrap lower bound at
  alpha_ci = 0.05 with the production `_block_bootstrap_lower_bound`,
  once with the old heuristic block length max(10, round(n^(1/3)*4)) = 32
  and once with the Politis-White block length (paired: same hits, same
  bootstrap seed).
- Report calibration P(LB <= true mean); nominal is 0.95.

Run:  python studies/study_blocklen.py
Time: ~1-2 minutes on a laptop CPU.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from statistics import NormalDist

import numpy as np
from scipy.signal import lfilter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_blocklen import politis_white_block_length
from src.horizon_scientific_eval import _block_bootstrap_lower_bound

BASE_SEED = 20260705
N_OBS = 500
N_REPS = 200
ALPHA_CI = 0.05
N_BOOT = 1000
PHIS = (0.0, 0.5, 0.9)
TARGET_RATE = 0.90
N_TRUE = 2_000_000
BURN = 10_000
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "outputs", "study_blocklen.csv")


def ar1(n, phi, rng, burn):
    """Gaussian AR(1): z_t = phi*z_{t-1} + eps_t, eps ~ N(0,1)."""
    eps = rng.standard_normal(n + burn)
    z = lfilter([1.0], [1.0, -phi], eps)
    return z[burn:]


def threshold_for(phi):
    """Threshold q with P(z_t <= q) = TARGET_RATE under the stationary law."""
    sd = 1.0 / np.sqrt(1.0 - phi * phi) if phi < 1.0 else 1.0
    return NormalDist().inv_cdf(TARGET_RATE) * sd


def old_heuristic_block_len(n):
    """Block length rule used before this study (horizon_scientific_eval)."""
    return max(10, int(round(n ** (1.0 / 3.0) * 4)))


def main():
    t0 = time.time()
    rows = []
    print(f"n={N_OBS}, reps={N_REPS}, alpha_ci={ALPHA_CI}, n_boot={N_BOOT}, "
          f"nominal calibration={1.0 - ALPHA_CI:.2f}")
    mc_se = np.sqrt(0.95 * 0.05 / N_REPS)
    print(f"Monte-Carlo SE on a calibration estimate at 0.95: {mc_se:.3f}\n")

    for phi in PHIS:
        q = threshold_for(phi)

        # True mean by simulation (long run, dedicated seed).
        rng_true = np.random.default_rng(BASE_SEED + int(phi * 1000))
        z_long = ar1(N_TRUE, phi, rng_true, BURN)
        true_mean = float(np.mean(z_long <= q))

        b_old = old_heuristic_block_len(N_OBS)
        lb_old = np.empty(N_REPS)
        lb_pw = np.empty(N_REPS)
        b_pw_all = np.empty(N_REPS, dtype=int)
        sample_means = np.empty(N_REPS)

        for rep in range(N_REPS):
            rng = np.random.default_rng(BASE_SEED + 100_000 + int(phi * 1000) * 1000 + rep)
            z = ar1(N_OBS, phi, rng, burn=500)
            hits = (z <= q).astype(int)
            sample_means[rep] = hits.mean()

            b_pw = politis_white_block_length(hits)
            b_pw_all[rep] = b_pw

            boot_seed = rep  # same seed for both arms: paired comparison
            lb_old[rep] = _block_bootstrap_lower_bound(
                hits, ALPHA_CI, n_boot=N_BOOT, block_len=b_old, seed=boot_seed
            )
            lb_pw[rep] = _block_bootstrap_lower_bound(
                hits, ALPHA_CI, n_boot=N_BOOT, block_len=b_pw, seed=boot_seed
            )

        cover_old = lb_old <= true_mean
        cover_pw = lb_pw <= true_mean
        calib_old = float(np.mean(cover_old))
        calib_pw = float(np.mean(cover_pw))
        slack_old = float(np.mean(true_mean - lb_old))
        slack_pw = float(np.mean(true_mean - lb_pw))
        # Paired (McNemar-style) disagreements: same hits + same bootstrap
        # seed in both arms, so these isolate the block-length effect.
        n_old_only = int(np.sum(cover_old & ~cover_pw))
        n_pw_only = int(np.sum(cover_pw & ~cover_old))

        print(f"phi={phi:.1f}  true_mean={true_mean:.4f}  "
              f"mean(sample_mean)={sample_means.mean():.4f}")
        print(f"  old heuristic : block_len={b_old:3d}            "
              f"calibration={calib_old:.3f}  mean(true-LB)={slack_old:.4f}")
        print(f"  Politis-White : block_len={b_pw_all.mean():5.1f} "
              f"(med {int(np.median(b_pw_all))}, min {b_pw_all.min()}, max {b_pw_all.max()})  "
              f"calibration={calib_pw:.3f}  mean(true-LB)={slack_pw:.4f}")
        print(f"  paired: covered by old only={n_old_only}, by PW only={n_pw_only} "
              f"(out of {N_REPS})\n")

        rows.append({
            "phi": phi,
            "true_mean": round(true_mean, 6),
            "n": N_OBS,
            "reps": N_REPS,
            "alpha_ci": ALPHA_CI,
            "nominal": 1.0 - ALPHA_CI,
            "block_len_old": b_old,
            "block_len_pw_mean": round(float(b_pw_all.mean()), 2),
            "block_len_pw_median": int(np.median(b_pw_all)),
            "block_len_pw_min": int(b_pw_all.min()),
            "block_len_pw_max": int(b_pw_all.max()),
            "calibration_old": round(calib_old, 4),
            "calibration_pw": round(calib_pw, 4),
            "mean_slack_old": round(slack_old, 5),
            "mean_slack_pw": round(slack_pw, 5),
            "paired_covered_old_only": n_old_only,
            "paired_covered_pw_only": n_pw_only,
        })

    out = os.path.abspath(OUTPUT_CSV)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results saved to {out}")

    # Diagnostic: at phi=0.9, is ANY fixed block length well calibrated?
    # If not, the miscalibration is a property of the percentile bootstrap
    # LB at n=500 under strong dependence, not of the block-length rule.
    phi = 0.9
    q = threshold_for(phi)
    rng_true = np.random.default_rng(BASE_SEED + int(phi * 1000))
    true_mean = float(np.mean(ar1(N_TRUE, phi, rng_true, BURN) <= q))
    print(f"\nDiagnostic sweep at phi={phi} (true_mean={true_mean:.4f}), "
          f"fixed block lengths:")
    for b_fix in (10, 20, 32, 50, 80, 120, 166):
        lbs = np.empty(N_REPS)
        for rep in range(N_REPS):
            rng = np.random.default_rng(BASE_SEED + 100_000 + int(phi * 1000) * 1000 + rep)
            hits = (ar1(N_OBS, phi, rng, burn=500) <= q).astype(int)
            lbs[rep] = _block_bootstrap_lower_bound(
                hits, ALPHA_CI, n_boot=N_BOOT, block_len=b_fix, seed=rep
            )
        print(f"  block_len={b_fix:3d}: calibration={np.mean(lbs <= true_mean):.3f}  "
              f"mean(true-LB)={np.mean(true_mean - lbs):.4f}")

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
