"""POST-HOC diagnostic of the pre-registered W2 failure on comed/naive.

This script is NOT part of the pre-registered protocol and writes no
evidence records. It explains a documented FAIL (coverage 0.8574 < 0.88,
bootstrap LB 0.8368 < 0.85 for the 24-hour-persistence model on COMED);
it must not be used to alter the recorded verdict.

Hypotheses tested (printed, not asserted):
  D1 label quantization: the naive model dies fast on COMED (H_med = 3 h);
     when most labels sit on {1..4}, the conformal margin can only move in
     integer steps and the effective coverage grid is coarse. Measure the
     test-label histogram and the coverage jump between adjacent integer
     bounds.
  D2 calib->test shift: compare the calibration-window and test-window
     label distributions (same model, same tolerance); a one-step downward
     shift in a 3-hour-median label regime is a ~5-10 point coverage hit.
  D3 control: same measurements on DOM/naive, which shares the model and
     protocol (verdict recorded independently).

Run: python studies/diag_comed_naive.py   (~5 min, read-only)
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_metrics import build_horizon_dataset
from studies.study_multidataset_validation import load, _standardized, DIM


def labels(series_std, lo, hi, hmax=72, tau=0.4, n_windows=800):
    class Naive:
        def predict(self, v):
            return float(v[1])

    seg = series_std[int(lo * series_std.size): int(hi * series_std.size)]
    _, H = build_horizon_dataset(
        Naive(), seg, DIM, 1, hmax, tau, max_windows=n_windows, seed=0,
        use_jacobian=False, error_mode="absolute", consecutive_k=2,
    )
    return np.asarray(H, dtype=float)


def report(ds):
    xs = _standardized(load(ds))
    h_cal = labels(xs, 0.75, 0.90)
    h_test = labels(xs, 0.90, 1.00)
    print(f"\n===== {ds}/naive =====")
    for name, H in (("calib", h_cal), ("test", h_test)):
        qs = np.percentile(H, [10, 25, 50, 75, 90])
        frac_small = float(np.mean(H <= 4))
        print(f"{name}: median={np.median(H):.0f}  q10/25/50/75/90={qs}  "
              f"P(H<=4)={frac_small:.2f}  n={H.size}")
    # D1: coverage achievable at each integer bound on the test labels
    print("test coverage if L == k (integer grid):")
    for k in range(1, 7):
        print(f"  L={k}: P(H>=L)={float(np.mean(h_test >= k)):.4f}")
    # D2: shift measured at matched quantiles
    for q in (10, 25, 50):
        d = np.percentile(h_test, q) - np.percentile(h_cal, q)
        print(f"shift test-calib at q{q}: {d:+.1f} steps")


if __name__ == "__main__":
    report("comed")
    report("dom")
