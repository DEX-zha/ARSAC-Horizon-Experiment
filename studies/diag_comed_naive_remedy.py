"""POST-HOC remedy demonstration for the comed/naive W2 failure.

NOT part of the pre-registered protocol; writes no evidence records and
must not alter the recorded verdict. Question: does drift-aware
calibration (the beyond-exchangeability machinery kept in
src/horizon_conformal_beyond.py exactly for this scenario) recover the
coverage lost to the measured seasonal shift (calib median H = 22 h vs
test median H = 4 h, studies/diag_comed_naive.py)?

Setup (declared): MARGINAL lower bound L (weighted alpha-quantile of past
labels, no conditional quantile model) — this isolates the calibration
question from the conditioning question. Walk-forward on the last 10% of
COMED with the 24-hour-persistence model, alpha = 0.085, tolerance 0.4,
hmax = 72. A label at window j uses data up to j+window+72, so at test
window i only labels with j <= i - 72 are available (enforced).

Schemes compared on the SAME test windows:
  static    uniform quantile of the fixed [75%, 90%) slice (mimics the
            failed campaign calibration)
  weighted  decay_weights over all past labels, half-life 720 windows
            (30 days)
  rolling   uniform over the last 1440 windows (60 days)

Run: python studies/diag_comed_naive_remedy.py   (~5 min, read-only)
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_conformal_beyond import decay_weights, weighted_conformal_quantile
from src.horizon_metrics import build_horizon_dataset
from studies.study_multidataset_validation import load, _standardized, DIM

HMAX, TAU, ALPHA = 72, 0.4, 0.085
HALF_LIFE, ROLL = 720, 1440


class Naive:
    def predict(self, v):
        return float(v[1])


def main():
    xs = _standardized(load("comed"))
    n = xs.size
    seg_lo = int(0.65 * n)
    seg = xs[seg_lo:]
    _, H = build_horizon_dataset(
        Naive(), seg, DIM, 1, HMAX, TAU, max_windows=None, seed=0,
        use_jacobian=False, error_mode="absolute", consecutive_k=2, stride=1,
    )
    H = np.asarray(H, dtype=float)
    # absolute series index of each window start
    starts = seg_lo + np.arange(H.size)
    i_cal_lo, i_cal_hi = int(0.75 * n), int(0.90 * n)
    static_scores = -H[(starts >= i_cal_lo) & (starts < i_cal_hi)]

    test_pos = np.where(starts >= i_cal_hi)[0]
    test_pos = test_pos[test_pos >= HMAX]
    hits = {"static": [], "weighted": [], "rolling": []}
    L_med = {k: [] for k in hits}
    for i in test_pos[::3]:  # stride 3 over ~2600 test windows
        past = -H[: i - HMAX + 1]
        l_static = -weighted_conformal_quantile(static_scores, ALPHA)
        w = decay_weights(past.size, HALF_LIFE)
        l_weighted = -weighted_conformal_quantile(past, ALPHA, weights=w)
        l_rolling = -weighted_conformal_quantile(past[-ROLL:], ALPHA)
        for key, L in (("static", l_static), ("weighted", l_weighted),
                       ("rolling", l_rolling)):
            hits[key].append(H[i] >= L)
            L_med[key].append(L)

    n_test = len(hits["static"])
    print(f"comed/naive walk-forward, {n_test} test windows, alpha={ALPHA}")
    for key in ("static", "weighted", "rolling"):
        cov = float(np.mean(hits[key]))
        lm = float(np.median(L_med[key]))
        print(f"  {key:9s}: coverage={cov:.4f}  L_med={lm:.1f}")


if __name__ == "__main__":
    main()
