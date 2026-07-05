"""ARSAC on real grid load: AEP hourly consumption (Kaggle/PJM, 2004-2018).

The product flow on an energy series:
1. profile_series  -> what kind of unpredictability is hourly load?
2. HorizonEstimator (internal linear forecaster) -> calibrated trust bound
3. HorizonEstimator (BYO: the industry seasonal-naive baseline, same-hour
   yesterday) -> the same calibrated audit applied to the model a grid
   operator actually starts from.

Data quirk handled: the Kaggle CSV is not sorted chronologically; rows are
sorted by timestamp and duplicate timestamps (DST) deduplicated before use.
Run: python studies/demo_energy.py
"""

import csv
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_estimator import HorizonEstimator
from src.horizon_profile import profile_series

CSV = os.path.join(os.path.dirname(__file__), "..", "AEP_hourly.csv", "AEP_hourly.csv")
N_HOURS = 26280  # last 3 years


def load_series():
    with open(CSV) as f:
        r = csv.reader(f)
        next(r)
        rows = [(datetime.fromisoformat(a), float(b)) for a, b in r]
    rows.sort(key=lambda t: t[0])
    seen, out = set(), []
    for ts, v in rows:
        if ts in seen:
            continue
        seen.add(ts)
        out.append(v)
    x = np.asarray(out, dtype=np.float64)[-N_HOURS:]
    return x


def show(name, est):
    rep = est.report()
    L = rep["lower_bound_median"]
    Hw = rep["horizon_window_median"]
    print(f"\n=== {name} ===")
    print(f"  couverture test        : {rep['coverage_test']:.3f} "
          f"(cible operationnelle 0.90, calibre a alpha={est.alpha:.3f})")
    print(f"  borne L mediane        : {L:.1f} h")
    print(f"  horizon H_w median     : {Hw:.1f} h")
    print(f"  tightness              : {rep['tightness']:.2f}")
    print(f"  regime                 : {rep['regime']}")
    print(f"  bruit estime           : {rep['sigma_obs_std_units']:.3f} std"
          if rep["sigma_obs_std_units"] is not None else "  bruit estime           : N/A")
    print(f"  lecture                : {rep['R_reading']}")


def main():
    x = load_series()
    print(f"AEP load: {x.size} h ({x.size / 8760:.1f} ans), "
          f"mean {x.mean():.0f} MW, std {x.std():.0f} MW")
    print(f"tolerance 0.4 std = {0.4 * x.std():.0f} MW d'erreur\n")

    print("--- 1. Profil de predictibilite ---")
    prof = profile_series(x)
    print(prof.summary())

    # alpha-margin remedy (documented calib->test shift of ~1.3 pt, see
    # docs/THEORY.md): calibrate at alpha - 0.015 to DELIVER the 0.90 target.
    common = dict(alpha=0.085, tolerance=0.4, horizon_max=72,
                  quantile_ensemble=1, mlp_epochs=40)

    est_lin = HorizonEstimator(model="linear", output_dir="outputs_energy", **common)
    est_lin.fit(x)
    show("Forecaster lineaire (appris par l'outil)", est_lin)

    # Industry baseline: same hour yesterday. Delay vector (dim=25, lag=1) is
    # (x_{t-24}, ..., x_t); predicting x_{t+1} seasonally = x_{t-23} = v[1].
    est_naive = HorizonEstimator(model=lambda v: v[1], dim=25, lag=1,
                                 output_dir="outputs_energy", **common)
    est_naive.fit(x)
    show("Naif saisonnier J-1 meme heure (baseline metier, via BYO)", est_naive)


if __name__ == "__main__":
    main()
