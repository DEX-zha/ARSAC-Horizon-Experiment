"""ARSAC on a real biosignal: BIDMC (PhysioNet) PPG + ECG, ICU patient 01.

Data: bidmc_01_Signals.csv, 125 Hz, 8 min. PLETH (pulse/PPG, smooth) and
II (ECG lead II, sharp QRS) are two contrasting waveforms. dt = 0.008 s.

Question answered per signal: how many samples ahead is a forecast
trustworthy (calibrated L), is the horizon model-limited or chaos/noise
limited (R), and how much of the gap is irreducible observation noise
(margin_real). Downsampled to keep CPU cost sane; horizon reported in
samples and in milliseconds.
"""

import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_estimator import HorizonEstimator

CSV = os.path.join(os.path.dirname(__file__), "..", "bidmc_01_Signals.csv")
FS = 125.0  # Hz


def load(col_name, decimate=2):
    with open(CSV) as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        idx = header.index(col_name)
        vals = [float(row[idx]) for row in r]
    x = np.asarray(vals, dtype=np.float64)[::decimate]
    return x, FS / decimate


def show(name, est, fs):
    rep = est.report()
    dt_ms = 1000.0 / fs
    L = rep["lower_bound_median"]
    Hw = rep["horizon_window_median"]
    print(f"\n=== {name} (fs={fs:.0f} Hz, {dt_ms:.1f} ms/sample) ===")
    print(f"  couverture test        : {rep['coverage_test']:.3f} (cible {1 - est.alpha:.2f})")
    print(f"  borne L mediane        : {L:.1f} samples = {L * dt_ms:.0f} ms")
    print(f"  horizon H_w median     : {Hw:.1f} samples = {Hw * dt_ms:.0f} ms")
    print(f"  tightness              : {rep['tightness']:.2f}")
    print(f"  labels identifiables   : {rep['label_identified']} (p_sat={rep['p_sat_test']:.2f})")
    r = rep["R_distance_to_chaos_floor"]
    s = rep["sigma_obs_std_units"]
    m = rep["margin_real"]
    print(f"  regime                 : {rep['regime']} (periodicite {rep['periodicity_index']:.2f})")
    print(f"  R (distance plancher)  : {'N/A hors regime' if r is None else f'{r:.1f}'}")
    print(f"  bruit d'observation est: {s if s is None else f'{s:.3f} std'}")
    print(f"  marge REELLE (bruit)   : {'N/A' if m is None else f'x{m:.1f}'}")
    print(f"  lecture                : {rep['R_reading']}")


def main():
    common = dict(model="mlp", alpha=0.1, tolerance=0.4, horizon_max=60,
                  quantile_ensemble=1, mlp_epochs=40, dim_max=8, lag_max=10)
    for sig in ["PLETH", "II"]:
        x, fs = load(sig, decimate=2)  # 62.5 Hz, ~30000 points
        # Use a stationary-ish 20000-sample chunk (~5 min) for speed.
        x = x[:20000]
        print(f"\n### {sig}: {x.size} samples @ {fs:.0f} Hz, "
              f"mean={x.mean():.3f} std={x.std():.3f}")
        est = HorizonEstimator(output_dir=f"outputs_bidmc_{sig.lower()}", **common)
        est.fit(x)
        show(sig, est, fs)


if __name__ == "__main__":
    main()
