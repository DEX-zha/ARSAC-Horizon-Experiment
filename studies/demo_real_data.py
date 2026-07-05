"""Real-data demo: what the tool answers on YOUR data with YOUR model.

Data: monthly mean sunspot number, SILSO (WDC-SILSO, Royal Observatory of
Belgium), 1749-2026, fetched from https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv
into data/sunspots_monthly.csv (semicolon-separated).

The three practitioner questions, answered end-to-end:
1. How far can I trust this forecast?      -> calibrated L(x), coverage checked
2. Is it worth improving my model?         -> R = distance to the chaos floor
3. Is my horizon window well-specified?    -> label_identified / p_sat gates

Run: python studies/demo_real_data.py
"""

import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_estimator import HorizonEstimator

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "sunspots_monthly.csv")


def load_sunspots():
    values = []
    with open(DATA, newline="") as f:
        for row in csv.reader(f, delimiter=";"):
            v = float(row[3])
            if v >= 0:  # -1 = missing
                values.append(v)
    return np.asarray(values, dtype=np.float64)


def show(name, est):
    rep = est.report()
    print(f"\n=== {name} ===")
    print(f"  couverture test        : {rep['coverage_test']:.3f} (cible {1 - est.alpha:.2f})")
    print(f"  borne L mediane        : {rep['lower_bound_median']:.1f} mois")
    print(f"  horizon H_w median     : {rep['horizon_window_median']:.1f} mois")
    print(f"  tightness              : {rep['tightness']:.2f}")
    print(f"  labels identifiables   : {rep['label_identified']} (p_sat={rep['p_sat_test']:.2f})")
    r = rep["R_distance_to_chaos_floor"]
    print(f"  R (distance au plancher): {r if r is None else f'{r:.1f}'}")
    s = rep["sigma_obs_std_units"]
    print(f"  bruit d'observation (est.): {s if s is None else f'{s:.3f} std'}")
    m = rep["margin_real"]
    print(f"  marge REELLE (bruit)   : {m if m is None else f'x{m:.1f}'}")
    print(f"  lecture                : {rep['R_reading']}")


def main():
    series = load_sunspots()
    print(f"Sunspots mensuels: {series.size} points, "
          f"moyenne {series.mean():.1f}, max {series.max():.1f}")

    common = dict(alpha=0.1, tolerance=0.5, horizon_max=36,
                  quantile_ensemble=1, mlp_epochs=40)

    # 1) Internal MLP forecaster.
    est_mlp = HorizonEstimator(model="mlp", output_dir="outputs_demo_sunspots",
                               **common)
    est_mlp.fit(series)
    show("MLP interne (modele appris par l'outil)", est_mlp)

    # 2) Bring-your-own model: seasonal persistence x(t+1) ~ x(t) (the naive
    #    baseline a practitioner might currently be using).
    est_naive = HorizonEstimator(model=lambda x: x[-1], dim=4, lag=1,
                                 output_dir="outputs_demo_sunspots",
                                 **common)
    est_naive.fit(series)
    show("Persistance (votre modele naif, via l'API BYO)", est_naive)

    print("\nConclusion produit : les deux modeles recoivent une borne L(x)"
          "\ncalibree et verifiable, et le diagnostic R dit s'il reste de la"
          "\nmarge a aller chercher en ameliorant le modele.")


if __name__ == "__main__":
    main()
