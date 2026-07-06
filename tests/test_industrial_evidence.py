"""Pins the industrial-validation evidence to its pre-registered criteria.

The versioned evidence file (docs/theory/data/industrial_validation.csv,
produced by studies/study_industrial_validation.py) backs the README claims.
This test re-derives every verdict from the raw recorded numbers on every CI
run: if the evidence is corrupted, edited, or a claim silently drifts from
what the file supports, the suite fails. Suspicion of error has a test.
"""

import ast
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

EV = os.path.join(os.path.dirname(__file__), "..", "docs", "theory", "data",
                  "industrial_validation.csv")


def _load():
    out = {}
    with open(EV, newline="") as f:
        r = csv.reader(f)
        next(r)
        for kind, payload in r:
            out[kind] = ast.literal_eval(payload)
    return out


def test_v1_regimes_quasi_periodic():
    R = _load()
    for ds in ("aep", "pjme"):
        v = R[f"v1_{ds}"]
        assert v["regime"] == "quasi-periodic"
        assert v["periodicity"] >= 0.5
        assert v["resolved"] is False


def test_v2_coverage_guarantee_held_6_of_6():
    R = _load()
    runs = {k: v for k, v in R.items() if k.startswith("v24_")}
    assert len(runs) == 6
    for name, v in runs.items():
        assert v["coverage"] >= 0.88, name
        assert v["boot_lb"] >= 0.85, name
        assert v["n"] > 1000, name


def test_v3_scaling_law_rederived_from_raw_medians():
    R = _load()
    for ds in ("aep", "pjme"):
        h_med = R[f"v3_{ds}"]["h_med"]
        taus = np.array(sorted(h_med), dtype=float)
        H = np.array([h_med[t] for t in taus])
        lt, lh = np.log(taus), np.log(H)
        s, b = np.polyfit(lt, lh, 1)
        r2_pow = 1 - np.var(lh - (s * lt + b)) / np.var(lh)
        a, c = np.polyfit(lt, H, 1)
        r2_log = 1 - np.var(lh - np.log(np.maximum(a * lt + c, 1e-9))) / np.var(lh)
        assert 1.2 <= s <= 2.8, ds
        assert r2_pow > 0.97, ds
        assert r2_pow > r2_log, ds
        sigma_eff = float(np.exp(-b / s))
        ratio = sigma_eff / R[f"v3_{ds}"]["sigma_hat"]
        assert 1 / 3 <= ratio <= 3, ds


def test_v4_decision_replicates_across_periods():
    R = _load()
    p1l, p2l = R["v24_aep_p1_linear"], R["v24_aep_p2_linear"]
    p1n, p2n = R["v24_aep_p1_naive"], R["v24_aep_p2_naive"]
    assert p1l["L_med"] > p1n["L_med"] and p2l["L_med"] > p2n["L_med"]
    assert 0.6 <= p2l["L_med"] / p1l["L_med"] <= 1.4
    assert p1l["spearman"] > 0.3 and p2l["spearman"] > 0.3
