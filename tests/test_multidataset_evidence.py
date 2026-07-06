"""Pins the multi-dataset validation evidence to its pre-registered criteria.

The versioned evidence file (docs/theory/data/multidataset_validation.csv,
produced by studies/study_multidataset_validation.py) backs the campaign-v2
claims: 3 datasets (COMED, DOM, UCI traffic) x 4 model families (linear,
naive, MLP, poly ridge), 26 pre-registered checks. This test re-derives
every verdict from the raw recorded numbers on every CI run: if the
evidence is corrupted, edited, or a claim silently drifts from what the
file supports, the suite fails.
"""

import ast
import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

EV = os.path.join(os.path.dirname(__file__), "..", "docs", "theory", "data",
                  "multidataset_validation.csv")

DATASETS = ("comed", "dom", "traffic")
MODELS = ("linear", "naive", "mlp", "poly")


def _load():
    out = {}
    with open(EV, newline="") as f:
        r = csv.reader(f)
        next(r)
        for kind, payload in r:
            out[kind] = ast.literal_eval(payload)
    return out


def test_w1_regimes_non_chaotic():
    R = _load()
    for ds in ("comed", "dom"):
        v = R[f"w1_{ds}"]
        assert v["regime"] == "quasi-periodic", ds
        assert v["resolved"] is False, ds
    assert R["w1_traffic"]["regime"] != "chaotic"


def test_w2_coverage_11_of_12_with_documented_drift_failure():
    """11/12 calibrations pass; comed/naive is a PRE-REGISTERED FAIL kept
    as such (seasonal block shift: calib median H = 22 h vs test median
    4 h; see docs/theory/multidataset_validation.md). This test pins BOTH
    outcomes: the 11 passes and the failure's recorded values — if either
    drifts, the suite fails."""
    R = _load()
    runs = {k: v for k, v in R.items() if k.startswith("w2_")}
    assert len(runs) == 12
    for ds in DATASETS:
        for m in MODELS:
            v = R[f"w2_{ds}_{m}"]
            assert v["boot_lb"] < v["coverage"], (ds, m)
            assert v["n"] > 1000, (ds, m)
            assert v["L_med"] <= v["H_med"], (ds, m)
            if (ds, m) == ("comed", "naive"):
                continue
            assert v["coverage"] >= 0.88, (ds, m)
            assert v["boot_lb"] >= 0.85, (ds, m)
    fail = R["w2_comed_naive"]
    assert fail["coverage"] == 0.8574 and fail["boot_lb"] == 0.8368


def _refit(h_med):
    taus = np.array(sorted(h_med), dtype=float)
    H = np.array([h_med[t] for t in taus])
    lt, lh = np.log(taus), np.log(H)
    s, b = np.polyfit(lt, lh, 1)
    r2_pow = 1 - np.var(lh - (s * lt + b)) / np.var(lh)
    a, c = np.polyfit(lt, H, 1)
    pred = a * lt + c
    r2_log = (1 - np.var(lh - np.log(np.maximum(pred, 1e-9))) / np.var(lh)
              if np.all(pred > 0) else -np.inf)
    sigma_eff = float(np.exp(-b / s))
    return s, r2_pow, r2_log, sigma_eff


def test_w3_scaling_law_rederived_from_raw_medians():
    R = _load()
    for ds in DATASETS:
        v = R[f"w3_{ds}_linear"]
        s, r2_pow, r2_log, sigma_eff = _refit(v["h_med"])
        lo = 1.0 if ds == "traffic" else 1.2
        assert lo <= s <= 2.8, (ds, s)
        assert r2_pow > r2_log, (ds, r2_pow, r2_log)
        ratio = sigma_eff / v["sigma_hat"]
        assert 1 / 3 <= ratio <= 3, (ds, ratio)
        # sanity: H(tau) strictly increasing on the pre-registered grid
        taus = sorted(v["h_med"])
        assert [round(t, 2) for t in taus] == [0.2, 0.3, 0.4, 0.6, 0.8], ds
        H = [v["h_med"][t] for t in taus]
        assert all(b > a for a, b in zip(H, H[1:])), (ds, H)


def test_w4_exponent_is_model_family_invariant():
    R = _load()
    sl, *_ = _refit(R["w3_comed_linear"]["h_med"])
    sp, r2_pow, r2_log, _ = _refit(R["w3_comed_poly"]["h_med"])
    assert abs(sp - sl) <= 0.5, (sl, sp)
    assert r2_pow > r2_log
