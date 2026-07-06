"""Multi-dataset, multi-model validation of ARSAC + theory (campaign v2).

PRE-REGISTERED PROTOCOL — all claims and thresholds below were written
BEFORE any run (2026-07-06), extending the 17/17 industrial validation
(studies/study_industrial_validation.py) along two axes it did not cover:
new datasets (different operators, one different DOMAIN) and new MODEL
FAMILIES (MLP, degree-2 polynomial ridge).

Datasets (hourly, sorted, timestamp-deduplicated, last 3 years):
  comed   COMED_MW        (Commonwealth Edison, Chicago; PJM mirror)
  dom     DOM_MW          (Dominion, Virginia; PJM mirror)
  traffic traffic_volume  (UCI Metro Interstate Traffic Volume, I-94
                           Minneapolis; NOT an energy series — different
                           domain, known gaps treated as consecutive)

Model families per dataset (all through the same HorizonEstimator API):
  linear  internal AR, window 25 (declared daily-cycle window)
  naive   BYO 24-hour persistence  (lambda v: v[1], dim=25)
  mlp     internal MLP forecaster (mlp_epochs=40)
  poly    BYO degree-2 polynomial ridge on the 25-window (350 features,
          ridge 1e-3), trained on the first 60% of the series (exactly the
          estimator's train split -> no leakage into calib/test)

W1 REGIME (theory: all three series are cycle+noise dominated, NOT chaotic)
    comed, dom  -> regime == 'quasi-periodic'
    traffic     -> regime != 'chaotic'  (any of quasi-periodic/stochastic/
                   regular acceptable; the falsifiable core is that the
                   profiler must not claim chaos on any of them)
    [3 checks]

W2 OPERATIONAL GUARANTEE, MODEL-AGNOSTIC (12 calibrations)
    For each dataset x {linear, naive, mlp, poly}, with the alpha-margin
    remedy FIXED IN ADVANCE (alpha_cal = 0.085 for a 0.90 target, unchanged
    from the Lorenz-validated remedy; tolerance 0.4 sigma, horizon_max 72):
      measured test coverage >= 0.88 AND bootstrap 95% LB >= 0.85.
    [12 checks]

W3 SCALING LAW REPLICATION (linear model, tau in {0.2,0.3,0.4,0.6,0.8})
    For each dataset: (a) log-log slope s in [1.2, 2.8] for the two load
    series, s in [1.0, 2.8] for traffic (different domain, wider prior,
    declared here before running); (b) power law beats the chaotic
    signature (R^2_pow > R^2_log); (c) sigma_eff within a factor 3 of the
    independent local-linear noise estimate.
    [9 checks]

W4 MODEL-FAMILY INVARIANCE OF THE EXPONENT (new falsifiable claim)
    The exponent s reflects the DATA's innovation-accumulation regime, not
    the model family: on comed, sweeping H(tau) with the poly model must
    give (a) |s_poly - s_linear| <= 0.5 and (b) R^2_pow > R^2_log for poly.
    (sigma_eff may differ between families — it tracks each model's own
    one-step error — only the exponent is constrained.)
    [2 checks]

TOTAL: 26 pre-registered checks.

Run (chunked, resumable via outputs/multidataset_validation.csv):
    python studies/study_multidataset_validation.py --arm profile
    python studies/study_multidataset_validation.py --arm estimator --dataset comed --model linear
    ...  (3 datasets x 4 models)
    python studies/study_multidataset_validation.py --arm sweep --dataset comed
    python studies/study_multidataset_validation.py --arm sweep --dataset dom
    python studies/study_multidataset_validation.py --arm sweep --dataset traffic
    python studies/study_multidataset_validation.py --arm sweep --dataset comed --model poly
    python studies/study_multidataset_validation.py --arm verdict
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scipy.stats import spearmanr

from src.horizon_estimator import HorizonEstimator
from src.horizon_noise import estimate_observation_noise
from src.horizon_profile import profile_series

CSV_OUT = os.path.join("outputs", "multidataset_validation.csv")
N_HOURS = 26280  # 3 years
TAUS = [0.2, 0.3, 0.4, 0.6, 0.8]
ALPHA_CAL = 0.085  # unchanged pre-fixed remedy (0.90 target)
DIM = 25

FILES = {
    "comed": (os.path.join("data", "COMED_hourly.csv"), "simple"),
    "dom": (os.path.join("data", "DOM_hourly.csv"), "simple"),
    "traffic": (os.path.join("data", "Metro_Interstate_Traffic_Volume.csv"), "uci"),
}


def load(dataset):
    path, kind = FILES[dataset]
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        if kind == "simple":
            for a, b in r:
                rows.append((datetime.fromisoformat(a), float(b)))
        else:  # uci traffic: date_time col 7, traffic_volume col 8
            for cols in r:
                rows.append((datetime.fromisoformat(cols[7]), float(cols[8])))
    rows.sort(key=lambda t: t[0])
    seen, out = set(), []
    for ts, v in rows:
        if ts in seen:
            continue
        seen.add(ts)
        out.append(v)
    return np.asarray(out, dtype=np.float64)[-N_HOURS:]


def record(kind, **kv):
    os.makedirs("outputs", exist_ok=True)
    new = not os.path.exists(CSV_OUT)
    with open(CSV_OUT, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["kind", "payload"])
        w.writerow([kind, repr(kv)])
    print(f"[recorded] {kind}: {kv}", flush=True)


def read_records():
    out = {}
    if not os.path.exists(CSV_OUT):
        return out
    with open(CSV_OUT, newline="") as f:
        r = csv.reader(f)
        next(r)
        for kind, payload in r:
            out[kind] = eval(payload)  # trusted local file
    return out


# --------------------------------------------------------------- models
def _train_linear_ar(series_std, dim=DIM):
    """Same declared baseline as the v1 protocol: daily-window linear AR,
    fit on the first 60% (the estimator's train split)."""
    n = series_std.size
    tr = series_std[: int(0.6 * n)]
    m = tr.size - dim
    X = np.column_stack([tr[i: i + m] for i in range(dim)])
    y = tr[dim: dim + m]
    w = np.linalg.solve(X.T @ X + 1e-4 * np.eye(dim), X.T @ y)

    class LinAR:
        def predict(self, v):
            return float(np.dot(np.asarray(v, dtype=np.float64), w))

    return LinAR()


def _poly_features(V):
    """[w, upper-triangle(w w^T)] for each row of V: 25 + 325 = 350 dims."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    iu = np.triu_indices(V.shape[1])
    quad = (V[:, :, None] * V[:, None, :])[:, iu[0], iu[1]]
    return np.hstack([V, quad])


def _train_poly_ridge(series_std, dim=DIM, lam=1e-3):
    """Degree-2 polynomial ridge on the 25-window (the NG-RC feature idea
    carried to real data), fit on the first 60%."""
    n = series_std.size
    tr = series_std[: int(0.6 * n)]
    m = tr.size - dim
    V = np.column_stack([tr[i: i + m] for i in range(dim)])
    X = _poly_features(V)
    y = tr[dim: dim + m]
    G = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(G, X.T @ y)

    class PolyRidge:
        def predict(self, v):
            return float(_poly_features(v)[0] @ w)

    return PolyRidge()


def _standardized(x):
    n = x.size
    tr = x[: int(0.6 * n)]
    return (x - tr.mean()) / tr.std()


# ------------------------------------------------------------------- arms
def arm_profile():
    for ds in FILES:
        x = load(ds)
        p = profile_series(x)
        record(f"w1_{ds}", regime=p.regime, periodicity=round(p.periodicity_index, 3),
               lam=round(p.lambda_per_step, 5), resolved=p.lambda_resolved,
               noise=round(p.noise_std_units, 4), n=int(x.size))


def arm_estimator(dataset, model_name):
    x = load(dataset)
    kw = dict(alpha=ALPHA_CAL, tolerance=0.4, horizon_max=72,
              quantile_ensemble=1, mlp_epochs=40, horizon_samples=99999,
              output_dir="outputs_multidataset")
    if model_name in ("linear", "mlp"):
        est = HorizonEstimator(model=model_name, **kw)
    elif model_name == "naive":
        est = HorizonEstimator(model=lambda v: v[1], dim=DIM, lag=1, **kw)
    else:  # poly: pre-train on the first 60% (estimator train split)
        poly = _train_poly_ridge(_standardized(x))
        est = HorizonEstimator(model=poly, dim=DIM, lag=1, **kw)
    est.fit(x)
    hits = np.asarray(est.result_.get("coverage_hit_series") or [], dtype=int)
    from src.horizon_scientific_eval import _block_bootstrap_lower_bound
    lb = _block_bootstrap_lower_bound(hits, 0.05) if hits.size else None
    rho, _ = spearmanr(est.lower_bounds_, est.test_horizons_)
    record(f"w2_{dataset}_{model_name}",
           coverage=round(float(est.coverage_), 4),
           boot_lb=None if lb is None else round(float(lb), 4),
           L_med=round(float(np.median(est.lower_bounds_)), 2),
           H_med=round(float(np.median(est.test_horizons_)), 2),
           spearman=round(float(rho), 3), n=int(hits.size),
           regime=est.result_.get("profile_regime"))


def arm_sweep(dataset, model_name="linear"):
    from src.horizon_metrics import build_horizon_dataset

    x = load(dataset)
    xs = _standardized(x)
    n = x.size
    i_test = int(0.9 * n)
    model = (_train_poly_ridge(xs) if model_name == "poly"
             else _train_linear_ar(xs))
    test = xs[i_test:]
    sig_hat, _ = estimate_observation_noise(xs[:i_test], dim=6, lag=1, n_samples=300)

    hmax = 600
    h_med = {}
    for tau in TAUS:
        _, H = build_horizon_dataset(
            model, test, DIM, 1, hmax, tau, max_windows=400, seed=0,
            use_jacobian=False, error_mode="absolute", consecutive_k=2,
        )
        h_med[tau] = float(np.median(H))
        print(f"{dataset}/{model_name} tau={tau}: H_med={h_med[tau]:.0f} "
              f"(censored {np.mean(H >= hmax):.2f})", flush=True)

    lt = np.log(np.asarray(TAUS))
    lh = np.log(np.asarray([h_med[t] for t in TAUS]))
    s, b = np.polyfit(lt, lh, 1)
    r2_pow = 1 - np.var(lh - (s * lt + b)) / np.var(lh)
    sigma_eff = float(np.exp(-b / s))
    a, c = np.polyfit(lt, np.exp(lh), 1)
    pred = a * lt + c
    r2_log = (1 - np.var(lh - np.log(np.maximum(pred, 1e-9))) / np.var(lh)
              if np.all(pred > 0) else -np.inf)
    record(f"w3_{dataset}_{model_name}", slope=round(float(s), 3),
           r2_pow=round(float(r2_pow), 4), r2_log=round(float(r2_log), 4),
           sigma_eff=round(sigma_eff, 4), sigma_hat=round(float(sig_hat), 4),
           ratio=round(sigma_eff / sig_hat, 3), h_med=h_med)


def arm_verdict():
    R = read_records()
    print("\n================ PRE-REGISTERED VERDICTS (26) ================")
    ok_all = True

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"{name}: {'PASS' if cond else 'FAIL'}  ({detail})")

    # W1 (3)
    for ds in ("comed", "dom"):
        v = R.get(f"w1_{ds}", {})
        check(f"W1 {ds} quasi-periodic", v.get("regime") == "quasi-periodic",
              f"regime={v.get('regime')} periodicity={v.get('periodicity')}")
    v = R.get("w1_traffic", {})
    check("W1 traffic not chaotic", v.get("regime") not in (None, "chaotic"),
          f"regime={v.get('regime')}")

    # W2 (12)
    for ds in FILES:
        for m in ("linear", "naive", "mlp", "poly"):
            v = R.get(f"w2_{ds}_{m}", {})
            check(f"W2 {ds}/{m} cov>=0.88 & LB>=0.85",
                  bool(v) and v["coverage"] >= 0.88 and (v["boot_lb"] or 0) >= 0.85,
                  f"cov={v.get('coverage')} lb={v.get('boot_lb')} n={v.get('n')}")

    # W3 (9)
    for ds in FILES:
        v = R.get(f"w3_{ds}_linear", {})
        lo = 1.0 if ds == "traffic" else 1.2
        if v:
            check(f"W3 {ds} slope in [{lo}, 2.8]", lo <= v["slope"] <= 2.8,
                  f"s={v['slope']}")
            check(f"W3 {ds} power beats chaotic log-law", v["r2_pow"] > v["r2_log"],
                  f"R2 {v['r2_pow']} vs {v['r2_log']}")
            check(f"W3 {ds} sigma ratio in [1/3, 3]", 1 / 3 <= v["ratio"] <= 3,
                  f"ratio={v['ratio']}")
        else:
            check(f"W3 {ds} (missing)", False, "no sweep record")

    # W4 (2)
    vl, vp = R.get("w3_comed_linear", {}), R.get("w3_comed_poly", {})
    if vl and vp:
        check("W4 |s_poly - s_linear| <= 0.5",
              abs(vp["slope"] - vl["slope"]) <= 0.5,
              f"s_lin={vl['slope']} s_poly={vp['slope']}")
        check("W4 poly power beats chaotic log-law", vp["r2_pow"] > vp["r2_log"],
              f"R2 {vp['r2_pow']} vs {vp['r2_log']}")
    else:
        check("W4 (missing)", False, "no poly sweep record")

    print(f"\nOVERALL: {'ALL PASS' if ok_all else 'AT LEAST ONE FAIL'}")


ARMS = {
    "profile": lambda a: arm_profile(),
    "estimator": lambda a: arm_estimator(a.dataset, a.model),
    "sweep": lambda a: arm_sweep(a.dataset, a.model if a.model == "poly" else "linear"),
    "verdict": lambda a: arm_verdict(),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dataset", choices=list(FILES), default="comed")
    ap.add_argument("--model", choices=["linear", "naive", "mlp", "poly"],
                    default="linear")
    a = ap.parse_args()
    ARMS[a.arm](a)
