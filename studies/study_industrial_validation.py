"""Industrial validation of ARSAC + theory on real grid-load data.

PRE-REGISTERED PROTOCOL — the four claims and their thresholds below were
written before any run. Datasets: AEP (American Electric Power, local CSV)
and PJME (PJM East, fetched from a public mirror) — two independent balancing
regions, hourly load, last 3 years each (sorted, DST-deduplicated).

V1 REGIME (theory: grid load is cycle+noise dominated, not chaotic)
    profile_series -> regime == 'quasi-periodic' on BOTH datasets
    (periodicity >= 0.5 and lambda not resolved).

V2 OPERATIONAL GUARANTEE (the product claim)
    With the alpha-margin remedy FIXED IN ADVANCE (alpha_cal = 0.085 for a
    0.90 target, calibrated on Lorenz before any grid data), for each
    dataset x {learned linear, seasonal-naive BYO}:
      measured test coverage >= 0.88  AND
      circular block-bootstrap 95% lower bound on coverage >= 0.85.

V3 SCALING LAW (the mathematical demonstration)
    Our validated error-accumulation framework in the lambda -> 0 limit:
    stable dynamics + per-step innovations sigma give e(h)^2 ~ e0^2 + h*sigma^2,
    hence H(tau) ~ (tau/sigma_eff)^2 — a LOG-LOG SLOPE ~2 — while the chaotic
    regime (validated on Lorenz/Rossler) gives H = ln(tau/e0)/(lambda*dt),
    i.e. H LINEAR IN ln(tau). Pre-registered on both datasets, tau sweep
    {0.2, 0.3, 0.4, 0.6, 0.8}:
      (a) power-law slope s in [1.2, 2.8];
      (b) power-law fits ln(H) better than the chaotic log-law (R^2_pow > R^2_log);
      (c) sigma_eff from the fit within a factor 3 of the INDEPENDENT
          local-linear noise estimate sigma_hat (documented x1.6-1.8 bias band).

V4 DECISION REPLICATION (out-of-period)
    AEP split into two disjoint 1.5-year periods:
      L_med(learned) > L_med(naive) in BOTH periods;
      L_med(learned) ratio period2/period1 in [0.6, 1.4];
      Spearman(L, realized H) > 0.3 in both periods.

Run (chunked, resumable via outputs/industrial_validation.csv):
    python studies/study_industrial_validation.py --arm profile
    python studies/study_industrial_validation.py --arm sweep --dataset aep
    python studies/study_industrial_validation.py --arm sweep --dataset pjme
    python studies/study_industrial_validation.py --arm estimator --dataset aep --period 1 --model linear
    ... (see ARMS below)
    python studies/study_industrial_validation.py --arm verdict
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

CSV_OUT = os.path.join("outputs", "industrial_validation.csv")
N_HOURS = 26280  # 3 years
TAUS = [0.2, 0.3, 0.4, 0.6, 0.8]
ALPHA_CAL = 0.085  # fixed in advance (Lorenz-validated remedy for 0.90 target)
FILES = {
    "aep": (os.path.join("AEP_hourly.csv", "AEP_hourly.csv"), "AEP_MW"),
    "pjme": (os.path.join("data", "PJME_hourly.csv"), "PJME_MW"),
}


def load(dataset):
    path, _ = FILES[dataset]
    with open(path) as f:
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


# ------------------------------------------------------------------- arms
def arm_profile():
    for ds in FILES:
        x = load(ds)
        p = profile_series(x)
        record(f"v1_{ds}", regime=p.regime, periodicity=round(p.periodicity_index, 3),
               lam=round(p.lambda_per_step, 5), resolved=p.lambda_resolved,
               noise=round(p.noise_std_units, 4))


def _train_linear_ar(series_std, dim=25):
    """Daily-window linear AR (dim=25 covers the 24 h cycle), declared in the
    protocol. Fit on the first 60%, labels measured on the last 10%."""
    n = series_std.size
    i_train = int(0.6 * n)
    tr = series_std[:i_train]
    m = tr.size - dim
    X = np.column_stack([tr[i: i + m] for i in range(dim)])
    y = tr[dim: dim + m]
    G = X.T @ X + 1e-4 * np.eye(dim)
    w = np.linalg.solve(G, X.T @ y)

    class LinAR:
        def predict(self, v):
            return float(np.dot(np.asarray(v, dtype=np.float64), w))

    return LinAR()


def arm_sweep(dataset):
    from src.horizon_metrics import build_horizon_dataset

    x = load(dataset)
    n = x.size
    i_train, i_test = int(0.6 * n), int(0.9 * n)
    mean, sd = x[:i_train].mean(), x[:i_train].std()
    xs = (x - mean) / sd
    model = _train_linear_ar(xs)
    test = xs[i_test:]
    sig_hat, _ = estimate_observation_noise(xs[:i_test], dim=6, lag=1, n_samples=300)

    hmax = 600
    h_med = {}
    for tau in TAUS:
        _, H = build_horizon_dataset(
            model, test, 25, 1, hmax, tau, max_windows=400, seed=0,
            use_jacobian=False, error_mode="absolute", consecutive_k=2,
        )
        h_med[tau] = float(np.median(H))
        print(f"{dataset} tau={tau}: H_med={h_med[tau]:.0f} "
              f"(censored {np.mean(H >= hmax):.2f})", flush=True)

    lt = np.log(np.asarray(TAUS))
    lh = np.log(np.asarray([h_med[t] for t in TAUS]))
    # power law: ln H = s ln tau + b  ->  H = (tau/sigma_eff)^s
    s, b = np.polyfit(lt, lh, 1)
    r2_pow = 1 - np.var(lh - (s * lt + b)) / np.var(lh)
    sigma_eff = float(np.exp(-b / s))
    # chaotic signature: H = a ln tau + c, compared in the same ln-H space
    a, c = np.polyfit(lt, np.exp(lh), 1)
    pred = a * lt + c
    r2_log = (1 - np.var(lh - np.log(np.maximum(pred, 1e-9))) / np.var(lh)
              if np.all(pred > 0) else -np.inf)
    record(f"v3_{dataset}", slope=round(float(s), 3), r2_pow=round(float(r2_pow), 4),
           r2_log=round(float(r2_log), 4), sigma_eff=round(sigma_eff, 4),
           sigma_hat=round(float(sig_hat), 4),
           ratio=round(sigma_eff / sig_hat, 3), h_med=h_med)


def arm_estimator(dataset, period, model_name):
    x = load(dataset)
    if period == "1":
        x = x[: x.size // 2]
    elif period == "2":
        x = x[x.size // 2:]
    kw = dict(alpha=ALPHA_CAL, tolerance=0.4, horizon_max=72,
              quantile_ensemble=1, mlp_epochs=40, horizon_samples=99999,
              output_dir="outputs_industrial")
    if model_name == "linear":
        est = HorizonEstimator(model="linear", **kw)
    else:
        est = HorizonEstimator(model=lambda v: v[1], dim=25, lag=1, **kw)
    est.fit(x)
    hits = np.asarray(est.result_.get("coverage_hit_series") or [], dtype=int)
    from src.horizon_scientific_eval import _block_bootstrap_lower_bound
    lb = _block_bootstrap_lower_bound(hits, 0.05) if hits.size else None
    rho, _ = spearmanr(est.lower_bounds_, est.test_horizons_)
    record(f"v24_{dataset}_p{period}_{model_name}",
           coverage=round(float(est.coverage_), 4),
           boot_lb=None if lb is None else round(float(lb), 4),
           L_med=round(float(np.median(est.lower_bounds_)), 2),
           H_med=round(float(np.median(est.test_horizons_)), 2),
           spearman=round(float(rho), 3), n=int(hits.size))


def arm_verdict():
    R = read_records()
    print("\n================ PRE-REGISTERED VERDICTS ================")
    ok_all = True

    def check(name, cond, detail):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"{name}: {'PASS' if cond else 'FAIL'}  ({detail})")

    for ds in FILES:
        v = R.get(f"v1_{ds}", {})
        check(f"V1 {ds} regime quasi-periodic",
              v.get("regime") == "quasi-periodic",
              f"regime={v.get('regime')} periodicity={v.get('periodicity')}")
    for key, v in R.items():
        if key.startswith("v24_"):
            check(f"V2 {key[4:]} coverage>=0.88 & bootLB>=0.85",
                  v["coverage"] >= 0.88 and (v["boot_lb"] or 0) >= 0.85,
                  f"cov={v['coverage']} lb={v['boot_lb']}")
    for ds in FILES:
        v = R.get(f"v3_{ds}", {})
        if v:
            check(f"V3 {ds} slope in [1.2, 2.8]", 1.2 <= v["slope"] <= 2.8,
                  f"s={v['slope']}")
            check(f"V3 {ds} power beats chaotic log-law", v["r2_pow"] > v["r2_log"],
                  f"R2 {v['r2_pow']} vs {v['r2_log']}")
            check(f"V3 {ds} sigma_eff/sigma_hat in [1/3, 3]",
                  1 / 3 <= v["ratio"] <= 3, f"ratio={v['ratio']}")
    p1l, p2l = R.get("v24_aep_p1_linear", {}), R.get("v24_aep_p2_linear", {})
    p1n, p2n = R.get("v24_aep_p1_naive", {}), R.get("v24_aep_p2_naive", {})
    if p1l and p2l and p1n and p2n:
        check("V4 learned > naive in both periods",
              p1l["L_med"] > p1n["L_med"] and p2l["L_med"] > p2n["L_med"],
              f"p1 {p1l['L_med']} vs {p1n['L_med']} | p2 {p2l['L_med']} vs {p2n['L_med']}")
        ratio = p2l["L_med"] / p1l["L_med"]
        check("V4 learned L_med stable across periods", 0.6 <= ratio <= 1.4,
              f"ratio={ratio:.2f}")
        check("V4 Spearman > 0.3 both periods",
              p1l["spearman"] > 0.3 and p2l["spearman"] > 0.3,
              f"{p1l['spearman']} / {p2l['spearman']}")
    print(f"\nOVERALL: {'ALL PASS' if ok_all else 'AT LEAST ONE FAIL'}")


ARMS = {
    "profile": lambda a: arm_profile(),
    "sweep": lambda a: arm_sweep(a.dataset),
    "estimator": lambda a: arm_estimator(a.dataset, a.period, a.model),
    "verdict": lambda a: arm_verdict(),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--dataset", choices=list(FILES), default="aep")
    ap.add_argument("--period", choices=["1", "2", "full"], default="full")
    ap.add_argument("--model", choices=["linear", "naive"], default="linear")
    a = ap.parse_args()
    ARMS[a.arm](a)
