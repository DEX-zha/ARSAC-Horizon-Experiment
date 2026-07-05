"""Study: do finite-time Lyapunov exponents (FTLE) explain / improve horizon labels?

Protocol (all seeded, CPU-only, Lorenz dt=0.01, series_len=4000):
  (1) Ground-truth check of the variational-equation FTLE estimator:
      mean lambda_T -> 0.906 as T grows, variance ~ 1/T.
  (2) FTLE of the LEARNED map (companion-Jacobian QR products along the
      predicted rollout, k=100 steps = 1.0 time unit) vs ground truth at
      the same trajectory points: means, stds, correlations.
  (3) Spearman correlation of model FTLE (and ground-truth FTLE oracle)
      with the horizon labels H_w from build_horizon_dataset, compared to
      the existing features jac_mean and resid1.
  (4) GradientBoostingRegressor(loss='quantile', alpha=0.1), 5-fold CV
      (contiguous folds, no shuffle: windows overlap): mean pinball loss
      with existing features vs existing+FTLE (and +oracle FTLE).

Run from the repo root:  python studies/study_ftle.py
"""

import os
import sys
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import KFold

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_ftle import (
    ftle_along_series,
    lorenz_ftle_ground_truth,
    lorenz_trajectory,
)
from src.horizon_metrics import build_horizon_dataset, evaluate_mse
from src.horizon_models import LinearAR, TorchWrapper
from src.horizon_training import train_mlp
from src.horizon_utils import build_supervised, set_seed, standardize_series

SEED = 0
DT = 0.01
DIM = 4
LAG = 10
K_FTLE = 100  # 100 steps x dt=0.01 -> T = 1.0 time unit
HORIZON_MAX = 30
TOLERANCE = 0.4  # absolute, standardized units
STRIDE = 2
ALPHA = 0.1
SERIES_LEN = 4000
TRAIN_END = 2400
VAL_END = 2800
LYAP_REF = 0.906


def _safe_spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _safe_pearson(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(pearsonr(a, b).statistic)


def part1_ground_truth_check():
    print("=" * 72)
    print("(1) Ground-truth FTLE estimator check (Lorenz variational + QR)")
    print("=" * 72)
    states = lorenz_trajectory(15000, dt=DT, warmup=2000, x0=(1.0, 1.0, 1.0))
    points = states[::50][:300]  # 300 points spaced 0.5 t.u. apart
    rows = {}
    for T in (0.5, 1.0, 2.0, 5.0, 10.0):
        lam = lorenz_ftle_ground_truth(points, T=T, dt=DT)
        rows[T] = lam
        print(
            f"  T={T:4.1f}  mean={lam.mean():+.4f}  std={lam.std():.4f}  "
            f"var={lam.var():.5f}  var*T={lam.var() * T:.5f}  "
            f"(reference lambda_1={LYAP_REF})"
        )
    v05, v20 = rows[0.5].var(), rows[2.0].var()
    print(f"  var(T=0.5)/var(T=2.0) = {v05 / v20:.2f}  (1/T scaling predicts 4.0; "
          "an O(1/T^2) alignment-transient term makes the observed decay faster)")
    return rows


def _fit_models(train_std, val_std, seed):
    x_tr, y_tr = build_supervised(train_std, DIM, LAG, horizon=1)
    x_va, y_va = build_supervised(val_std, DIM, LAG, horizon=1)

    linear = LinearAR(reg=1e-4).fit(x_tr, y_tr)

    set_seed(seed)
    net, _ = train_mlp(
        x_tr, y_tr, x_va, y_va,
        input_dim=DIM, hidden_dim=32, epochs=20, lr=1e-3,
        batch_size=64, patience=20, device="cpu",
    )
    mlp = TorchWrapper(net, device="cpu")
    mse = {
        "linear": evaluate_mse(linear, x_va, y_va),
        "mlp": evaluate_mse(mlp, x_va, y_va),
    }
    return {"linear": linear, "mlp": mlp}, mse


def _cv_pinball(x, y, alpha=ALPHA, n_splits=5):
    """Contiguous 5-fold CV mean pinball loss of a quantile GBR."""
    losses = []
    for tr_idx, te_idx in KFold(n_splits=n_splits, shuffle=False).split(x):
        gbr = GradientBoostingRegressor(
            loss="quantile", alpha=alpha, random_state=SEED
        )
        gbr.fit(x[tr_idx], y[tr_idx])
        losses.append(mean_pinball_loss(y[te_idx], gbr.predict(x[te_idx]), alpha=alpha))
    return float(np.mean(losses))


def _cv_pinball_constant(y, alpha=ALPHA, n_splits=5):
    """Constant empirical-quantile baseline under the same CV."""
    losses = []
    y = np.asarray(y, dtype=np.float64)
    for tr_idx, te_idx in KFold(n_splits=n_splits, shuffle=False).split(y):
        q = np.quantile(y[tr_idx], alpha)
        losses.append(mean_pinball_loss(y[te_idx], np.full(len(te_idx), q), alpha=alpha))
    return float(np.mean(losses))


def analyze_model(name, model, ev_std, states, ev_offset, seed):
    print("=" * 72)
    print(f"(2)+(3)+(4) Model: {name} (seed {seed})")
    print("=" * 72)
    t0 = time.perf_counter()

    feats, labels = build_horizon_dataset(
        model, ev_std, DIM, LAG, HORIZON_MAX, TOLERANCE,
        max_windows=None, seed=seed, use_jacobian=True,
        error_mode="absolute", stride=STRIDE,
    )
    window_len = (DIM - 1) * LAG + 1
    n_label = len(ev_std) - window_len - HORIZON_MAX
    label_starts = np.arange(0, n_label, STRIDE, dtype=np.int64)
    assert len(label_starts) == len(labels)

    ftle_step, ftle_starts = ftle_along_series(
        model, ev_std, DIM, LAG, k=K_FTLE, sample_stride=STRIDE,
        max_windows=None, seed=seed,
    )
    ftle_ut = ftle_step / DT  # exponent per unit time

    # Sensitivity: FTLE over k=30 steps, the scale of the labels (Hmax=30).
    ftle30_step, ftle30_starts = ftle_along_series(
        model, ev_std, DIM, LAG, k=HORIZON_MAX, sample_stride=STRIDE,
        max_windows=None, seed=seed,
    )

    # All start grids are arange(0, n, STRIDE): the common prefix aligns.
    m = min(len(label_starts), len(ftle_starts), len(ftle30_starts))
    assert np.array_equal(label_starts[:m], ftle_starts[:m])
    assert np.array_equal(label_starts[:m], ftle30_starts[:m])
    labels_m = labels[:m]
    feats_m = feats[:m]
    ftle_m = ftle_ut[:m]
    ftle30_m = ftle30_step[:m] / DT

    # Ground-truth FTLE at the same trajectory points (T = k*dt = 1.0).
    anchor_idx = ev_offset + label_starts[:m] + (DIM - 1) * LAG
    gt_m = lorenz_ftle_ground_truth(states[anchor_idx], T=K_FTLE * DT, dt=DT)

    print(f"  windows: {m}   p_sat (H_w == Hmax): {np.mean(labels_m == HORIZON_MAX):.3f}")
    print(f"  model FTLE /t.u.: mean={ftle_m.mean():+.4f} std={ftle_m.std():.4f}")
    print(f"  truth FTLE /t.u.: mean={gt_m.mean():+.4f} std={gt_m.std():.4f}")
    print(f"  corr(model FTLE, truth FTLE): pearson={_safe_pearson(ftle_m, gt_m):+.3f} "
          f"spearman={_safe_spearman(ftle_m, gt_m):+.3f}")

    resid1 = feats_m[:, DIM + 3]
    jac_mean = feats_m[:, DIM + 4]
    log_h = np.log(np.maximum(labels_m, 1.0))
    corr = {
        "ftle_H": _safe_spearman(ftle_m, labels_m),
        "ftle_logH": _safe_spearman(ftle_m, log_h),
        "ftle30_H": _safe_spearman(ftle30_m, labels_m),
        "gt_H": _safe_spearman(gt_m, labels_m),
        "jac_mean_H": _safe_spearman(jac_mean, labels_m),
        "resid1_H": _safe_spearman(resid1, labels_m),
    }
    print(f"  Spearman(model FTLE k=100, H_w)    = {corr['ftle_H']:+.3f}")
    print(f"  Spearman(model FTLE k=100, log H_w)= {corr['ftle_logH']:+.3f}")
    print(f"  Spearman(model FTLE k=30,  H_w)    = {corr['ftle30_H']:+.3f}  [sensitivity]")
    print(f"  Spearman(truth FTLE,       H_w)    = {corr['gt_H']:+.3f}  [oracle]")
    print(f"  Spearman(jac_mean,         H_w)    = {corr['jac_mean_H']:+.3f}  [existing]")
    print(f"  Spearman(resid1,           H_w)    = {corr['resid1_H']:+.3f}  [existing]")
    # Direct test of the core relation H(x) ~ ln(tol/e0) / lambda_T(x)
    # using the TRUE flow FTLE and the observed one-step error e0=resid1.
    e0 = np.maximum(resid1, 1e-6)
    lam_step = np.maximum(gt_m * DT, 1e-3)  # per-step rate, floored at ~0
    h_theory = np.log(TOLERANCE / e0) / lam_step
    corr["h_theory_H"] = _safe_spearman(h_theory, labels_m)
    print(f"  Spearman(ln(tol/e0)/lambda_T,H_w)  = {corr['h_theory_H']:+.3f}"
          "  [theory-direct, oracle lambda]")

    noncens = labels_m < HORIZON_MAX
    if noncens.sum() >= 30:
        print(f"  non-censored subset (n={int(noncens.sum())}): "
              f"Spearman(FTLE, H_w)={_safe_spearman(ftle_m[noncens], labels_m[noncens]):+.3f} "
              f"Spearman(truth, H_w)={_safe_spearman(gt_m[noncens], labels_m[noncens]):+.3f}")

    x_base = feats_m
    x_ftle = np.column_stack([feats_m, ftle_m])
    x_ftle30 = np.column_stack([feats_m, ftle30_m])
    x_gt = np.column_stack([feats_m, gt_m])
    pin_const = _cv_pinball_constant(labels_m)
    pin_base = _cv_pinball(x_base, labels_m)
    pin_ftle = _cv_pinball(x_ftle, labels_m)
    pin_ftle30 = _cv_pinball(x_ftle30, labels_m)
    pin_gt = _cv_pinball(x_gt, labels_m)
    gain_ftle = 100.0 * (pin_base - pin_ftle) / pin_base
    gain_ftle30 = 100.0 * (pin_base - pin_ftle30) / pin_base
    gain_gt = 100.0 * (pin_base - pin_gt) / pin_base
    print(f"  pinball@{ALPHA} 5-fold CV: const-quantile baseline    = {pin_const:.4f}")
    print(f"  pinball@{ALPHA} 5-fold CV: existing features          = {pin_base:.4f}")
    print(f"  pinball@{ALPHA} 5-fold CV: existing + FTLE k=100      = {pin_ftle:.4f} "
          f"({gain_ftle:+.2f}% vs existing)")
    print(f"  pinball@{ALPHA} 5-fold CV: existing + FTLE k=30       = {pin_ftle30:.4f} "
          f"({gain_ftle30:+.2f}% vs existing)  [sensitivity]")
    print(f"  pinball@{ALPHA} 5-fold CV: existing + truth FTLE      = {pin_gt:.4f} "
          f"({gain_gt:+.2f}% vs existing)  [oracle]")
    print(f"  [{name}] analysis took {time.perf_counter() - t0:.1f}s")

    return {
        "corr": corr,
        "pinball": {
            "const": pin_const, "base": pin_base, "ftle": pin_ftle,
            "ftle30": pin_ftle30, "gt": pin_gt,
        },
        "gain_ftle_pct": gain_ftle,
        "gain_ftle30_pct": gain_ftle30,
        "gain_gt_pct": gain_gt,
        "p_sat": float(np.mean(labels_m == HORIZON_MAX)),
        "n_windows": int(m),
        "ftle_mean": float(ftle_m.mean()),
        "ftle_std": float(ftle_m.std()),
        "gt_mean": float(gt_m.mean()),
        "gt_std": float(gt_m.std()),
        "corr_model_truth": _safe_pearson(ftle_m, gt_m),
    }


def run_seed(seed, x0):
    """Trains both models on a fresh trajectory and runs parts (2)-(4)."""
    # Single RK4 trajectory: full 3D states kept so ground-truth FTLE can be
    # anchored at the exact points where the forecaster windows sit.
    states = lorenz_trajectory(SERIES_LEN, dt=DT, warmup=3000, x0=x0)
    series = states[:, 0]
    train, val, ev = series[:TRAIN_END], series[TRAIN_END:VAL_END], series[VAL_END:]
    train_std, mu, sd = standardize_series(train)
    val_std, _, _ = standardize_series(val, mean=mu, std=sd)
    ev_std, _, _ = standardize_series(ev, mean=mu, std=sd)

    models, mse = _fit_models(train_std, val_std, seed)
    print(f"[seed {seed}] one-step val MSE: linear={mse['linear']:.2e}  "
          f"mlp={mse['mlp']:.2e}")

    results = {}
    for name in ("linear", "mlp"):
        results[name] = analyze_model(name, models[name], ev_std, states,
                                      VAL_END, seed)
    return results


def main():
    t_start = time.perf_counter()
    set_seed(SEED)

    part1_ground_truth_check()

    seed_x0 = {0: (-6.0, 8.0, 27.0), 1: (4.0, -3.0, 21.0)}
    all_results = {seed: run_seed(seed, x0) for seed, x0 in seed_x0.items()}

    print("=" * 72)
    print("DECISION (criterion: pinball gain >= 5% OR (|Spearman(FTLE,H_w)| >= 0.3"
          " AND stronger than jac_mean)); FTLE = model FTLE k=100 per spec")
    print("=" * 72)
    for name in ("linear", "mlp"):
        gains = [all_results[s][name]["gain_ftle_pct"] for s in seed_x0]
        s_ftles = [all_results[s][name]["corr"]["ftle_H"] for s in seed_x0]
        s_jacs = [all_results[s][name]["corr"]["jac_mean_H"] for s in seed_x0]
        mean_gain = float(np.mean(gains))
        mean_abs_ftle = float(np.mean(np.abs(s_ftles)))
        mean_abs_jac = float(np.mean(np.abs(s_jacs)))
        cond_pinball = mean_gain >= 5.0
        cond_corr = (
            np.isfinite(mean_abs_ftle)
            and mean_abs_ftle >= 0.3
            and (not np.isfinite(mean_abs_jac) or mean_abs_ftle > mean_abs_jac)
        )
        verdict = "INTEGRATE" if (cond_pinball or cond_corr) else "SHELVE"
        per_seed = "  ".join(
            f"seed{s}: gain={all_results[s][name]['gain_ftle_pct']:+.2f}% "
            f"rho={all_results[s][name]['corr']['ftle_H']:+.3f}"
            for s in seed_x0
        )
        print(f"  {name}: mean pinball gain {mean_gain:+.2f}% | "
              f"mean |Spearman(FTLE,H)|={mean_abs_ftle:.3f} vs "
              f"|jac_mean|={mean_abs_jac:.3f} -> {verdict}")
        print(f"          {per_seed}")
    print(f"total runtime: {time.perf_counter() - t_start:.1f}s")
    return all_results


if __name__ == "__main__":
    main()
