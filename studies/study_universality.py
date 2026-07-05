"""Universality test: do learned-forecaster horizons obey a Lyapunov scaling law?

Candidate law (the only 'fundamental discovery'-grade hypothesis in reach of
this project): for a window with one-step error e0_w, the horizon behaves as

    H_w  ~=  ln(tau / e0_w) / (Lambda_eff * dt)

with an effective growth rate Lambda_eff whose ratio to the system's global
Lyapunov exponent lambda_1 is (a) roughly constant across TOLERANCES within a
system, and (b) of the same order across SYSTEMS. If (a)+(b) hold, horizons of
learned forecasters are governed by the attractor's Lyapunov structure through
a universal O(1) constant; if they fail, horizons are model-error-limited and
the failure mode is informative.

Protocol (pre-registered):
- systems: lorenz, rossler, mackey_glass (per-system dt, reference lambda_1
  from literature, validated by tests/test_physics_chaos.py);
- forecaster: linear (2 seeds) and MLP on lorenz (robustness check);
- tolerances tau in {0.2, 0.4, 0.8} std;
- per window: Lambda_eff_w = ln(tau / e0_w) / (H_w * dt), with e0_w = resid1
  (the window's one-step absolute error, feature dim+3);
- filters: windows with e0_w < tau/4 (identifiable growth range) and
  H_w < Hmax (non-censored);
- report the distribution of R_w = Lambda_eff_w / lambda_1 per (system, tau):
  median, IQR;
- verdicts: SCALING-IN-TAU holds if median R varies < 30% across tau while
  median raw H varies > 60%; CROSS-SYSTEM holds if median R spans < factor 2
  across systems at tau=0.4.

Reproducible: python studies/study_universality.py  (~3-6 min CPU)
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_cli import DEFAULT_DT, DEFAULT_LAMBDA, build_parser, load_config
from src.horizon_data import DataManager
from src.horizon_forecast import Forecaster
from src.horizon_metrics import build_horizon_dataset

SYSTEMS = {
    "lorenz": {"series_len": 12000, "warmup": 1000, "hmax": 400},
    "rossler": {"series_len": 12000, "warmup": 1000, "hmax": 400},
    "mackey_glass": {"series_len": 6000, "warmup": 400, "hmax": 280},
}
TAUS = [0.2, 0.4, 0.8]
SEEDS = [0, 1]


def run_one(dataset, seed, model_name):
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    parser = build_parser()
    parser.set_defaults(**config)
    args = parser.parse_args([])
    args.dataset = dataset
    args.seed = seed
    args.model = model_name
    args.series_len = SYSTEMS[dataset]["series_len"]
    args.warmup = SYSTEMS[dataset]["warmup"]
    args.train_ratio = 0.6
    args.val_ratio = 0.15
    args.calib_ratio = 0.15
    args.mlp_epochs = 30
    args.dim_max = 6
    args.lag_max = 6
    args.progress = False
    args.use_cuda = False

    import torch

    from src.horizon_utils import set_seed

    set_seed(seed)
    device = torch.device("cpu")
    dm = DataManager(args)
    train, val, calib, test = dm.prepare_data()
    fc = Forecaster(args, device)
    best = fc.select_embedding(train, val)
    model = fc.train_final_model(train, val)

    dim, lag = best["dim"], best["lag"]
    dt = DEFAULT_DT[dataset]
    lam = DEFAULT_LAMBDA[dataset]
    hmax = SYSTEMS[dataset]["hmax"]

    out = []
    for tau in TAUS:
        feats, H = build_horizon_dataset(
            model, test, dim, lag, hmax, tau,
            max_windows=500, seed=seed, use_jacobian=False,
            error_mode="absolute", consecutive_k=2,
        )
        e0 = feats[:, dim + 3]  # resid1: per-window one-step abs error
        keep = (e0 > 0) & (e0 < tau / 4.0) & (H < hmax)
        n_censored = int(np.sum(H >= hmax))
        if keep.sum() < 30:
            out.append((tau, None, None, None, int(keep.sum()), n_censored))
            continue
        lam_eff = np.log(tau / e0[keep]) / (H[keep] * dt)
        ratio = lam_eff / lam
        out.append(
            (
                tau,
                float(np.median(H[keep])),
                float(np.median(ratio)),
                (float(np.quantile(ratio, 0.25)), float(np.quantile(ratio, 0.75))),
                int(keep.sum()),
                n_censored,
            )
        )
    return dim, lag, out


def main():
    results = {}
    for dataset in SYSTEMS:
        for seed in SEEDS:
            dim, lag, rows = run_one(dataset, seed, "linear")
            results[(dataset, "linear", seed)] = rows
            for tau, h_med, r_med, r_iqr, n, n_cens in rows:
                print(
                    f"{dataset:13s} linear seed={seed} dim={dim} lag={lag} "
                    f"tau={tau:.1f} H_med={h_med} R_med={r_med} IQR={r_iqr} "
                    f"n={n} censored={n_cens}",
                    flush=True,
                )
    # Robustness: MLP on lorenz, seed 0.
    dim, lag, rows = run_one("lorenz", 0, "mlp")
    results[("lorenz", "mlp", 0)] = rows
    for tau, h_med, r_med, r_iqr, n, n_cens in rows:
        print(
            f"lorenz        mlp    seed=0 dim={dim} lag={lag} tau={tau:.1f} "
            f"H_med={h_med} R_med={r_med} IQR={r_iqr} n={n} censored={n_cens}",
            flush=True,
        )

    print("\n=== VERDICTS (pre-registered) ===")
    for (dataset, model, seed), rows in sorted(results.items()):
        vals = [(t, h, r) for t, h, r, _, n, _ in rows if r is not None]
        if len(vals) < 2:
            print(f"{dataset}/{model}/s{seed}: insufficient windows")
            continue
        r_meds = [r for _, _, r in vals]
        h_meds = [h for _, h, _ in vals]
        r_var = (max(r_meds) - min(r_meds)) / np.mean(r_meds)
        h_var = (max(h_meds) - min(h_meds)) / np.mean(h_meds)
        ok = "HOLDS" if (r_var < 0.30 and h_var > 0.60) else "FAILS"
        print(
            f"{dataset}/{model}/s{seed}: scaling-in-tau {ok} "
            f"(R spread {r_var:.2f}, raw-H spread {h_var:.2f}, R_med~{np.mean(r_meds):.2f})"
        )
    at04 = [
        r for (ds, m, s), rows in results.items() if m == "linear"
        for t, h, r, _, n, _ in rows if t == 0.4 and r is not None
    ]
    if at04:
        print(
            f"cross-system at tau=0.4 (linear): R_med range "
            f"[{min(at04):.2f}, {max(at04):.2f}] -> "
            f"{'HOLDS' if max(at04) / max(min(at04), 1e-9) < 2.0 else 'FAILS'} (factor-2 criterion)"
        )


if __name__ == "__main__":
    main()
