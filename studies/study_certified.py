"""Study: certified (Lipschitz-based) horizon bound vs empirical H_w labels.

For Lorenz and Rossler (per-system dt defaults) and two forecasters
(LinearAR, small MLP), this study reports:
  - L_inf(f), L_2(f): Lipschitz upper bounds of the one-step model,
  - G = max(1, L_inf), delta = sup one-step residual on train+val+calib,
  - h_cert: certified horizon (first step where the closed-form bound
    delta * (G^h - 1) / (G - 1) can reach the tolerance),
  - violations: number of test windows with H_w < h_cert (0 = sound),
  - usefulness ratio h_cert / median(H_w),
  - delta_test / delta: does the empirical residual sup transfer to test?

Also reports two conservative variants (delta inflated x1.5; delta from
the calib segment only) to quantify the proposed fix if violations occur.

Run from the repo root:  python studies/study_certified.py
Runtime: ~2-4 minutes on CPU. All randomness is seeded.
"""

import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.horizon_certified import (
    certified_horizon,
    empirical_delta_sup,
    lipschitz_l2,
    lipschitz_linf,
)
from src.horizon_metrics import build_horizon_dataset
from src.horizon_models import TorchWrapper
from src.horizon_training import train_mlp
from src.horizon_utils import (
    build_supervised,
    generate_lorenz,
    generate_rossler,
    horizon_from_model_bound_by_growth,
    set_seed,
    split_series,
    standardize_series,
)
from src.horizon_models import LinearAR

SEED = 0
SERIES_LEN = 8000
DIM = 4
LAG = 2
TOLERANCE = 0.4  # absolute, in std units (valid-time convention)
HORIZON_MAX = 400
MAX_WINDOWS = 500
CONSECUTIVE_K = 2
MLP_HIDDEN = 32
MLP_EPOCHS = 15

# Per-system sampling dt (DEFAULT_DT in src/horizon_cli.py) and the
# literature largest Lyapunov exponent per unit time, used only to
# express horizons in Lyapunov times.
SYSTEMS = [
    {"name": "lorenz", "dt": 0.01, "lyap_ut": 0.906, "gen": generate_lorenz},
    {"name": "rossler", "dt": 0.05, "lyap_ut": 0.071, "gen": generate_rossler},
]


def _h_cert_from_delta(growth, delta, tolerance):
    """Recomputes h_cert (1-indexed step) for a modified delta."""
    if delta <= 0.0:
        return float("inf")
    if delta >= tolerance:
        return 1.0
    steps = horizon_from_model_bound_by_growth(growth, delta, delta, tolerance)
    return float("inf") if math.isinf(steps) else float(steps) + 1.0


def _violations(h_w, h_cert):
    return int(np.sum(h_w < h_cert))


def run_config(system, model_name, model, splits_std, dt, lyap_ut):
    train_std, val_std, calib_std, test_std = splits_std
    t0 = time.time()

    l_inf = lipschitz_linf(model, input_dim=DIM)
    l_2 = lipschitz_l2(model, input_dim=DIM)
    h_cert, growth, delta = certified_horizon(
        model, [train_std, val_std, calib_std], DIM, LAG, TOLERANCE
    )
    delta_test = empirical_delta_sup(model, test_std, DIM, LAG)

    _, h_w = build_horizon_dataset(
        model,
        test_std,
        DIM,
        LAG,
        horizon_max=HORIZON_MAX,
        tolerance=TOLERANCE,
        max_windows=MAX_WINDOWS,
        seed=SEED,
        use_jacobian=False,
        error_mode="absolute",
        consecutive_k=CONSECUTIVE_K,
    )

    n_windows = int(h_w.size)
    n_censored = int(np.sum(h_w >= HORIZON_MAX))
    median_hw = float(np.median(h_w)) if n_windows else float("nan")
    min_hw = float(np.min(h_w)) if n_windows else float("nan")
    violations = _violations(h_w, h_cert)
    ratio = h_cert / median_hw if median_hw > 0 else float("nan")

    # Conservative variants (the proposed fixes if violations > 0).
    h_cert_infl = _h_cert_from_delta(growth, 1.5 * delta, TOLERANCE)
    delta_disjoint = empirical_delta_sup(model, calib_std, DIM, LAG)
    h_cert_disj = _h_cert_from_delta(growth, delta_disjoint, TOLERANCE)

    lyap_time_steps = 1.0 / (lyap_ut * dt)  # Lyapunov time in samples
    elapsed = time.time() - t0
    return {
        "system": system,
        "model": model_name,
        "L_inf": l_inf,
        "L_2": l_2,
        "G": growth,
        "delta": delta,
        "delta_test": delta_test,
        "delta_test_over_delta": delta_test / delta if delta > 0 else float("nan"),
        "h_cert": h_cert,
        "h_cert_lyap": h_cert / lyap_time_steps,
        "n_windows": n_windows,
        "n_censored": n_censored,
        "median_Hw": median_hw,
        "min_Hw": min_hw,
        "violations": violations,
        "ratio_h_cert_over_median_Hw": ratio,
        "h_cert_delta_x1.5": h_cert_infl,
        "violations_delta_x1.5": _violations(h_w, h_cert_infl),
        "delta_disjoint_calib": delta_disjoint,
        "h_cert_delta_disjoint": h_cert_disj,
        "violations_delta_disjoint": _violations(h_w, h_cert_disj),
        "eval_seconds": elapsed,
    }


def main():
    set_seed(SEED)
    t_start = time.time()
    results = []

    for sysconf in SYSTEMS:
        name, dt, gen = sysconf["name"], sysconf["dt"], sysconf["gen"]
        print(f"=== {name} (dt={dt}) ===", flush=True)
        series = gen(SERIES_LEN, dt=dt)
        train, val, calib, test = split_series(
            series, train_ratio=0.5, val_ratio=0.15, calib_ratio=0.15
        )
        train_std, mean, std = standardize_series(train)
        val_std, _, _ = standardize_series(val, mean, std)
        calib_std, _, _ = standardize_series(calib, mean, std)
        test_std, _, _ = standardize_series(test, mean, std)
        splits_std = (train_std, val_std, calib_std, test_std)

        x_train, y_train = build_supervised(train_std, DIM, LAG, horizon=1)
        x_val, y_val = build_supervised(val_std, DIM, LAG, horizon=1)

        # LinearAR
        linear = LinearAR(reg=1e-4).fit(x_train, y_train)
        res = run_config(name, "linear", linear, splits_std, dt, sysconf["lyap_ut"])
        results.append(res)
        print(_fmt(res), flush=True)

        # Small MLP (Tanh, 2 hidden layers of MLP_HIDDEN)
        set_seed(SEED)
        mlp, _ = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=DIM,
            hidden_dim=MLP_HIDDEN,
            epochs=MLP_EPOCHS,
            patience=MLP_EPOCHS,
            device="cpu",
        )
        mlp.eval()
        wrapper = TorchWrapper(mlp, "cpu")
        res = run_config(name, "mlp", wrapper, splits_std, dt, sysconf["lyap_ut"])
        results.append(res)
        print(_fmt(res), flush=True)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "study_certified_results.csv")
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {out_path}")
    print(f"Total runtime: {time.time() - t_start:.1f} s")

    total_viol = sum(r["violations"] for r in results)
    print(f"Total violations across configs: {total_viol} "
          f"({'SOUND' if total_viol == 0 else 'NOT SOUND - see diagnostics'})")


def _fmt(r):
    return (
        f"{r['system']:8s} {r['model']:6s} "
        f"L_inf={r['L_inf']:.4g} L_2={r['L_2']:.4g} G={r['G']:.4g} "
        f"delta={r['delta']:.4g} h_cert={r['h_cert']:.0f} "
        f"({r['h_cert_lyap']:.3f} T_lyap) | "
        f"median(H_w)={r['median_Hw']:.0f} min(H_w)={r['min_Hw']:.0f} "
        f"censored={r['n_censored']}/{r['n_windows']} "
        f"violations={r['violations']} ratio={r['ratio_h_cert_over_median_Hw']:.4f} "
        f"delta_test/delta={r['delta_test_over_delta']:.3f} "
        f"[x1.5: h={r['h_cert_delta_x1.5']:.0f} v={r['violations_delta_x1.5']}] "
        f"[disjoint: h={r['h_cert_delta_disjoint']:.0f} "
        f"v={r['violations_delta_disjoint']}] "
        f"({r['eval_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    main()
