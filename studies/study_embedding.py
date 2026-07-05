"""Study: theory-grounded embedding (MI + FNN) for the chaos estimators.

For the 4 systems at their audited per-system dt, selects (dim, lag) with
mutual information (Fraser & Swinney) + false nearest neighbors (Kennel)
and compares the Rosenstein Lyapunov estimate (auto params) under this
embedding vs the naive default (dim=3, lag=1), against literature values.

Decision criterion (Point 6): integrate select_embedding as the default
source of lyap_dim/lyap_lag when the user leaves them None if >= 2 systems
get closer to the literature value and none degrades by more than 50
points of relative error.

Reproducible: seeded, CPU-only, numpy/scipy only. Runtime ~2-5 min.
Run from the repo root:  python studies/study_embedding.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_embedding import select_embedding
from src.horizon_utils import (
    estimate_lyapunov,
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
    set_seed,
)

SEED = 0
ROBUSTNESS_SEEDS = [0, 1, 2]  # perturbed initial conditions per seed
NAIVE_DIM, NAIVE_LAG = 3, 1  # naive default embedding under comparison
MAX_DEGRADATION = 0.50  # max allowed increase in relative error (50 points)

# name, generator(seed), dt, literature lambda_1 per unit time. The seed
# perturbs the initial condition (same attractor, different trajectory).
SYSTEMS = [
    (
        "lorenz",
        lambda s: generate_lorenz(6000, dt=0.01, warmup=1000, x0=1.0 + 0.1 * s),
        0.01,
        0.906,
    ),
    (
        "rossler",
        lambda s: generate_rossler(8000, dt=0.05, warmup=1000, x0=1.0 + 0.1 * s),
        0.05,
        0.071,
    ),
    (
        "mackey_glass",
        lambda s: generate_mackey_glass(3000, tau=17.0, dt=1.0, warmup=300 + 50 * s),
        1.0,
        0.006,
    ),
    (
        # x0 = 0.25 would land on the exact fixed point 0.75 after one
        # step (degenerate orbit), so perturb by 0.03 instead of 0.05.
        "logistic",
        lambda s: generate_logistic_map(4000, x0=0.2 + 0.03 * s),
        1.0,
        math.log(2.0),
    ),
]


def lyap_per_unit_time(series, dim, lag, dt):
    """Rosenstein estimate (auto params) converted to per-unit-time."""
    slope, _ = estimate_lyapunov(series, dim=dim, lag=lag, dt=dt)
    return slope / dt


def run_one(name, gen, dt, lit, seed):
    """Selects the theory embedding and compares both Lyapunov estimates."""
    set_seed(seed)
    series = gen(seed)
    t1 = time.time()
    sel = select_embedding(series, max_dim=10, max_lag=100, seed=seed)
    t_sel = time.time() - t1
    lam_theory = lyap_per_unit_time(series, sel["dim"], sel["lag"], dt)
    lam_naive = lyap_per_unit_time(series, NAIVE_DIM, NAIVE_LAG, dt)
    return dict(
        name=name,
        seed=seed,
        dt=dt,
        dim=sel["dim"],
        lag=sel["lag"],
        lag_time=sel["lag"] * dt,
        fnn=sel["fnn_fractions"],
        lit=lit,
        lam_theory=lam_theory,
        lam_naive=lam_naive,
        rel_theory=abs(lam_theory - lit) / abs(lit),
        rel_naive=abs(lam_naive - lit) / abs(lit),
        t_sel=t_sel,
    )


def main():
    t0 = time.time()
    rows = [run_one(name, gen, dt, lit, SEED) for name, gen, dt, lit in SYSTEMS]

    print("\n=== Selected embeddings (MI + FNN) ===")
    hdr = f"{'system':<14}{'dt':>6}{'dim':>5}{'lag':>5}{'lag*dt':>8}  fnn_fractions"
    print(hdr)
    for r in rows:
        fnn_txt = np.array2string(np.round(r["fnn"], 3), max_line_width=120)
        print(
            f"{r['name']:<14}{r['dt']:>6}{r['dim']:>5}{r['lag']:>5}"
            f"{r['lag_time']:>8.2f}  {fnn_txt}  ({r['t_sel']:.1f}s)"
        )

    print("\n=== Sanity checks ===")
    lorenz = next(r for r in rows if r["name"] == "lorenz")
    mg = next(r for r in rows if r["name"] == "mackey_glass")
    print(
        f"lorenz lag*dt = {lorenz['lag_time']:.2f} t.u. "
        f"(expected ~0.1-0.2): {'OK' if 0.05 <= lorenz['lag_time'] <= 0.3 else 'FAIL'}"
    )
    print(
        f"mackey_glass dim = {mg['dim']} "
        f"(expected 4-7): {'OK' if 4 <= mg['dim'] <= 7 else 'FAIL'}"
    )

    print("\n=== Lyapunov per unit time: theory embedding vs naive (3, 1) ===")
    print(
        f"{'system':<14}{'lit.':>8}{'theory':>10}{'naive':>10}"
        f"{'relerr_th':>11}{'relerr_nv':>11}{'closer':>8}{'degraded':>10}"
    )
    n_closer = 0
    n_degraded = 0
    for r in rows:
        closer = r["rel_theory"] < r["rel_naive"]
        degraded = r["rel_theory"] > r["rel_naive"] + MAX_DEGRADATION
        n_closer += int(closer)
        n_degraded += int(degraded)
        print(
            f"{r['name']:<14}{r['lit']:>8.3f}{r['lam_theory']:>10.4f}"
            f"{r['lam_naive']:>10.4f}{r['rel_theory']:>11.3f}{r['rel_naive']:>11.3f}"
            f"{'yes' if closer else 'no':>8}{'yes' if degraded else 'no':>10}"
        )

    integrate = n_closer >= 2 and n_degraded == 0
    print(
        f"\nDecision (seed {SEED}): closer on {n_closer}/4 systems, "
        f"{n_degraded} degraded by > {MAX_DEGRADATION:.0%} relative error"
    )
    print(f"Verdict: {'INTEGRATE' if integrate else 'DO NOT INTEGRATE (as-is)'}")

    print("\n=== Robustness: perturbed initial conditions, seeds", ROBUSTNESS_SEEDS, "===")
    print(
        f"{'system':<14}{'seed':>6}{'dim':>5}{'lag':>5}"
        f"{'theory':>10}{'naive':>10}{'relerr_th':>11}{'relerr_nv':>11}{'closer':>8}"
    )
    all_rows = []
    for name, gen, dt, lit in SYSTEMS:
        for seed in ROBUSTNESS_SEEDS:
            r = rows[[s[0] for s in SYSTEMS].index(name)] if seed == SEED else run_one(
                name, gen, dt, lit, seed
            )
            all_rows.append(r)
            print(
                f"{r['name']:<14}{r['seed']:>6}{r['dim']:>5}{r['lag']:>5}"
                f"{r['lam_theory']:>10.4f}{r['lam_naive']:>10.4f}"
                f"{r['rel_theory']:>11.3f}{r['rel_naive']:>11.3f}"
                f"{'yes' if r['rel_theory'] < r['rel_naive'] else 'no':>8}"
            )
    n_runs = len(all_rows)
    n_closer_all = sum(r["rel_theory"] < r["rel_naive"] for r in all_rows)
    n_degraded_all = sum(
        r["rel_theory"] > r["rel_naive"] + MAX_DEGRADATION for r in all_rows
    )
    worst = max(r["rel_theory"] - r["rel_naive"] for r in all_rows)
    print(
        f"\nAcross {n_runs} runs: closer {n_closer_all}/{n_runs}, "
        f"degraded > {MAX_DEGRADATION:.0%}: {n_degraded_all}, "
        f"worst rel-err increase: {worst:+.3f}"
    )
    print(f"Total runtime: {time.time() - t0:.1f}s")
    return rows, all_rows, integrate


if __name__ == "__main__":
    main()
