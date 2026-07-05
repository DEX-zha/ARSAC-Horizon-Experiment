"""Noise-aware floor: validate the noise estimator and the reachable-floor law.

Extends the chaos-floor result (docs/theory/chaos_floor.md) to NOISY data —
the bridge that makes the R diagnostic honest on real-world series where
twins are impossible. Three pre-registered claims, tested on Lorenz with
KNOWN synthetic observation noise sigma in {1e-3, 1e-2, 3e-2} (std units):

C1 (estimator): sigma_hat/sigma in [0.5, 2.0] for every level.
C2 (law): the MEASURED noisy floor (one-shot twin at eps=sigma) matches
    H = ln(tau/sigma)/(lambda_1*dt) within [0.8, 1.3] — the validated
    one-shot law transported to the noise scale.
C3 (bound + saturation): an NG-RC trained ON the noisy data satisfies
    H_model <= floor (the bound is real), and approaches it
    (rho_noisy = H_model/H_floor, reported; >= 0.6 NEAR / >= 0.8 TOUCHED).

Run: python studies/study_noisy_floor.py   (~5-8 min CPU)
Results appended to outputs/noisy_floor.csv.
"""

import csv
import itertools
import os
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from study_chaos_floor import (  # noqa: E402
    DT,
    LAMBDA1,
    TAU,
    NGRCFull,
    generate_full_state,
    horizon_from_errors,
    rollout_errors_full,
    twin_horizons_vec,
)
import study_chaos_floor as scf  # noqa: E402

from src.horizon_noise import estimate_observation_noise  # noqa: E402

SIGMAS = [1e-3, 1e-2, 3e-2]  # observation noise, units of x std
SERIES_LEN = 30000
WARMUP = 1000
HMAX = 1200
N_WINDOWS = 150
SEED = 0
CSV_PATH = os.path.join("outputs", "noisy_floor.csv")


def select_ngrc_noisy(states_std, i_train, i_val, seed):
    """Grid selection on noisy validation rollouts (ridge must filter noise)."""
    rng = np.random.default_rng(seed)
    best, best_vt = None, -1.0
    probe_hmax = 1000
    for deg, alpha in itertools.product((3, 4, 5, 6), (1e-8, 1e-6, 1e-4, 1e-2)):
        try:
            m = NGRCFull(1, deg, alpha).fit(states_std[:i_train])
        except np.linalg.LinAlgError:
            continue
        limit = i_val - i_train - 1 - probe_hmax
        if limit <= 25:
            continue
        starts = i_train + rng.choice(limit, size=15, replace=False)
        hs = []
        for s in starts:
            errs = rollout_errors_full(m, states_std, None, s, probe_hmax)
            h, _ = horizon_from_errors(errs, TAU, hmax=probe_hmax)
            hs.append(h)
        vt = float(np.median(hs))
        if vt > best_vt:
            best_vt, best = vt, (deg, alpha)
    deg, alpha = best
    final = NGRCFull(1, deg, alpha).fit(states_std[:i_val])
    print(f"  NG-RC(noisy) selected: degree={deg} alpha={alpha} "
          f"(val valid time {best_vt:.0f})", flush=True)
    return final


def main():
    t0 = time.time()
    states_clean = generate_full_state(SERIES_LEN, WARMUP, DT)
    n = len(states_clean)
    i_train, i_val, i_test = int(0.6 * n), int(0.75 * n), int(0.9 * n)
    rows = []
    for sigma in SIGMAS:
        rng = np.random.default_rng(SEED + int(sigma * 1e6))
        x_std_raw = states_clean[:i_train, 0].std()
        comp_sd_clean = states_clean[:i_train].std(axis=0)
        noisy = states_clean + rng.normal(size=states_clean.shape) * (sigma * comp_sd_clean)

        # Standardize the NOISY data (that's all a practitioner has).
        mean = noisy[:i_train].mean(axis=0)
        sd = noisy[:i_train].std(axis=0)
        noisy_std = (noisy - mean) / sd
        x_series = noisy_std[:, 0]  # observable, sigma in ~std units

        # --- C1: noise estimator on the noisy observable ---
        sig_hat, _ = estimate_observation_noise(x_series[:i_val], dim=6, lag=1, seed=0)
        c1 = 0.5 <= sig_hat / sigma <= 2.0
        print(f"sigma={sigma:.0e}: sigma_hat={sig_hat:.2e} "
              f"(ratio {sig_hat / sigma:.2f}) C1={'PASS' if c1 else 'FAIL'}", flush=True)

        # --- C2: measured noisy floor vs law ---
        scf.SEED = SEED
        limit = n - i_test - 1 - HMAX
        starts = np.sort(np.random.default_rng(1).choice(limit, size=N_WINDOWS, replace=False)) + i_test
        t0s = starts  # current state index = start (delays=1)
        eps = np.full(N_WINDOWS, sigma, dtype=np.float64)
        floor_h = twin_horizons_vec(states_clean, x_std_raw, t0s, eps, "oneshot",
                                    HMAX, np.random.default_rng(2))
        law_h = np.log(TAU / sigma) / (LAMBDA1 * DT)
        ratio_law = float(np.median(floor_h)) / law_h
        c2 = 0.8 <= ratio_law <= 1.3
        print(f"  floor measured med={np.median(floor_h):.0f} steps, law={law_h:.0f} "
              f"(ratio {ratio_law:.2f}) C2={'PASS' if c2 else 'FAIL'}", flush=True)

        # --- C3: NG-RC trained on noisy data vs that floor ---
        model = select_ngrc_noisy(noisy_std, i_train, i_val, seed=3)
        hm = []
        for s in starts:
            errs = rollout_errors_full(model, noisy_std, None, s, HMAX)
            h, _ = horizon_from_errors(errs, TAU, hmax=HMAX)
            hm.append(h)
        hm = np.array(hm, dtype=float)
        rho = hm / np.maximum(floor_h, 1.0)
        rho_med = float(np.median(rho))
        bound_ok = rho_med <= 1.1  # the floor must actually bound the model
        sat = ("TOUCHED" if rho_med >= 0.8 else "NEAR" if rho_med >= 0.6 else "BELOW")
        print(f"  H_model med={np.median(hm):.0f} rho_noisy={rho_med:.3f} "
              f"IQR [{np.quantile(rho, 0.25):.3f}, {np.quantile(rho, 0.75):.3f}] "
              f"bound={'OK' if bound_ok else 'VIOLATED'} saturation={sat}", flush=True)

        rows.append([sigma, f"{sig_hat:.4e}", f"{sig_hat / sigma:.3f}", int(c1),
                     f"{np.median(floor_h):.0f}", f"{law_h:.0f}",
                     f"{ratio_law:.3f}", int(c2), f"{np.median(hm):.0f}",
                     f"{rho_med:.4f}", int(bound_ok), sat])

    os.makedirs("outputs", exist_ok=True)
    header = ["sigma", "sigma_hat", "sigma_ratio", "C1_pass", "floor_med_steps",
              "law_steps", "law_ratio", "C2_pass", "H_model_med", "rho_noisy",
              "bound_ok", "saturation"]
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)
    print(f"\nDone in {time.time() - t0:.0f}s -> {CSV_PATH}")


if __name__ == "__main__":
    main()
