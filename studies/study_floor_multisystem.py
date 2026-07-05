"""Multi-system replication of the chaos-floor result (the paper's key claim).

Same protocol as studies/study_chaos_floor.py (paired twins at the model's own
one-step error, pre-registered rho >= 0.8 threshold, positive control on the
twin's R), transported to:

- ROSSLER (a=0.2, b=0.2, c=5.7, dt=0.05, lambda_1 ~ 0.071): quadratic vector
  field like Lorenz -> polynomial NG-RC is expected to reach the floor.
- MACKEY-GLASS (tau=17, dt=1.0, lambda_1 ~ 0.006): DELAY system with a
  NON-polynomial (Hill) nonlinearity -> the genuine generality test. The
  'full state' is the sampled delay segment [x(t), ..., x(t-19)] (m=20 covers
  the tau=17 history); twins integrate the true DDE with the WHOLE history
  segment perturbed by iid noise of size e0 (documented convention).

Run:  python studies/study_floor_multisystem.py --system rossler
      python studies/study_floor_multisystem.py --system mackey_glass
Appends to outputs/chaos_floor_multisystem.csv.
"""

import argparse
import csv
import itertools
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.dirname(__file__))

from study_chaos_floor import NGRCFull, horizon_from_errors  # noqa: E402

TAU = 0.4
K_CONSEC = 2
N_TWIN_DIRS = 3
CSV_PATH = os.path.join("outputs", "chaos_floor_multisystem.csv")

CFG = {
    "rossler": dict(dt=0.05, lam=0.071, series_len=90000, warmup=1000,
                    hmax=7500, n_windows=100, degrees=(5,),
                    alphas=(1e-8,), probe=4500),
    "mackey_glass": dict(dt=1.0, lam=0.006, series_len=25000, warmup=500,
                         hmax=2000, n_windows=120, degrees=(2, 3),
                         alphas=(1e-6, 1e-4, 1e-2), probe=1800, m_state=20),
}


# ------------------------------------------------------------------ Rossler
def rossler_rhs_vec(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    return np.column_stack([-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)])


def generate_rossler_states(n, warmup, dt):
    def rhs(t, s):
        x, y, z = s
        return [-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)]
    total = n + warmup
    t_grid = np.linspace(0.0, (total - 1) * dt, total)
    sol = solve_ivp(rhs, (0.0, t_grid[-1]), [1.0, 0.0, 0.0], t_eval=t_grid,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y[:, warmup:].T


def ode_twin_horizons(rhs_vec, states_raw, x_std, t0s, eps_std, hmax, dt, rng,
                      substeps=4, mode="oneshot"):
    """One-shot paired twins for an ODE system (mutual distance, common-mode
    integrator error), vectorized across windows x directions."""
    m = len(t0s)
    h = dt / substeps
    eps_raw = np.repeat(np.asarray(eps_std) * x_std, N_TWIN_DIRS)
    A = np.repeat(states_raw[t0s], N_TWIN_DIRS, axis=0).astype(np.float64)
    B = A.copy()
    if mode == "oneshot":
        delta = rng.normal(size=A.shape)
        delta *= (eps_raw / np.linalg.norm(delta, axis=1))[:, None]
        B = A + delta

    def rk4(S):
        k1 = rhs_vec(S)
        k2 = rhs_vec(S + 0.5 * h * k1)
        k3 = rhs_vec(S + 0.5 * h * k2)
        k4 = rhs_vec(S + h * k3)
        return S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    hor = np.full(len(A), hmax, dtype=np.int64)
    run = np.zeros(len(A), dtype=np.int64)
    done = np.zeros(len(A), dtype=bool)
    for step in range(1, hmax + 1):
        for _ in range(substeps):
            A = rk4(A)
            B = rk4(B)
        if mode == "inject":
            noise = rng.normal(size=B.shape)
            noise *= (eps_raw / np.linalg.norm(noise, axis=1))[:, None]
            B = B + noise
        err = np.abs(A[:, 0] - B[:, 0]) / x_std
        run = np.where(err >= TAU, run + 1, 0)
        newly = (~done) & (run >= K_CONSEC)
        hor[newly] = step - (K_CONSEC - 1)
        done |= newly
        if done.all():
            break
    return np.median(hor.reshape(m, N_TWIN_DIRS), axis=1).astype(float)


# -------------------------------------------------------------- Mackey-Glass
MG_BETA, MG_GAMMA, MG_N, MG_TAU = 0.2, 0.1, 10, 17.0


def generate_mg_series(n, warmup, dt, dt_int=0.1):
    from src.horizon_utils import generate_mackey_glass
    return generate_mackey_glass(n, tau=MG_TAU, beta=MG_BETA, gamma=MG_GAMMA,
                                 n=MG_N, dt=dt, warmup=warmup, integrator="rk4",
                                 dt_int=dt_int)


def mg_twin_horizons(series_raw, x_std, t0s, eps_std, hmax, dt, rng,
                     dt_int=0.1):
    """DDE paired twins: both members integrate the true Mackey-Glass DDE from
    the window's recorded history; the perturbed member's WHOLE history segment
    gets iid noise of size eps (the delay state is a function, not a point).
    Mutual head distance; common-mode integrator (method of steps, RK4,
    linear interpolation of the delayed value at half steps)."""
    substeps = max(1, int(round(dt / dt_int)))
    h = dt / substeps
    n_delay = max(1, int(round(MG_TAU / h)))
    m = len(t0s)
    total = m * N_TWIN_DIRS
    eps_raw = np.repeat(np.asarray(eps_std) * x_std, N_TWIN_DIRS)

    # History buffers: (total, n_delay + 1), index -1 = current head.
    hist_len = n_delay + 1
    H_A = np.empty((total, hist_len))
    for i, t0 in enumerate(np.repeat(t0s, N_TWIN_DIRS)):
        # reconstruct internal-step history by linear interp of the sampled series
        samples_needed = int(np.ceil(hist_len * h / dt)) + 2
        seg = series_raw[t0 - samples_needed + 1: t0 + 1]
        t_seg = np.arange(seg.size) * dt
        t_fine = t_seg[-1] - np.arange(hist_len - 1, -1, -1) * h
        H_A[i] = np.interp(t_fine, t_seg, seg)
    H_B = H_A + rng.normal(size=H_A.shape) * eps_raw[:, None]

    def mg_rhs(x_val, x_del):
        return MG_BETA * x_del / (1.0 + x_del ** MG_N) - MG_GAMMA * x_val

    def step_buffer(H):
        x = H[:, -1]
        xd0 = H[:, 0]
        xd1 = H[:, 1]
        xdh = 0.5 * (xd0 + xd1)
        k1 = mg_rhs(x, xd0)
        k2 = mg_rhs(x + 0.5 * h * k1, xdh)
        k3 = mg_rhs(x + 0.5 * h * k2, xdh)
        k4 = mg_rhs(x + h * k3, xd1)
        new = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        H[:, :-1] = H[:, 1:]
        H[:, -1] = new
        return H

    hor = np.full(total, hmax, dtype=np.int64)
    run = np.zeros(total, dtype=np.int64)
    done = np.zeros(total, dtype=bool)
    for step in range(1, hmax + 1):
        for _ in range(substeps):
            H_A = step_buffer(H_A)
            H_B = step_buffer(H_B)
        err = np.abs(H_A[:, -1] - H_B[:, -1]) / x_std
        run = np.where(err >= TAU, run + 1, 0)
        newly = (~done) & (run >= K_CONSEC)
        hor[newly] = step - (K_CONSEC - 1)
        done |= newly
        if done.all():
            break
    return np.median(hor.reshape(m, N_TWIN_DIRS), axis=1).astype(float)


# ------------------------------------------------------------------ Common
def build_state_matrix(series, m_state):
    """Delay-state matrix with x(t) LAST (component -1 = head).

    For the model we use component 0 = head convention of NGRCFull error
    measurement, so we flip: column 0 = x(t), then x(t-1), ..."""
    n = len(series) - m_state + 1
    cols = [series[m_state - 1 - j: m_state - 1 - j + n] for j in range(m_state)]
    return np.column_stack(cols)  # col 0 = x(t), col j = x(t-j)


def rollout_state_model(model, states_std, start, hmax):
    k = model.delays  # = 1
    hist = [states_std[start + i].copy() for i in range(k)]
    errs = np.empty(hmax)
    for hh in range(hmax):
        nxt = model.step(hist)
        errs[hh] = abs(nxt[0] - states_std[start + k + hh][0])
        hist.append(nxt)
        hist.pop(0)
    return errs


def select_model(states_std, i_train, i_val, degrees, alphas, probe, seed=3):
    rng = np.random.default_rng(seed)
    best, best_vt = None, -1.0
    for deg, alpha in itertools.product(degrees, alphas):
        try:
            mdl = NGRCFull(1, deg, alpha).fit(states_std[:i_train])
        except np.linalg.LinAlgError:
            continue
        limit = i_val - i_train - 1 - probe
        if limit <= 20:
            continue
        starts = i_train + rng.choice(limit, size=12, replace=False)
        hs = []
        for s in starts:
            errs = rollout_state_model(mdl, states_std, s, probe)
            hh, _ = horizon_from_errors(errs, TAU, hmax=probe)
            hs.append(hh)
        vt = float(np.median(hs))
        if vt > best_vt:
            best_vt, best = vt, (deg, alpha)
    deg, alpha = best
    final = NGRCFull(1, deg, alpha).fit(states_std[:i_val])
    print(f"  selected: degree={deg} alpha={alpha} (val valid time {best_vt:.0f})",
          flush=True)
    return final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["rossler", "mackey_glass"], required=True)
    system = ap.parse_args().system
    cfg = CFG[system]
    t_start = time.time()
    rng = np.random.default_rng(0)

    if system == "rossler":
        states_raw = generate_rossler_states(cfg["series_len"], cfg["warmup"], cfg["dt"])
        x_raw = states_raw[:, 0]
    else:
        x_raw = generate_mg_series(cfg["series_len"], cfg["warmup"], cfg["dt"])
        states_raw = build_state_matrix(x_raw, cfg["m_state"])  # (n', m)

    n = len(states_raw)
    i_train, i_val, i_test = int(0.6 * n), int(0.75 * n), int(0.9 * n)
    mean = states_raw[:i_train].mean(axis=0)
    sd = states_raw[:i_train].std(axis=0)
    states_std = (states_raw - mean) / sd
    x_std = float(x_raw[: i_train].std()) if system == "rossler" else float(sd[0] * 1.0)
    if system == "rossler":
        x_std = float(states_raw[:i_train, 0].std())

    print(f"{system}: n={n}, training NG-RC...", flush=True)
    model = select_model(states_std, i_train, i_val, cfg["degrees"],
                         cfg["alphas"], cfg["probe"])

    hmax, nw = cfg["hmax"], cfg["n_windows"]
    limit = n - i_test - 1 - hmax
    if limit <= nw:
        sys.exit(f"test split too short: {limit}")
    starts = np.sort(rng.choice(limit, size=nw, replace=False)) + i_test

    e0_l, hm_l = [], []
    for s in starts:
        errs = rollout_state_model(model, states_std, s, hmax)
        e0_l.append(max(float(errs[0]), 1e-12))
        hh, _ = horizon_from_errors(errs, TAU, hmax=hmax)
        hm_l.append(hh)
    e0 = np.array(e0_l)
    hm = np.array(hm_l, dtype=float)

    t0s = starts  # current state index (delays=1)
    ht_inj = None
    if system == "rossler":
        ht = ode_twin_horizons(rossler_rhs_vec, states_raw, x_std, t0s, e0,
                               hmax, cfg["dt"], np.random.default_rng(7))
        ht_inj = ode_twin_horizons(rossler_rhs_vec, states_raw, x_std, t0s, e0,
                                   hmax, cfg["dt"], np.random.default_rng(8),
                                   mode="inject")
    else:
        # map state index back to series index of the head x(t)
        head_idx = t0s + cfg["m_state"] - 1
        ht = mg_twin_horizons(x_raw, x_std, head_idx, e0, hmax, cfg["dt"],
                              np.random.default_rng(7))

    lam, dt = cfg["lam"], cfg["dt"]
    rho = hm / np.maximum(ht, 1.0)
    r_model = np.log(TAU / e0) / (hm * dt) / lam
    r_twin = np.log(TAU / e0) / (ht * dt) / lam
    med = lambda a: float(np.median(a))
    verdict = ("FLOOR TOUCHED" if med(rho) >= 0.8
               else "NEAR FLOOR" if med(rho) >= 0.5 else "MODEL-LIMITED")
    cens_m = int(np.sum(hm >= hmax))
    cens_t = int(np.sum(ht >= hmax))
    if cens_t > 0.1 * nw:
        print(f"WARNING: {cens_t}/{nw} twins censored at Hmax={hmax} -> "
              "H_twin is a LOWER bound, rho is INFLATED; increase hmax. "
              "Verdict must not be trusted.", flush=True)
    print(f"\nsystem={system} n_win={nw} censored(model/twin)={cens_m}/{cens_t}")
    print(f"e0 med = {med(e0):.2e} std | H_model med = {med(hm):.0f} steps"
          f" = {med(hm) * dt * lam:.2f} T_lyap | H_twin med = {med(ht):.0f}"
          f" = {med(ht) * dt * lam:.2f} T_lyap")
    print(f"rho median {med(rho):.3f} IQR [{np.quantile(rho, 0.25):.3f},"
          f" {np.quantile(rho, 0.75):.3f}] | R_model {med(r_model):.2f}"
          f" | R_twin {med(r_twin):.2f}")
    if ht_inj is not None:
        rho_inj = hm / np.maximum(ht_inj, 1.0)
        lam_dt = lam * dt
        h_inj_theory = np.log(TAU * lam_dt / e0) / lam_dt
        print(f"INJECTION floor: H_inj med={med(ht_inj):.0f} (theory "
              f"{med(h_inj_theory):.0f}, ratio {med(ht_inj / h_inj_theory):.2f}) | "
              f"rho_inj {med(rho_inj):.3f} IQR [{np.quantile(rho_inj, 0.25):.3f},"
              f" {np.quantile(rho_inj, 0.75):.3f}] -> "
              f"{'TOUCHED' if med(rho_inj) >= 0.8 else 'NEAR' if med(rho_inj) >= 0.5 else 'BELOW'}")
    print(f"VERDICT: {verdict}  ({time.time() - t_start:.0f}s)")

    os.makedirs("outputs", exist_ok=True)
    new = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["system", "n_windows", "e0_med", "H_model_med",
                        "H_twin_med", "rho_med", "rho_q25", "rho_q75",
                        "R_model", "R_twin", "censored_model", "censored_twin",
                        "verdict"])
        w.writerow([system, nw, f"{med(e0):.3e}", med(hm), med(ht),
                    f"{med(rho):.4f}", f"{np.quantile(rho, 0.25):.4f}",
                    f"{np.quantile(rho, 0.75):.4f}", f"{med(r_model):.3f}",
                    f"{med(r_twin):.3f}", cens_m, cens_t, verdict])


if __name__ == "__main__":
    main()
