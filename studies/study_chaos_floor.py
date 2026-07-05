"""Chaos-floor experiment: touch the physical predictability limit and PROVE it.

Proof device (no hypotheses): the TWIN EXPERIMENT. For each test window, the
true Lorenz dynamics is integrated from the window's exact full state with a
perturbation of size equal to the forecaster's own one-step error e0_w. The
twin's horizon H_twin (same tolerance, same K-consecutive rule, measured on
the same observable) is the *measured physical floor* at that error level: no
forecaster can systematically beat the true dynamics at equal initial error.

  floor touched  <=>  paired ratio rho_w = H_model_w / H_twin_w -> 1

Pre-registered verdicts (tau = 0.4 std, K = 2, Lorenz dt = 0.01):
- FLOOR TOUCHED:  median rho >= 0.8
- NEAR FLOOR:     0.5 <= median rho < 0.8
- MODEL-LIMITED:  median rho < 0.5
Also reported: R = Lambda_eff/lambda_1 (should approach ~1-3 at the floor,
it embeds the FTLE fluctuations), censoring rates, and the transition curve
rho(model) across {linear, NG-RC} (MLP point from study_universality).

Forecaster candidate: NG-RC (next-generation reservoir computing, Gauthier
et al. 2021) = ridge regression on polynomial monomials of delay coordinates,
observable-only (x component). Model selection on validation rollouts.

Run:  python studies/study_chaos_floor.py --arm ngrc   (or linear / mlp)
Results are appended to outputs/chaos_floor.csv (resumable per arm).
"""

import argparse
import csv
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
DT = 0.01
LAMBDA1 = 0.906  # /u.t., literature; validated by tests/test_physics_chaos.py
TAU = 0.4
K_CONSEC = 2
HMAX = 1200  # 12 t.u. ~ 10.9 Lyapunov times
SERIES_LEN = 22000
WARMUP = 1000
SEED = 0
N_WINDOWS = 250
N_TWIN_DIRS = 3
CSV_PATH = os.path.join("outputs", "chaos_floor.csv")


def lorenz_rhs(t, s):
    x, y, z = s
    return [SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z]


def generate_full_state(n, warmup, dt, x0=(1.0, 1.0, 1.0)):
    total = n + warmup
    t_grid = np.linspace(0.0, (total - 1) * dt, total)
    sol = solve_ivp(lorenz_rhs, (0.0, t_grid[-1]), list(x0), t_eval=t_grid,
                    method="RK45", rtol=1e-9, atol=1e-9)
    return sol.y[:, warmup:].T  # (n, 3)


def horizon_from_errors(step_err, tau, k=K_CONSEC, hmax=None):
    """First 1-indexed h with k consecutive errors >= tau, else len (censored)."""
    n = len(step_err)
    run = 0
    for h in range(n):
        run = run + 1 if step_err[h] >= tau else 0
        if run >= k:
            return h + 1 - (k - 1), False
    return (hmax or n), True


# ---------------------------------------------------------------- NG-RC model
class NGRC:
    """Ridge regression on polynomial monomials of delay coordinates."""

    def __init__(self, delays, lag, degree, alpha):
        self.delays, self.lag, self.degree, self.alpha = delays, lag, degree, alpha
        self.w = None
        self.combos = None

    def _features(self, V):
        # V: (n, d) delay matrix -> (n, n_feat) [1, linear, deg-2, (deg-3)]
        n, d = V.shape
        if self.combos is None:
            combos = []
            for deg in range(2, self.degree + 1):
                combos.extend(itertools.combinations_with_replacement(range(d), deg))
            self.combos = combos
        cols = [np.ones((n, 1)), V]
        for c in self.combos:
            cols.append(np.prod(V[:, list(c)], axis=1, keepdims=True))
        return np.concatenate(cols, axis=1)

    def fit(self, series):
        d, L = self.delays, self.lag
        n = len(series) - (d - 1) * L - 1
        V = np.column_stack([series[i * L: i * L + n] for i in range(d)])
        y = series[(d - 1) * L + 1: (d - 1) * L + 1 + n]
        F = self._features(V)
        A = F.T @ F + self.alpha * np.eye(F.shape[1])
        self.w = np.linalg.solve(A, F.T @ y)
        return self

    def predict(self, x):
        F = self._features(np.asarray(x, dtype=np.float64).reshape(1, -1))
        return float(np.clip((F @ self.w)[0], -6.0, 6.0))


def rollout_errors(model, series, start, d, lag, hmax):
    """Autoregressive rollout from window at `start`; per-step abs errors."""
    window_len = (d - 1) * lag + 1
    hist = list(series[start: start + window_len])
    errs = np.empty(hmax)
    for h in range(hmax):
        x = [hist[i * lag] for i in range(d)]
        pred = model.predict(x)
        errs[h] = abs(pred - series[start + window_len + h])
        hist.append(pred)
        hist.pop(0)
    return errs


def median_valid_time(model, series, d, lag, n_probe=25, hmax=600, seed=0):
    rng = np.random.default_rng(seed)
    limit = len(series) - (d - 1) * lag - 1 - hmax
    if limit <= n_probe:
        return 0.0
    starts = rng.choice(limit, size=n_probe, replace=False)
    hs = []
    for s in starts:
        errs = rollout_errors(model, series, s, d, lag, hmax)
        h, _ = horizon_from_errors(errs, TAU, hmax=hmax)
        hs.append(h)
    return float(np.median(hs))


def select_ngrc(train, val):
    best, best_vt = None, -1.0
    trainval = np.concatenate([train, val])
    for d, L, deg, alpha in itertools.product((4, 6, 8), (1, 2), (2, 3), (1e-8, 1e-6, 1e-4)):
        try:
            m = NGRC(d, L, deg, alpha).fit(train)
            vt = median_valid_time(m, val, d, L)
        except np.linalg.LinAlgError:
            continue
        if vt > best_vt:
            best_vt, best = vt, (d, L, deg, alpha)
    d, L, deg, alpha = best
    final = NGRC(d, L, deg, alpha).fit(trainval)
    print(f"NG-RC selected: delays={d} lag={L} degree={deg} alpha={alpha} "
          f"(val median valid time {best_vt:.0f} steps)", flush=True)
    return final, d, L


def fit_linear(train, val):
    from src.horizon_models import LinearAR
    best, best_vt = None, -1.0
    for d in (4, 6, 8):
        m = LinearAR(reg=1e-4).fit(*_supervised(train, d, 1))
        vt = median_valid_time(m, val, d, 1)
        if vt > best_vt:
            best_vt, best = vt, (m, d)
    m, d = best
    m = LinearAR(reg=1e-4).fit(*_supervised(np.concatenate([train, val]), d, 1))
    print(f"linear selected: dim={d} (val median valid time {best_vt:.0f})", flush=True)
    return m, d, 1


def fit_mlp(train, val):
    import torch
    from src.horizon_models import TorchWrapper
    from src.horizon_training import train_mlp
    x_t, y_t = _supervised(train, 6, 1)
    x_v, y_v = _supervised(val, 6, 1)
    net, _ = train_mlp(x_t, y_t, x_v, y_v, input_dim=6, hidden_dim=128,
                       epochs=120, lr=1e-3, batch_size=64, patience=20,
                       device=torch.device("cpu"), show_progress=False)
    return TorchWrapper(net, torch.device("cpu")), 6, 1


def _supervised(series, d, lag):
    n = len(series) - (d - 1) * lag - 1
    x = np.column_stack([series[i * lag: i * lag + n] for i in range(d)])
    y = series[(d - 1) * lag + 1: (d - 1) * lag + 1 + n]
    return x, y


def twin_horizon(states_raw, x_mean, x_std, t0, eps_std, rng, hmax):
    """Physical floor at this state and error level: perturb the TRUE dynamics."""
    hs = []
    horizon_budget = min(hmax, len(states_raw) - t0 - 1)
    t_grid = np.arange(1, horizon_budget + 1) * DT
    for _ in range(N_TWIN_DIRS):
        delta = rng.normal(size=3)
        delta *= (eps_std * x_std) / np.linalg.norm(delta)
        sol = solve_ivp(lorenz_rhs, (0.0, t_grid[-1]), states_raw[t0] + delta,
                        t_eval=t_grid, method="RK45", rtol=1e-9, atol=1e-9)
        x_pert = sol.y[0]
        x_true = states_raw[t0 + 1: t0 + 1 + horizon_budget, 0]
        step_err = np.abs(x_pert - x_true) / x_std
        h, _ = horizon_from_errors(step_err, TAU, hmax=horizon_budget)
        hs.append(h)
    return float(np.median(hs))


class NGRCFull:
    """Full-state NG-RC: ridge on polynomial monomials of stacked states.

    Lorenz's vector field is quadratic, so polynomial features of the full
    state can capture the one-step flow map AND its Jacobian: error growth
    should then approach the true local Lyapunov rate (R -> 1)."""

    def __init__(self, delays, degree, alpha):
        self.delays, self.degree, self.alpha = delays, degree, alpha
        self.W = None
        self.combos = None

    def _features(self, S):
        n, d = S.shape
        if self.combos is None:
            combos = []
            for deg in range(2, self.degree + 1):
                combos.extend(itertools.combinations_with_replacement(range(d), deg))
            self.combos = combos
        cols = [np.ones((n, 1)), S]
        for c in self.combos:
            cols.append(np.prod(S[:, list(c)], axis=1, keepdims=True))
        return np.concatenate(cols, axis=1)

    def fit(self, states_std):
        k = self.delays
        n = len(states_std) - k
        S = np.concatenate([states_std[i: i + n] for i in range(k)], axis=1)
        Y = states_std[k: k + n]
        F = self._features(S)
        A = F.T @ F + self.alpha * np.eye(F.shape[1])
        self.W = np.linalg.solve(A, F.T @ Y)
        return self

    def step(self, hist):
        # hist: list of last `delays` standardized states (each shape (3,))
        S = np.concatenate(hist).reshape(1, -1)
        out = (self._features(S) @ self.W)[0]
        return np.clip(out, -8.0, 8.0)


def rollout_errors_full(model, states_std, x_series_std, start, hmax):
    """Full-state rollout; per-step abs error on the standardized x observable."""
    k = model.delays
    hist = [states_std[start + i].copy() for i in range(k)]
    errs = np.empty(hmax)
    for h in range(hmax):
        nxt = model.step(hist)
        errs[h] = abs(nxt[0] - states_std[start + k + h][0]) * _XRATIO[0]
        hist.append(nxt)
        hist.pop(0)
    return errs


_XRATIO = [1.0]  # states-std x-sigma / observable x-sigma (set in main)


def lorenz_rhs_vec(S):
    # S: (m, 3) -> (m, 3)
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    return np.column_stack([SIGMA * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def rk4_step_vec(S, h):
    k1 = lorenz_rhs_vec(S)
    k2 = lorenz_rhs_vec(S + 0.5 * h * k1)
    k3 = lorenz_rhs_vec(S + 0.5 * h * k2)
    k4 = lorenz_rhs_vec(S + h * k3)
    return S + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def twin_horizons_vec(states_raw, x_std, t0s, eps_std, mode, hmax, rng, substeps=4):
    """Vectorized dual-floor twins. Both pair members use the SAME fixed-step
    RK4 integrator, so integrator error is common-mode; their mutual distance
    follows the true variational dynamics.

    mode='oneshot': clean vs (clean + one perturbation of size eps at t0)
                    -> the classical physical floor at initial error eps.
    mode='inject':  clean vs (clean + fresh isotropic noise of size eps after
                    every dt step) -> the floor for STEP-WISE forecasters,
                    whose per-step residual is eps (theory:
                    H ~ ln(tau*lambda*dt/eps)/(lambda*dt) steps).
    Returns per-window horizons (median over N_TWIN_DIRS directions).
    """
    m = len(t0s)
    reps = N_TWIN_DIRS
    h = DT / substeps
    eps_raw = np.repeat(np.asarray(eps_std) * x_std, reps)
    A = np.repeat(states_raw[t0s], reps, axis=0).astype(np.float64)  # clean
    B = A.copy()
    if mode == "oneshot":
        delta = rng.normal(size=(m * reps, 3))
        delta *= (eps_raw / np.linalg.norm(delta, axis=1))[:, None]
        B = B + delta
    hor = np.full(m * reps, hmax, dtype=np.int64)
    run = np.zeros(m * reps, dtype=np.int64)
    done = np.zeros(m * reps, dtype=bool)
    for step in range(1, hmax + 1):
        for _ in range(substeps):
            A = rk4_step_vec(A, h)
            B = rk4_step_vec(B, h)
        if mode == "inject":
            noise = rng.normal(size=(m * reps, 3))
            noise *= (eps_raw / np.linalg.norm(noise, axis=1))[:, None]
            B = B + noise
        err = np.abs(A[:, 0] - B[:, 0]) / x_std
        exceed = err >= TAU
        run = np.where(exceed, run + 1, 0)
        newly = (~done) & (run >= K_CONSEC)
        hor[newly] = step - (K_CONSEC - 1)
        done |= newly
        if done.all():
            break
    return np.median(hor.reshape(m, reps), axis=1).astype(float)


def select_ngrc_full(states_std, x_series_std, i_train, i_val):
    train_states = states_std[:i_train]
    best, best_vt = None, -1.0
    rng = np.random.default_rng(123)
    for k, deg, alpha in itertools.product((1,), (4, 5, 6), (1e-8, 1e-10, 1e-12)):
        try:
            m = NGRCFull(k, deg, alpha).fit(train_states)
        except np.linalg.LinAlgError:
            continue
        probe_hmax = 2500  # must exceed the best candidates' valid time,
        # otherwise all good configs saturate and selection is arbitrary
        limit = i_val - i_train - k - probe_hmax
        if limit <= 30:
            continue
        starts = i_train + rng.choice(limit, size=20, replace=False)
        hs = []
        for s in starts:
            errs = rollout_errors_full(m, states_std, x_series_std, s, probe_hmax)
            h, _ = horizon_from_errors(errs, TAU, hmax=probe_hmax)
            hs.append(h)
        vt = float(np.median(hs))
        if vt > best_vt:
            best_vt, best = vt, (k, deg, alpha)
    k, deg, alpha = best
    final = NGRCFull(k, deg, alpha).fit(states_std[:i_val])
    print(f"NG-RC-full selected: delays={k} degree={deg} alpha={alpha} "
          f"(val median valid time {best_vt:.0f} steps)", flush=True)
    return final, k


def main():
    global SEED, SERIES_LEN, HMAX
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["ngrc", "linear", "mlp", "ngrc_full"], default="ngrc")
    ap.add_argument("--seed", type=int, default=0)
    opts = ap.parse_args()
    arm = opts.arm
    SEED = opts.seed
    if arm == "ngrc_full":
        SERIES_LEN, HMAX = 40000, 3000

    rng = np.random.default_rng(SEED)
    t_start = time.time()
    ic = (1.0 + 0.05 * SEED, 1.0, 1.0)  # seed shifts the initial condition
    states = generate_full_state(SERIES_LEN, WARMUP, DT, x0=ic)
    x_raw = states[:, 0]
    n = len(x_raw)
    i_train, i_val, i_test = int(0.60 * n), int(0.75 * n), int(0.90 * n)
    x_mean, x_std = x_raw[:i_train].mean(), x_raw[:i_train].std()
    series = (x_raw - x_mean) / x_std
    train, val = series[:i_train], series[i_train:i_val]
    test_offset = i_test
    test = series[test_offset:]

    rows = []
    if arm == "ngrc_full":
        state_mean = states[:i_train].mean(axis=0)
        state_sd = states[:i_train].std(axis=0)
        states_std = (states - state_mean) / state_sd
        _XRATIO[0] = float(state_sd[0] / x_std)  # = 1.0 (same train stats)
        model, k = select_ngrc_full(states_std, series, i_train, i_val)
        d, lag = 3 * k, 1  # reporting only
        limit = n - test_offset - k - HMAX
        if limit <= N_WINDOWS:
            sys.exit(f"test split too short: limit={limit}")
        starts = np.sort(rng.choice(limit, size=N_WINDOWS, replace=False)) + test_offset
        e0_l, hm_l, cm_l, t0_l = [], [], [], []
        for s in starts:
            errs = rollout_errors_full(model, states_std, series, s, HMAX)
            e0_l.append(max(float(errs[0]), 1e-12))
            h_model, censored_m = horizon_from_errors(errs, TAU, hmax=HMAX)
            hm_l.append(h_model)
            cm_l.append(int(censored_m))
            t0_l.append(s + k - 1)  # index of the current (last known) state
        t0_arr, e0_arr = np.array(t0_l), np.array(e0_l)
        ht_one = twin_horizons_vec(states, x_std, t0_arr, e0_arr, "oneshot", HMAX, rng)
        ht_inj = twin_horizons_vec(states, x_std, t0_arr, e0_arr, "inject", HMAX, rng)
        rows = list(zip(starts, e0_l, hm_l, cm_l, ht_one))
        # --- injection floor: the fair bound for STEP-WISE forecasters ---
        hm_a = np.array(hm_l, dtype=float)
        rho_inj = hm_a / np.maximum(ht_inj, 1.0)
        arg = TAU * LAMBDA1 * DT / e0_arr
        h_inj_theory = np.where(arg > 1.0, np.log(arg) / (LAMBDA1 * DT), 1.0)
        print(f"\n--- Injection floor (step-wise fair bound) ---")
        print(f"H_inj measured med = {np.median(ht_inj):.0f} steps "
              f"({np.median(ht_inj) * DT * LAMBDA1:.2f} T_lyap); "
              f"theory ln(tau*lambda*dt/e0)/(lambda*dt) med = {np.median(h_inj_theory):.0f} steps "
              f"(ratio meas/theory {np.median(ht_inj / h_inj_theory):.3f})")
        print(f"rho_inj = H_model/H_inj: median {np.median(rho_inj):.3f} "
              f"IQR [{np.quantile(rho_inj, 0.25):.3f}, {np.quantile(rho_inj, 0.75):.3f}]")
        inj_verdict = ("INJECTION FLOOR TOUCHED" if np.median(rho_inj) >= 0.8
                       else "NEAR" if np.median(rho_inj) >= 0.5 else "MODEL-LIMITED")
        print(f"VERDICT vs injection floor: {inj_verdict}", flush=True)
    else:
        if arm == "ngrc":
            model, d, lag = select_ngrc(train, val)
        elif arm == "linear":
            model, d, lag = fit_linear(train, val)
        else:
            model, d, lag = fit_mlp(train, val)

        window_len = (d - 1) * lag + 1
        limit = len(test) - window_len - HMAX
        if limit <= N_WINDOWS:
            sys.exit(f"test split too short: limit={limit}")
        starts = np.sort(rng.choice(limit, size=N_WINDOWS, replace=False))

        for s in starts:
            errs = rollout_errors(model, test, s, d, lag, HMAX)
            e0 = max(float(errs[0]), 1e-9)
            h_model, censored_m = horizon_from_errors(errs, TAU, hmax=HMAX)
            t0_global = test_offset + s + window_len - 1  # index of current state
            h_twin = twin_horizon(states, x_mean, x_std, t0_global, e0, rng, HMAX)
            rows.append((s, e0, h_model, int(censored_m), h_twin))

    e0s = np.array([r[1] for r in rows])
    hm = np.array([r[2] for r in rows], dtype=float)
    cm = np.array([r[3] for r in rows])
    ht = np.array([r[4] for r in rows], dtype=float)
    valid = (e0s < TAU / 4.0)
    rho = hm[valid] / np.maximum(ht[valid], 1.0)
    lam_eff = np.log(TAU / e0s[valid]) / (hm[valid] * DT)
    r_diag = lam_eff / LAMBDA1
    lam_eff_twin = np.log(TAU / e0s[valid]) / (ht[valid] * DT)
    r_twin = lam_eff_twin / LAMBDA1

    med = lambda a: float(np.median(a)) if a.size else float("nan")
    verdict = ("FLOOR TOUCHED" if med(rho) >= 0.8
               else "NEAR FLOOR" if med(rho) >= 0.5 else "MODEL-LIMITED")
    print(f"\narm={arm} dim={d} lag={lag} n={int(valid.sum())} "
          f"censored_model={int(cm.sum())}", flush=True)
    print(f"e0 median = {med(e0s[valid]):.2e} std units")
    print(f"H_model med = {med(hm[valid]):.0f} steps = {med(hm[valid]) * DT * LAMBDA1:.2f} T_lyap")
    print(f"H_twin  med = {med(ht[valid]):.0f} steps = {med(ht[valid]) * DT * LAMBDA1:.2f} T_lyap")
    print(f"paired rho = H_model/H_twin: median {med(rho):.3f} "
          f"IQR [{np.quantile(rho, 0.25):.3f}, {np.quantile(rho, 0.75):.3f}]")
    print(f"R_model = Lambda_eff/lambda1: median {med(r_diag):.2f}")
    print(f"R_twin  (floor's own R):      median {med(r_twin):.2f}")
    print(f"VERDICT: {verdict}  ({time.time() - t_start:.0f}s)")

    os.makedirs("outputs", exist_ok=True)
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["arm", "dim", "lag", "n", "censored", "e0_med",
                        "H_model_med", "H_twin_med", "rho_med", "rho_q25",
                        "rho_q75", "R_model_med", "R_twin_med", "verdict"])
        w.writerow([arm, d, lag, int(valid.sum()), int(cm.sum()),
                    f"{med(e0s[valid]):.3e}", med(hm[valid]), med(ht[valid]),
                    f"{med(rho):.4f}", f"{np.quantile(rho, 0.25):.4f}",
                    f"{np.quantile(rho, 0.75):.4f}", f"{med(r_diag):.3f}",
                    f"{med(r_twin):.3f}", verdict])


if __name__ == "__main__":
    main()
