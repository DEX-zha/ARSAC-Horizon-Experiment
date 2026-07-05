"""Study: conformal calibration under temporal dependence and drift.

Empirical validation for docs/theory/conformal_dependence.md (audit item E1).

Two seeded Monte-Carlo experiments on synthetic nonconformity scores drawn
from a stationary Gaussian AR(1) chain (phi = 0.9, unit marginal variance),
n_calib = 500, n_test = 500 (the test segment CONTINUES the calibration
chain, as in the real pipeline), 300 replications, alpha = 0.10.

(1) STATIONARY DEPENDENCE: compares
      - overlapping : classical conformal quantile on all 500 correlated
                      scores (what the current pipeline does),
      - disjoint    : thinning with gap=8 -> 63 near-decorrelated scores
                      (approximate exchangeability under mixing),
      - weighted    : Barber et al. decaying weights, half_life = n/4
                      (measured here for its stationary margin inflation),
      - oracle      : the TRUE marginal quantile Phi^-1(0.9) - its coverage
                      fluctuation is the irreducible test-side floor that no
                      calibration method can beat.
    plus sensitivity rows over the thinning gap.

(2) DRIFT: the score mean shifts by +0.5 sigma at the calibration midpoint
    (second half of calibration and the whole test segment are shifted).
    Compares standard (uniform) vs weighted conformal quantiles over a grid
    of half-lives, plus the Barber coverage-loss bound computed with the
    exact Gaussian d_TV of the mean shift.

Metrics per method: mean/std of per-replication empirical test coverage,
P(coverage < 0.88); and the calibration-conditional TRUE coverage
Phi(c - mu_test), which isolates the calibration effect from test-side noise.

Decision criteria (from the project directive) are evaluated at the
prescribed configurations (gap=8, half_life=n/4); tuned configurations are
confirmed on a SECOND independent seed to avoid tuning on the noise.

Runtime: ~1 s. All randomness is seeded.

Usage: python studies/study_conformal_dependence.py
"""

import os
import sys
from math import erf, sqrt

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_conformal_beyond import (
    coverage_gap_bound,
    decay_weights,
    disjoint_indices,
    weighted_conformal_quantile,
)

SEED = 1234
CONFIRM_SEED = 20260705
ALPHA = 0.10
N_CALIB = 500
N_TEST = 500
N_REPS = 300
PHI = 0.9
GAP = 8  # ceil(500 / 8) = 63 retained scores (~60 effective points)
HALF_LIFE = N_CALIB / 4.0  # prescribed half-life = 125
DRIFT = 0.5  # mean shift in units of the marginal std (sigma = 1)
FLOOR = 0.88  # target 0.90 minus 0.02


def norm_cdf(x):
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def ar1_chain(rng, n, phi):
    """Stationary Gaussian AR(1) with unit MARGINAL variance."""
    innov_sd = sqrt(1.0 - phi**2)
    x = np.empty(n, dtype=np.float64)
    x[0] = rng.normal(0.0, 1.0)
    eps = rng.normal(0.0, innov_sd, size=n - 1)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t - 1]
    return x


def run_scenario(seed, shift_calib_half=False, methods=()):
    """Runs N_REPS replications; returns per-method metric dicts.

    methods: list of (name, callable(calib_scores) -> margin c).
    If shift_calib_half, adds +DRIFT to the second half of calibration and to
    the whole test segment (drift scenario); test marginal mean is +DRIFT.
    """
    rng = np.random.default_rng(seed)
    mu_test = DRIFT if shift_calib_half else 0.0
    shift = np.zeros(N_CALIB + N_TEST)
    if shift_calib_half:
        shift[N_CALIB // 2:] = DRIFT
    cov = {name: [] for name, _ in methods}
    true_cov = {name: [] for name, _ in methods}
    margins = {name: [] for name, _ in methods}
    for _ in range(N_REPS):
        chain = ar1_chain(rng, N_CALIB + N_TEST, PHI) + shift
        calib, test = chain[:N_CALIB], chain[N_CALIB:]
        for name, fn in methods:
            c = fn(calib)
            cov[name].append(float(np.mean(test <= c)))
            true_cov[name].append(float(norm_cdf(c - mu_test)))
            margins[name].append(c)
    out = {}
    for name, _ in methods:
        cv = np.asarray(cov[name])
        tc = np.asarray(true_cov[name])
        mg = np.asarray(margins[name])
        out[name] = {
            "mean_cov": cv.mean(),
            "std_cov": cv.std(ddof=1),
            "p_below": np.mean(cv < FLOOR),
            "mean_true": tc.mean(),
            "p_true_below": np.mean(tc < FLOOR),
            "mean_margin": mg.mean(),
            "raw_cov": cv,  # kept for paired comparisons
        }
    return out


def paired_gain(res, name_new, name_ref):
    """Mean per-replication coverage gain and its paired standard error."""
    diff = res[name_new]["raw_cov"] - res[name_ref]["raw_cov"]
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(diff.size))


def print_table(title, results, order):
    print(f"\n== {title} ==")
    header = (
        f"{'method':<24}{'mean cov':>9}{'std cov':>9}{'P(<.88)':>9}"
        f"{'true cov':>10}{'P(true<.88)':>13}{'margin c':>10}"
    )
    print(header)
    print("-" * len(header))
    for name in order:
        r = results[name]
        print(
            f"{name:<24}{r['mean_cov']:>9.4f}{r['std_cov']:>9.4f}"
            f"{r['p_below']:>9.4f}{r['mean_true']:>10.4f}"
            f"{r['p_true_below']:>13.4f}{r['mean_margin']:>10.4f}"
        )


def std_quantile(calib):
    return weighted_conformal_quantile(calib, ALPHA)


def make_disjoint(gap):
    idx = disjoint_indices(N_CALIB, gap)
    return lambda calib: weighted_conformal_quantile(calib[idx], ALPHA)


def make_weighted(half_life):
    w = decay_weights(N_CALIB, half_life)
    return lambda calib: weighted_conformal_quantile(calib, ALPHA, w)


def oracle(_calib):
    # True marginal (1-alpha)-quantile of N(0,1) scores; in the drift
    # scenario the shifted mean cancels in (c - mu_test), handled by caller.
    return 1.2815515655446004  # Phi^-1(0.90)


def oracle_drift(_calib):
    return 1.2815515655446004 + DRIFT


def main():
    print(
        f"Study: conformal under dependence (seed={SEED}, alpha={ALPHA}, "
        f"phi={PHI}, n_calib={N_CALIB}, n_test={N_TEST}, reps={N_REPS})"
    )

    # ------------------------------------------------------------------ #
    # Scenario 1: stationary dependence                                    #
    # ------------------------------------------------------------------ #
    methods1 = [
        ("overlapping (n=500)", std_quantile),
        ("disjoint gap=8 (n=63)", make_disjoint(8)),
        ("weighted hl=n/4", make_weighted(N_CALIB / 4.0)),
        ("oracle Phi^-1(0.9)", oracle),
        # sensitivity on the gap
        ("disjoint gap=4 (n=125)", make_disjoint(4)),
        ("disjoint gap=16 (n=32)", make_disjoint(16)),
        ("disjoint gap=25 (n=20)", make_disjoint(25)),
        # sensitivity on the half-life (for stationary inflation)
        ("weighted hl=n/8", make_weighted(N_CALIB / 8.0)),
    ]
    res1 = run_scenario(SEED, shift_calib_half=False, methods=methods1)
    print_table(
        f"Scenario 1: stationary AR(1), alpha={ALPHA}",
        res1,
        [name for name, _ in methods1],
    )

    # ------------------------------------------------------------------ #
    # Scenario 2: drift                                                    #
    # ------------------------------------------------------------------ #
    methods2 = [
        ("standard (uniform)", std_quantile),
        ("weighted hl=n/4", make_weighted(N_CALIB / 4.0)),
        ("oracle (shifted)", oracle_drift),
        # sensitivity on the half-life
        ("weighted hl=n/2", make_weighted(N_CALIB / 2.0)),
        ("weighted hl=n/8", make_weighted(N_CALIB / 8.0)),
        ("weighted hl=n/16", make_weighted(N_CALIB / 16.0)),
    ]
    res2 = run_scenario(SEED + 1, shift_calib_half=True, methods=methods2)
    print_table(
        f"Scenario 2: +{DRIFT} sigma shift at calibration midpoint",
        res2,
        [name for name, _ in methods2],
    )

    # Barber coverage-loss bound with the exact Gaussian d_TV of the shift:
    # d_TV(N(0,1), N(delta,1)) = 2*Phi(delta/2) - 1.
    dtv_shift = 2.0 * float(norm_cdf(DRIFT / 2.0)) - 1.0
    dtv = np.concatenate(
        [np.full(N_CALIB // 2, dtv_shift), np.zeros(N_CALIB - N_CALIB // 2)]
    )
    print(
        f"\nBarber coverage-loss bound (exact d_TV of +{DRIFT} sigma shift = "
        f"{dtv_shift:.4f} on the pre-drift half):"
    )
    for label, w in (
        ("standard (uniform)", np.ones(N_CALIB)),
        ("weighted hl=n/4", decay_weights(N_CALIB, N_CALIB / 4.0)),
        ("weighted hl=n/8", decay_weights(N_CALIB, N_CALIB / 8.0)),
    ):
        loss = coverage_gap_bound(w, dtv)
        print(
            f"  {label:<20} guaranteed coverage >= {1 - ALPHA - loss:.4f} "
            f"(loss term {loss:.4f})"
        )

    # ------------------------------------------------------------------ #
    # Decision criteria (prescribed configurations)                       #
    # ------------------------------------------------------------------ #
    print("\n== Decision criteria (prescribed configs) ==")
    p_over = res1["overlapping (n=500)"]["p_below"]
    p_disj = res1["disjoint gap=8 (n=63)"]["p_below"]
    p_oracle = res1["oracle Phi^-1(0.9)"]["p_below"]
    ratio = p_over / p_disj if p_disj > 0 else np.inf
    print(
        f"[disjoint gap=8]  P(cov<0.88): overlapping={p_over:.4f}, "
        f"disjoint={p_disj:.4f} (x{ratio:.2f}; need >= x2), "
        f"ORACLE floor={p_oracle:.4f}"
        f" -> {'PASS' if p_disj <= p_over / 2 else 'FAIL'}"
    )
    print(
        "  note: the oracle floor is the test-side fluctuation no "
        "calibration can remove; criterion on calibration-conditional "
        "true coverage below."
    )
    pt_over = res1["overlapping (n=500)"]["p_true_below"]
    pt_disj = res1["disjoint gap=8 (n=63)"]["p_true_below"]
    ratio_t = pt_over / pt_disj if pt_disj > 0 else np.inf
    print(
        f"[disjoint gap=8, true-cov]  P(Phi(c)<0.88): "
        f"overlapping={pt_over:.4f}, disjoint={pt_disj:.4f} (x{ratio_t:.2f})"
        f" -> {'PASS' if pt_disj <= pt_over / 2 else 'FAIL'}"
    )

    gain, se = paired_gain(res2, "weighted hl=n/4", "standard (uniform)")
    m_std = res1["overlapping (n=500)"]["mean_margin"]
    m_w4 = res1["weighted hl=n/4"]["mean_margin"]
    infl4 = (m_w4 - m_std) / abs(m_std)
    ok4 = gain >= 0.03 and infl4 <= 0.10
    print(
        f"[weighted hl=n/4] drift gain={gain * 100:.2f} +/- {se * 100:.2f} pts "
        f"(paired SE; need >= 3), "
        f"stationary inflation={infl4 * 100:.2f}% (need <= 10%) "
        f"-> {'PASS' if ok4 else 'FAIL'}"
    )

    gain8, se8 = paired_gain(res2, "weighted hl=n/8", "standard (uniform)")
    m_w8 = res1["weighted hl=n/8"]["mean_margin"]
    infl8 = (m_w8 - m_std) / abs(m_std)
    ok8 = gain8 >= 0.03 and infl8 <= 0.10
    print(
        f"[weighted hl=n/8] drift gain={gain8 * 100:.2f} +/- {se8 * 100:.2f} pts, "
        f"stationary inflation={infl8 * 100:.2f}% "
        f"-> {'PASS' if ok8 else 'FAIL'} (tuned config)"
    )

    # ------------------------------------------------------------------ #
    # Confirmation of tuned configs on an independent seed                #
    # ------------------------------------------------------------------ #
    print(f"\n== Confirmation run (independent seed={CONFIRM_SEED}) ==")
    conf1 = run_scenario(
        CONFIRM_SEED,
        shift_calib_half=False,
        methods=[
            ("overlapping (n=500)", std_quantile),
            ("disjoint gap=8 (n=63)", make_disjoint(8)),
            ("weighted hl=n/8", make_weighted(N_CALIB / 8.0)),
        ],
    )
    conf2 = run_scenario(
        CONFIRM_SEED + 1,
        shift_calib_half=True,
        methods=[
            ("standard (uniform)", std_quantile),
            ("weighted hl=n/4", make_weighted(N_CALIB / 4.0)),
            ("weighted hl=n/8", make_weighted(N_CALIB / 8.0)),
        ],
    )
    print_table("Confirmation, scenario 1", conf1, list(conf1))
    print_table("Confirmation, scenario 2", conf2, list(conf2))
    cgain8, cse8 = paired_gain(conf2, "weighted hl=n/8", "standard (uniform)")
    cinfl8 = (
        conf1["weighted hl=n/8"]["mean_margin"]
        - conf1["overlapping (n=500)"]["mean_margin"]
    ) / abs(conf1["overlapping (n=500)"]["mean_margin"])
    cp_over = conf1["overlapping (n=500)"]["p_below"]
    cp_disj = conf1["disjoint gap=8 (n=63)"]["p_below"]
    print(
        f"\n[confirm disjoint gap=8] P(cov<0.88) overlapping={cp_over:.4f} "
        f"vs disjoint={cp_disj:.4f} "
        f"(x{(cp_over / cp_disj if cp_disj > 0 else np.inf):.2f})"
    )
    print(
        f"[confirm weighted hl=n/8] drift gain={cgain8 * 100:.2f} "
        f"+/- {cse8 * 100:.2f} pts (paired SE), "
        f"stationary inflation={cinfl8 * 100:.2f}% "
        f"-> {'PASS' if cgain8 >= 0.03 and cinfl8 <= 0.10 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
