"""Study: right-censored labels — naive pinball vs Powell (1986) loss.

Synthetic 1D heteroscedastic data with analytically known conditional
quantiles, right-censored at the 70th percentile of the training labels
(~30% censoring), mimicking horizon labels capped at Hmax. For each seed a
tiny quantile MLP is fitted twice on the SAME recorded (censored) labels:

  - naive : plain pinball loss on recorded y (current pipeline behavior),
  - powell: censored pinball loss pinball(y_rec, min(pred, C)).

Both are then pushed through the project's one-sided conformal step
(margin c = conformal quantile of s = q_hat - y_rec on a censored
calibration set; lower bound L = q_hat - c) and evaluated against the TRUE
uncensored labels of a test set.

Reported per design, aggregated over seeds:
  - bias of q_hat vs the analytic Q_alpha(y|x) (mean signed + mean abs),
  - conformal margin c, coverage P(y_true >= L), mean L (tightness),
  - oracle-calibration margin (uncensored calib labels) to illustrate the
    conservativeness theorem: c_censored >= c_oracle, coverage held in both.

Designs:
  - primary  : y = 5 + 4x + LogNormal(sigma(x)=0.3+0.4x)  -> the censoring
    cap C intersects Q_0.1(y|x) on ~18% of the x-range (the regime the
    Powell loss is built for),
  - sensitivity: y = 5 + 2x + LogNormal(sigma(x)=0.4+0.4x) -> C almost
    never reaches Q_0.1(y|x) (~6% of x-range): naive pinball is already
    nearly consistent there and Powell is expected to change little.

Run: python studies/study_censoring.py   (seeded, ~1-2 min on CPU)
"""

import os
import sys
import time

import numpy as np
import torch
from scipy.stats import norm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_censoring import censored_pinball_loss, saturation_gate  # noqa: E402
from src.horizon_conformal import conformal_quantile  # noqa: E402
from src.horizon_models import MLPPredictor, TorchWrapper  # noqa: E402
from src.horizon_utils import set_seed  # noqa: E402

ALPHA = 0.1
N_SEEDS = 20
N_TRAIN = 1200
N_CALIB = 800
N_TEST = 4000
CENSOR_PCT = 70.0
HIDDEN = 32
EPOCHS = 400
LR = 1e-2
DEVICE = "cpu"

DESIGNS = {
    "primary(slope=4)": {"slope": 4.0, "sig0": 0.3, "sig1": 0.4},
    "sensitivity(slope=2)": {"slope": 2.0, "sig0": 0.4, "sig1": 0.4},
}


def simulate(rng, n, design):
    """Draws x ~ U(0,2), y = 5 + slope*x + LogNormal(sigma(x))."""
    x = rng.uniform(0.0, 2.0, size=n)
    sigma = design["sig0"] + design["sig1"] * x
    y = 5.0 + design["slope"] * x + np.exp(sigma * rng.standard_normal(n))
    return x, y


def true_quantile(x, design, tau=ALPHA):
    """Analytic conditional tau-quantile of the simulated y given x."""
    sigma = design["sig0"] + design["sig1"] * x
    return 5.0 + design["slope"] * x + np.exp(sigma * norm.ppf(tau))


def fit_quantile_mlp(x, y, cap, seed):
    """Fits a tiny quantile MLP with (censored) pinball loss, full batch."""
    set_seed(seed)
    model = MLPPredictor(input_dim=1, hidden_dim=HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    x_t = torch.tensor(x[:, None], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.float32, device=DEVICE)
    for _ in range(EPOCHS):
        opt.zero_grad()
        pred = model(x_t)
        loss = censored_pinball_loss(pred, y_t, ALPHA, cap)
        loss.backward()
        opt.step()
    model.eval()
    return TorchWrapper(model, DEVICE)


def run_seed(seed, design):
    """One replication: returns metrics for naive and powell fits."""
    rng = np.random.default_rng(1000 + seed)
    x_train, y_train = simulate(rng, N_TRAIN, design)
    x_calib, y_calib = simulate(rng, N_CALIB, design)
    x_test, y_test = simulate(rng, N_TEST, design)

    cap = float(np.percentile(y_train, CENSOR_PCT))
    y_train_rec = np.minimum(y_train, cap)
    y_calib_rec = np.minimum(y_calib, cap)
    gate = saturation_gate(y_train_rec, cap, ALPHA)

    q_true_test = true_quantile(x_test, design)
    out = {"p_sat": gate["p_sat"], "identified": gate["identified"]}
    for name, loss_cap in (("naive", None), ("powell", cap)):
        wrapper = fit_quantile_mlp(x_train, y_train_rec, loss_cap, seed=seed)
        q_calib = wrapper.predict_batch(x_calib[:, None]).astype(np.float64)
        q_test = wrapper.predict_batch(x_test[:, None]).astype(np.float64)

        # Quantile bias vs the analytic truth.
        err = q_test - q_true_test
        # Conformal margin from CENSORED calib labels (realistic setting).
        c_rec = conformal_quantile(q_calib - y_calib_rec, ALPHA)
        # Oracle margin from TRUE calib labels (theorem (a) reference).
        c_true = conformal_quantile(q_calib - y_calib, ALPHA)
        # Censoring-aware score (Powell transform applied to the conformal
        # step too): s = min(q_hat, C) - y_rec, bound L = min(q_hat, C) - c.
        c_cap = conformal_quantile(np.minimum(q_calib, cap) - y_calib_rec, ALPHA)
        lower = q_test - c_rec
        lower_oracle = q_test - c_true
        lower_cap = np.minimum(q_test, cap) - c_cap
        top = x_test > 1.5  # most-censored quarter of the input range
        out[name] = {
            "bias": float(np.mean(err)),
            "abs_bias": float(np.mean(np.abs(err))),
            "margin_rec": float(c_rec),
            "margin_oracle": float(c_true),
            "margin_cap": float(c_cap),
            "coverage": float(np.mean(y_test >= lower)),
            "coverage_oracle": float(np.mean(y_test >= lower_oracle)),
            "coverage_cap": float(np.mean(y_test >= lower_cap)),
            "coverage_top": float(np.mean(y_test[top] >= lower[top])),
            "coverage_cap_top": float(np.mean(y_test[top] >= lower_cap[top])),
            "mean_lower": float(np.mean(lower)),
            "mean_lower_cap": float(np.mean(lower_cap)),
        }
    return out


def aggregate(records, model):
    keys = records[0][model].keys()
    return {k: np.array([r[model][k] for r in records]) for k in keys}


def main():
    start = time.time()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    print(f"alpha={ALPHA}  seeds={N_SEEDS}  censoring at p{CENSOR_PCT:.0f}")
    verdicts = {}
    for design_name, design in DESIGNS.items():
        records = [run_seed(s, design) for s in range(N_SEEDS)]
        p_sat = np.mean([r["p_sat"] for r in records])
        print(f"\n=== design {design_name} ===")
        print(f"p_sat={p_sat:.3f}  gate identified={records[0]['identified']}")
        stats = {m: aggregate(records, m) for m in ("naive", "powell")}
        cols = [
            ("bias", "bias"),
            ("abs_bias", "|bias|"),
            ("margin_rec", "margin"),
            ("margin_oracle", "m_orcl"),
            ("margin_cap", "m_cap"),
            ("coverage", "cover"),
            ("coverage_oracle", "cov_or"),
            ("coverage_cap", "cov_cap"),
            ("coverage_top", "cov_top"),
            ("coverage_cap_top", "covctop"),
            ("mean_lower", "mean_L"),
            ("mean_lower_cap", "mean_Lc"),
        ]
        print(f"{'model':8s} " + " ".join(f"{label:>8s}" for _, label in cols))
        for m in ("naive", "powell"):
            s = stats[m]
            print(
                f"{m:8s} "
                + " ".join(f"{s[key].mean():8.3f}" for key, _ in cols)
            )
            print(
                f"{'':8s} "
                + " ".join(f"±{s[key].std():7.3f}" for key, _ in cols)
            )
        abs_bias_naive = stats["naive"]["abs_bias"].mean()
        abs_bias_powell = stats["powell"]["abs_bias"].mean()
        reduction = 1.0 - abs_bias_powell / abs_bias_naive
        cov_powell = stats["powell"]["coverage"].mean()
        cov_naive = stats["naive"]["coverage"].mean()
        cover_ok = cov_powell >= 1.0 - ALPHA - 0.02
        margin_delta = (
            stats["powell"]["margin_rec"].mean() - stats["naive"]["margin_rec"].mean()
        )
        tightness_gain_cap = (
            stats["powell"]["mean_lower_cap"].mean()
            - stats["naive"]["mean_lower"].mean()
        )
        print(
            f"|bias| reduction (powell vs naive): {100 * reduction:+.1f}%  "
            f"coverage naive={cov_naive:.3f} powell={cov_powell:.3f} "
            f"(threshold {1.0 - ALPHA - 0.02:.2f})  margin delta={margin_delta:+.3f}"
        )
        print(
            f"tightness: mean_L powell+capped-score vs naive: "
            f"{tightness_gain_cap:+.3f} "
            f"(cov {stats['powell']['coverage_cap'].mean():.3f} vs {cov_naive:.3f})"
        )
        # Theorem (a) check: censored margin >= oracle margin, seed by seed.
        thm_a = {
            m: bool(
                np.all(stats[m]["margin_rec"] >= stats[m]["margin_oracle"] - 1e-12)
            )
            for m in ("naive", "powell")
        }
        print(f"thm(a) c_censored >= c_oracle on every seed: {thm_a}")
        verdicts[design_name] = {
            "reduction": reduction,
            "cover_ok": cover_ok,
            "integrate": bool(reduction >= 0.30 and cover_ok),
        }
    print(f"\nDecision (criterion: |bias| -30% and coverage >= {1 - ALPHA - 0.02:.2f}):")
    for name, v in verdicts.items():
        print(
            f"  {name}: reduction={100 * v['reduction']:+.1f}% "
            f"coverage_ok={v['cover_ok']} -> "
            f"{'INTEGRATE' if v['integrate'] else 'not on its own'}"
        )
    print(f"runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
