"""Generates assets/multidataset_validation.png — campaign-v2 proof figure,
self-checked.

Reads the VERSIONED evidence (docs/theory/data/multidataset_validation.csv,
written by the pre-registered protocol in study_multidataset_validation.py),
re-derives every fit from the raw recorded numbers, and ASSERTS all 26
pre-registered criteria before rendering: if any check fails, the figure
refuses to exist. Left: the scaling law H(tau) on three datasets from two
domains, with the model-family invariance of the exponent (poly overlay on
COMED). Right: the twelve coverage calibrations (3 datasets x 4 model
families) with circular-block-bootstrap lower bounds.

Run: python studies/make_multidataset_figure.py   (~5 s, no recomputation)
"""

import ast
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EV = os.path.join(os.path.dirname(__file__), "..", "docs", "theory", "data",
                  "multidataset_validation.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "assets",
                   "multidataset_validation.png")

DATASETS = ("comed", "dom", "traffic")
MODELS = ("linear", "naive", "mlp", "poly")


def load_evidence():
    out = {}
    with open(EV, newline="") as f:
        r = csv.reader(f)
        next(r)
        for kind, payload in r:
            out[kind] = ast.literal_eval(payload)
    return out


def refit(h_med):
    taus = np.array(sorted(h_med), dtype=float)
    H = np.array([h_med[t] for t in taus])
    lt, lh = np.log(taus), np.log(H)
    s, b = np.polyfit(lt, lh, 1)
    r2_pow = 1 - np.var(lh - (s * lt + b)) / np.var(lh)
    a, c = np.polyfit(lt, H, 1)
    pred = a * lt + c
    r2_log = (1 - np.var(lh - np.log(np.maximum(pred, 1e-9))) / np.var(lh)
              if np.all(pred > 0) else -np.inf)
    sigma_eff = float(np.exp(-b / s))
    return taus, H, s, b, r2_pow, r2_log, sigma_eff


def main():
    R = load_evidence()
    checks = 0

    # ---- generation tests: re-derive and re-assert all 26 criteria ----
    for ds in ("comed", "dom"):
        assert R[f"w1_{ds}"]["regime"] == "quasi-periodic", ds
        checks += 1
    assert R["w1_traffic"]["regime"] != "chaotic"
    checks += 1

    # 11/12 pass; comed/naive is the documented pre-registered FAIL
    # (seasonal block shift, see docs/theory/multidataset_validation.md) —
    # the figure asserts BOTH outcomes so the render matches the record.
    for ds in DATASETS:
        for m in MODELS:
            v = R[f"w2_{ds}_{m}"]
            if (ds, m) == ("comed", "naive"):
                assert v["coverage"] < 0.88 and v["boot_lb"] < 0.85, (ds, m)
            else:
                assert v["coverage"] >= 0.88 and v["boot_lb"] >= 0.85, (ds, m)
            checks += 1

    fits = {}
    for ds in DATASETS:
        v = R[f"w3_{ds}_linear"]
        taus, H, s, b, r2p, r2l, sig = refit(v["h_med"])
        lo = 1.0 if ds == "traffic" else 1.2
        assert lo <= s <= 2.8, (ds, s)
        assert r2p > r2l, (ds, r2p, r2l)
        assert 1 / 3 <= sig / v["sigma_hat"] <= 3, ds
        checks += 3
        fits[ds] = (taus, H, s, b, r2p, sig, v["sigma_hat"])

    vp = R["w3_comed_poly"]
    tausp, Hp, sp, bp, r2pp, r2lp, sigp = refit(vp["h_med"])
    assert abs(sp - fits["comed"][2]) <= 0.5, (fits["comed"][2], sp)
    assert r2pp > r2lp
    checks += 2
    assert checks == 26, checks
    print(f"generation tests: {checks} pre-registered checks re-derived and "
          f"CONFIRMED (25 PASS + 1 documented FAIL, comed/naive)", flush=True)

    # ---- render ----
    bg, grey, txt = "#0b0e14", "#3a4356", "#c8d6f0"
    colors = {"comed": "#7fb4ff", "dom": "#e0a458", "traffic": "#6fd08c"}
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(12.5, 4.6), dpi=150,
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.24})
    fig.patch.set_facecolor(bg)
    for ax in (axL, axR):
        ax.set_facecolor(bg)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors=grey, labelsize=8)
        ax.grid(color=grey, alpha=0.25, linewidth=0.6)

    # Left: scaling law, three datasets + poly overlay on comed
    tt = np.linspace(np.log(0.17), np.log(0.95), 100)
    for ds in DATASETS:
        taus, H, s, b, r2p, sig, sig_hat = fits[ds]
        axL.plot(np.log(taus), np.log(H), "o", color=colors[ds], ms=6,
                 label=f"{ds.upper()} — s={s:.2f}, R²={r2p:.3f}, "
                       f"σ_fit/σ̂={sig / sig_hat:.2f}")
        axL.plot(tt, s * tt + b, color=colors[ds], linewidth=1.0, alpha=0.8)
    axL.plot(np.log(tausp), np.log(Hp), "s", mfc="none", mec=colors["comed"],
             ms=8, label=f"COMED poly ridge — s={sp:.2f} "
                         f"(Δs={abs(sp - fits['comed'][2]):.2f})")
    axL.plot(tt, sp * tt + bp, color=colors["comed"], linewidth=1.0,
             alpha=0.6, linestyle=":")
    axL.set_xlabel("ln τ  (failure tolerance, σ units)", color=grey, fontsize=9)
    axL.set_ylabel("ln H  (measured horizon, hours)", color=grey, fontsize=9)
    axL.set_title("Scaling law on three datasets, two domains —\n"
                  "exponent invariant across model families (COMED: linear vs poly)",
                  color=txt, fontsize=9, loc="left", pad=6)
    leg = axL.legend(loc="upper left", fontsize=7, frameon=False)
    for t in leg.get_texts():
        t.set_color(txt)

    # Right: 12 coverage calibrations grouped by dataset
    labels, cov, lb, cols, hatches = [], [], [], [], []
    for ds in DATASETS:
        for m in MODELS:
            v = R[f"w2_{ds}_{m}"]
            labels.append(f"{m}")
            cov.append(v["coverage"])
            lb.append(v["boot_lb"])
            cols.append(colors[ds])
            hatches.append("///" if (ds, m) == ("comed", "naive") else "")
    xs = np.arange(len(labels), dtype=float)
    for i, ds in enumerate(DATASETS):  # visual group gaps
        xs[i * 4:(i + 1) * 4] += i * 0.8
    bars = axR.bar(xs, cov, width=0.62, color=cols, alpha=0.55)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    axR.plot(xs, lb, "v", color="#e8e3d3", ms=6,
             label="bootstrap 95% lower bound")
    axR.axhline(0.90, color=txt, linewidth=0.9, linestyle="--", alpha=0.8)
    axR.text(xs[-1] + 0.3, 0.902, "target 0.90", color=txt, fontsize=8,
             ha="right")
    i_fail = labels.index("naive")  # first naive == comed/naive
    axR.annotate("pre-registered FAIL\n(seasonal drift, diagnosed)",
                 xy=(xs[i_fail], cov[i_fail]), xytext=(xs[i_fail] + 1.6, 0.845),
                 color="#e07a7a", fontsize=7,
                 arrowprops=dict(arrowstyle="->", color="#e07a7a", lw=0.8))
    axR.set_xticks(xs, labels, fontsize=7, color=grey, rotation=45)
    for i, ds in enumerate(DATASETS):
        axR.text(xs[i * 4 + 1] + 0.5, 0.795, ds.upper(),
                 color=colors[ds], fontsize=8, ha="center", fontweight="bold")
    axR.set_ylim(0.78, 1.0)
    axR.set_ylabel("measured coverage  P(H ≥ L)", color=grey, fontsize=9)
    axR.set_title("Guarantee held 11/12 across model families —\n"
                  "the one failure is pre-registered, kept, and diagnosed "
                  "(seasonal shift)",
                  color=txt, fontsize=9, loc="left", pad=6)
    leg = axR.legend(loc="lower right", fontsize=7.5, frameon=False)
    for t in leg.get_texts():
        t.set_color(txt)

    fig.savefig(OUT, bbox_inches="tight", facecolor=bg)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
