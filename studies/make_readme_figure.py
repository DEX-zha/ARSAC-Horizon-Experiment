"""Generates assets/predictability_map.png for the README — from a real run.

Repo rule: no hand-made or stale figures. This script runs the actual pipeline
(Lorenz, linear forecaster, calibrated bounds exported per window) and renders
the per-window trust bound L(x_t) under the trajectory it belongs to, styled to
match the ARSAC logo (dark background, blue gradient).

Run: python studies/make_readme_figure.py   (~30 s)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_estimator import HorizonEstimator
from src.horizon_utils import generate_lorenz

OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "predictability_map.png")


def main():
    series = generate_lorenz(6000, dt=0.01, warmup=1000)
    est = HorizonEstimator(
        model="linear", alpha=0.1, tolerance=0.4, horizon_max=30,
        quantile_ensemble=1, mlp_epochs=30, horizon_samples=99999,
        output_dir="outputs_readme_fig",
    )
    est.fit(series)
    L = est.lower_bounds_
    n_windows = L.size
    print(f"windows={n_windows}  L in [{L.min():.1f}, {L.max():.1f}] "
          f"med={np.median(L):.1f}  coverage={est.coverage_:.3f}")

    # Reconstruct the standardized test segment for the top panel (same split
    # convention as the estimator: train 0.6 / val 0.15 / calib 0.15 / test 0.1).
    n = series.size
    i_train, i_test = int(0.6 * n), int(0.9 * n)
    mean, sd = series[:i_train].mean(), series[:i_train].std()
    test = (series[i_test:] - mean) / sd
    dim = est.result_.get("dim")
    lag = est.result_.get("lag")
    window_len = (dim - 1) * lag + 1
    t = np.arange(n_windows)  # window w starts at test[w]; current time index w+window_len-1
    x_now = test[window_len - 1: window_len - 1 + n_windows]

    bg, blue_hi, blue_lo, grey = "#0b0e14", "#7fb4ff", "#1f4e8c", "#3a4356"
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(11, 4.6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.3], "hspace": 0.12},
    )
    fig.patch.set_facecolor(bg)
    for ax in (ax0, ax1):
        ax.set_facecolor(bg)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=grey, labelsize=8)
        ax.grid(axis="y", color=grey, alpha=0.25, linewidth=0.6)

    ax0.plot(t, x_now, color=blue_hi, linewidth=0.8, alpha=0.95)
    ax0.set_ylabel("x(t)  (Lorenz, std)", color=grey, fontsize=9)
    ax0.set_yticks([-2, 0, 2])

    ax1.fill_between(t, 0, L, step="mid", color=blue_lo, alpha=0.85)
    ax1.plot(t, L, drawstyle="steps-mid", color=blue_hi, linewidth=1.0)
    ax1.set_ylabel("trustworthy steps  L(x$_t$)", color=grey, fontsize=9)
    ax1.set_xlabel("time (test windows)", color=grey, fontsize=9)
    ax1.set_ylim(0, max(10, L.max() + 6))
    med = float(np.median(L))
    ax1.axhline(med, color="#c8d6f0", linewidth=0.8, linestyle="--", alpha=0.6)
    ax1.text(t[-1], med + 0.5, f"median {med:.0f}", color="#c8d6f0",
             fontsize=8, ha="right", alpha=0.8)
    ax1.set_title(
        f"How many forecast steps can be trusted if you start NOW? — "
        f"coverage {est.coverage_:.2f} (target 0.90), same system, same model",
        color="#c8d6f0", fontsize=9, loc="left", pad=6,
    )

    # Annotate two example instants: a comfortable start and a risky one.
    i_lo = int(np.argmin(L))
    smooth = np.convolve(L, np.ones(15) / 15, mode="same")
    i_hi = int(np.argmax(smooth[30:-30])) + 30
    for i, label, dy in (
        (i_hi, f"start here:\n≥{L[i_hi]:.0f} steps safe", 4.5),
        (i_lo, f"start here:\nonly {L[i_lo]:.0f} guaranteed", 4.5),
    ):
        ax1.annotate(
            label, xy=(t[i], L[i]), xytext=(t[i], L[i] + dy),
            color="#e8eefc", fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="#e8eefc", lw=0.9),
        )
        for ax in (ax0, ax1):
            ax.axvline(t[i], color="#e8eefc", linewidth=0.6, alpha=0.25)

    fig.savefig(OUT, bbox_inches="tight", facecolor=bg)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
