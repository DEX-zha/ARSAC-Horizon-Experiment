"""Plotting helpers for horizon experiments."""

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_rmse(
    rmse_by_h,
    horizon_real,
    horizon_theory,
    horizon_model,
    horizon_model_cal,
    horizon_est,
    out_path,
):
    """Saves RMSE vs horizon plot."""
    x = np.arange(1, len(rmse_by_h) + 1, dtype=np.float64)
    mask = np.isfinite(rmse_by_h)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], rmse_by_h[mask], label="RMSE")
    plt.axvline(horizon_real, color="tab:red", linestyle="--", label="H_real")
    if math.isfinite(horizon_theory):
        plt.axvline(horizon_theory, color="tab:green", linestyle="--", label="H_theory")
    if horizon_model is not None and math.isfinite(horizon_model):
        plt.axvline(
            horizon_model,
            color="tab:purple",
            linestyle="--",
            label="H_model",
        )
    if horizon_model_cal is not None and math.isfinite(horizon_model_cal):
        plt.axvline(
            horizon_model_cal,
            color="tab:orange",
            linestyle="--",
            label="H_cal",
        )
    if horizon_est is not None and math.isfinite(horizon_est):
        plt.axvline(
            horizon_est,
            color="tab:cyan",
            linestyle="--",
            label="H_est",
        )
    plt.xlabel("Horizon (steps)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_log_divergence(rmse_by_h, lyap_step, out_path):
    """Saves log-divergence plot with Lyapunov slope."""
    eps = 1e-8
    log_err = np.log(rmse_by_h + eps)
    x = np.arange(len(log_err), dtype=np.float64)
    mask = np.isfinite(log_err)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], log_err[mask], label="log(RMSE)")
    if np.isfinite(log_err[0]):
        line = log_err[0] + lyap_step * x
        plt.plot(x[mask], line[mask], label="Lyapunov slope")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("log(RMSE)")
    plt.title("Log Divergence vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_predictability_map(l_values, jac_values, resid_values, out_path):
    """Saves a predictability heatmap for L(x_t) plus optional correlates."""
    l_values = np.asarray(l_values, dtype=np.float64)
    if l_values.size == 0:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), gridspec_kw={"height_ratios": [1, 2]})
    heat = l_values.reshape(1, -1)
    axes[0].imshow(heat, aspect="auto", cmap="viridis")
    axes[0].set_yticks([])
    axes[0].set_xlabel("t")
    axes[0].set_title("Predictability map: L(x_t)")

    axes[1].plot(l_values, label="L(x_t)")
    if jac_values is not None and len(jac_values) == l_values.size:
        axes[1].plot(jac_values, alpha=0.7, label="jac_norm")
    if resid_values is not None and len(resid_values) == l_values.size:
        axes[1].plot(resid_values, alpha=0.7, label="resid1")
    axes[1].set_xlabel("t")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
