"""Tests for censoring-aware quantile losses and the saturation gate."""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_censoring import (
    censored_pinball_loss,
    censored_pinball_np,
    saturation_gate,
)
from src.horizon_conformal import conformal_quantile
from src.horizon_training import pinball_loss


def test_cap_none_matches_plain_pinball():
    torch.manual_seed(0)
    pred = torch.randn(64)
    target = torch.randn(64)
    for q in (0.1, 0.5, 0.9):
        expected = pinball_loss(pred, target, q)
        got = censored_pinball_loss(pred, target, q, cap=None)
        assert torch.isclose(got, expected)


def test_torch_and_numpy_versions_agree():
    rng = np.random.default_rng(1)
    pred = rng.normal(size=50)
    target = rng.normal(size=50)
    for cap in (None, 0.5):
        loss_t = censored_pinball_loss(
            torch.tensor(pred), torch.tensor(target), 0.1, cap
        ).item()
        loss_np = censored_pinball_np(pred, target, 0.1, cap)
        assert np.isclose(loss_t, loss_np)


def test_loss_flat_and_gradient_zero_above_cap():
    # All recorded targets censored at C: for pred >= C the Powell loss is
    # flat (min(pred, C) = C) while the naive pinball keeps growing.
    cap = 2.0
    target = torch.full((32,), cap, dtype=torch.float64)
    q = 0.1

    loss_at_cap = censored_pinball_np(np.full(32, cap), target.numpy(), q, cap)
    loss_above = censored_pinball_np(np.full(32, cap + 5.0), target.numpy(), q, cap)
    assert np.isclose(loss_at_cap, loss_above)

    naive_at_cap = censored_pinball_np(np.full(32, cap), target.numpy(), q, None)
    naive_above = censored_pinball_np(np.full(32, cap + 5.0), target.numpy(), q, None)
    assert naive_above > naive_at_cap

    pred = torch.full((32,), cap + 1.0, dtype=torch.float64, requires_grad=True)
    loss = censored_pinball_loss(pred, target, q, cap)
    loss.backward()
    assert torch.all(pred.grad == 0.0)


def test_population_minimizer_identified_region():
    # When Q_tau(Y) < C both losses recover the true quantile.
    rng = np.random.default_rng(2)
    y_true = np.exp(0.5 * rng.standard_normal(20000))
    tau = 0.2
    q_true = float(np.quantile(y_true, tau))
    cap = float(np.quantile(y_true, 0.7))
    assert q_true < cap
    y_rec = np.minimum(y_true, cap)

    grid = np.linspace(0.01, cap + 1.0, 400)
    powell = [censored_pinball_np(np.full_like(y_rec, g), y_rec, tau, cap) for g in grid]
    naive = [censored_pinball_np(np.full_like(y_rec, g), y_rec, tau, None) for g in grid]
    assert abs(grid[int(np.argmin(powell))] - q_true) < 0.05
    assert abs(grid[int(np.argmin(naive))] - q_true) < 0.05


def test_naive_biased_powell_flat_when_quantile_censored():
    # When Q_tau(Y) > C the naive loss drags the minimizer down to C while
    # the Powell loss is indifferent on [C, inf) (no downward pull).
    rng = np.random.default_rng(3)
    y_true = 5.0 + rng.standard_normal(20000)
    tau = 0.5
    q_true = float(np.quantile(y_true, tau))
    cap = q_true - 1.0  # censor strictly below the target quantile
    y_rec = np.minimum(y_true, cap)

    # Naive: loss at the true quantile is strictly worse than at the cap.
    naive_at_cap = censored_pinball_np(np.full_like(y_rec, cap), y_rec, tau, None)
    naive_at_true = censored_pinball_np(np.full_like(y_rec, q_true), y_rec, tau, None)
    assert naive_at_true > naive_at_cap
    # Powell: flat between the cap and the true quantile (no penalty).
    powell_at_cap = censored_pinball_np(np.full_like(y_rec, cap), y_rec, tau, cap)
    powell_at_true = censored_pinball_np(np.full_like(y_rec, q_true), y_rec, tau, cap)
    assert np.isclose(powell_at_cap, powell_at_true)


def test_saturation_gate_basic():
    y = np.array([1.0, 2.0, 3.0, 10.0, 10.0, 10.0, 4.0, 5.0, 6.0, 7.0])
    gate = saturation_gate(y, horizon_max=10, alpha=0.2)
    assert np.isclose(gate["p_sat"], 0.3)
    assert gate["identified"] is True  # 0.3 <= 0.8
    assert isinstance(gate["message"], str) and gate["message"]

    gate_bad = saturation_gate(y, horizon_max=10, alpha=0.75)
    assert gate_bad["identified"] is False  # 0.3 > 0.25

    gate_empty = saturation_gate(np.array([]), horizon_max=10, alpha=0.1)
    assert gate_empty["p_sat"] is None
    assert gate_empty["identified"] is False


def test_saturation_gate_rejects_bad_alpha():
    y = np.ones(5)
    for alpha in (0.0, 1.0, -0.1, 1.5):
        try:
            saturation_gate(y, horizon_max=1.0, alpha=alpha)
        except ValueError:
            continue
        raise AssertionError(f"alpha={alpha} should raise ValueError")


def test_censoring_is_conservative_for_lower_bound():
    # Theorem (a): recording y_rec = min(y, C) can only increase the signed
    # score s = q_hat - y, hence the conformal margin, hence lower L, and
    # coverage of the true y is preserved.
    rng = np.random.default_rng(4)
    alpha = 0.1
    n = 500
    y_calib = np.exp(rng.standard_normal(n))
    q_hat_calib = np.full(n, float(np.quantile(y_calib, alpha)))
    cap = float(np.quantile(y_calib, 0.7))
    y_calib_rec = np.minimum(y_calib, cap)

    scores_true = q_hat_calib - y_calib
    scores_rec = q_hat_calib - y_calib_rec
    assert np.all(scores_rec >= scores_true)

    c_true = conformal_quantile(scores_true, alpha)
    c_rec = conformal_quantile(scores_rec, alpha)
    assert c_rec >= c_true

    y_test = np.exp(rng.standard_normal(2000))
    q_hat_test = np.full(2000, float(np.quantile(y_calib, alpha)))
    cover_rec = np.mean(y_test >= q_hat_test - c_rec)
    cover_true = np.mean(y_test >= q_hat_test - c_true)
    assert cover_rec >= cover_true
    assert cover_rec >= 1.0 - alpha - 0.03
