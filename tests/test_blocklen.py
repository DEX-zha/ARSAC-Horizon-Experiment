"""Tests for the Politis-White automatic block-length estimator."""

import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_blocklen import (
    flat_top_lambda,
    politis_white_block_length,
    sample_autocovariances,
)


def _ar1(n, phi, seed, burn=500):
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n + burn)
    z = np.empty(n + burn)
    z[0] = eps[0]
    for t in range(1, n + burn):
        z[t] = phi * z[t - 1] + eps[t]
    return z[burn:]


def test_flat_top_kernel_values():
    values = flat_top_lambda([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, -0.75])
    expected = np.array([1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5])
    assert np.allclose(values, expected)


def test_sample_autocovariances_matches_definition():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(50)
    xc = x - x.mean()
    gamma = sample_autocovariances(x, 3)
    for k in range(4):
        expected = float(np.dot(xc[: 50 - k], xc[k:])) / 50
        assert np.isclose(gamma[k], expected)
    assert gamma[0] > 0


def test_sample_autocovariances_rejects_bad_lag():
    with pytest.raises(ValueError):
        sample_autocovariances(np.ones(5), 5)


def test_constant_series_returns_min_block():
    assert politis_white_block_length(np.ones(500)) == 10
    assert politis_white_block_length(np.zeros(200)) == 10
    assert politis_white_block_length(np.full(300, 7.3)) == 10


def test_degenerate_inputs_return_min_block():
    assert politis_white_block_length(np.array([1.0])) == 10
    assert politis_white_block_length(np.array([])) == 10
    assert politis_white_block_length(np.array([1.0, np.nan, 2.0] * 50)) == 10


def test_iid_series_gets_small_block():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    b = politis_white_block_length(x)
    assert b == 10  # theoretical b_opt for white noise is tiny; clamped at 10


def test_binary_hits_supported():
    rng = np.random.default_rng(1)
    hits = (rng.random(500) < 0.9).astype(int)
    b = politis_white_block_length(hits)
    assert 10 <= b <= 500 // 3


def test_block_length_grows_with_dependence():
    n = 2000
    b_weak = politis_white_block_length(_ar1(n, 0.0, seed=7))
    b_strong = politis_white_block_length(_ar1(n, 0.9, seed=7))
    assert b_strong > b_weak
    assert b_weak == 10


def test_ar1_phi09_matches_theory_order_of_magnitude():
    # For AR(1) with phi=0.9 the Politis-White b_opt (stationary bootstrap
    # constants) is ~40-70 at n=4000; check the right order of magnitude.
    b = politis_white_block_length(_ar1(4000, 0.9, seed=11))
    assert 20 <= b <= 120


def test_clamped_to_upper_bound():
    # Near-unit-root series on a short sample must respect the n//3 cap.
    n = 60
    x = _ar1(n, 0.995, seed=5)
    b = politis_white_block_length(x)
    assert 10 <= b <= max(10, n // 3)


def test_deterministic():
    x = _ar1(800, 0.7, seed=9)
    assert politis_white_block_length(x) == politis_white_block_length(x)


def test_scientific_eval_uses_politis_white_when_block_len_none():
    from src.horizon_scientific_eval import _block_bootstrap_lower_bound

    hits = (_ar1(400, 0.5, seed=13) < 1.0).astype(int)
    lb_auto = _block_bootstrap_lower_bound(hits, alpha=0.05, n_boot=200, seed=0)
    b_pw = politis_white_block_length(hits)
    lb_explicit = _block_bootstrap_lower_bound(
        hits, alpha=0.05, n_boot=200, block_len=b_pw, seed=0
    )
    assert lb_auto is not None and lb_explicit is not None
    assert np.isclose(lb_auto, lb_explicit)
