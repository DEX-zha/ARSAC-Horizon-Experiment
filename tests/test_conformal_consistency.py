"""Consistency tests for duplicated conformal logic."""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import src.horizon_calibration as hcal
import src.horizon_conformal as hconf


def test_conformal_quantile_consistency():
    scores = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.inf], dtype=np.float64)
    alpha = 0.2

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    q1 = hconf.conformal_quantile(scores, alpha, rng=rng1, tie_jitter=0.01)
    q2 = hcal.conformal_quantile(scores, alpha, rng=rng2, tie_jitter=0.01)

    assert np.isclose(q1, q2)


def test_block_conformal_margin_consistency():
    scores = np.linspace(-1.0, 1.0, 41)
    alpha = 0.1

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    m1 = hconf.block_conformal_margin(
        scores, alpha, block_count=5, block_quantile=0.8, rng=rng1, tie_jitter=0.02
    )
    m2 = hcal.block_conformal_margin(
        scores, alpha, block_count=5, block_quantile=0.8, rng=rng2, tie_jitter=0.02
    )

    assert np.isclose(m1, m2)


def test_bins_and_assign_consistency():
    rng = np.random.default_rng(0)
    values = rng.normal(size=100)
    edges1 = hconf.compute_bin_edges(values, bins=4)
    edges2 = hcal.compute_bin_edges(values, bins=4)

    np.testing.assert_allclose(edges1, edges2)

    features = rng.normal(size=(50, 2))
    edges_list = [edges1, hconf.compute_bin_edges(features[:, 1], bins=3)]

    ids1, count1 = hconf.assign_bin_ids(features, edges_list)
    ids2, count2 = hcal.assign_bin_ids(features, edges_list)

    np.testing.assert_array_equal(ids1, ids2)
    assert count1 == count2


def test_fit_mondrian_bins_consistency():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(80, 1))
    scores = rng.normal(size=80)
    edges = hconf.compute_bin_edges(features[:, 0], bins=3)

    c1, g1, n1 = hconf.fit_mondrian_bins(
        features,
        scores,
        alpha=0.1,
        edges_list=[edges],
        min_bin=5,
        shrinkage=2.0,
        global_c=0.5,
        rng=np.random.default_rng(7),
        tie_jitter=0.0,
    )
    c2, g2, n2 = hcal.fit_mondrian_bins(
        features,
        scores,
        alpha=0.1,
        edges_list=[edges],
        min_bin=5,
        shrinkage=2.0,
        global_c=0.5,
        rng=np.random.default_rng(7),
        tie_jitter=0.0,
    )

    np.testing.assert_allclose(c1, c2)
    np.testing.assert_array_equal(g1, g2)
    np.testing.assert_array_equal(n1, n2)


def test_tree_estimator_consistency():
    # Audit E3: horizon_calibration re-exports ConformalTreeEstimator from
    # horizon_conformal (single audited implementation, temporal split fit).
    # Guard against reintroducing a diverging duplicate.
    assert hcal.ConformalTreeEstimator is hconf.ConformalTreeEstimator
    rng = np.random.default_rng(2)
    X = rng.normal(size=(35, 4))
    scores = rng.normal(size=35)
    alpha = 0.15

    t1 = hconf.ConformalTreeEstimator(min_samples_leaf=10, max_depth=3)
    t2 = hcal.ConformalTreeEstimator(min_samples_leaf=10, max_depth=3)

    t1.fit(X, scores, alpha)
    t2.fit(X, scores, alpha)

    p1 = t1.predict(X)
    p2 = t2.predict(X)

    np.testing.assert_allclose(p1, p2)


def test_extract_bin_features_consistency():
    dim = 3
    x_raw = np.arange(5 * (dim + 8), dtype=np.float64).reshape(5, dim + 8)

    for mode in ("resid", "jac_log", "both"):
        f1 = hconf.extract_bin_features(x_raw, dim, mode)
        f2 = hcal.extract_bin_features(x_raw, dim, mode)
        np.testing.assert_array_equal(f1, f2)
