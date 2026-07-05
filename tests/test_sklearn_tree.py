
import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from src.horizon_experiment import ConformalTreeEstimator, conformal_quantile

def test_conformal_tree_temporal_split():
    """Audit E3: with n >= 4*min_samples_leaf, fit() uses a temporal split.

    Tree structure is learned on the first half; leaf quantiles (and the
    global fallback) come from the held-out second half. Leaves with < 5
    held-out samples fall back to the global quantile.
    """
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    scores = y - 0.5 # Fake residuals

    alpha = 0.1
    tree = ConformalTreeEstimator(min_samples_leaf=10, max_depth=3)
    tree.fit(X, scores, alpha)

    preds = tree.predict(X)

    # Predictions are constant per leaf
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)
    for leaf in unique_leaves:
        p = preds[leaf_ids == leaf]
        assert np.all(p == p[0])

    # Leaf values match conformal_quantile of the SECOND-HALF scores per leaf
    half = len(y) // 2
    hold_scores = scores[half:]
    hold_leaves = tree.apply(X[half:])
    expected_global = conformal_quantile(hold_scores, alpha)
    assert abs(tree.global_fallback - expected_global) <= 1e-12

    for leaf in np.unique(hold_leaves):
        leaf_scores = hold_scores[hold_leaves == leaf]
        if leaf_scores.size < 5:
            expected = expected_global
        else:
            expected = conformal_quantile(leaf_scores, alpha)
        assert abs(tree.leaf_quantiles[leaf] - expected) <= 1e-6

    # Leaves never seen among the held-out half use the global fallback
    for leaf in unique_leaves:
        if leaf not in tree.leaf_quantiles:
            p = preds[leaf_ids == leaf]
            assert np.all(np.abs(p - expected_global) <= 1e-12)


def test_conformal_tree_small_data_fallback():
    """With n < 4*min_samples_leaf the historical single-set fit is kept."""
    np.random.seed(0)
    X = np.random.rand(30, 5)
    scores = np.random.rand(30) - 0.5

    alpha = 0.1
    tree = ConformalTreeEstimator(min_samples_leaf=10, max_depth=3)
    tree.fit(X, scores, alpha)

    preds = tree.predict(X)
    leaf_ids = tree.apply(X)

    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        p = preds[mask]
        assert np.all(p == p[0])
        expected = conformal_quantile(scores[mask], alpha)
        assert abs(p[0] - expected) <= 1e-6

if __name__ == "__main__":
    test_conformal_tree_temporal_split()
    test_conformal_tree_small_data_fallback()
