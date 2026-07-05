
import os
import sys

import numpy as np

sys.path.append(os.getcwd())

from src.horizon_calibration import compute_bin_edges, fit_mondrian_bins, validate_coverage


def test_robustness():
    # validate_coverage sanity
    y_true = np.array([10, 12, 14, 16, 18])
    y_lower = np.array([9, 11, 13, 15, 17])
    y_upper = np.array([11, 13, 15, 17, 19])
    cov = validate_coverage(y_true, y_lower, y_upper, 0.9)
    assert abs(cov - 1.0) <= 1e-6

    y_lower_bad = np.array([11, 13, 15, 17, 19])
    cov_bad = validate_coverage(y_true, y_lower_bad, y_upper, 0.9)
    assert abs(cov_bad - 0.0) <= 1e-6

    # fit_mondrian_bins fallback when bins are too small
    features = np.random.rand(10, 1)
    scores = np.ones(10) * 1.0
    global_c = 2.0
    edges = compute_bin_edges(features, 5)

    c_groups, _, counts = fit_mondrian_bins(
        features,
        scores,
        alpha=0.1,
        edges_list=[edges],
        min_bin=5,
        shrinkage=0.0,
        global_c=global_c,
    )

    small_indices = np.where(counts < 5)[0]
    for idx in small_indices:
        assert abs(c_groups[idx] - global_c) <= 1e-6


if __name__ == "__main__":
    test_robustness()
