
import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from src.horizon_calibration import fit_mondrian_bins, validate_coverage, compute_bin_edges

def test_robustness():
    print("Testing Robustness...")
    
    # Test validate_coverage
    y_true = np.array([10, 12, 14, 16, 18])
    y_lower = np.array([9, 11, 13, 15, 17])
    y_upper = np.array([11, 13, 15, 17, 19])
    cov = validate_coverage(y_true, y_lower, y_upper, 0.9)
    print(f"Coverage (perfect): {cov}")
    if abs(cov - 1.0) > 1e-6:
        print("FAIL: Expected 1.0 coverage")
        return False

    y_lower_bad = np.array([11, 13, 15, 17, 19]) # All higher than true
    cov_bad = validate_coverage(y_true, y_lower_bad, y_upper, 0.9)
    print(f"Coverage (bad): {cov_bad}")
    if abs(cov_bad - 0.0) > 1e-6:
        print("FAIL: Expected 0.0 coverage")
        return False

    # Test fit_mondrian_bins with small samples
    # We will use 10 samples and 5 bins, so 2 per bin on average.
    # If we set min_bin=5, all bins should be "small" and fall back to global_c.
    
    features = np.random.rand(10, 1)
    scores = np.ones(10) * 1.0 # Constant score
    global_c = 2.0
    edges = compute_bin_edges(features, 5)
    
    c_groups, _, counts = fit_mondrian_bins(
        features, scores, alpha=0.1, edges_list=[edges], 
        min_bin=5, shrinkage=0.0, global_c=global_c
    )
    
    print(f"Bin counts: {counts}")
    print(f"C groups: {c_groups}")
    
    # Since all counts should be < 5 (avg 2, max likely <5), they should all be global_c
    if np.any(counts < 5):
        # Verification: check if a small bin has global_c
        small_indices = np.where(counts < 5)[0]
        for idx in small_indices:
            if abs(c_groups[idx] - global_c) > 1e-6:
                print(f"FAIL: Bin {idx} with count {counts[idx]} did not fallback to global_c")
                return False
    
    print("SUCCESS: Robustness tests passed.")
    return True

if __name__ == "__main__":
    if test_robustness():
        sys.exit(0)
    else:
        sys.exit(1)
