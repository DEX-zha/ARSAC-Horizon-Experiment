
import sys
import os
import numpy as np

sys.path.append(os.getcwd())
from src.horizon_experiment import ConformalTreeEstimator, conformal_quantile

def test_conformal_tree():
    print("Testing ConformalTreeEstimator...")
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    scores = y - 0.5 # Fake residuals
    
    alpha = 0.1
    tree = ConformalTreeEstimator(min_samples_leaf=10, max_depth=3)
    tree.fit(X, scores, alpha)
    
    preds = tree.predict(X)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions mean: {preds.mean():.4f}")
    
    # Check if predictions are constant per leaf
    leaf_ids = tree.apply(X)
    unique_leaves = np.unique(leaf_ids)
    print(f"Unique leaves: {len(unique_leaves)}")
    
    if len(unique_leaves) < 2:
        print("WARNING: Only 1 leaf found. Might be due to small data or pruning.")
    
    for leaf in unique_leaves:
        mask = leaf_ids == leaf
        p = preds[mask]
        if not np.all(p == p[0]):
             print(f"FAIL: Leaf {leaf} has non-constant predictions")
             return False
        
        # Verify the value matches conformal_quantile of the leaf scores
        leaf_scores = scores[mask]
        expected = conformal_quantile(leaf_scores, alpha)
        if abs(p[0] - expected) > 1e-6:
             print(f"FAIL: Leaf {leaf} value {p[0]} != expected {expected}")
             return False
             
    print("SUCCESS: ConformalTreeEstimator seems to work.")
    return True

if __name__ == "__main__":
    if test_conformal_tree():
        sys.exit(0)
    else:
        sys.exit(1)
