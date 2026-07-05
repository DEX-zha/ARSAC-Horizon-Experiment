
import logging
import os
import sys

sys.path.append(os.getcwd())

from src.horizon_experiment import run_experiment

# Mocking args for linear sanity test
class MockArgs:
    def __init__(self):
        self.dataset = "logistic" 
        self.series_len = 1000
        self.warmup = 100
        self.seed = 42
        self.train_ratio = 0.5
        self.val_ratio = 0.25
        self.calib_ratio = 0.1 # Total = 0.85, leaves 0.15 for test
        self.dim_min = 1
        self.dim_max = 5
        self.lag_min = 1
        self.lag_max = 5
        self.model = "linear"
        self.linear_reg = 0.0
        self.train_multistep = False
        self.progress = False
        self.output_dir = "tests/output_sanity"
        self.plot = False
        self.selection_metric = "val_mse"
        self.error_mode = "relative"
        self.error_factor = 2.0
        self.error_tolerance = 0.1
        self.horizon_max = 20
        self.selection_horizon_max = 10
        self.lyap_dim = None
        self.lyap_lag = None
        self.lyap_max_t = 20
        self.lyap_theiler = 5
        self.lyap_fit_start = 1
        self.lyap_fit_end = 5
        self.dt = 1.0 # Logistic map is discrete
        self.r = 4.0 # Chaotic
        self.x0 = 0.2
        self.conformal_mode = "bins" # Default fallback
        self.delta_mode = "quantile"
        self.delta_quantile = 0.95
        self.delta_scale = 3.0
        self.delta_local = False
        self.growth_source = "error"
        self.expansion_horizon = 5
        self.expansion_quantile = 0.95
        self.expansion_samples = 100
        self.expansion_theiler = 1
        self.expansion_dim = None
        self.expansion_lag = None
        self.bound_mode = "probabilistic"
        self.calibration_alpha = 0.1
        self.calibration_floor = 1.0
        self.calibrate_coverage = False
        # Add missing args required by horizon_experiment properties access
        self.scale_eps = 1e-6
        self.scale_quantile_high = 0.9
        self.quantile_ensemble = 3
        self.quantile_ensemble_stride = 10
        self.mlp_hidden = 10
        self.mlp_epochs = 1
        self.mlp_lr = 0.01
        self.mlp_batch = 32
        self.mlp_patience = 5
        self.lstm_hidden = 10
        self.lstm_layers = 1
        self.lstm_epochs = 1
        self.lstm_lr = 0.01
        self.lstm_batch = 32
        self.lstm_patience = 5
        self.tf_start = 1.0
        self.tf_end = 0.1
        self.tf_val = 0.0
        self.use_cuda = False
        
        # Conformal args
        self.conformal_bins = 4
        self.conformal_min_bin = 5
        self.conformal_bin_shrinkage = 1.0
        self.conformal_tie_jitter = 1e-6
        self.conformal_bin_feature = "resid"
        self.conformal_cv_folds = 1
        self.conformal_no_sigma = False
        self.conformal_feature = "pred"
        self.conformal_tree_depth = 3
        self.conformal_min_leaf = 10
        self.conformal_tree_bins = 5
        self.conformal_tree_min_gain = 0.0
        self.delta_local_k = 10
        self.delta_local_quantile = None
        self.delta_local_samples = 100
        self.horizon_samples = None
        self.horizon_consecutive_k = 2
        self.horizon_feature_horizon = 2
        self.horizon_thin = 1
        self.horizon_calib_thin = 1
        self.scale_floor = 1e-3
        self.scale_cap = None
        self.scale_from_quantiles = True
        self.scale_cap_quantile = 0.99
        self.scale_floor_quantile = 0.01
        self.block_count = 1
        self.block_quantile = 0.9
        self.offset_calibration = False
        self.offset_quantile = None
        self.horizon_use_jacobian = False
        self.plot_prefix = "sanity"

def test_linear_sanity():
    args = MockArgs()
    logging.basicConfig(level=logging.INFO)
    results = run_experiment(args)

    assert "val_loss" in results
    assert "horizon_real" in results

if __name__ == "__main__":
    test_linear_sanity()
