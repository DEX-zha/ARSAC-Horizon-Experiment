"""CLI module for the horizon experiment."""

import argparse
import logging
import os
import sys

import yaml


def build_parser(add_help=True):
    """Builds the argument parser for the experiment CLI."""
    parser = argparse.ArgumentParser(
        description="AI-driven prediction horizon experiment for chaotic series",
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["logistic", "lorenz", "rossler", "mackey_glass"],
        default="lorenz",
    )
    parser.add_argument("--series-len", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--calib-ratio", type=float, default=0.05)

    parser.add_argument("--dim-min", type=int, default=2)
    parser.add_argument("--dim-max", type=int, default=8)
    parser.add_argument("--lag-min", type=int, default=1)
    parser.add_argument("--lag-max", type=int, default=8)
    parser.add_argument(
        "--model", type=str, choices=["linear", "mlp", "lstm"], default="mlp"
    )
    parser.add_argument("--linear-reg", type=float, default=1e-4)

    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch", type=int, default=64)
    parser.add_argument("--mlp-patience", type=int, default=12)
    parser.add_argument("--train-multistep", action="store_true")
    parser.add_argument("--train-horizon", type=int, default=5)
    parser.add_argument("--tf-start", type=float, default=1.0)
    parser.add_argument("--tf-end", type=float, default=0.2)
    parser.add_argument("--tf-val", type=float, default=0.0)

    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-epochs", type=int, default=80)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch", type=int, default=64)
    parser.add_argument("--lstm-patience", type=int, default=12)

    parser.add_argument("--horizon-max", type=int, default=50)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["val_mse", "horizon"],
        default="val_mse",
    )
    parser.add_argument("--selection-horizon-max", type=int, default=20)
    parser.add_argument(
        "--error-mode",
        type=str,
        choices=["absolute", "relative"],
        default="relative",
    )
    parser.add_argument("--error-factor", type=float, default=10.0)
    parser.add_argument("--error-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--bound-mode",
        type=str,
        choices=["probabilistic", "horizon_conformal"],
        default="probabilistic",
    )
    parser.add_argument("--horizon-quantile", type=float, default=None)
    parser.add_argument(
        "--conformal-mode",
        type=str,
        choices=["global", "normalized", "tree", "bins"],
        default="bins",
    )
    parser.add_argument("--conformal-bins", type=int, default=4)
    parser.add_argument("--conformal-min-bin", type=int, default=5)
    parser.add_argument("--conformal-bin-shrinkage", type=float, default=20.0)
    parser.add_argument("--conformal-tie-jitter", type=float, default=1e-6)
    parser.add_argument(
        "--conformal-bin-feature",
        type=str,
        choices=["resid", "jac_log", "both"],
        default="resid",
    )
    parser.add_argument("--conformal-cv-folds", type=int, default=1)
    parser.add_argument(
        "--conformal-no-sigma",
        dest="conformal_no_sigma",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--conformal-feature",
        type=str,
        choices=["pred", "pred_jacobian"],
        default="pred",
    )
    parser.add_argument("--conformal-tree-depth", type=int, default=4)
    parser.add_argument("--conformal-min-leaf", type=int, default=500)
    parser.add_argument("--conformal-tree-bins", type=int, default=9)
    parser.add_argument("--conformal-tree-min-gain", type=float, default=1e-6)
    parser.add_argument(
        "--delta-mode",
        type=str,
        choices=["quantile", "max", "mean_std"],
        default="quantile",
    )
    parser.add_argument("--delta-quantile", type=float, default=0.95)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--delta-local", action="store_true")
    parser.add_argument("--delta-local-k", type=int, default=20)
    parser.add_argument("--delta-local-quantile", type=float, default=None)
    parser.add_argument("--delta-local-samples", type=int, default=500)
    parser.add_argument("--horizon-samples", type=int, default=None)
    parser.add_argument("--horizon-consecutive-k", type=int, default=2)
    parser.add_argument("--horizon-feature-horizon", type=int, default=3)
    parser.add_argument("--horizon-thin", type=int, default=1)
    parser.add_argument("--horizon-calib-thin", type=int, default=1)
    parser.add_argument("--scale-eps", type=float, default=1e-6)
    parser.add_argument("--scale-floor", type=float, default=1e-3)
    parser.add_argument("--scale-cap", type=float, default=None)
    parser.add_argument(
        "--scale-from-quantiles",
        dest="scale_from_quantiles",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-scale-from-quantiles",
        dest="scale_from_quantiles",
        action="store_false",
    )
    parser.add_argument("--scale-quantile-high", type=float, default=0.9)
    parser.add_argument("--scale-cap-quantile", type=float, default=0.95)
    parser.add_argument("--scale-floor-quantile", type=float, default=0.1)
    parser.add_argument("--quantile-ensemble", type=int, default=3)
    parser.add_argument("--quantile-ensemble-stride", type=int, default=1000)
    parser.add_argument("--block-count", type=int, default=5)
    parser.add_argument("--block-quantile", type=float, default=0.9)
    parser.add_argument("--coverage-guard-quantile", type=float, default=None)
    parser.add_argument("--coverage-guard-margin", type=float, default=0.02)
    parser.add_argument("--coverage-guard-min-scale", type=float, default=0.0)
    parser.add_argument("--debias-scale", type=float, default=0.0)
    parser.add_argument("--debias-quantile", type=float, default=None)
    parser.add_argument(
        "--offset-calibration",
        dest="offset_calibration",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-offset-calibration",
        dest="offset_calibration",
        action="store_false",
    )
    parser.add_argument("--offset-quantile", type=float, default=None)
    parser.add_argument(
        "--horizon-use-jacobian",
        dest="horizon_use_jacobian",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-horizon-jacobian",
        dest="horizon_use_jacobian",
        action="store_false",
    )
    parser.add_argument(
        "--growth-source",
        type=str,
        choices=["state", "error", "jacobian"],
        default="error",
    )
    parser.add_argument("--expansion-quantile", type=float, default=0.95)
    parser.add_argument("--expansion-samples", type=int, default=500)
    parser.add_argument("--expansion-theiler", type=int, default=10)
    parser.add_argument("--expansion-dim", type=int, default=None)
    parser.add_argument("--expansion-lag", type=int, default=None)
    parser.add_argument("--expansion-horizon", type=int, default=10)
    parser.add_argument("--calibrate-coverage", action="store_true")
    parser.add_argument("--calibration-alpha", type=float, default=0.1)
    parser.add_argument("--calibration-floor", type=float, default=1.0)

    parser.add_argument("--lyap-max-t", type=int, default=25)
    parser.add_argument("--lyap-theiler", type=int, default=10)
    parser.add_argument("--lyap-fit-start", type=int, default=1)
    parser.add_argument("--lyap-fit-end", type=int, default=10)
    parser.add_argument("--lyap-dim", type=int, default=None)
    parser.add_argument("--lyap-lag", type=int, default=None)

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", type=str, default="horizon")
    parser.add_argument("--predictability-map", action="store_true", default=False)
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")

    parser.add_argument("--r", type=float, default=4.0)
    parser.add_argument("--x0", type=float, default=0.2)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--integrator",
        type=str,
        choices=["euler", "rk4"],
        default="euler",
    )
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)

    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=5.7)

    parser.add_argument("--tau", type=int, default=17)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    return parser


def load_config(config_path):
    """Loads configuration from a YAML file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir, log_file="experiment.log"):
    """Sets up logging to console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, log_file)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging configured. Output: {log_path}")


def main():
    """CLI entry point."""
    from src.horizon_experiment import run_experiment
    
    # 1. Parse ONLY --config first to get the path
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    known_args, _ = pre_parser.parse_known_args()
    
    # 2. Load YAML config
    config = load_config(known_args.config)
    
    # 3. Build main parser and set defaults from config
    parser = build_parser()
    parser.set_defaults(**config)
    
    # 4. Parse full args (CLI overrides config defaults)
    args = parser.parse_args()
    
    # 5. Setup Logging
    setup_logging(args.output_dir)
    logging.info(f"Loaded configuration from {known_args.config}")
    
    run_experiment(args)


if __name__ == "__main__":
    main()
