"""AI-driven prediction horizon experiment for chaotic time series."""

import time

from src.horizon_cli import build_parser, main  # noqa: F401
from src.horizon_conformal import ConformalTreeEstimator, conformal_quantile  # noqa: F401
from src.horizon_experiment_conformal import _run_conformal
from src.horizon_experiment_core import _experiment_setup
from src.horizon_experiment_io import _build_return, _log_summary, _maybe_plot, _write_csv
from src.horizon_experiment_probabilistic import _run_probabilistic
from src.horizon_models import LinearAR  # noqa: F401
from src.horizon_progress import ProgressBar  # noqa: F401
from src.horizon_training import train_lstm, train_mlp  # noqa: F401


def run_experiment(args):
    """Runs a full horizon experiment and writes summary CSV output."""
    t0 = time.time()
    ctx, base, lyap, stats, exp_dim, exp_lag, dt = _experiment_setup(args)
    if args.bound_mode == "horizon_conformal":
        stats = _run_conformal(ctx, base, lyap, stats, dt)
    else:
        stats = _run_probabilistic(ctx, base, lyap, stats, dt)
    _write_csv(args, ctx.best, base, lyap, stats, exp_dim, exp_lag)
    _maybe_plot(args, base, lyap, stats)
    elapsed = time.time() - t0
    _log_summary(args, ctx.best, base, lyap, stats, elapsed)
    return _build_return(ctx.best, base, lyap, stats, args, exp_dim, exp_lag)


if __name__ == "__main__":
    main()
