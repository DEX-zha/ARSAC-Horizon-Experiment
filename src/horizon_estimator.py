"""High-level API: calibrated lower bounds on prediction horizons (Plan V2 Phase 5).

MVP facade over the chaos pipeline for USER-SUPPLIED series::

    from src.horizon_estimator import HorizonEstimator

    est = HorizonEstimator(model="mlp", alpha=0.1, tolerance=0.4)
    est.fit(my_series)                    # 1-D array-like, >= ~1000 points
    est.lower_bounds_                     # per-test-window lower bounds L(x)
    est.coverage_                         # empirical P(H_w >= L) on test
    print(est.report())

Semantics and caveats (see docs/THEORY.md):
- ``tolerance`` is in standardized units (fraction of the series std): the
  "valid time" convention. The bound is EMPIRICALLY calibrated (level (c) of
  THEORY.md); windows overlap, so exchangeability is approximate.
- ``horizon_max``: with unknown dynamics there is no Lyapunov reference, so
  the auto default is a conservative 50 steps; pass an explicit value sized
  to your data (the ``label_identified`` diagnostic flags a too-short window).
- The forecaster is internal (linear / mlp / lstm). Bringing your own model
  requires the full pipeline API (deferred, Plan V2 Phase 5 next step).
"""

from __future__ import annotations

import numpy as np

from src.horizon_cli import build_parser
from src.horizon_experiment import run_experiment


class HorizonEstimator:
    """Calibrated lower bounds on the prediction horizon of a time series."""

    def __init__(
        self,
        model="mlp",
        alpha=0.05,
        tolerance=0.4,
        horizon_max=None,
        dt=None,
        use_cuda=False,
        output_dir="outputs_estimator",
        **overrides,
    ):
        self.model = model
        self.alpha = float(alpha)
        self.tolerance = float(tolerance)
        self.horizon_max = horizon_max
        self.dt = dt
        self.use_cuda = bool(use_cuda)
        self.output_dir = output_dir
        self.overrides = overrides
        self.result_ = None

    def _build_args(self, series):
        args = build_parser().parse_args([])
        args.dataset = "custom"
        args.custom_series = np.asarray(series, dtype=np.float64).reshape(-1)
        args.series_len = int(args.custom_series.size)
        args.model = self.model
        args.calibration_alpha = self.alpha
        args.error_mode = "absolute"
        args.error_tolerance = self.tolerance
        args.horizon_max = self.horizon_max
        args.dt = self.dt
        args.bound_mode = "horizon_conformal"
        args.export_bounds = True
        # Library defaults: a real calibration split (CLI default is 0.05).
        args.train_ratio = 0.6
        args.val_ratio = 0.15
        args.calib_ratio = 0.15
        args.use_cuda = self.use_cuda
        args.progress = False
        args.output_dir = self.output_dir
        for key, value in self.overrides.items():
            if not hasattr(args, key):
                raise TypeError(f"Unknown pipeline option: {key!r}")
            setattr(args, key, value)
        return args

    def fit(self, series):
        """Runs the full pipeline (train -> label -> quantile -> conformal)."""
        args = self._build_args(series)
        self.result_ = run_experiment(args)
        self.horizon_max_ = int(args.horizon_max)
        self.lower_bounds_ = np.asarray(
            self.result_.get("l_test_values") or [], dtype=np.float64
        )
        self.test_horizons_ = np.asarray(
            self.result_.get("h_test_values") or [], dtype=np.float64
        )
        self.coverage_ = self.result_.get("coverage_test")
        self.tightness_ = self.result_.get("tightness_ratio")
        self.horizon_certified_ = self.result_.get("horizon_certified")
        self.label_identified_ = self.result_.get("label_identified", True)
        return self

    def report(self):
        """Key diagnostics as a plain dict (see docs/THEORY.md for semantics)."""
        if self.result_ is None:
            raise RuntimeError("call fit(series) first")
        r = self.result_
        return {
            "alpha": self.alpha,
            "tolerance_std_units": self.tolerance,
            "horizon_max": self.horizon_max_,
            "coverage_test": self.coverage_,
            "tightness": self.tightness_,
            "slack_p90": r.get("slack_p90"),
            "lower_bound_median": (
                float(np.median(self.lower_bounds_)) if self.lower_bounds_.size else None
            ),
            "horizon_window_median": r.get("horizon_real_window_median"),
            "p_sat_test": r.get("p_sat_test"),
            "label_identified": self.label_identified_,
            "horizon_certified": self.horizon_certified_,
            "lyapunov_per_step": r.get("lyapunov_step"),
            "embedding": (r.get("dim"), r.get("lag")),
            "guarantee_level": "empirical (see docs/THEORY.md section 2c)",
        }
