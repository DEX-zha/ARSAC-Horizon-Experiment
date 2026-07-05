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


class _UserModelWrapper:
    """Adapts a user forecaster (object with .predict or plain callable).

    The pipeline feeds delay vectors x of shape (dim,) built from the
    STANDARDIZED series (mean/std of the train split) and expects the
    standardized next value as a float.
    """

    def __init__(self, model):
        if callable(getattr(model, "predict", None)):
            self._predict = model.predict
        elif callable(model):
            self._predict = model
        else:
            raise TypeError("user model must be callable or expose .predict(x)")

    def predict(self, x):
        return float(self._predict(np.asarray(x, dtype=np.float64)))

    def predict_batch(self, xs):
        xs = np.asarray(xs, dtype=np.float64)
        return np.array([self.predict(x) for x in xs], dtype=np.float64)


class HorizonEstimator:
    """Calibrated lower bounds on the prediction horizon of a time series.

    ``model`` is either the name of an internal forecaster ("linear", "mlp",
    "lstm") or YOUR OWN forecaster (callable or object with .predict) — in
    that case pass ``dim`` (and optionally ``lag``): the input your model
    expects is the delay vector (x_{t-(dim-1)lag}, ..., x_t) of the
    standardized series, and the output the standardized next value.
    """

    def __init__(
        self,
        model="mlp",
        alpha=0.05,
        tolerance=0.4,
        horizon_max=None,
        dt=None,
        dim=None,
        lag=1,
        use_cuda=False,
        output_dir="outputs_estimator",
        **overrides,
    ):
        self.model = model
        self.alpha = float(alpha)
        self.tolerance = float(tolerance)
        self.horizon_max = horizon_max
        self.dt = dt
        self.dim = dim
        self.lag = lag
        self.use_cuda = bool(use_cuda)
        self.output_dir = output_dir
        self.overrides = overrides
        self.result_ = None

    def _build_args(self, series):
        args = build_parser().parse_args([])
        args.dataset = "custom"
        args.custom_series = np.asarray(series, dtype=np.float64).reshape(-1)
        args.series_len = int(args.custom_series.size)
        if isinstance(self.model, str):
            args.model = self.model
        else:
            if self.dim is None:
                raise TypeError("bring-your-own model requires dim= (delay vector length)")
            args.model = "linear"  # unused placeholder for arg validation paths
            args.user_model = _UserModelWrapper(self.model)
            args.user_dim = int(self.dim)
            args.user_lag = int(self.lag)
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
        self._compute_r_diagnostic()
        self._compute_noise_floor(args.custom_series, args.train_ratio)
        return self

    def _compute_noise_floor(self, raw, train_ratio):
        """Noise-aware reachable floor: separates noise from model deficit.

        sigma_obs is estimated by local-linear residuals (src/horizon_noise,
        validated on known synthetic noise); the reachable horizon under that
        noise follows the one-shot floor law H = ln(tau/sigma)/(lambda*dt)
        validated by paired twins (docs/theory/chaos_floor.md). margin_real_
        = H_reachable / median(H_w): how much horizon a PERFECT model of the
        dynamics could still add given the noise in YOUR data.
        """
        from src.horizon_noise import estimate_observation_noise, reachable_horizon_steps

        self.sigma_obs_ = None
        self.h_reachable_ = None
        self.margin_real_ = None
        try:
            raw = np.asarray(raw, dtype=np.float64).reshape(-1)
            i_train = max(100, int(train_ratio * raw.size))
            mean, sd = raw[:i_train].mean(), raw[:i_train].std()
            if sd <= 0:
                return
            sig_hat, _ = estimate_observation_noise((raw - mean) / sd, dim=6, lag=1, seed=0)
            lam_step = self.result_.get("lyapunov_step")
            if not np.isfinite(sig_hat):
                return
            self.sigma_obs_ = float(sig_hat)
            h_reach = reachable_horizon_steps(self.tolerance, sig_hat, lam_step)
            self.h_reachable_ = float(min(h_reach, 50 * self.horizon_max_))
            h_med = self.result_.get("horizon_real_window_median")
            if h_med and h_med > 0 and np.isfinite(h_reach):
                self.margin_real_ = float(h_reach / h_med)
        except Exception:  # diagnostic must never break a fit
            return

    def _compute_r_diagnostic(self):
        """R = Lambda_eff/lambda_1: distance of the model to the chaos floor.

        Validated on Lorenz against paired physical twins
        (docs/theory/chaos_floor.md): R ~ 1 means the forecaster has saturated
        the system's physical predictability (improving the model cannot buy
        more horizon); R >> 1 means the horizon is model-limited (improving
        the model CAN buy ~x(R) more horizon, logarithmically in precision).
        """
        self.R_median_ = None
        self.chaos_limited_ = None
        e0 = np.asarray(self.result_.get("e0_test_values") or [], dtype=np.float64)
        lam_step = self.result_.get("lyapunov_step")
        if not e0.size or e0.size != self.test_horizons_.size or not lam_step or lam_step <= 0:
            return
        h = self.test_horizons_
        keep = (e0 > 0) & (e0 < self.tolerance / 4.0) & (h < self.horizon_max_)
        if keep.sum() < 20:
            return
        r = np.log(self.tolerance / e0[keep]) / (h[keep] * float(lam_step))
        self.R_median_ = float(np.median(r))
        self.chaos_limited_ = bool(self.R_median_ < 2.0)

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
            "R_distance_to_chaos_floor": self.R_median_,
            "chaos_limited": self.chaos_limited_,
            "sigma_obs_std_units": self.sigma_obs_,
            "H_reachable_given_noise": self.h_reachable_,
            "margin_real": self.margin_real_,
            "R_reading": self._reading(),
            "guarantee_level": "empirical (see docs/THEORY.md section 2c)",
        }

    def _reading(self):
        """One actionable sentence combining R and the noise-aware margin."""
        if self.R_median_ is None:
            return None
        if self.chaos_limited_:
            return ("model at the physical predictability floor: a better model "
                    "cannot buy more horizon; invest in better measurements/state "
                    "estimation")
        base = (f"horizon is model-limited (error grows ~{self.R_median_:.1f}x "
                "faster than the deterministic chaos floor)")
        if self.margin_real_ is None:
            return base + "; improving the model can extend the horizon"
        if self.margin_real_ < 1.5:
            return (base + f"; BUT the estimated observation noise "
                    f"(sigma~{self.sigma_obs_:.2g} std) caps the reachable horizon "
                    f"at ~{self.margin_real_:.1f}x the current one: the margin is "
                    "mostly noise, not model deficit")
        return (base + f"; accounting for the estimated noise "
                f"(sigma~{self.sigma_obs_:.2g} std), a better model could still "
                f"reach ~{self.margin_real_:.1f}x the current horizon")
