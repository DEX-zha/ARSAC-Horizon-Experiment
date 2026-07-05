"""Conformal pipeline orchestration for horizon_experiment."""

from __future__ import annotations

from src.horizon_experiment_core import ConformalModel
from src.horizon_experiment_conformal_calibration import _calibrate_conformal
from src.horizon_experiment_conformal_data import (
    _conformal_predictions,
    _horizon_sets,
    _normalize_horizon_sets,
    _resolve_quantile,
    _use_sigma,
)
from src.horizon_experiment_conformal_stats import _extra_conformal_stats, _finalize_conformal, _test_conformal


def _run_conformal(ctx, base, lyap, stats, dt):
    calib_series = ctx.data.calib_series()
    sets = _horizon_sets(ctx, calib_series)
    feat_mean, feat_std = _normalize_horizon_sets(sets)
    use_sigma = _use_sigma(ctx.args)
    quantile = _resolve_quantile(ctx.args)
    preds = _conformal_predictions(ctx, sets, calib_series, quantile, use_sigma, ctx.constants, feat_mean, feat_std)
    model = ConformalModel(ctx.args.conformal_mode, stats["c_global"])
    if preds.pred_calib.size:
        model = _calibrate_conformal(ctx, sets, preds, use_sigma, ctx.constants, stats)
    pred_test_cal, leaf_ids = _test_conformal(ctx, sets, preds, model, use_sigma, ctx.constants, stats)
    _extra_conformal_stats(pred_test_cal, leaf_ids, sets, ctx.constants, stats, ctx.best, ctx.args)
    if getattr(ctx.args, "export_bounds", False):
        # Opt-in per-window export for the HorizonEstimator API. These keys
        # are not in CSV_HEADER, so the CSV writer ignores them; they travel
        # through the run_experiment return dict.
        stats["l_test_values"] = [float(v) for v in pred_test_cal]
        stats["h_test_values"] = [float(v) for v in sets.y_test]
    return _finalize_conformal(stats, dt, ctx.args)


