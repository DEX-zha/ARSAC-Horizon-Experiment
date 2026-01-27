"""Conformal evaluation stats for horizon_experiment."""

from __future__ import annotations

import numpy as np

from src.horizon_experiment_core import _clip_horizon, _const, _horizon_time
from src.horizon_experiment_conformal_calibration import _conformal_c_test, _score_quantiles


def _test_predictions(pred_test, c_test, sigma_test, use_sigma, args, constants):
    sigma_term = sigma_test if use_sigma else np.ones_like(pred_test)
    return _clip_horizon(pred_test - c_test * sigma_term, args, constants)



def _horizon_point_stats(pred_test, pred_test_cal):
    if not pred_test.size:
        return 0.0, 0.0, 0.0
    return float(np.median(pred_test)), float(np.mean(pred_test)), float(np.median(pred_test_cal))



def _window_stats(y_test):
    if not y_test.size:
        return None, None
    return float(np.median(y_test)), float(np.mean(y_test))



def _coverage_stats(y_test, pred_test_cal, constants):
    if not y_test.size or not pred_test_cal.size:
        return None, None, None, None
    slack = y_test - pred_test_cal
    slack_q = float(_const(constants, "slack_quantile"))
    tightness = float(np.median(pred_test_cal) / np.median(y_test)) if np.median(y_test) > 0.0 else None
    return float(np.mean(y_test >= pred_test_cal)), tightness, float(np.median(slack)), float(np.quantile(slack, slack_q))



def _jacobian_coverages(pred_test_cal, y_test, x_test_raw, best, constants):
    if not pred_test_cal.size or not x_test_raw.size or not y_test.size:
        return None
    bins = int(_const(constants, "jacobian_bins"))
    jac_values = x_test_raw[:, best["dim"] + 2]
    edges = np.quantile(jac_values, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    jac_bins = np.digitize(jac_values, edges[1:-1], right=False)
    out = {}
    for b in range(bins):
        mask = jac_bins == b
        out[f"jac_q{b + 1}"] = float(np.mean(y_test[mask] >= pred_test_cal[mask])) if np.any(mask) else None
    return out



def _leaf_coverage_stats(leaf_ids, y_test, pred_test_cal, constants):
    if pred_test_cal.size == 0 or leaf_ids is None or y_test.size == 0:
        return None
    coverages = [
        float(np.mean(y_test[leaf_ids == leaf] >= pred_test_cal[leaf_ids == leaf]))
        for leaf in np.unique(leaf_ids)
        if np.any(leaf_ids == leaf)
    ]
    if not coverages:
        return None
    coverages = np.asarray(coverages, dtype=np.float64)
    q10, _, _ = _score_quantiles(constants)
    return {"leaf_count": int(coverages.size), "leaf_min": float(np.min(coverages)), "leaf_p10": float(np.quantile(coverages, q10)), "leaf_med": float(np.median(coverages)), "leaf_mean": float(np.mean(coverages))}



def _test_conformal(ctx, sets, preds, model, use_sigma, constants, stats):
    if not preds.pred_test.size:
        return np.array([], dtype=np.float64), None
    c_test, leaf_ids = _conformal_c_test(model, preds, sets, constants, ctx)
    pred_test_cal = _test_predictions(preds.pred_test, c_test, preds.sigma_test, use_sigma, ctx.args, constants)
    h_model, h_est, h_cal = _horizon_point_stats(preds.pred_test, pred_test_cal)
    stats["horizon_model_steps"] = h_model
    stats["horizon_est_steps"] = h_est
    stats["horizon_model_cal"] = h_cal
    stats["horizon_window_median"], stats["horizon_window_mean"] = _window_stats(sets.y_test)
    coverage, tightness, slack_med, slack_p90 = _coverage_stats(sets.y_test, pred_test_cal, constants)
    stats["coverage_test"] = coverage
    stats["tightness_ratio"] = tightness
    stats["slack_median"] = slack_med
    stats["slack_p90"] = slack_p90
    return pred_test_cal, leaf_ids



def _extra_conformal_stats(pred_test_cal, leaf_ids, sets, constants, stats, best):
    stats["jac_quantile_coverages"] = _jacobian_coverages(
        pred_test_cal, sets.y_test, sets.x_test_raw, best, constants
    )
    stats["leaf_coverage_stats"] = _leaf_coverage_stats(
        leaf_ids, sets.y_test, pred_test_cal, constants
    )



def _finalize_conformal(stats, dt, args):
    stats["horizon_model_time"] = _horizon_time(stats["horizon_model_steps"], dt)
    stats["horizon_est_time"] = _horizon_time(stats["horizon_est_steps"], dt)
    stats["horizon_model_cal_time"] = _horizon_time(stats["horizon_model_cal"], dt)
    stats["model_error_mode"] = f"conformal_{args.conformal_mode}"
    stats["scale"] = stats["c_global"]
    stats["growth_source"] = "conformal"
    return stats


