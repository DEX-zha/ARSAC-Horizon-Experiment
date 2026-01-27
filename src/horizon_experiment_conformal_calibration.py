"""Conformal calibration helpers for horizon_experiment."""

from __future__ import annotations

import numpy as np

from src.horizon_conformal import (
    ConformalTreeEstimator,
    assign_bin_ids,
    block_conformal_margin,
    compute_bin_edges,
    extract_bin_features,
    fit_mondrian_bins,
)
from src.horizon_experiment_core import (
    ConformalModel,
    _clip_horizon,
    _const,
    _seed_offset,
)


def _compute_scores(pred_calib, y_calib, sigma_calib, use_sigma, args):
    signed = pred_calib - y_calib
    if use_sigma:
        scores = signed / np.maximum(sigma_calib, args.scale_floor)
    else:
        scores = signed
    return scores, signed



def _score_quantiles(constants):
    q = _const(constants, "score_quantiles")
    return float(q[0]), float(q[1]), float(q[2])



def _score_distribution_stats(scores, constants):
    if not scores.size:
        return {}
    q10, q50, q90 = _score_quantiles(constants)
    return {
        "score_pos_frac": float(np.mean(scores > 0.0)),
        "score_neg_frac": float(np.mean(scores < 0.0)),
        "score_zero_frac": float(np.mean(scores == 0.0)),
        "score_p10": float(np.quantile(scores, q10)),
        "score_p50": float(np.median(scores)),
        "score_p90": float(np.quantile(scores, q90)),
        "score_mean": float(np.mean(scores)),
    }



def _score_signal_stats(signed, pred_calib, y_calib, sigma_calib, use_sigma, constants):
    stats = {}
    if signed.size:
        stats["signed_med"] = float(np.median(signed))
    if pred_calib.size:
        stats["pred_calib_med"] = float(np.median(pred_calib))
    if y_calib.size:
        stats["y_calib_med"] = float(np.median(y_calib))
    if use_sigma and sigma_calib.size:
        _, _, q90 = _score_quantiles(constants)
        stats["sigma_med"] = float(np.median(sigma_calib))
        stats["sigma_p90"] = float(np.quantile(sigma_calib, q90))
        stats["sigma_max"] = float(np.max(sigma_calib))
    return stats



def _global_margin(scores, args, rng):
    return block_conformal_margin(
        scores,
        args.calibration_alpha,
        args.block_count,
        block_quantile=args.block_quantile,
        rng=rng,
        tie_jitter=args.conformal_tie_jitter,
    )



def _feature_indices(best_dim):
    return {
        "jac": best_dim + 2,
        "jac_log": best_dim + 5,
        "resid": best_dim + 3,
        "err_var": best_dim + 6,
        "pred_var": best_dim + 7,
    }



def _safe_feature_column(x_raw, idx, ref):
    return x_raw[:, idx] if x_raw.size else np.zeros_like(ref)



def _tree_features(x_raw, pred, sigma, indices):
    return np.column_stack(
        [
            pred,
            sigma,
            _safe_feature_column(x_raw, indices["jac"], pred),
            _safe_feature_column(x_raw, indices["jac_log"], pred),
            _safe_feature_column(x_raw, indices["resid"], pred),
            _safe_feature_column(x_raw, indices["err_var"], pred),
            _safe_feature_column(x_raw, indices["pred_var"], pred),
        ]
    )



def _min_leaf_eff(args, scores_size, constants):
    floor = int(_const(constants, "tree_min_leaf_floor"))
    if scores_size:
        return min(args.conformal_min_leaf, max(floor, int(scores_size // 4)))
    return args.conformal_min_leaf



def _tree_estimator(args, scores_size, constants):
    return ConformalTreeEstimator(
        min_samples_leaf=_min_leaf_eff(args, scores_size, constants),
        max_depth=args.conformal_tree_depth,
        min_gain=args.conformal_tree_min_gain,
    )



def _fit_tree_conformal(pred_calib, sigma_calib, scores, x_calib_raw, args, best, constants, rng):
    indices = _feature_indices(best["dim"])
    features = _tree_features(x_calib_raw, pred_calib, sigma_calib, indices)
    tree = _tree_estimator(args, scores.size, constants)
    tree.fit(features, scores, args.calibration_alpha, rng=rng, tie_jitter=args.conformal_tie_jitter)
    return tree, tree.predict(features)



def _predict_tree_c(tree, pred_test, sigma_test, x_test_raw, best):
    indices = _feature_indices(best["dim"])
    features = _tree_features(x_test_raw, pred_test, sigma_test, indices)
    return tree.predict(features), tree.apply(features)



def _bin_features(args, x_raw, best):
    return extract_bin_features(x_raw, best["dim"], args.conformal_bin_feature)



def _bin_pool(bin_features_train, bin_features_calib):
    if bin_features_train.size and bin_features_calib.size:
        return np.vstack([bin_features_train, bin_features_calib])
    return bin_features_calib if bin_features_calib.size else bin_features_train



def _bin_edges_list(bin_pool, bins):
    bin_dim = bin_pool.shape[1] if bin_pool.ndim == 2 else 1
    return [compute_bin_edges(bin_pool[:, col], bins) for col in range(bin_dim)]



def _bin_stats(bin_counts, c_groups):
    if not bin_counts.size:
        return {}
    nonzero = bin_counts[bin_counts > 0]
    if not nonzero.size:
        return {}
    stats = {"bin_count": int(nonzero.size), "bin_min_count": int(np.min(nonzero)), "bin_med_count": float(np.median(nonzero))}
    c_used = c_groups[bin_counts > 0]
    if c_used.size:
        stats.update({"bin_c_min": float(np.min(c_used)), "bin_c_med": float(np.median(c_used)), "bin_c_max": float(np.max(c_used))})
    return stats



def _fit_bin_conformal(pred_calib, scores, x_train_raw, x_calib_raw, args, best, constants, rng, c_global):
    bin_train = _bin_features(args, x_train_raw, best)
    bin_calib = _bin_features(args, x_calib_raw, best)
    bin_pool = _bin_pool(bin_train, bin_calib)
    edges_list = _bin_edges_list(bin_pool, args.conformal_bins)
    c_groups, bin_ids, bin_counts = fit_mondrian_bins(
        bin_calib, scores, args.calibration_alpha, edges_list, args.conformal_min_bin,
        args.conformal_bin_shrinkage, c_global, rng=rng, tie_jitter=args.conformal_tie_jitter,
    )
    bin_model = {"edges": edges_list, "c_groups": c_groups, "counts": bin_counts}
    c_calib = c_groups[bin_ids] if bin_ids.size else np.full_like(pred_calib, c_global)
    return bin_model, c_calib, _bin_stats(bin_counts, c_groups)



def _predict_bin_c(bin_model, pred_test, x_test_raw, args, best, c_global):
    bin_features_test = _bin_features(args, x_test_raw, best)
    bin_ids, _ = assign_bin_ids(bin_features_test, bin_model["edges"])
    c_test = bin_model["c_groups"][bin_ids] if bin_ids.size else np.full_like(pred_test, c_global)
    return c_test, bin_ids



def _calib_interval(pred_calib, c_calib, sigma_calib, use_sigma, args, constants):
    sigma_term = sigma_calib if use_sigma else np.ones_like(c_calib)
    return _clip_horizon(pred_calib - c_calib * sigma_term, args, constants)



def _calib_stats(l_calib, y_calib):
    if not l_calib.size:
        return None, None, 0
    coverage = float(np.mean(y_calib >= l_calib)) if y_calib.size else None
    return float(np.median(l_calib)), coverage, int(len(y_calib))



def _fit_conformal_model(ctx, sets, preds, scores, use_sigma, constants, rng, stats):
    if ctx.args.conformal_mode == "tree":
        tree, c_calib = _fit_tree_conformal(preds.pred_calib, preds.sigma_calib, scores, sets.x_calib_raw, ctx.args, ctx.best, constants, rng)
        return ConformalModel("tree", stats["c_global"], tree=tree), c_calib
    if ctx.args.conformal_mode == "bins":
        bin_model, c_calib, bin_stats = _fit_bin_conformal(preds.pred_calib, scores, sets.x_train_raw, sets.x_calib_raw, ctx.args, ctx.best, constants, rng, stats["c_global"])
        stats.update(bin_stats)
        return ConformalModel("bins", stats["c_global"], bin_model=bin_model), c_calib
    c_calib = np.full_like(preds.pred_calib, stats["c_global"])
    return ConformalModel("global", stats["c_global"]), c_calib



def _conformal_c_test(model, preds, sets, constants, ctx):
    if model.mode == "tree" and model.tree is not None:
        c_test, leaf_ids = _predict_tree_c(model.tree, preds.pred_test, preds.sigma_test, sets.x_test_raw, ctx.best)
        return c_test, leaf_ids
    if model.mode == "bins" and model.bin_model is not None:
        return _predict_bin_c(model.bin_model, preds.pred_test, sets.x_test_raw, ctx.args, ctx.best, model.c_global)
    return np.full_like(preds.pred_test, model.c_global), None



def _calibrate_conformal(ctx, sets, preds, use_sigma, constants, stats):
    scores, signed = _compute_scores(preds.pred_calib, sets.y_calib, preds.sigma_calib, use_sigma, ctx.args)
    stats.update(_score_distribution_stats(scores, constants))
    stats.update(_score_signal_stats(signed, preds.pred_calib, sets.y_calib, preds.sigma_calib, use_sigma, constants))
    rng = np.random.default_rng(_seed_offset(ctx.args.seed, constants, "tie_rng"))
    stats["c_global"] = _global_margin(scores, ctx.args, rng)
    model, c_calib = _fit_conformal_model(ctx, sets, preds, scores, use_sigma, constants, rng, stats)
    l_calib = _calib_interval(preds.pred_calib, c_calib, preds.sigma_calib, use_sigma, ctx.args, constants)
    l_med, coverage, samples = _calib_stats(l_calib, sets.y_calib)
    stats["l_calib_med"] = l_med
    stats["coverage"] = coverage
    stats["calibration_samples"] = samples
    return model


