"""Conformal prediction module for horizon estimation."""

import math
import numpy as np

from src.horizon_training import train_quantile_mlp, train_mlp
from src.horizon_models import TorchWrapper
from src.horizon_utils import set_seed


def conformal_quantile(scores, alpha, rng=None, tie_jitter=0.0):
    """Computes the (1-alpha) conformal quantile with finite-sample correction."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    if tie_jitter and tie_jitter > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        scale = float(np.std(scores))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        jitter = tie_jitter * scale
        if jitter > 0.0:
            scores = scores + rng.uniform(0.0, jitter, size=scores.shape)
    n = scores.size
    rank = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    rank = max(0, min(rank, n - 1))
    return float(np.sort(scores)[rank])


def block_conformal_margin(
    scores,
    alpha,
    block_count,
    block_quantile=0.9,
    rng=None,
    tie_jitter=0.0,
):
    """Computes a conservative margin using a high quantile of block margins."""
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0
    block_count = max(1, int(block_count))
    if block_count <= 1 or scores.size <= 1:
        return conformal_quantile(scores, alpha, rng=rng, tie_jitter=tie_jitter)
    block_count = min(block_count, scores.size)
    block_size = max(1, scores.size // block_count)
    margins = []
    for i in range(block_count):
        start = i * block_size
        end = scores.size if i == block_count - 1 else start + block_size
        margins.append(
            conformal_quantile(scores[start:end], alpha, rng=rng, tie_jitter=tie_jitter)
        )
    if not margins:
        return conformal_quantile(scores, alpha, rng=rng, tie_jitter=tie_jitter)
    block_quantile = float(min(max(block_quantile, 0.0), 1.0))
    return float(np.quantile(np.asarray(margins, dtype=np.float64), block_quantile))


def compute_bin_edges(values, bins):
    """Computes quantile-based bin edges with infinite caps."""
    values = np.asarray(values, dtype=np.float64)
    bins = max(1, int(bins))
    if values.size == 0 or bins <= 1:
        return np.array([-np.inf, np.inf], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = edges.astype(np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bin_ids(features, edges_list):
    """Assigns deterministic bin ids for a set of features."""
    features = np.asarray(features, dtype=np.float64)
    n = features.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), 1
    group_ids = np.zeros(n, dtype=np.int64)
    multiplier = 1
    for col, edges in enumerate(edges_list):
        bins = max(1, len(edges) - 1)
        if bins <= 1:
            bin_idx = np.zeros(n, dtype=np.int64)
        else:
            bin_idx = np.digitize(features[:, col], edges[1:-1], right=False)
        group_ids += bin_idx * multiplier
        multiplier *= bins
    return group_ids, max(1, multiplier)


class ConformalTreeEstimator:
    """Wraps Scikit-Learn DecisionTree for quantile prediction."""
    
    def __init__(self, min_samples_leaf, max_depth, min_gain=0.0):
        from sklearn.tree import DecisionTreeRegressor
        self.tree = DecisionTreeRegressor(
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            min_impurity_decrease=min_gain
        )
        self.leaf_quantiles = {}
        self.global_fallback = 0.0

    def fit(self, X, y, alpha, rng=None, tie_jitter=0.0):
        """Fit the tree and compute quantiles in each leaf."""
        self.tree.fit(X, y)
        leaf_ids = self.tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        self.global_fallback = conformal_quantile(y, alpha, rng=rng, tie_jitter=tie_jitter)

        for leaf in unique_leaves:
            mask = leaf_ids == leaf
            leaf_y = y[mask]
            self.leaf_quantiles[leaf] = conformal_quantile(
                leaf_y, alpha, rng=rng, tie_jitter=tie_jitter
            )

    def predict(self, X):
        """Predict calibrated constants for each sample."""
        leaf_ids = self.tree.apply(X)
        return np.array([self.leaf_quantiles.get(leaf_id, self.global_fallback) for leaf_id in leaf_ids])

    def apply(self, X):
        """Return leaf IDs for each sample."""
        return self.tree.apply(X)


def fit_mondrian_bins(
    features,
    scores,
    alpha,
    edges_list,
    min_bin,
    shrinkage,
    global_c,
    rng=None,
    tie_jitter=0.0,
):
    """Fits per-bin conformal constants with shrinkage toward the global constant."""
    features = np.asarray(features, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    group_ids, group_count = assign_bin_ids(features, edges_list)
    c_groups = np.full(group_count, global_c, dtype=np.float64)
    counts = (
        np.bincount(group_ids, minlength=group_count)
        if group_ids.size
        else np.zeros(group_count, dtype=np.int64)
    )
    for gid in range(group_count):
        count = int(counts[gid])
        if count < min_bin:
            continue
        group_scores = scores[group_ids == gid]
        if group_scores.size == 0:
            continue
        c_group = conformal_quantile(
            group_scores, alpha, rng=rng, tie_jitter=tie_jitter
        )
        if shrinkage > 0.0:
            weight = count / (count + shrinkage)
            c_group = weight * c_group + (1.0 - weight) * global_c
        c_groups[gid] = c_group
    return c_groups, group_ids, counts


def extract_bin_features(x_raw, dim, mode):
    """Extracts regime features for forced Mondrian bins."""
    if x_raw.size == 0:
        return np.empty((0, 1), dtype=np.float64)
    jac_log_idx = dim + 5
    resid_idx = dim + 3
    if mode == "jac_log":
        return x_raw[:, [jac_log_idx]]
    if mode == "both":
        return np.column_stack([x_raw[:, jac_log_idx], x_raw[:, resid_idx]])
    return x_raw[:, [resid_idx]]


def predict_quantile_ensemble(
    x_train,
    y_train,
    x_val,
    y_val,
    x_calib,
    x_test,
    quantile,
    args,
    device,
    seed_base,
):
    """Trains an ensemble of quantile models and aggregates their predictions."""
    members = max(1, int(args.quantile_ensemble))
    calib_preds = []
    test_preds = []
    for m in range(members):
        set_seed(seed_base + m * args.quantile_ensemble_stride)
        model, _ = train_quantile_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=x_train.shape[1],
            quantile=quantile,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch,
            patience=args.mlp_patience,
            device=device,
            show_progress=args.progress,
        )
        wrapper = TorchWrapper(model, device)
        if x_calib.size:
            calib_preds.append(wrapper.predict_batch(x_calib).reshape(-1))
        if x_test.size:
            test_preds.append(wrapper.predict_batch(x_test).reshape(-1))
    if calib_preds:
        pred_calib = np.median(np.stack(calib_preds, axis=0), axis=0)
    else:
        pred_calib = np.array([])
    if test_preds:
        pred_test = np.median(np.stack(test_preds, axis=0), axis=0)
    else:
        pred_test = np.array([])
    return pred_calib, pred_test


def predict_sigma_quantile_ensemble(
    x_train,
    y_train,
    x_val,
    y_val,
    x_pred,
    args,
    device,
    seed_base,
):
    """Predicts sigma via quantile spread using an ensemble."""
    q_high = float(min(max(args.scale_quantile_high, 0.5), 0.99))
    med_pred, _ = predict_quantile_ensemble(
        x_train,
        y_train,
        x_val,
        y_val,
        x_pred,
        np.empty((0, x_train.shape[1]), dtype=np.float64),
        0.5,
        args,
        device,
        seed_base=seed_base,
    )
    high_pred, _ = predict_quantile_ensemble(
        x_train,
        y_train,
        x_val,
        y_val,
        x_pred,
        np.empty((0, x_train.shape[1]), dtype=np.float64),
        q_high,
        args,
        device,
        seed_base=seed_base + 500,
    )
    if med_pred.size == 0:
        return med_pred
    return np.maximum(high_pred - med_pred, 0.0)


def predict_sigma_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
    x_pred,
    args,
    device,
    seed_base,
):
    """Predicts sigma via a residual-scale MLP."""
    set_seed(seed_base)
    median_model, _ = train_quantile_mlp(
        x_train,
        y_train,
        x_val,
        y_val,
        input_dim=x_train.shape[1],
        quantile=0.5,
        hidden_dim=args.mlp_hidden,
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        batch_size=args.mlp_batch,
        patience=args.mlp_patience,
        device=device,
        show_progress=args.progress,
    )
    median_wrapper = TorchWrapper(median_model, device)
    med_train = median_wrapper.predict_batch(x_train).reshape(-1)
    med_val = (
        median_wrapper.predict_batch(x_val).reshape(-1) if x_val.size else med_train
    )
    scale_train = np.log(np.abs(y_train - med_train) + args.scale_eps)
    scale_val = (
        np.log(np.abs(y_val - med_val) + args.scale_eps) if x_val.size else scale_train
    )
    x_scale_val = x_val if x_val.size else x_train
    scale_model, _ = train_mlp(
        x_train,
        scale_train,
        x_scale_val,
        scale_val,
        input_dim=x_train.shape[1],
        hidden_dim=args.mlp_hidden,
        epochs=args.mlp_epochs,
        lr=args.mlp_lr,
        batch_size=args.mlp_batch,
        patience=args.mlp_patience,
        device=device,
        show_progress=args.progress,
    )
    scale_wrapper = TorchWrapper(scale_model, device)
    if x_pred.size:
        return np.exp(scale_wrapper.predict_batch(x_pred).reshape(-1))
    return np.array([], dtype=np.float64)


def make_contiguous_folds(n, folds):
    """Splits indices into contiguous folds."""
    folds = max(1, int(folds))
    folds = min(folds, n) if n > 0 else 1
    sizes = [n // folds] * folds
    for i in range(n % folds):
        sizes[i] += 1
    ranges = []
    start = 0
    for size in sizes:
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges
