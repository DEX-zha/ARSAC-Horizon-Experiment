"""Integration tests for the theory-study wiring (P2 censoring, P4 certified, P6 embedding).

Covers: CLI flag parsing, the censored score capping, the always-on saturation
gate, the certified-horizon export (stats + CSV header), and the theory
embedding default in _lyapunov_metrics.
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

from src.horizon_cli import build_parser
from src.horizon_embedding import select_embedding
from src.horizon_experiment import run_experiment
from src.horizon_experiment_conformal_calibration import _censored_pred, _compute_scores
from src.horizon_experiment_core import DataSplits, _lyapunov_metrics
from src.horizon_experiment_io import CSV_HEADER
from src.horizon_training import quantile_cap
from src.horizon_utils import generate_logistic_map


def _fast_args(tmp_path, extra=None):
    """Small end-to-end configuration: linear model, 1200 points, Hmax=10."""
    argv = [
        "--dataset", "logistic",
        "--model", "linear",
        "--series-len", "1200",
        "--warmup", "100",
        "--horizon-max", "10",
        "--bound-mode", "horizon_conformal",
        "--conformal-mode", "global",
        "--conformal-cv-folds", "1",
        "--quantile-ensemble", "1",
        "--mlp-epochs", "2",
        "--mlp-hidden", "8",
        "--horizon-samples", "60",
        "--calib-ratio", "0.1",
        "--no-progress",
        "--output-dir", str(tmp_path),
    ]
    if extra:
        argv.extend(extra)
    args = build_parser().parse_args(argv)
    return args


def test_censored_quantile_flag_parse():
    args_off = build_parser().parse_args([])
    args_on = build_parser().parse_args(["--censored-quantile"])
    assert args_off.censored_quantile is False
    assert args_on.censored_quantile is True


def test_quantile_cap_context_sets_and_restores():
    import src.horizon_training as ht

    assert ht._DEFAULT_QUANTILE_CAP is None
    with quantile_cap(10.0):
        assert ht._DEFAULT_QUANTILE_CAP == 10.0
    assert ht._DEFAULT_QUANTILE_CAP is None
    # No-op form used when the flag is off.
    with quantile_cap(None):
        assert ht._DEFAULT_QUANTILE_CAP is None


def test_censored_pred_caps_only_when_flag_on():
    args_on = build_parser().parse_args(["--censored-quantile", "--horizon-max", "10"])
    args_off = build_parser().parse_args(["--horizon-max", "10"])
    pred = np.array([3.0, 10.0, 14.5])
    np.testing.assert_allclose(_censored_pred(pred, args_on), [3.0, 10.0, 10.0])
    np.testing.assert_allclose(_censored_pred(pred, args_off), pred)


def test_compute_scores_capped_scores_match_study_convention():
    # Study P2: signed score must be min(pred, Hmax) - y when the flag is on.
    args_on = build_parser().parse_args(["--censored-quantile", "--horizon-max", "10"])
    pred = np.array([12.0, 4.0])
    y = np.array([10.0, 6.0])
    sigma = np.ones_like(pred)
    scores, signed = _compute_scores(pred, y, sigma, False, args_on)
    np.testing.assert_allclose(signed, [0.0, -2.0])
    np.testing.assert_allclose(scores, signed)


def test_csv_header_has_certified_columns():
    for col in ("horizon_certified", "lipschitz_G", "delta_sup"):
        assert col in CSV_HEADER


def test_lyapunov_metrics_uses_theory_embedding_when_unset():
    series = generate_logistic_map(1500, r=4.0, x0=0.2)
    split = int(0.8 * len(series))
    data = DataSplits(
        train_std=series[:split], val_std=series[split:],
        calib_std=np.array([]), test_std=np.array([]),
        train_raw=series[:split], val_raw=series[split:],
        calib_raw=np.array([]), test_raw=np.array([]),
    )
    best = {"dim": 6, "lag": 7}  # deliberately unlike the theory embedding

    class A:
        lyap_dim = None
        lyap_lag = None
        lyap_max_t = 20
        lyap_theiler = 5
        lyap_fit_start = 1
        lyap_fit_end = 5

    lyap = _lyapunov_metrics(data, best, A(), base_err=0.1, tolerance=0.4, dt=1.0)
    emb = select_embedding(np.concatenate([data.train_raw, data.val_raw]))
    assert lyap.dim == int(emb["dim"])
    assert lyap.lag == int(emb["lag"])
    # Explicit args still win over the theory embedding.
    A.lyap_dim = 3
    A.lyap_lag = 2
    lyap_explicit = _lyapunov_metrics(data, best, A(), base_err=0.1, tolerance=0.4, dt=1.0)
    assert lyap_explicit.dim == 3
    assert lyap_explicit.lag == 2


def test_end_to_end_censored_flag_on(tmp_path):
    torch.manual_seed(0)
    args = _fast_args(tmp_path, extra=["--censored-quantile"])
    result = run_experiment(args)
    # P2: gate always on -> label_identified is a bool in the returned stats.
    assert isinstance(result["label_identified"], bool)
    # P4: certified diagnostics exported and sane for a linear model.
    assert result["lipschitz_G"] >= 1.0
    assert result["delta_sup"] >= 0.0
    assert result["horizon_certified"] >= 0.0
    # The calibrated bound stays inside [horizon_min, horizon_max].
    assert 0.0 <= result["horizon_model_cal"] <= args.horizon_max


def test_end_to_end_baseline_flags_off(tmp_path):
    torch.manual_seed(0)
    args = _fast_args(tmp_path)
    assert args.censored_quantile is False
    result = run_experiment(args)
    assert isinstance(result["label_identified"], bool)
    assert result["lipschitz_G"] >= 1.0
    assert "horizon_certified" in result


def test_probabilistic_path_has_gate_and_certified(tmp_path):
    torch.manual_seed(0)
    args = _fast_args(tmp_path, extra=["--bound-mode", "probabilistic"])
    result = run_experiment(args)
    assert isinstance(result["label_identified"], bool)
    assert result["lipschitz_G"] >= 1.0
    assert "delta_sup" in result
