# Horizon Experiment (standalone)

This repo studies **prediction horizons** for chaotic time series (Lorenz / Rossler).
Goal: compute a **conservative lower bound** L(x) such that
P(H_real_window >= L(x)) >= 1 - alpha, while keeping L(x) as tight as possible.

Use cases:
- quantify how far a one-step model can be trusted on chaotic systems
- compare stability across seeds / regimes
- produce calibrated lower bounds for deployment or safety checks

Key ideas implemented:
- per-window horizon labels (H_w) instead of global RMSE thresholds
- quantile regression for a conservative base estimate
- one-sided conformal calibration (signed scores)
- heteroscedastic normalization (sigma from quantile spread)
- Mondrian bins (deterministic, shrinkage) + CV+ cross-fitting

## Setup

Create a venv and install deps:

```
python -m venv venv
./venv/bin/pip install -r requirements.txt
```

GPU note (CUDA): install the correct PyTorch wheel for your CUDA version.
Example:

```
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(See https://pytorch.org/get-started/locally/ for the exact command.)

## Quick run

Single experiment:

```
./venv/bin/python -m src.horizon_experiment --dataset lorenz --use-cuda
```

Rigorous benchmark (long):

```
./venv/bin/python -m src.horizon_benchmark --use-cuda
```

You can disable the progress bar with `--no-progress`.

Probabilistic model-aware bound options (calibration + quantiles):

```
./venv/bin/python -m src.horizon_experiment \
  --calib-ratio 0.05 --delta-mode quantile --delta-quantile 0.95 \
  --expansion-quantile 0.95 --expansion-samples 500 --expansion-horizon 10 \
  --calibrate-coverage --calibration-alpha 0.1
```

The run reports three horizons:
- `horizon_model`: probabilistic bound from quantile growth and residuals.
- `horizon_est`: precision-focused estimate from mean growth and mean residuals.
- `horizon_cal`: conservative bound after coverage calibration.

Growth source options:
- default: `--growth-source error` (uses model error growth).
- fallback: `--growth-source state` (uses embedded state expansion).
- experimental: `--growth-source jacobian` (uses local Jacobian norms).

Local residual option:
- `--delta-local` enables kNN-based local residual quantiles.

Multi-step training:
- `--train-multistep` enables teacher-forcing training with `--train-horizon`.
- `--tf-start`, `--tf-end` set the linear schedule, `--tf-val` for validation.

## Recommended conformal evaluation (current best)

We found the most stable tightness across seeds with:
- CV+ cross-fitting (5 folds)
- more data (series_len=10000, calib_ratio=0.1)
- conformal bins=2 on residual regime
- horizon quantile = 0.15

**Lorenz (mlp, 5 seeds, alpha=0.1) results:**
- coverage median ~0.95
- tightness median ~0.75
- slack median ~2.0

Reproduce Lorenz:
```
python -m src.horizon_conformal_eval \
  --datasets lorenz --models mlp --seeds 0,1,2,3,4 \
  --series-len 10000 --warmup 500 --calib-ratio 0.1 \
  --horizon-max 60 --horizon-samples 800 \
  --conformal-cv-folds 5 --horizon-quantile 0.15 --conformal-bins 2 \
  --no-progress \
  --output-runs outputs/horizon_conformal_runs_cv5_long_opt.csv \
  --output-summary outputs/horizon_conformal_summary_cv5_long_opt.csv
```

Reproduce Rossler (same settings):
```
python -m src.horizon_conformal_eval \
  --datasets rossler --models mlp --seeds 0,1,2,3,4 \
  --series-len 10000 --warmup 500 --calib-ratio 0.1 \
  --horizon-max 60 --horizon-samples 800 \
  --conformal-cv-folds 5 --horizon-quantile 0.15 --conformal-bins 2 \
  --no-progress \
  --output-runs outputs/horizon_conformal_runs_cv5_long_opt_rossler.csv \
  --output-summary outputs/horizon_conformal_summary_cv5_long_opt_rossler.csv
```

## Outputs

- `outputs/horizon_benchmark.md` summary table
- `outputs/horizon_benchmark_runs.csv` all runs (per seed)
- `outputs/horizon_benchmark_table.tex` LaTeX table
- `outputs/horizon_conformal_runs_*.csv` conformal evaluation (per seed)
- `outputs/horizon_conformal_summary_*.csv` conformal evaluation summary

## Tests

```
./venv/bin/python -m unittest tests/test_horizon_utils.py
./venv/bin/python -m unittest tests/test_horizon_models.py
```
