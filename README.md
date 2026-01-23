# Horizon Experiment (standalone)

Standalone repo for the AI-driven prediction horizon experiments on Lorenz/Rossler.

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

## Outputs

- `outputs/horizon_benchmark.md` summary table
- `outputs/horizon_benchmark_runs.csv` all runs (per seed)
- `outputs/horizon_benchmark_table.tex` LaTeX table

## Tests

```
./venv/bin/python -m unittest tests/test_horizon_utils.py
./venv/bin/python -m unittest tests/test_horizon_models.py
```
