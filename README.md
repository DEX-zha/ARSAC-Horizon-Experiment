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

## Outputs

- `outputs/horizon_benchmark.md` summary table
- `outputs/horizon_benchmark_runs.csv` all runs (per seed)
- `outputs/horizon_benchmark_table.tex` LaTeX table

## Tests

```
./venv/bin/python -m unittest tests/test_horizon_utils.py
./venv/bin/python -m unittest tests/test_horizon_models.py
```
