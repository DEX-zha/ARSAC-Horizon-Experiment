# ARSAC Horizon Experiment

This repo estimates **prediction horizons** for chaotic time series (Lorenz / Rössler / Mackey‑Glass).
The goal is a **conservative lower bound** L(x) such that:

**P(H_window >= L(x)) >= 1 − α**

while keeping L(x) as tight as possible.

Use cases:
- quantify how far a one‑step model can be trusted on chaotic systems
- compare stability across seeds / regimes
- produce calibrated lower bounds for deployment and safety checks

What’s implemented:
- **Per‑window horizon labels (H_w)** instead of global RMSE thresholds
- **Quantile regression** for a conservative base estimate
- **One‑sided conformal calibration** with signed scores
- **Heteroscedastic normalization** (σ from quantile spread)
- **CV+ cross‑fitting** and **Mondrian bins** (deterministic + shrinkage)

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

You can switch the ODE integrator with:
```
./venv/bin/python -m src.horizon_experiment --dataset lorenz --integrator rk4
```
For Mackey‑Glass, RK4 uses the same delayed value within the step (approximation).
**Note:** Lorenz/Rössler currently use SciPy RK45 internally; the `--integrator`
flag only affects Mackey‑Glass in the current codebase.

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

## Performance (latest runs)

**Metrics**
- **Coverage** = mean(H_w >= L_cal) on test (target ≥ 1 − α)
- **Tightness** = median(L_cal) / median(H_w) (closer to 1 is tighter)
- **Slack** = H_w − L_cal (median + P90)

### Profile A — Standard (Lorenz, RK4, α=0.10)
Settings: CV+ (5 folds), bins=2, horizon_quantile=0.15, block_quantile=0.99,
series_len=10000, calib_ratio=0.1.  
Source: `outputs/horizon_conformal_runs_cv5_adv_rk4.csv`

**Summary (5 seeds):**
- Coverage median **0.9368** (min 0.9127, max 0.9459)
- Tightness median **0.8705** (min 0.6979, max 0.9191)
- Slack median **1.09**, Slack P90 median **11.86**

### Profile B — Critical/Finance‑like (Lorenz, RK4, α=0.05)
Settings: CV+ (5 folds), bins=2, horizon_quantile=0.05, block_quantile=0.995,
series_len=20000, calib_ratio=0.15, **no offset calibration**.  
Source: `outputs/horizon_conformal_runs_critical_rk4.csv`

**Summary (5 seeds):**
- Coverage median **0.9568** (min 0.9463, max 0.9688)
- Tightness median **0.8380** (min 0.8075, max 0.9017)
- Slack median **1.33**, Slack P90 median **11.26**

**Interpretation**
- Coverage is **stable and high** in both profiles, especially in critical mode.
- Tightness is **decent but still variable** across seeds (more in standard mode).
- P90 slack is still heavy‑tailed → conservative in worst‑case regimes.

## Reproduce (recommended)

Standard Lorenz (RK4):
```
python -m src.horizon_conformal_eval \
  --datasets lorenz --models mlp --seeds 0,1,2,3,4 \
  --series-len 10000 --warmup 500 --calib-ratio 0.1 \
  --horizon-max 60 --horizon-samples 800 \
  --conformal-cv-folds 5 --horizon-quantile 0.15 --conformal-bins 2 \
  --block-quantile 0.99 --integrator rk4 \
  --no-progress \
  --output-runs outputs/horizon_conformal_runs_cv5_adv_rk4.csv \
  --output-summary outputs/horizon_conformal_summary_cv5_adv_rk4.csv
```

Critical Lorenz (RK4, finance‑like):
```
python -m src.horizon_conformal_eval \
  --datasets lorenz --models mlp --seeds 0,1,2,3,4 \
  --series-len 20000 --warmup 500 --train-ratio 0.7 --val-ratio 0.1 --calib-ratio 0.15 \
  --horizon-max 60 --horizon-samples 1200 \
  --conformal-cv-folds 5 --horizon-quantile 0.05 --calibration-alpha 0.05 \
  --conformal-bins 2 --block-quantile 0.995 --scale-cap-quantile 0.99 \
  --no-offset-calibration --integrator rk4 \
  --no-progress \
  --output-runs outputs/horizon_conformal_runs_critical_rk4.csv \
  --output-summary outputs/horizon_conformal_summary_critical_rk4.csv
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
