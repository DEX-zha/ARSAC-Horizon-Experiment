# ARSAC Horizon Experiment

This repo estimates **prediction horizons** for chaotic time series (Lorenz / Rössler / Mackey‑Glass).
The goal is a **conservative lower bound** L(x) targeting:

**P(H_window >= L(x)) ≈ 1 − α** (empirical coverage)

while keeping L(x) as tight as possible.

**Honest statement of the guarantee:** the calibration provides *empirical* coverage
targeting 1 − α, valid within the calibrated regime (in-distribution). Because
horizon labels come from overlapping windows of a single time series, the
exchangeability assumption behind finite-sample conformal guarantees is violated;
no distribution-free finite-sample guarantee is claimed. See `AUDIT_MATH.md` for
the full audit and `PLAN_V2.md` for the remediation plan.

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

## Using your own data (HorizonEstimator API)

The pipeline is not limited to the built-in chaotic systems — hand any 1-D
series to the `HorizonEstimator` facade (Plan V2 Phase 5 MVP):

```python
from src.horizon_estimator import HorizonEstimator

est = HorizonEstimator(model="mlp", alpha=0.1, tolerance=0.4, horizon_max=30)
est.fit(my_series)          # array-like, >= ~1000 points recommended
est.lower_bounds_           # per-test-window calibrated lower bounds L(x)
est.coverage_               # empirical P(H_w >= L) on held-out windows
print(est.report())         # diagnostics incl. label_identified, horizon_certified
```

`tolerance` is a fraction of the series std ("valid time" convention). The
coverage is empirically calibrated (see `docs/THEORY.md`, level (c)); the
`label_identified` diagnostic warns when `horizon_max` is too short for the
target quantile.

**Bring your own forecaster** — the tool calibrates bounds FOR your model:

```python
est = HorizonEstimator(model=my_model, dim=6, lag=1, alpha=0.1, tolerance=0.4,
                       horizon_max=30)
est.fit(my_series)   # my_model: callable or object with .predict(x)->float,
                     # fed delay vectors of the standardized series
```

The report includes **R = Λ_eff/λ₁, the measured distance of your model to the
physical predictability floor** (validated against paired ground-truth twins on
Lorenz, `docs/theory/chaos_floor.md`): R ≈ 1 means your model has saturated the
system's predictability (improving it cannot buy more horizon — invest in
better measurements); R ≫ 1 means the horizon is model-limited and a better
model can extend it. Caveat on noisy real-world data: R measures the distance
to the *deterministic* floor — observation noise raises the actually reachable
floor, so part of a large R can be irreducible noise rather than model deficit.
Real-data end-to-end demo: `python studies/demo_real_data.py`
(monthly sunspots, internal MLP vs a user persistence model).

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

## Censored labels (H_w = Hmax means H_w >= Hmax)

Horizon labels are right-censored at `horizon_max` (audit C3). Two mechanisms
handle this:

- **Always on**: a saturation gate checks whether the target alpha-quantile is
  identified from the censored calibration labels (`p_sat <= 1 - alpha`). The
  result is exported as `label_identified`; a warning is logged when the
  quantile sits in the censored region (increase `horizon_max` in that case —
  the calibrated bound is then driven by saturated labels).
- **Opt-in**: `--censored-quantile` trains the quantile models with the Powell
  (1986) censored pinball loss (`cap = horizon_max`) and caps predictions at
  `horizon_max` inside the conformal score and margin computations. Enable it
  whenever `p_sat_calib > 0`; on uncensored data it is an exact no-op (the
  measured study showed the naive pinball is biased low when the censoring cap
  crosses the target quantile: bias -59% with Powell + capped scores, coverage
  preserved, bound unchanged elsewhere).

## Certified horizon diagnostic (non-statistical)

Each run also exports a certified lower bound computed from the model's
Lipschitz constant (`docs/theory/` study P4): `horizon_certified` (first step
where `delta_sup * (G^h - 1)/(G - 1)` can reach the tolerance), `lipschitz_G`
(sup-norm Lipschitz bound; exact for linear models, layer product for MLPs)
and `delta_sup` (max one-step residual on the calibration series). Every
window label satisfies `H_w >= horizon_certified` as long as the residual
bound holds; a test window violating it is an out-of-distribution signal.
This is a diagnostic only — the conformal `L(x)` remains the operational
bound (h_cert is typically 5-11x below the median H_w). For LSTM models the
layer-product bound is invalid; the columns fall back to `0.0` with a warning.

## Chaos-estimator embedding

When `--lyap-dim` and `--lyap-lag` are both unset, the chaos estimators
(Rosenstein Lyapunov, expansion) now use a theory-grounded Takens embedding:
lag from the first minimum of the time-delayed mutual information (Fraser &
Swinney), dimension from false nearest neighbors (Kennel), with a `lag=1`
guard for discrete maps (`src/horizon_embedding.py`). Measured on the four
reference systems, the Rosenstein lambda gets closer to the literature values
on 4/4 systems (e.g. Lorenz rel. err 0.19 -> 0.03). Explicit `--lyap-dim` /
`--lyap-lag` still win, and the forecaster keeps its own val-MSE embedding.

## Units and time scales

- **Mackey-Glass `tau` is now in TIME units** (default `17.0`), no longer in
  integration steps. The generator converts internally via `round(tau / dt)`.
- **`dt` now defaults per system** instead of a shared global value:
  - lorenz: `dt = 0.01`
  - rossler: `dt = 0.05`
  - mackey_glass: `dt = 1.0`
  - logistic: `dt = 1` (map iteration)

Reference largest Lyapunov exponents (per unit time) and the resulting Lyapunov
time in steps at the default `dt`:

| System | λ₁ (reference) | Lyapunov time 1/λ₁ | dt | Lyapunov time (steps) |
|---|---|---|---|---|
| Lorenz | 0.906 | ≈ 1.10 t.u. | 0.01 | ≈ 110 |
| Rössler | 0.071 | ≈ 14.1 t.u. | 0.05 | ≈ 282 |
| Mackey-Glass (τ=17) | ≈ 0.006 | ≈ 167 t.u. | 1.0 | ≈ 167 |
| Logistic (r=4) | ln 2 ≈ 0.693 | ≈ 1.44 iters | 1 | ≈ 1.4 |

For meaningful horizon studies, `horizon_max` should be at least ~3 Lyapunov
times (in steps) for the system under study. **This is now the default**: when
`--horizon-max` is left unset (`null` in `config.yaml`), it auto-resolves to
`max(horizon_lyap_factor Lyapunov times, 1.2 × ln(τ/e₀)/λ₁)` — extended when
the model's one-step error `e₀` lets it see further than 3 T_λ — clamped to
[30, 400] steps and to the available test/calib data (a warning logs when the
target is cut: horizons beyond the cap are right-censored). The default
tolerance is now **absolute 0.4** (fraction of the standardized attractor
scale, the "valid time" convention): relative mode ties the horizon to model
precision instead of the attractor (audit C2) and is kept for diagnostics only.

## Performance (latest runs)

Regenerated 2026-07-05 from `outputs/benchmark_final.csv` by
`studies/make_results_tables.py` (no hand-copied numbers). Settings: α=0.05,
absolute tolerance 0.4·std, auto `horizon_max` (Lyapunov times), post-audit
defaults (no debias, guard on, bins=2), 5 seeds, ratios 0.6/0.15/0.15/0.10.
Reproduce: `python studies/benchmark_final.py` (resumable), then
`python studies/make_results_tables.py`.

**Metrics**
- **Coverage** = mean(H_w >= L_cal) on test (target ≥ 1 − α = 0.95)
- **Tightness** = median(L_cal) / median(H_w) (closer to 1 is tighter)
- **Slack p90** = 90th percentile of H_w − L_cal (steps)

| System | Model | Coverage med [min] | Tightness | Slack p90 | p_sat | H_w med (steps) | Hmax |
|---|---|---|---|---|---|---|---|
| logistic | linear | 0.967 [0.960] | 1.000 | 1.9 | 0.000 | 1 | 30 |
| logistic | mlp | 0.970 [0.955] | 0.832 | 4.5 | 0.000 | 8 | 30 |
| lorenz | linear | 0.943 [0.936] | 0.757 | 30.0 | 0.000 | 23 | 400 |
| lorenz | mlp | 0.963 [0.930] | 0.617 | 80.5 | 0.000 | 58 | 400 |
| mackey_glass | linear | 0.960 [0.954] | 0.768 | 12.4 | 0.000 | 11 | 292 |
| mackey_glass | mlp | 0.987 [0.956] | 0.464 | 174.2 | 0.038 | 200 | 286 |
| rossler | linear | 0.930 [0.923] | 0.786 | 35.9 | 0.000 | 34 | 400 |
| rossler | mlp | 0.813 [0.700] | 0.698 | 51.0 | 0.000 | 44 | 400 |

**Honest interpretation**
- Coverage is at or near target on logistic, Mackey-Glass and Lorenz (Lorenz
  linear keeps its documented ~1pt calib→test shift; remedy: calibrate at
  α_cal ≈ α − 0.015, validated 0.951 [0.947], or `conformal_mode: global`,
  0.961 [0.953]).
- **Rössler + MLP is reported as a negative result**: coverage 0.813 [0.700]
  with huge seed variance. Cause: with λ₁ ≈ 0.071 and dt = 0.05, the test
  split (60 t.u.) spans only ~4 Lyapunov times — too few independent regimes
  for stable coverage (the irreducible test-side fluctuation identified in
  `docs/theory/conformal_dependence.md`). Stable Rössler+MLP claims need
  series several times longer; do not deploy at this profile.
- MLP models see further than linear (H_w med 58 vs 23 on Lorenz, 200 vs 11 on
  Mackey-Glass) but the bound is then more conservative (tightness 0.46-0.62):
  tightening L(x) for strong models is the main open optimization.

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

