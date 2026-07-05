# AGENT.MD – ARSAC Horizon Experiment (Full Technical Specification)

This document is the **complete English specification** of the project. It contains:
- full architecture,
- exhaustive module inventory,
- all mathematical definitions and formulas,
- all pipelines (chaos only),
- all calibration details, safeguards, and diagnostics,
- all CLI arguments and runtime behaviors,
- all known limitations.

It intentionally **omits usage examples** and focuses on exhaustive technical detail.

---

# 1) Project Goal and Scope

The project estimates **conservative prediction horizons** for chaotic time series.
The main output is a lower bound **L(x)** for the horizon, intended to satisfy:

**P(H_w ≥ L(x)) = 1 - α** (empirical coverage on calibrated domain).

The system is designed to be:
- **model-agnostic** (usable with any forecaster),
- **robust** to heteroscedasticity and regime variation,
- **calibration-aware** (conformal guarantees in-distribution),
- **auditable** (diagnostics, correlations, warnings, logs).

The system includes one major pipeline:
1) **Chaos pipeline** (Lorenz/Rössler/Mackey-Glass/logistic)

---

# 2) Mathematical Definitions

## 2.1 Per-Window Horizon Label (H_w)
For each window **w** and horizon **h = 1..Hmax**, compute the **per-step
absolute error** at horizon h (not a cumulative RMSE):

```
e_w(h) = | y_{t+h} - ŷ_{t+h} |
```

Define tolerance **τ**:
- **absolute mode**: τ = error_tolerance
- **relative mode**: τ = median over windows of the one-step residual,
  multiplied by error_factor. The tolerance is shared across windows
  (no longer a per-window single-sample tolerance based on rmse_w(1)).

Define **K-consecutive exceedance horizon**:

```
H_w = min{ h | e_w(h..h+K-1) > τ }, else Hmax
```

i.e. the label is the first h at which K consecutive per-step absolute errors
exceed τ. H_w is the training target for the horizon model. H_w = Hmax is
right-censored, not exact.

---

## 2.2 Horizon Model (Quantile Lower Bound)
Learn a conditional quantile **q̂_α(x)** with pinball loss:

```
q̂_α(x) ≈ Q_α(H_w | x)
```

This produces a conservative base estimate.

---

## 2.3 Conformal Lower Bound (Signed Score)
Define signed nonconformity scores:

```
α_i = q̂_α(x_i) - y_i
```

Optionally normalized by scale:

```
α_i = (q̂_α(x_i) - y_i) / ŝ(x_i)
```

Compute conformal constant:

```
c = Quantile_{1-α}(α_i)
```

Final lower bound:

```
L(x) = q̂_α(x) - c · ŝ(x)
```

**Signed scores** allow upward correction when q̂ is overly pessimistic.

---

## 2.4 Heteroscedastic Scale (ŝ)
ŝ(x) can be estimated by:
- quantile spread (q̂_0.9 - q̂_0.5), or
- residual-MLP: log(|y - q̂_0.5| + eps)

Then apply caps/floors:

```
ŝ ← max(ŝ, scale_floor)
ŝ ← min(ŝ, scale_cap or quantile cap)
```

---

## 2.5 Mondrian Bins + Shrinkage
Partition a regime feature into bins. For bin g:

```
c_g = Quantile_{1-α}(α_i | bin=g)
```
Shrink towards global constant:

```
w_g = n_g / (n_g + λ)
c_g = w_g · c_g + (1 - w_g) · c_global
```

Used to stabilize per-bin calibration.

---

## 2.6 Coverage Guard (Post-Calib Safety)
Define:

```
z = L_calib - y_calib
```

Guard is a quantile of z:

```
guard = Quantile_q(z)
```

Then:

```
L ← L - guard
```

Guard ensures that coverage does not collapse on test.

---

## 2.7 Debias Post-Conformal
Debias re-centers the calibrated bound to reduce over-coverage:

```
δ = Quantile_q(z)
L ← L - debias_scale · δ
```

---

## 2.8 Gated Multi-Step Rollout
For each x:

```
rollout steps = floor(L(x))
stop afterwards
```

---

## 2.9 Adaptive MPC Horizon

```
H_mpc = clamp( floor(L), 1, Hmax )
```

---

## 2.10 Lyapunov Estimation (Rosenstein)
The largest Lyapunov exponent is estimated with the Rosenstein method:
- **Theiler window is selected automatically** as the first zero crossing of
  the autocorrelation function (temporal decorrelation of neighbors).
- The **linear region of the mean log-divergence curve is fitted
  automatically** (before saturation at the attractor size), instead of a
  fixed 1..10-step fit.
- λ is reported per step and converted with `dt`; the Lyapunov time is 1/λ.
  Reference values: Lorenz 0.906, Rössler 0.071, Mackey-Glass (τ=17) ≈ 0.006,
  logistic ln 2 ≈ 0.693 (per unit time / per iteration).

---

## 2.11 Probabilistic Coverage Calibration (Lower Bound)
The probabilistic path calibrates a **lower** bound:

```
scale = Quantile_α( h_real / h_model ), capped at 1 (shrink-only)
L     = h_model * scale
coverage = P( H_real >= L )
```

A hit is counted when `h_real >= h_model * scale`. `calibration_floor`
defaults to **0.0** (no floor forcing the scale above 1). This replaces the
previous inverted scheme, which calibrated an upper bound
(quantile_{1-α} with a floor at 1.0 and hits when the bound exceeded h_real).

---

## 2.12 Coverage Lower Bound (coverage_lb)
`coverage_lb` is computed by **block bootstrap** over coverage hits (block
size at least the correlation range of overlapping windows), not by a Wilson
interval, since the hits are not i.i.d. Bernoulli.

---

# 3) Repository Architecture (Exhaustive)

```
root/
  AGENT.MD
  README.md
  config.yaml
  experiment_constants.yaml
  outputs/
  examples/
  src/
  tests/
```

## 3.1 src/ (Chaos Pipeline)

### horizon_data.py
- Generates time series for chaotic systems.
- Supports Lorenz, Rössler, logistic, Mackey-Glass.
- Mackey-Glass is integrated as a true DDE via the **method of steps**, with
  interpolation of the delayed history at intermediate stages. **tau is
  expressed in time units** (default 17.0, the chaotic regime) and converted to
  steps internally via `round(tau / dt)`. Each system has its own default `dt`
  (lorenz 0.01, rossler 0.05, mackey_glass 1.0, logistic 1).
- Handles split into train/val/calib/test.

### horizon_forecast.py
- Embedding search (dim/lag) with validation metric.
- Model selection: linear / MLP / LSTM.
- Supports one-step and multistep training.

### horizon_metrics.py
- RMSE calculation (rolling, per-window).
- Horizon definitions (global, window-based).
- Jacobian estimation if available.
- Added: `gated_rollout` for L(x) gating.

### horizon_conformal.py
- Conformal quantile and block-quantile margins.
- Mondrian bin edges.
- Tree estimator for conformal constants, fitted with a **temporal split**:
  the tree structure is built on the first part of the calibration data and
  the leaf quantiles are computed on the held-out later part (no reuse of the
  same samples for both).
- Ensemble quantile prediction.

### horizon_experiment_core.py
- Main orchestration for chaos pipeline.
- Loads constants, datasets, trains model, computes metrics.
- Exports stats into result dict.

### horizon_experiment_conformal_data.py
- Builds horizon dataset (features + H_w labels).
- Handles cross-fitting CV predictions.
- Builds ŝ and quantile predictions.

### horizon_experiment_conformal_calibration.py
- Computes signed scores.
- Applies global/bins/tree calibration.
- Coverage guard + debias.

### horizon_experiment_conformal_stats.py
- Coverage/tightness/slack.
- Jacobian coverage quantiles.
- Predictability correlations (L vs jac/resid).
- Predictability map plotting.

### horizon_experiment_io.py
- CSV logging.
- Plotting of RMSE/log divergence.

### horizon_cli.py
- Central CLI for experiments (all parameters).

### horizon_conformal_eval.py
- Multi-seed conformal evaluation runner.
- Writes per-seed and summary CSVs.

---

# 4) Runtime Behavior: Chaos Pipeline

1. Generate chaotic series.
2. Split into train/val/calib/test.
3. Select embedding (dim, lag).
4. Train final model on train+val.
5. Compute rolling RMSE and global horizon_real.
6. Build horizon dataset with H_w labels.
7. Quantile horizon model + ŝ estimation.
8. Conformal calibration using signed scores.
9. Compute L(x) per window.
10. Coverage/tightness/slack evaluation on test.
11. Optional predictability map.

---

# 5) Diagnostics and Warnings

- **p_sat_calib**: fraction of windows with H_w = Hmax
  - warning if > 0.2
- **coverage_test** < target
  - indicates calibration failure
- **slack_p90** large
  - indicates conservative collapse
- **predictability_corr_jac/resid**
  - indicates which uncertainty proxy dominates

---

# 6) Conformal Stability Fixes Implemented

- Signed nonconformity scores.
- Heteroscedastic ŝ normalization.
- Bins + shrinkage to avoid empty bins.
- Block-quantile calibration (robust to non-IID).
- Coverage guard (post-calib safety).
- Debias (post-calib recentering).
- Ensemble quantile median aggregation.

---

# 7) Default Parameters (config.yaml)

Current defaults were rewritten to match stable M7-style settings (chaos pipeline):
- horizon_max = null (auto: max(horizon_lyap_factor Lyapunov times, 1.2·ln(τ/e₀)/λ₁),
  clamped to [30, 400] and to the test/calib data budget; censoring warning when cut)
- horizon_lyap_factor = 3.0
- error_mode = absolute, error_tolerance = 0.4 (attractor-scale "valid time"
  convention; relative mode is diagnostic-only — it ties the horizon to model
  precision, audit C2)
- horizon_samples = 1200
- calib_ratio = 0.2
- horizon_quantile = 0.10
- calibration_alpha = 0.05
- bins = 2
- block_count = 10
- coverage_guard_quantile = 0.80
- coverage_guard_margin = 0.02
- coverage_guard_min_scale = 0.25
- debias_scale = 0.0 (ablation 2026-07-05: le debias coutait de la couverture sans gain de tightness sur labels corriges)
- scale_cap_quantile = 0.98
- integrator = rk4
- use_cuda = true

---

# 8) Predictability Map

The map is a heatmap of L(x_t) over time, plus optional overlays.
Correlations are computed between L(x_t) and:
- jac_norm (if available)
- resid1 (always available)

---

# 9) Tests

- Horizon utils tests
- Model tests
- Tree binning tests

---

# 10) Known Limitations

- Coverage is valid only within calibrated distribution.
- Small calibration sizes cause unstable coverage.
- Heavy computation for large datasets.

---

# 11) Internal CSV Metrics (Chaos Pipeline)

- coverage_test
- tightness_ratio
- slack_median / slack_p90
- leaf_* stats (tree)
- jac_quantile coverages
- p_sat_calib / p_sat_test
- c_global, signed_med, sigma stats
- debias, guard

---

# 12) Full Parameter Inventory (CLI)

The CLI accepts all parameters from `src/horizon_cli.py`:
- dataset, series_len, warmup, train_ratio, val_ratio, calib_ratio
- model, mlp/lstm hyperparams
- horizon_max, horizon_samples, horizon_quantile
- calibration_alpha
- conformal_mode, bins, shrinkage, min_bin
- block_count, block_quantile
- coverage_guard_quantile, coverage_guard_margin, coverage_guard_min_scale
- debias_scale, debias_quantile
- scale_cap_quantile, scale_floor
- integrator (euler/rk4)
- use_cuda
- predictability_map

---

# 13) End Notes

This document is intended as the canonical reference for the ARSAC Horizon Experiment.
Any modifications to core calibration logic, label definitions, or forecasting interfaces
should be reflected here.

---

End of AGENT.MD.