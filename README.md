<div align="center">

<img src="assets/logo.jpg" alt="ARSAC — Lorenz butterfly logo" width="380"/>

# ARSAC Horizon

**How far can you trust a forecast on a chaotic system? Get a number — a calibrated one.**

[![tests](https://img.shields.io/badge/tests-225%20passing-brightgreen?style=flat-square)](tests/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](pyproject.toml)
[![status](https://img.shields.io/badge/status-research%20beta-orange?style=flat-square)](docs/THEORY.md)
[![theory](https://img.shields.io/badge/theory-docs%2FTHEORY.md-8A2BE2?style=flat-square)](docs/THEORY.md)

</div>

---

ARSAC Horizon takes **your time series** and **your forecaster** and answers the three
questions every practitioner actually has:

| Question | Answer | How it's backed |
|---|---|---|
| 🎯 *How far ahead can I trust this forecast?* | A calibrated lower bound **L(x)** per window, `P(H ≥ L) ≥ 1−α` | Conformal calibration, coverage **measured** on held-out windows |
| 📈 *Is it worth improving my model?* | **R = Λ_eff/λ₁**, your measured distance to the physical predictability floor | Validated against ground-truth paired twins ([the chaos-floor study](docs/theory/chaos_floor.md)) |
| 🔊 *…or is it just noise?* | **margin_real**: the reachable margin once estimated observation noise is deducted | Noise-transported floor law, verified on known synthetic noise |

In plain words: for each window of your series the pipeline *measures* how long the
forecast stays inside a tolerance band (the horizon **H**), then turns those
measurements into a per-window lower bound **L(x)** that is wrong at most a chosen
fraction α of the time. **R** and **margin_real** then tell you *why* the horizon
stops where it does — model error, chaos itself, or measurement noise.

To our knowledge, no other packaged tool answers the second and third questions
with a **calibrated instrument**.

<div align="center">
<img src="assets/predictability_map.png" alt="Predictability map: calibrated lower bound L(x_t) along one trajectory" width="760"/>
<br/>
<sub>One pipeline output: the calibrated bound <b>L(x_t)</b> along a single trajectory.
Same system, same model — between 2 and 8 trustworthy steps depending on where you sit
on the attractor. Predictability is a property of the moment; ARSAC measures it per window.</sub>
</div>

## 🚀 Quickstart

```bash
git clone https://github.com/DEX-zha/ARSAC-Horizon-Experiment && cd ARSAC-Horizon-Experiment
pip install -e .        # Python ≥ 3.10; numpy/scipy/torch/scikit-learn/PyYAML
```

Copy-paste runnable — internal forecaster on a chaotic series:

```python
import numpy as np
from src.horizon_estimator import HorizonEstimator

x = np.empty(4000); x[0] = 0.2
for t in range(3999):                     # logistic map, r=4: fully chaotic
    x[t + 1] = 4.0 * x[t] * (1.0 - x[t])

est = HorizonEstimator(model="mlp", alpha=0.1, tolerance=0.4).fit(x)
est.lower_bounds_                 # calibrated per-window lower bounds L(x)
est.coverage_                     # empirical P(H ≥ L) on held-out windows
print(est.report())               # R, sigma_obs, margin_real, gates, ...
```

Or bring **your own forecaster** — a callable (or object with `.predict`) fed delay
vectors of the standardized series, returning the next standardized value:

```python
est = HorizonEstimator(model=my_model, dim=6, lag=1,
                       alpha=0.1, tolerance=0.4, horizon_max=30)
est.fit(my_series)                # any 1-D series, ≥ ~1000 points
```

Real-data demos:

```bash
python studies/demo_real_data.py     # 277 years of monthly sunspots (data included)
# → coverage 0.979 (target 0.90) · calibrated bound: 6.5 months
# → R = 35.9, but margin_real = ×4.2  ("most of that R is noise, not model deficit")

python studies/demo_bidmc.py         # ICU biosignals: PPG + ECG @ 125 Hz
# (expects bidmc_01_Signals.csv from PhysioNet's BIDMC dataset in the repo root)
```

## 🔧 How it works

1. **Embed** — the series is standardized and delay-embedded (`dim`/`lag` selected by
   validation when not given).
2. **Forecast** — your model (or an internal `linear`/`mlp`/`lstm`) is rolled forward
   from each window.
3. **Label** — each window gets its measured horizon `H_w`: the first excursion of the
   rolling forecast error beyond the tolerance band (right-censored labels are
   detected and gated, not silently kept).
4. **Calibrate** — a quantile model predicts a per-window bound, and a split-conformal
   correction on held-out windows makes it honest: `P(H ≥ L) ≥ 1−α`, with the achieved
   coverage reported, not assumed.

## 🦋 The floor: what R is calibrated against

A forecaster's horizon is limited either by **its own error** (improvable) or by
**chaos itself** (not improvable — invest in better measurements instead). We measure
the boundary directly: *paired twin experiments* integrate the true dynamics from the
same state with a perturbation equal to the model's own one-step error — the physical
floor at that error level. Pre-registered criteria, positive controls, replication:

| System | Vector field | ρ = H_model/H_floor | R model | R floor (control) | Verdict |
|---|---|---|---|---|---|
| Lorenz | polynomial | **0.83 / 0.86** (2 seeds) | 1.15 / 1.21 | 0.96–1.03 ✓ | **floor reached** — 18–19 Lyapunov times of valid forecast |
| Rössler | polynomial | 0.60 (0.71 vs step-wise floor) | 1.70 | 1.01 ✓ | near floor |
| Mackey-Glass | non-polynomial (Hill) | 0.045 | 20.9 | 0.91 ✓ | model-limited — the method's mapped generality boundary |

Two quantitative laws fall out and replicate across systems and regimes
(ratios measured/theory 0.88–1.21): the **injection floor**
`H ≈ ln(τ·λ₁·dt/ε)/(λ₁·dt)` for step-wise forecasters, and its transport to
observation noise — which is what makes R honest on real data.
Full story, controls and evidence CSVs: [`docs/theory/chaos_floor.md`](docs/theory/chaos_floor.md).

## 📊 Benchmark (regenerated by script, no hand-copied numbers)

α = 0.05, attractor-scale tolerance 0.4·σ, auto horizon window in Lyapunov times,
5 seeds — produced by `studies/benchmark_final.py` → `studies/make_results_tables.py`:

| System | Model | Coverage med [min] | Tightness | H_w med (steps) |
|---|---|---|---|---|
| logistic | linear / mlp | 0.967 / 0.970 | 1.00 / 0.83 | 1 / 8 |
| lorenz | linear / mlp | 0.943 / 0.963 | 0.76 / 0.62 | 23 / 58 |
| mackey_glass | linear / mlp | 0.960 / 0.987 | 0.77 / 0.46 | 11 / 200 |
| rossler | linear / mlp | 0.930 / **0.813 [0.700]** | 0.79 / 0.70 | 34 / 44 |

The Rössler+MLP row is a **published negative result**: at λ₁ ≈ 0.071 the test split
spans only ~4 Lyapunov times — a concrete data-budget requirement for slow chaotic
systems ([details](docs/theory/eval_results.md)). Lorenz-linear keeps a documented
~1-pt calib→test shift (remedy: `α_cal ≈ α − 0.015`, validated 0.951 [0.947]).

## 🧪 The science, in layers

Everything is reproducible (seeded studies, evidence CSVs versioned) and every
guarantee is labeled with its actual strength:

| Level | What | Where |
|---|---|---|
| **Certified** | Lipschitz bound `horizon_certified` — holds for *every* window (0 violations / 2000) | [`docs/theory/certified_horizon.md`](docs/theory/certified_horizon.md) |
| **Measured law** | injection/noise floor laws, chaos-floor saturation | [`docs/theory/chaos_floor.md`](docs/theory/chaos_floor.md) |
| **Empirical** | conformal coverage of L(x) (overlapping windows: exchangeability is approximate) | [`docs/THEORY.md`](docs/THEORY.md) |

- 📘 [`docs/THEORY.md`](docs/THEORY.md) — the unified theory: definitions, guarantee levels, decisions with numbers
- 🔬 [`docs/theory/`](docs/theory/) — 8 studies (conformal under dependence, censoring, FTLE, certified bound, block bootstrap, embedding, chaos floor, noisy floor) with **pre-registered accept/shelve verdicts** — including the honest negatives
- 🧾 [`AUDIT_MATH.md`](AUDIT_MATH.md) — the full math audit that rebuilt this pipeline (the original Mackey-Glass generator wasn't even chaotic)
- 🗺️ [`PLAN_V2.md`](PLAN_V2.md) — roadmap and phase status
- 📄 [`paper.tex`](paper.tex) — manuscript with script-generated tables

Safety rails are built in and have caught real errors in-house: right-censoring
gates (`label_identified`, Powell loss via `--censored-quantile`), twin-censoring
guards, positive controls on every floor measurement.

## ⚙️ CLI (research pipeline)

```bash
arsac-horizon --dataset lorenz --model mlp        # console entry point (pip install -e .)
python -m src.horizon_experiment --dataset lorenz --model mlp     # same, as a module
python studies/benchmark_final.py                                # full benchmark (resumable)
python chaos_quick_test.py --dataset mackey_glass                 # is my series chaotic?
```

~100 flags documented in [`AGENT.MD`](AGENT.MD); production defaults in
[`config.yaml`](config.yaml) (ablation-validated: debias removed, guard kept).
Physics you can rely on: per-system `dt`, Mackey-Glass integrated as a true DDE
(`tau` in **time units**), Rosenstein λ validated against literature on 4/4 systems
by [pinned physics tests](tests/test_physics_chaos.py):

| System | λ₁ (lit.) | Lyapunov time | default dt |
|---|---|---|---|
| Lorenz | 0.906 | ≈ 1.1 t.u. (110 steps) | 0.01 |
| Rössler | 0.071 | ≈ 14 t.u. (282 steps) | 0.05 |
| Mackey-Glass (τ=17) | ≈ 0.006 | ≈ 167 t.u. (167 steps) | 1.0 |
| Logistic (r=4) | ln 2 | ≈ 1.4 iters | 1 |

## 🗂️ Repository map

```
src/                  pipeline + HorizonEstimator API (import path: src.*)
studies/              reproducible experiments (every number in the docs)
docs/THEORY.md        unified theory & guarantee levels
docs/theory/          per-study memos + versioned evidence CSVs
tests/                225 tests incl. physics pins (λ vs literature)
data/                 real datasets (SILSO monthly sunspots)
paper.tex             manuscript (tables generated by studies/, not hand-copied)
```

Run the test suite:

```bash
pip install -e .[dev]
pytest -q             # 225 tests, incl. pinned physics checks
```

## ⚠️ Honest limitations

- Coverage of L(x) is **empirically** calibrated (overlapping windows break strict
  exchangeability); worst-seed behavior is documented per system.
- The floor results are established on simulated systems (Lorenz fully, Rössler
  partially, Mackey-Glass = mapped boundary); on real data, R relies on a
  Rosenstein λ estimate and a conservative (upward-biased) noise estimator.
- Import path is `src.*` until the package rename planned before any PyPI release.
- Real-world evidence: one fully documented case study (sunspots) plus a biosignal
  demo in progress (BIDMC PPG/ECG, [`studies/demo_bidmc.py`](studies/demo_bidmc.py)) —
  bring your series and widen the sample.

---

<div align="center">
<sub>ARSAC Horizon — measure the edge of predictability instead of guessing it.</sub>
</div>
