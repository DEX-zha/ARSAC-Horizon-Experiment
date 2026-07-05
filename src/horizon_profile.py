"""Predictability profiler: WHY is your series (un)predictable?

Entry point of the product (born from the BIDMC biosignal test, where the
chaos-floor diagnostic R fired outside its validity regime and produced a
division-by-lambda artifact). Before any horizon diagnostic, classify the
series into one of four predictability regimes and route accordingly:

- 'stochastic'      noise dominates (local-linear residual ~ signal std):
                    modeling will buy little; expect L ~ 1-2 steps.
- 'chaotic'         a positive Lyapunov exponent is RESOLVED on the observed
                    window: L(x), R (distance to the chaos floor) and
                    margin_real (noise-aware margin) all apply.
- 'quasi-periodic'  strong recurrence (heartbeats, machines, seasons):
                    L(x) applies; R does NOT (lambda ~ 0 -> division
                    artifact); horizon is set by cycle stability + noise.
- 'regular'         predictable structure without resolved exponential
                    divergence (e.g. drifts, random walks at attractor
                    scale): L(x) applies; chaos diagnostics do not.

Classification signals (each individually validated elsewhere):
periodicity index (autocorrelation peak past the first zero), Rosenstein
lambda on the MI+FNN theory embedding (tests/test_physics_chaos.py), and
the local-linear noise estimator (studies/study_noisy_floor.py). Lambda is
'resolved' when the measured divergence grows by at least a factor e over
the fitting window (lambda * window >= 1) — an absolute per-step threshold
would misclassify slow chaos (audit: Lorenz at dt=0.01 has lambda_step
0.009).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.horizon_embedding import select_embedding
from src.horizon_noise import estimate_observation_noise
from src.horizon_utils import estimate_lyapunov, standardize_series

STOCHASTIC_NOISE_LEVEL = 0.7   # sigma_hat (std units) above which noise dominates
PERIODICITY_THRESHOLD = 0.5    # autocorr recurrence above which R is unsafe
GROWTH_RESOLVED = 1.0          # lambda * window >= 1 <=> divergence grew by e
STRUCTURE_RATIO_MAX = 0.5      # sigma_hat / sigma_persistence: 'chaotic' requires
# predictive structure beyond persistence (a random walk has none, yet fools
# Rosenstein with diffusive sqrt(t) divergence that fits as a positive slope)


@dataclass
class PredictabilityProfile:
    regime: str
    periodicity_index: float
    period_samples: Optional[int]
    lambda_per_step: float
    lambda_resolved: bool
    noise_std_units: float
    structure_ratio: float
    embedding: tuple
    n_samples: int
    reading: str

    def summary(self):
        lines = [
            f"regime            : {self.regime}",
            f"periodicity index : {self.periodicity_index:.2f}"
            + (f" (period ~ {self.period_samples} samples)"
               if self.period_samples and self.periodicity_index >= PERIODICITY_THRESHOLD else ""),
            f"lambda per step   : {self.lambda_per_step:.4f} "
            f"({'resolved' if self.lambda_resolved else 'not resolved on this window'})",
            f"noise (std units) : {self.noise_std_units:.3f}",
            f"embedding (dim,lag): {self.embedding}",
            f"reading           : {self.reading}",
        ]
        return "\n".join(lines)


def _periodicity(x):
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0 or x.size < 200:
        return 0.0, None
    ac = np.correlate(x, x, "full")[x.size - 1:] / denom
    neg = np.where(ac < 0)[0]
    if not neg.size:
        return 1.0, None
    seg = ac[neg[0]: neg[0] + max(10, x.size // 4)]
    if not seg.size:
        return 0.0, None
    return float(np.max(seg)), int(neg[0] + int(np.argmax(seg)))


def _reading(regime, periodicity, lam, noise):
    if regime == "stochastic":
        return ("noise dominates (local-linear residual "
                f"~{noise:.2f} of the signal std): no model will forecast far; "
                "expect horizons of 1-2 steps and invest in better data, not models")
    if regime == "chaotic":
        return ("resolved positive Lyapunov exponent: the full instrument applies "
                "— calibrated L(x), R (distance to the physical predictability "
                "floor) and the noise-aware margin_real")
    if regime == "quasi-periodic":
        return (f"strongly recurrent signal (periodicity {periodicity:.2f}): "
                "L(x) applies; the chaos-floor diagnostic R does NOT (lambda ~ 0 "
                "makes it a division artifact). Horizon is set by cycle "
                "stability and noise, not chaotic divergence")
    return ("predictable structure without resolved exponential divergence on "
            "this window: L(x) applies; chaos diagnostics do not. If you believe "
            "the system is chaotic, the sampling may be too fine or the window "
            "too short to resolve lambda — downsample or provide lambda_per_step")


def profile_series(series, lam_step=None, embedding=None, max_points=8000, seed=0):
    """Classifies the predictability regime of a 1-D series.

    ``lam_step``/``embedding`` can be supplied to reuse pipeline estimates
    (HorizonEstimator does); otherwise the validated MI+FNN embedding and
    Rosenstein estimator are run on (a subsample of) the series.
    """
    raw = np.asarray(series, dtype=np.float64).reshape(-1)
    x, _, _ = standardize_series(raw)
    if x.size > max_points:
        x = x[:max_points]
    n = int(x.size)

    periodicity, period = _periodicity(x)

    if embedding is None:
        try:
            emb = select_embedding(x)
            embedding = (int(emb["dim"]), int(emb["lag"]))
        except Exception:
            embedding = (6, 1)
    dim, lag = embedding

    if lam_step is None:
        try:
            lam_step, _ = estimate_lyapunov(x, dim=dim, lag=lag)
        except Exception:
            lam_step = 0.0
    lam_step = float(lam_step)

    sigma_hat, _ = estimate_observation_noise(x, dim=min(dim, 8), lag=1,
                                              n_samples=300, seed=seed)
    if not np.isfinite(sigma_hat):
        sigma_hat = 0.0

    n_embedded = max(2, n - (dim - 1) * lag)
    window = min(400, max(20, n_embedded // 4))
    lambda_resolved = bool(lam_step > 0 and lam_step * window >= GROWTH_RESOLVED)

    # Determinism beyond persistence: a random walk shows diffusive sqrt(t)
    # divergence that Rosenstein fits as a positive slope, but its local-linear
    # residual is no better than the persistence residual.
    sigma_pers = float(np.std(np.diff(x))) if n > 10 else 0.0
    structure_ratio = float(sigma_hat / sigma_pers) if sigma_pers > 0 else float("inf")

    if sigma_hat >= STOCHASTIC_NOISE_LEVEL:
        regime = "stochastic"
    elif lambda_resolved and structure_ratio < STRUCTURE_RATIO_MAX:
        regime = "chaotic"
    elif periodicity >= PERIODICITY_THRESHOLD:
        regime = "quasi-periodic"
    else:
        regime = "regular"

    return PredictabilityProfile(
        regime=regime,
        periodicity_index=periodicity,
        period_samples=period,
        lambda_per_step=lam_step,
        lambda_resolved=lambda_resolved,
        noise_std_units=float(sigma_hat),
        structure_ratio=structure_ratio,
        embedding=(dim, lag),
        n_samples=n,
        reading=_reading(regime, periodicity, lam_step, sigma_hat),
    )
