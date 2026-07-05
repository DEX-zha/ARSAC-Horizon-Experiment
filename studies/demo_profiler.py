"""Showcase: the predictability profiler on six signals of different nature.

The product's entry point: before any horizon diagnostic, answer
"WHY is this series (un)predictable?" and route to the right instrument.
Run: python studies/demo_profiler.py   (~1 min)
"""

import csv
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_profile import profile_series
from src.horizon_utils import generate_logistic_map, generate_lorenz

BIDMC = os.path.join(os.path.dirname(__file__), "..", "bidmc_01_Signals.csv")


def signals():
    rng = np.random.default_rng(0)
    yield "bruit blanc", rng.normal(size=6000)
    yield "marche aleatoire", np.cumsum(rng.normal(size=6000))
    t = np.linspace(0, 300 * np.pi, 6000)
    yield "sinus + bruit", np.sin(t) + 0.05 * rng.normal(size=6000)
    yield "logistique (chaos)", generate_logistic_map(6000)
    yield "Lorenz (chaos)", generate_lorenz(6000, dt=0.01, warmup=1000)
    if os.path.exists(BIDMC):
        with open(BIDMC) as f:
            r = csv.reader(f)
            h = [c.strip() for c in next(r)]
            idx = h.index("PLETH")
            x = np.array([float(row[idx]) for row in r])[::2][:8000]
        yield "PPG patient ICU (BIDMC)", x


def main():
    print(f"{'signal':24s} {'regime':16s} {'period.':>7} {'lambda/pas':>10} "
          f"{'bruit':>6} {'structure':>9}")
    for name, x in signals():
        p = profile_series(x)
        print(f"{name:24s} {p.regime:16s} {p.periodicity_index:7.2f} "
              f"{p.lambda_per_step:10.4f} {p.noise_std_units:6.3f} "
              f"{min(p.structure_ratio, 99):9.2f}")
    print("\nRoutage: chaotic -> L(x) + R + margin_real | quasi-periodic/regular"
          " -> L(x) (R retenu) | stochastic -> n'insistez pas.")


if __name__ == "__main__":
    main()
