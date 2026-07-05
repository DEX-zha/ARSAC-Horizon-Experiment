"""Mini script to assess chaos via Lyapunov and expansion quantile."""

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__))

from src.horizon_cli import resolve_dt
from src.horizon_utils import (
    estimate_expansion_quantile,
    estimate_lyapunov,
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
)


def _generate_series(args):
    if args.dataset == "logistic":
        return generate_logistic_map(
            args.series_len, r=args.r, x0=args.x0, warmup=args.warmup
        )
    if args.dataset == "lorenz":
        return generate_lorenz(
            args.series_len,
            dt=args.dt,
            sigma=args.sigma,
            rho=args.rho,
            beta=args.beta,
            warmup=args.warmup,
            integrator=args.integrator,
        )
    if args.dataset == "rossler":
        return generate_rossler(
            args.series_len,
            dt=args.dt,
            a=args.a,
            b=args.b,
            c=args.c,
            warmup=args.warmup,
            integrator=args.integrator,
        )
    if args.dataset == "mackey_glass":
        return generate_mackey_glass(
            args.series_len,
            tau=args.tau,
            beta=args.mg_beta,
            gamma=args.gamma,
            n=args.n,
            dt=args.dt,
            warmup=args.warmup,
            integrator=args.integrator,
        )
    raise ValueError(f"Unknown dataset: {args.dataset}")


def main():
    parser = argparse.ArgumentParser(description="Quick chaos test")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["logistic", "lorenz", "rossler", "mackey_glass"],
        default="logistic",
    )
    parser.add_argument("--series-len", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Integration timestep; None picks the per-system default (audit A1/A3)",
    )
    parser.add_argument("--r", type=float, default=4.0)
    parser.add_argument("--x0", type=float, default=0.2)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)
    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=5.7)
    parser.add_argument(
        "--tau", type=float, default=17.0, help="Mackey-Glass delay in TIME units"
    )
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--integrator", type=str, choices=["euler", "rk4"], default="rk4")
    parser.add_argument("--lyap-max-t", type=int, default=None, help="None = auto")
    parser.add_argument("--lyap-theiler", type=int, default=None, help="None = auto")
    parser.add_argument("--lyap-fit-start", type=int, default=None, help="None = auto")
    parser.add_argument("--lyap-fit-end", type=int, default=None, help="None = auto")
    parser.add_argument("--exp-quantile", type=float, default=0.9)
    parser.add_argument("--exp-theiler", type=int, default=5)
    parser.add_argument("--exp-max-pairs", type=int, default=200)
    parser.add_argument("--exp-horizon", type=int, default=3)
    args = parser.parse_args()
    args.dt = resolve_dt(args.dataset, args.dt)

    try:
        series = _generate_series(args)
    except ImportError as exc:
        print(f"Import error: {exc}")
        print("You may need scipy for lorenz/rossler.")
        sys.exit(1)

    lyap, _ = estimate_lyapunov(
        series,
        dim=args.dim,
        lag=args.lag,
        max_t=args.lyap_max_t,
        theiler=args.lyap_theiler,
        fit_start=args.lyap_fit_start,
        fit_end=args.lyap_fit_end,
        dt=args.dt,
    )
    lq, _ = estimate_expansion_quantile(
        series,
        dim=args.dim,
        lag=args.lag,
        quantile=args.exp_quantile,
        theiler=args.exp_theiler,
        max_pairs=args.exp_max_pairs,
        seed=0,
        horizon=args.exp_horizon,
    )

    chaos_flag = (lyap > 0.0) and (lq > 1.0)
    strength = "chaotic" if chaos_flag else "non-chaotic"

    print(f"Dataset: {args.dataset}")
    print(f"Resolved dt: {args.dt}")
    print(f"lambda_per_step: {lyap:.6f}")
    print(f"lambda_per_unit_time: {lyap / args.dt:.6f}")
    print(f"Expansion Lq: {lq:.6f}")
    print(f"Assessment: {strength}")


if __name__ == "__main__":
    main()
