
import numpy as np
from src.horizon_utils import (
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
    split_series,
    standardize_series,
)

class DataManager:
    """Handles data loading, splitting, and standardization."""

    def __init__(self, args):
        self.args = args
        self.raw_series = None
        self.train = None
        self.val = None
        self.calib = None
        self.test = None
        self.mean = None
        self.std = None

    def load_series(self):
        """Generates the selected chaotic time series."""
        if self.args.dataset == "logistic":
            self.raw_series = generate_logistic_map(
                self.args.series_len, r=self.args.r, x0=self.args.x0, warmup=self.args.warmup
            )
        elif self.args.dataset == "lorenz":
            self.raw_series = generate_lorenz(
                self.args.series_len,
                dt=self.args.dt,
                sigma=self.args.sigma,
                rho=self.args.rho,
                beta=self.args.beta,
                warmup=self.args.warmup,
                integrator=self.args.integrator,
            )
        elif self.args.dataset == "rossler":
            self.raw_series = generate_rossler(
                self.args.series_len,
                dt=self.args.dt,
                a=self.args.a,
                b=self.args.b,
                c=self.args.c,
                warmup=self.args.warmup,
                integrator=self.args.integrator,
            )
        elif self.args.dataset == "mackey_glass":
            self.raw_series = generate_mackey_glass(
                self.args.series_len,
                tau=self.args.tau,
                beta=self.args.mg_beta,
                gamma=self.args.gamma,
                n=self.args.n,
                dt=self.args.dt,
                warmup=self.args.warmup,
                integrator=self.args.integrator,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.args.dataset}")
        
        return self.raw_series

    def prepare_data(self):
        """Splits and standardizes the data."""
        if self.raw_series is None:
            self.load_series()

        train_raw, val_raw, calib_raw, test_raw = split_series(
            self.raw_series,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            calib_ratio=self.args.calib_ratio,
        )

        self.train, self.mean, self.std = standardize_series(train_raw)
        self.val = (val_raw - self.mean) / self.std
        self.calib = (calib_raw - self.mean) / self.std if calib_raw.size else self.val
        self.test = (test_raw - self.mean) / self.std
        
        return self.train, self.val, self.calib, self.test

    def get_raw_splits(self):
        """Returns raw splits (unstandardized)."""
        # Re-calc separation indices logic or just use split_series again
        # For simplicity, we can rely on what we just did or re-call split_series
        return split_series(
            self.raw_series,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            calib_ratio=self.args.calib_ratio,
        )
