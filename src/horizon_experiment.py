"""AI-driven prediction horizon experiment for chaotic time series."""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from src.horizon_utils import (
    build_supervised,
    estimate_lyapunov,
    estimate_expansion_quantile,
    generate_logistic_map,
    generate_lorenz,
    generate_mackey_glass,
    generate_rossler,
    horizon_from_model_bound,
    horizon_from_model_bound_by_growth,
    set_seed,
    split_series,
    standardize_series,
)

class ProgressBar:
    """Simple progress bar with ETA."""

    def __init__(self, total, label="progress"):
        self.total = max(1, int(total))
        self.label = label
        self.start = time.time()
        self.current = 0

    def update(self, step=1, extra=""):
        """Advances the progress bar and prints the current status."""
        self.current = min(self.total, self.current + step)
        frac = self.current / self.total
        width = 24
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = time.time() - self.start
        rate = self.current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.current) / rate if rate > 0 else 0.0
        msg = (
            f"\r{self.label} [{bar}] {self.current}/{self.total} "
            f"{frac*100:5.1f}% ETA {eta:5.1f}s {extra}"
        )
        print(msg, end="", flush=True)

    def close(self):
        """Ends the progress bar line."""
        print()


class LinearAR:
    """Ridge-regularized linear auto-regressive predictor."""

    def __init__(self, reg=1e-4):
        self.reg = reg
        self.weights = None

    def fit(self, x, y):
        """Fits the linear model parameters."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        x_aug = np.concatenate([x, ones], axis=1)
        xtx = x_aug.T @ x_aug
        xtx += self.reg * np.eye(xtx.shape[0], dtype=np.float64)
        self.weights = np.linalg.solve(xtx, x_aug.T @ y)
        return self

    def predict(self, x):
        """Predicts a single value from a feature vector."""
        x = np.asarray(x, dtype=np.float64)
        return float(np.dot(self.weights[:-1], x) + self.weights[-1])

    def predict_batch(self, x):
        """Predicts a batch of values from feature vectors."""
        x = np.asarray(x, dtype=np.float64)
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        x_aug = np.concatenate([x, ones], axis=1)
        return x_aug @ self.weights


class MLPPredictor(nn.Module):
    """Small MLP for one-step prediction."""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMPredictor(nn.Module):
    """Single-layer LSTM predictor for one-step forecasting."""

    def __init__(self, hidden_dim=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.readout(last).squeeze(-1)


class TorchWrapper:
    """Wraps a torch model for numpy-friendly predict calls."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        """Predicts one step for a single input vector."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(x_t).item())

    def predict_batch(self, x):
        """Predicts one step for a batch of input vectors."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model(x_t).cpu().numpy()


class TorchSeqWrapper:
    """Wraps sequence models expecting (batch, time, 1) inputs."""

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        """Predicts one step for a single input vector."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            return float(self.model(x_t).item())

    def predict_batch(self, x):
        """Predicts one step for a batch of input vectors."""
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        x_t = x_t.view(x_t.shape[0], x_t.shape[1], 1)
        with torch.no_grad():
            return self.model(x_t).cpu().numpy()


def train_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
    input_dim,
    hidden_dim=64,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cpu",
    show_progress=False,
):
    """Trains an MLP with early stopping.

    Args:
        x_train: Training inputs.
        y_train: Training targets.
        x_val: Validation inputs.
        y_val: Validation targets.
        input_dim: Input dimensionality.
        hidden_dim: Hidden layer width.
        epochs: Max epochs to train.
        lr: Learning rate.
        batch_size: Batch size.
        patience: Early stopping patience.
        device: Torch device string.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (trained model, best validation loss).
    """
    model = MLPPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    progress = ProgressBar(epochs, label="train-mlp") if show_progress else None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.tensor(x_val, dtype=torch.float32, device=device))
            val_loss = criterion(
                val_pred, torch.tensor(y_val, dtype=torch.float32, device=device)
            ).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if progress:
                    progress.update(epochs - progress.current)
                break
        if progress:
            progress.update(1, extra=f"val={val_loss:.4f}")

    if progress:
        progress.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def train_lstm(
    x_train,
    y_train,
    x_val,
    y_val,
    hidden_dim=64,
    num_layers=1,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cpu",
    show_progress=False,
):
    """Trains an LSTM with early stopping.

    Args:
        x_train: Training inputs.
        y_train: Training targets.
        x_val: Validation inputs.
        y_val: Validation targets.
        hidden_dim: Hidden size.
        num_layers: LSTM layers.
        epochs: Max epochs.
        lr: Learning rate.
        batch_size: Batch size.
        patience: Early stopping patience.
        device: Torch device string.
        show_progress: Whether to show a progress bar.

    Returns:
        Tuple of (trained model, best validation loss).
    """
    model = LSTMPredictor(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32).view(-1, x_train.shape[1], 1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    train_ds = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    best_val = float("inf")
    best_state = None
    wait = 0
    progress = ProgressBar(epochs, label="train-lstm") if show_progress else None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
            x_val_t = x_val_t.view(x_val_t.shape[0], x_val_t.shape[1], 1)
            val_pred = model(x_val_t)
            val_loss = criterion(
                val_pred, torch.tensor(y_val, dtype=torch.float32, device=device)
            ).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if progress:
                    progress.update(epochs - progress.current)
                break
        if progress:
            progress.update(1, extra=f"val={val_loss:.4f}")

    if progress:
        progress.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def evaluate_mse(model, x, y, device="cpu"):
    """Computes mean squared error for a model."""
    if hasattr(model, "predict_batch"):
        pred = model.predict_batch(x)
        return float(np.mean((pred - y) ** 2))

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        pred = model(x_t)
        return float(torch.mean((pred - y_t) ** 2).item())


def estimate_model_error(
    model, series_std, dim, lag, mode="quantile", quantile=0.95, scale=3.0
):
    """Estimates a probabilistic model error bound from one-step residuals."""
    try:
        x_calib, y_calib = build_supervised(series_std, dim, lag, horizon=1)
    except ValueError:
        return 0.0, "none"

    if x_calib.size == 0:
        return 0.0, "none"

    if hasattr(model, "predict_batch"):
        preds = model.predict_batch(x_calib)
    else:
        preds = np.array([model.predict(x) for x in x_calib], dtype=np.float64)

    preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    residuals = np.abs(preds - y_calib)
    if residuals.size == 0:
        return 0.0, "none"

    if mode == "max":
        return float(np.max(residuals)), "max"
    if mode == "mean_std":
        delta = float(np.mean(residuals) + scale * np.std(residuals))
        return delta, f"mean+{scale:.1f}std"
    delta = float(np.quantile(residuals, quantile))
    return delta, f"quantile@{quantile:.2f}"


def rolling_rmse(model, series_std, dim, lag, horizon_max):
    """Computes multi-step RMSE by rolling autoregression."""
    series_std = np.asarray(series_std, dtype=np.float64)
    window_len = (dim - 1) * lag + 1
    n = len(series_std) - window_len - horizon_max
    if n <= 0:
        return np.full(horizon_max, np.nan, dtype=np.float64)

    errors = np.zeros(horizon_max, dtype=np.float64)
    count = np.zeros(horizon_max, dtype=np.float64)
    for start in range(n):
        history = list(series_std[start : start + window_len])
        for h in range(horizon_max):
            x = [history[i * lag] for i in range(dim)]
            pred = model.predict(x)
            true = series_std[start + (dim - 1) * lag + h + 1]
            errors[h] += (pred - true) ** 2
            count[h] += 1.0
            history.append(pred)
            history.pop(0)

    rmse = np.sqrt(errors / np.maximum(count, 1.0))
    return rmse


def horizon_from_rmse(rmse, tolerance):
    """Returns the first horizon where RMSE exceeds tolerance."""
    for idx, value in enumerate(rmse, start=1):
        if value >= tolerance:
            return idx
    return len(rmse)


def horizon_from_lyapunov(lyapunov, init_err, tolerance):
    """Estimates theoretical horizon from Lyapunov growth."""
    if lyapunov <= 0 or init_err <= 0:
        return float("inf")
    if tolerance <= init_err:
        return 0.0
    return math.log(tolerance / init_err) / lyapunov


def plot_rmse(rmse_by_h, horizon_real, horizon_theory, horizon_model, out_path):
    """Saves RMSE vs horizon plot."""
    x = np.arange(1, len(rmse_by_h) + 1, dtype=np.float64)
    mask = np.isfinite(rmse_by_h)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], rmse_by_h[mask], label="RMSE")
    plt.axvline(horizon_real, color="tab:red", linestyle="--", label="H_real")
    if math.isfinite(horizon_theory):
        plt.axvline(horizon_theory, color="tab:green", linestyle="--", label="H_theory")
    if horizon_model is not None and math.isfinite(horizon_model):
        plt.axvline(
            horizon_model,
            color="tab:purple",
            linestyle="--",
            label="H_model",
        )
    plt.xlabel("Horizon (steps)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_log_divergence(rmse_by_h, lyap_step, out_path):
    """Saves log-divergence plot with Lyapunov slope."""
    eps = 1e-8
    log_err = np.log(rmse_by_h + eps)
    x = np.arange(len(log_err), dtype=np.float64)
    mask = np.isfinite(log_err)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x[mask], log_err[mask], label="log(RMSE)")
    if np.isfinite(log_err[0]):
        line = log_err[0] + lyap_step * x
        plt.plot(x[mask], line[mask], label="Lyapunov slope")
    plt.xlabel("Horizon (steps)")
    plt.ylabel("log(RMSE)")
    plt.title("Log Divergence vs Horizon")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def get_series(args):
    """Generates the selected chaotic time series."""
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
        )
    if args.dataset == "rossler":
        return generate_rossler(
            args.series_len,
            dt=args.dt,
            a=args.a,
            b=args.b,
            c=args.c,
            warmup=args.warmup,
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
        )
    raise ValueError("Unknown dataset")


def select_embedding(args, train_series, val_series, dim_values, lag_values, device):
    """Selects the best (dim, lag) embedding based on validation criteria."""
    best = None
    progress = None
    if args.progress:
        progress = ProgressBar(len(dim_values) * len(lag_values), label="embed-search")
    for dim in dim_values:
        for lag in lag_values:
            try:
                x_train, y_train = build_supervised(train_series, dim, lag, horizon=1)
                x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
            except ValueError:
                if progress:
                    progress.update(1, extra=f"dim={dim} lag={lag}")
                continue

            if args.model == "linear":
                model = LinearAR(reg=args.linear_reg).fit(x_train, y_train)
                val_loss = evaluate_mse(model, x_val, y_val)
                wrapped = model
            elif args.model == "mlp":
                model, val_loss = train_mlp(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    input_dim=dim,
                    hidden_dim=args.mlp_hidden,
                    epochs=args.mlp_epochs,
                    lr=args.mlp_lr,
                    batch_size=args.mlp_batch,
                    patience=args.mlp_patience,
                    device=device,
                    show_progress=False,
                )
                wrapped = TorchWrapper(model, device)
            else:
                model, val_loss = train_lstm(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    hidden_dim=args.lstm_hidden,
                    num_layers=args.lstm_layers,
                    epochs=args.lstm_epochs,
                    lr=args.lstm_lr,
                    batch_size=args.lstm_batch,
                    patience=args.lstm_patience,
                    device=device,
                    show_progress=False,
                )
                wrapped = TorchSeqWrapper(model, device)

            selection = {
                "metric": args.selection_metric,
                "score": -val_loss,
                "horizon": None,
            }
            if args.selection_metric == "horizon":
                rmse_val = rolling_rmse(
                    wrapped, val_series, dim, lag, args.selection_horizon_max
                )
                base_err = rmse_val[0] if rmse_val.size > 0 else 0.0
                if args.error_mode == "relative":
                    tolerance = base_err * args.error_factor
                else:
                    tolerance = args.error_tolerance
                if not np.isfinite(tolerance) or tolerance <= 0:
                    horizon_val = 0
                else:
                    horizon_val = horizon_from_rmse(rmse_val, tolerance)
                selection["score"] = horizon_val
                selection["horizon"] = horizon_val

            if best is None:
                best = {
                    "dim": dim,
                    "lag": lag,
                    "val_loss": val_loss,
                    "model": wrapped,
                    "selection": selection,
                }
                if progress:
                    progress.update(
                        1,
                        extra=f"dim={dim} lag={lag} val={val_loss:.4f}",
                    )
                continue

            if args.selection_metric == "horizon":
                if selection["score"] > best["selection"]["score"]:
                    best = {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": val_loss,
                        "model": wrapped,
                        "selection": selection,
                    }
                elif selection["score"] == best["selection"]["score"]:
                    if val_loss < best["val_loss"]:
                        best = {
                            "dim": dim,
                            "lag": lag,
                            "val_loss": val_loss,
                            "model": wrapped,
                            "selection": selection,
                        }
            else:
                if val_loss < best["val_loss"]:
                    best = {
                        "dim": dim,
                        "lag": lag,
                        "val_loss": val_loss,
                        "model": wrapped,
                        "selection": selection,
                    }
            if progress:
                extra = f"dim={dim} lag={lag} val={val_loss:.4f}"
                if selection["horizon"] is not None:
                    extra += f" h={selection['horizon']}"
                progress.update(1, extra=extra)
    if best is None:
        raise RuntimeError("No valid embedding configuration found.")
    if progress:
        progress.close()
    return best


def train_final_model(
    args,
    train_series,
    val_series,
    dim,
    lag,
    device,
    show_progress=False,
):
    """Trains the final model on train+val with the chosen embedding."""
    merged = np.concatenate([train_series, val_series], axis=0)
    x_train, y_train = build_supervised(merged, dim, lag, horizon=1)
    x_val, y_val = build_supervised(val_series, dim, lag, horizon=1)
    if args.model == "linear":
        model = LinearAR(reg=args.linear_reg).fit(x_train, y_train)
        return model
    if args.model == "mlp":
        model, _ = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=dim,
            hidden_dim=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch,
            patience=args.mlp_patience,
            device=device,
            show_progress=show_progress,
        )
        return TorchWrapper(model, device)
    model, _ = train_lstm(
        x_train,
        y_train,
        x_val,
        y_val,
        hidden_dim=args.lstm_hidden,
        num_layers=args.lstm_layers,
        epochs=args.lstm_epochs,
        lr=args.lstm_lr,
        batch_size=args.lstm_batch,
        patience=args.lstm_patience,
        device=device,
        show_progress=show_progress,
    )
    return TorchSeqWrapper(model, device)


def run_experiment(args):
    """Runs a full horizon experiment and writes summary CSV output."""
    set_seed(args.seed)
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    series = get_series(args)

    train_raw, val_raw, calib_raw, test_raw = split_series(
        series,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        calib_ratio=args.calib_ratio,
    )
    train_std, mean, std = standardize_series(train_raw)
    val_std = (val_raw - mean) / std
    calib_std = (calib_raw - mean) / std if calib_raw.size else val_std
    test_std = (test_raw - mean) / std

    dim_values = list(range(args.dim_min, args.dim_max + 1))
    lag_values = list(range(args.lag_min, args.lag_max + 1))

    t0 = time.time()
    best = select_embedding(
        args,
        train_std,
        val_std,
        dim_values,
        lag_values,
        device,
    )
    model = train_final_model(
        args,
        train_std,
        val_std,
        best["dim"],
        best["lag"],
        device,
        show_progress=args.progress,
    )

    x_test, y_test = build_supervised(test_std, best["dim"], best["lag"], horizon=1)
    test_mse = evaluate_mse(model, x_test, y_test, device=device)

    rmse_by_h = rolling_rmse(
        model, test_std, best["dim"], best["lag"], args.horizon_max
    )
    base_err = rmse_by_h[0] if rmse_by_h.size > 0 else 0.0
    if args.error_mode == "relative":
        tolerance = base_err * args.error_factor
    else:
        tolerance = args.error_tolerance
    horizon_real = horizon_from_rmse(rmse_by_h, tolerance)

    lyap_dim = args.lyap_dim if args.lyap_dim is not None else best["dim"]
    lyap_lag = args.lyap_lag if args.lyap_lag is not None else best["lag"]
    lyap_step, _ = estimate_lyapunov(
        np.concatenate([train_raw, val_raw]),
        dim=lyap_dim,
        lag=lyap_lag,
        max_t=args.lyap_max_t,
        theiler=args.lyap_theiler,
        fit_start=args.lyap_fit_start,
        fit_end=args.lyap_fit_end,
        dt=args.dt if args.dataset != "logistic" else 1.0,
    )
    init_err = rmse_by_h[0] if rmse_by_h.size > 0 else 0.0
    dt = args.dt if args.dataset != "logistic" else 1.0
    lyap_time = lyap_step / dt if dt > 0 else 0.0
    horizon_theory_steps = horizon_from_lyapunov(lyap_step, base_err, tolerance)
    horizon_theory_time = (
        horizon_theory_steps * dt if math.isfinite(horizon_theory_steps) else float("inf")
    )
    horizon_real_time = horizon_real * dt

    model_error, model_error_mode = estimate_model_error(
        model,
        calib_std,
        best["dim"],
        best["lag"],
        mode=args.delta_mode,
        quantile=args.delta_quantile,
        scale=args.delta_scale,
    )
    exp_dim = args.expansion_dim if args.expansion_dim is not None else lyap_dim
    exp_lag = args.expansion_lag if args.expansion_lag is not None else lyap_lag
    exp_series = np.concatenate([train_std, val_std], axis=0)
    expansion_q, _ = estimate_expansion_quantile(
        exp_series,
        dim=exp_dim,
        lag=exp_lag,
        quantile=args.expansion_quantile,
        theiler=args.expansion_theiler,
        max_pairs=args.expansion_samples,
        seed=args.seed,
    )
    horizon_model_steps = horizon_from_model_bound_by_growth(
        expansion_q, base_err, model_error, tolerance
    )
    horizon_model_time = (
        horizon_model_steps * dt if math.isfinite(horizon_model_steps) else float("inf")
    )

    os.makedirs(args.output_dir, exist_ok=True)
    csv_name = "horizon_results.csv"
    csv_path = os.path.join(args.output_dir, csv_name)
    header = [
        "dataset",
        "model",
        "dim",
        "lag",
        "val_mse",
        "selection_metric",
        "selection_horizon",
        "test_mse",
        "lyapunov_step",
        "lyapunov_time",
        "lyapunov_dim",
        "lyapunov_lag",
        "horizon_real",
        "horizon_real_time",
        "horizon_theory",
        "horizon_theory_time",
        "error_mode",
        "error_factor",
        "error_tolerance",
        "error_tolerance_used",
        "calib_ratio",
        "model_error",
        "model_error_mode",
        "horizon_model",
        "horizon_model_time",
        "expansion_quantile",
        "expansion_samples",
        "expansion_theiler",
        "expansion_dim",
        "expansion_lag",
        "expansion_Lq",
        "bound_mode",
    ]

    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header != header:
            base, ext = os.path.splitext(csv_name)
            suffix = 2
            while True:
                alt_name = f"{base}_v{suffix}{ext}"
                alt_path = os.path.join(args.output_dir, alt_name)
                if not os.path.exists(alt_path):
                    csv_path = alt_path
                    break
                suffix += 1
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                args.dataset,
                args.model,
                best["dim"],
                best["lag"],
                f"{best['val_loss']:.6f}",
                args.selection_metric,
                best["selection"]["horizon"]
                if best.get("selection") and best["selection"]["horizon"] is not None
                else "",
                f"{test_mse:.6f}",
                f"{lyap_step:.6f}",
                f"{lyap_time:.6f}",
                lyap_dim,
                lyap_lag,
                horizon_real,
                f"{horizon_real_time:.3f}",
                f"{horizon_theory_steps:.3f}"
                if math.isfinite(horizon_theory_steps)
                else "inf",
                f"{horizon_theory_time:.3f}"
                if math.isfinite(horizon_theory_time)
                else "inf",
                args.error_mode,
                args.error_factor,
                args.error_tolerance,
                f"{tolerance:.6f}",
                f"{args.calib_ratio:.3f}",
                f"{model_error:.6f}",
                model_error_mode,
                f"{horizon_model_steps:.3f}"
                if math.isfinite(horizon_model_steps)
                else "inf",
                f"{horizon_model_time:.3f}"
                if math.isfinite(horizon_model_time)
                else "inf",
                f"{args.expansion_quantile:.3f}",
                args.expansion_samples,
                args.expansion_theiler,
                exp_dim,
                exp_lag,
                f"{expansion_q:.6f}",
                "probabilistic",
            ]
        )

    if args.plot:
        rmse_path = os.path.join(args.output_dir, f"{args.plot_prefix}_rmse.png")
        log_path = os.path.join(args.output_dir, f"{args.plot_prefix}_log.png")
        plot_rmse(
            rmse_by_h, horizon_real, horizon_theory_steps, horizon_model_steps, rmse_path
        )
        plot_log_divergence(rmse_by_h, lyap_step, log_path)
        print(f"Plots saved to {rmse_path} and {log_path}")

    elapsed = time.time() - t0
    selection_note = ""
    if best.get("selection") and best["selection"]["horizon"] is not None:
        selection_note = f" sel_h={best['selection']['horizon']}"
    print(
        f"Best dim={best['dim']} lag={best['lag']} val={best['val_loss']:.6f} "
        f"test={test_mse:.6f} lyap_step={lyap_step:.4f} lyap_time={lyap_time:.4f} "
        f"horizon_real={horizon_real} horizon_real_time={horizon_real_time:.2f} "
        f"horizon_theory={horizon_theory_steps:.2f} horizon_theory_time={horizon_theory_time:.2f} "
        f"horizon_model={horizon_model_steps:.2f} horizon_model_time={horizon_model_time:.2f} "
        f"delta={model_error:.4f} Lq={expansion_q:.4f} tol={tolerance:.6f}"
        f"{selection_note} elapsed={elapsed:.1f}s"
    )
    return {
        "dim": best["dim"],
        "lag": best["lag"],
        "val_loss": best["val_loss"],
        "test_mse": test_mse,
        "lyapunov_step": lyap_step,
        "lyapunov_time": lyap_time,
        "lyapunov_dim": lyap_dim,
        "lyapunov_lag": lyap_lag,
        "horizon_real": horizon_real,
        "horizon_theory": horizon_theory_steps,
        "horizon_real_time": horizon_real_time,
        "horizon_theory_time": horizon_theory_time,
        "horizon_model": horizon_model_steps,
        "horizon_model_time": horizon_model_time,
        "model_error": model_error,
        "model_error_mode": model_error_mode,
        "calib_ratio": args.calib_ratio,
        "expansion_quantile": args.expansion_quantile,
        "expansion_samples": args.expansion_samples,
        "expansion_theiler": args.expansion_theiler,
        "expansion_dim": exp_dim,
        "expansion_lag": exp_lag,
        "expansion_Lq": expansion_q,
        "bound_mode": "probabilistic",
        "error_tolerance_used": tolerance,
        "selection_metric": args.selection_metric,
        "selection_horizon": best["selection"]["horizon"]
        if best.get("selection") and best["selection"]["horizon"] is not None
        else None,
    }


def build_parser(add_help=True):
    """Builds the argument parser for the experiment CLI."""
    parser = argparse.ArgumentParser(
        description="AI-driven prediction horizon experiment for chaotic series",
        add_help=add_help,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["logistic", "lorenz", "rossler", "mackey_glass"],
        default="lorenz",
    )
    parser.add_argument("--series-len", type=int, default=4000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--calib-ratio", type=float, default=0.05)

    parser.add_argument("--dim-min", type=int, default=2)
    parser.add_argument("--dim-max", type=int, default=8)
    parser.add_argument("--lag-min", type=int, default=1)
    parser.add_argument("--lag-max", type=int, default=8)
    parser.add_argument(
        "--model", type=str, choices=["linear", "mlp", "lstm"], default="mlp"
    )
    parser.add_argument("--linear-reg", type=float, default=1e-4)

    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=80)
    parser.add_argument("--mlp-lr", type=float, default=1e-3)
    parser.add_argument("--mlp-batch", type=int, default=64)
    parser.add_argument("--mlp-patience", type=int, default=12)

    parser.add_argument("--lstm-hidden", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--lstm-epochs", type=int, default=80)
    parser.add_argument("--lstm-lr", type=float, default=1e-3)
    parser.add_argument("--lstm-batch", type=int, default=64)
    parser.add_argument("--lstm-patience", type=int, default=12)

    parser.add_argument("--horizon-max", type=int, default=50)
    parser.add_argument(
        "--selection-metric",
        type=str,
        choices=["val_mse", "horizon"],
        default="val_mse",
    )
    parser.add_argument("--selection-horizon-max", type=int, default=20)
    parser.add_argument(
        "--error-mode",
        type=str,
        choices=["absolute", "relative"],
        default="relative",
    )
    parser.add_argument("--error-factor", type=float, default=10.0)
    parser.add_argument("--error-tolerance", type=float, default=1.0)
    parser.add_argument(
        "--delta-mode",
        type=str,
        choices=["quantile", "max", "mean_std"],
        default="quantile",
    )
    parser.add_argument("--delta-quantile", type=float, default=0.95)
    parser.add_argument("--delta-scale", type=float, default=3.0)
    parser.add_argument("--expansion-quantile", type=float, default=0.95)
    parser.add_argument("--expansion-samples", type=int, default=500)
    parser.add_argument("--expansion-theiler", type=int, default=10)
    parser.add_argument("--expansion-dim", type=int, default=None)
    parser.add_argument("--expansion-lag", type=int, default=None)

    parser.add_argument("--lyap-max-t", type=int, default=25)
    parser.add_argument("--lyap-theiler", type=int, default=10)
    parser.add_argument("--lyap-fit-start", type=int, default=1)
    parser.add_argument("--lyap-fit-end", type=int, default=10)
    parser.add_argument("--lyap-dim", type=int, default=None)
    parser.add_argument("--lyap-lag", type=int, default=None)

    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-prefix", type=str, default="horizon")
    parser.add_argument("--progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")

    parser.add_argument("--r", type=float, default=4.0)
    parser.add_argument("--x0", type=float, default=0.2)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--rho", type=float, default=28.0)
    parser.add_argument("--beta", type=float, default=8.0 / 3.0)

    parser.add_argument("--a", type=float, default=0.2)
    parser.add_argument("--b", type=float, default=0.2)
    parser.add_argument("--c", type=float, default=5.7)

    parser.add_argument("--tau", type=int, default=17)
    parser.add_argument("--mg-beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--n", type=int, default=10)
    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
