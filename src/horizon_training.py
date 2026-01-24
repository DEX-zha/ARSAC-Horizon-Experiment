"""Training utilities for horizon experiments."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.horizon_models import LSTMPredictor, MLPPredictor
from src.horizon_progress import ProgressBar


def build_multistep_supervised(series, dim, lag, horizon):
    """Builds supervised inputs with multi-step targets."""
    series = np.asarray(series, dtype=np.float64)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n = len(series) - (dim - 1) * lag - horizon
    if n <= 0:
        raise ValueError("series too short for given dim/lag/horizon")
    x = np.empty((n, dim), dtype=np.float64)
    for i in range(dim):
        start = i * lag
        x[:, i] = series[start : start + n]
    y = np.empty((n, horizon), dtype=np.float64)
    y_start = (dim - 1) * lag + 1
    for h in range(horizon):
        y[:, h] = series[y_start + h : y_start + h + n]
    return x, y


def schedule_teacher_forcing(epoch, total_epochs, start, end):
    """Linear teacher forcing schedule."""
    if total_epochs <= 1:
        return float(end)
    ratio = start + (end - start) * (epoch / (total_epochs - 1))
    return float(min(max(ratio, 0.0), 1.0))


def multistep_loss(model, history, targets, criterion, tf_ratio, is_lstm):
    """Computes multi-step loss with scheduled teacher forcing."""
    batch_size, horizon = targets.shape
    total_loss = 0.0
    state = history
    for step in range(horizon):
        if is_lstm:
            pred = model(state.view(batch_size, state.shape[1], 1))
        else:
            pred = model(state)
        pred = pred.view(-1)
        target = targets[:, step].view(-1)
        total_loss = total_loss + criterion(pred, target)
        if tf_ratio >= 1.0:
            next_val = target
        elif tf_ratio <= 0.0:
            next_val = pred
        else:
            mask = torch.rand(batch_size, device=state.device) < tf_ratio
            next_val = torch.where(mask, target, pred)
        state = torch.cat([state[:, 1:], next_val.unsqueeze(1)], dim=1)
    return total_loss / float(horizon)


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
    """Trains an MLP with early stopping."""
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


def pinball_loss(pred, target, quantile):
    """Computes the pinball loss for quantile regression."""
    diff = target - pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1.0) * diff))


def train_quantile_mlp(
    x_train,
    y_train,
    x_val,
    y_val,
    input_dim,
    quantile=0.9,
    hidden_dim=64,
    epochs=50,
    lr=1e-3,
    batch_size=64,
    patience=10,
    device="cpu",
    show_progress=False,
):
    """Trains an MLP for quantile regression with early stopping."""
    if x_val.size == 0 or y_val.size == 0:
        x_val = x_train
        y_val = y_train

    model = MLPPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )

    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    wait = 0
    progress = ProgressBar(epochs, label="train-quantile") if show_progress else None
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = pinball_loss(pred, yb, quantile)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = pinball_loss(val_pred, y_val_t, quantile).item()

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
    """Trains an LSTM with early stopping."""
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


def train_mlp_multistep(
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
    tf_start=1.0,
    tf_end=0.2,
    tf_val=0.0,
    device="cpu",
    show_progress=False,
):
    """Trains an MLP with multi-step teacher forcing."""
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
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    wait = 0
    progress = ProgressBar(epochs, label="train-mlp") if show_progress else None
    for epoch in range(epochs):
        model.train()
        tf_ratio = schedule_teacher_forcing(epoch, epochs, tf_start, tf_end)
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = multistep_loss(model, xb, yb, criterion, tf_ratio, is_lstm=False)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = multistep_loss(
                model, x_val_t, y_val_t, criterion, tf_val, is_lstm=False
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


def train_lstm_multistep(
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
    tf_start=1.0,
    tf_end=0.2,
    tf_val=0.0,
    device="cpu",
    show_progress=False,
):
    """Trains an LSTM with multi-step teacher forcing."""
    model = LSTMPredictor(hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    wait = 0
    progress = ProgressBar(epochs, label="train-lstm") if show_progress else None
    for epoch in range(epochs):
        model.train()
        tf_ratio = schedule_teacher_forcing(epoch, epochs, tf_start, tf_end)
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = multistep_loss(model, xb, yb, criterion, tf_ratio, is_lstm=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = multistep_loss(
                model, x_val_t, y_val_t, criterion, tf_val, is_lstm=True
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
