"""Model definitions and wrappers for horizon experiments."""

import numpy as np
import torch
import torch.nn as nn


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
