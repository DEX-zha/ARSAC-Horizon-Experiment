"""Unit tests for horizon_experiment models."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_experiment import LinearAR, train_lstm, train_mlp
from src.horizon_utils import set_seed


class TestHorizonModels(unittest.TestCase):
    """Tests for MLP/LSTM training and linear baseline."""

    def test_linear_ar_fit(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3))
        y = 2.0 * x[:, 0] - 0.5 * x[:, 1] + 0.1
        model = LinearAR(reg=1e-8).fit(x, y)
        pred = model.predict_batch(x)
        mse = np.mean((pred - y) ** 2)
        self.assertLess(mse, 1e-10)

    def test_mlp_train(self):
        set_seed(0)
        rng = np.random.default_rng(1)
        x = rng.normal(size=(256, 2))
        y = x[:, 0] - 0.25 * x[:, 1]
        x_train, y_train = x[:200], y[:200]
        x_val, y_val = x[200:], y[200:]
        model, val_loss = train_mlp(
            x_train,
            y_train,
            x_val,
            y_val,
            input_dim=2,
            hidden_dim=16,
            epochs=50,
            lr=1e-2,
            batch_size=32,
            patience=10,
            device="cpu",
        )
        self.assertLess(val_loss, 1e-3)

    def test_lstm_train(self):
        set_seed(1)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(256, 3))
        y = x[:, -1]
        x_train, y_train = x[:200], y[:200]
        x_val, y_val = x[200:], y[200:]
        model, val_loss = train_lstm(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dim=16,
            num_layers=1,
            epochs=50,
            lr=1e-2,
            batch_size=32,
            patience=10,
            device="cpu",
        )
        self.assertLess(val_loss, 1e-3)


if __name__ == "__main__":
    unittest.main()
