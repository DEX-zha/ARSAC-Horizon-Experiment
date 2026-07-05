
import os
import sys

import numpy as np

# Add repo root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_utils import generate_lorenz, generate_rossler

def test_lorenz():
    length = 1000
    data = generate_lorenz(length, dt=0.01)
    assert data.shape[0] == length
    assert not np.isnan(data).any()

def test_rossler():
    length = 1000
    data = generate_rossler(length, dt=0.05)
    assert data.shape[0] == length
    assert not np.isnan(data).any()

if __name__ == "__main__":
    test_lorenz()
    test_rossler()
