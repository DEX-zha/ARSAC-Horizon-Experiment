
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from horizon_utils import generate_lorenz, generate_rossler

def test_lorenz():
    print("Testing Lorenz...")
    length = 1000
    try:
        data = generate_lorenz(length, dt=0.01)
        if data.shape[0] != length:
            print(f"FAIL: Expected length {length}, got {data.shape[0]}")
            return False
        if np.isnan(data).any():
            print("FAIL: NaNs in output")
            return False
        print(f"SUCCESS: Generated {length} points. Mean: {data.mean():.4f}")
        return True
    except Exception as e:
        print(f"FAIL: Exception {e}")
        return False

def test_rossler():
    print("Testing Rossler...")
    length = 1000
    try:
        data = generate_rossler(length, dt=0.05)
        if data.shape[0] != length:
            print(f"FAIL: Expected length {length}, got {data.shape[0]}")
            return False
        if np.isnan(data).any():
            print("FAIL: NaNs in output")
            return False
        print(f"SUCCESS: Generated {length} points. Mean: {data.mean():.4f}")
        return True
    except Exception as e:
        print(f"FAIL: Exception {e}")
        return False

if __name__ == "__main__":
    ok_l = test_lorenz()
    ok_r = test_rossler()
    if ok_l and ok_r:
        print("All Physics Tests Passed")
        sys.exit(0)
    else:
        sys.exit(1)
