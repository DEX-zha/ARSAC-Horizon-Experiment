
import sys
import os
sys.path.append(os.getcwd())

print("Importing src.horizon_data...")
try:
    import src.horizon_data
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing src.horizon_forecast...")
try:
    import src.horizon_forecast
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing src.horizon_calibration...")
try:
    import src.horizon_calibration
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")

print("Importing src.horizon_experiment...")
try:
    import src.horizon_experiment
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
