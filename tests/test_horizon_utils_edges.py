"""Edge-case unit tests for horizon_utils."""

import math
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_utils import adaptive_horizon, horizon_from_model_bound_by_growth


def test_horizon_from_model_bound_by_growth_cases():
    h_inf = horizon_from_model_bound_by_growth(
        1.0, init_err=0.01, delta=0.0, tolerance=0.1
    )
    h_lin = horizon_from_model_bound_by_growth(
        1.0, init_err=0.01, delta=0.02, tolerance=0.1
    )
    h_zero = horizon_from_model_bound_by_growth(
        0.0, init_err=0.01, delta=0.02, tolerance=0.1
    )

    assert math.isinf(h_inf)
    assert h_lin == 5
    assert h_zero == 0.0


def test_adaptive_horizon_clamps_scalar_and_array():
    assert adaptive_horizon(2.9, horizon_max=5) == 2

    arr = adaptive_horizon(np.array([-1.0, 1.2, 9.9]), horizon_max=5)
    np.testing.assert_array_equal(arr, np.array([1, 1, 5]))
