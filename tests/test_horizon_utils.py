"""Unit tests for horizon_utils."""

import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.horizon_utils import (
    build_supervised,
    embed_series,
    estimate_expansion_quantile,
    estimate_lyapunov,
    generate_logistic_map,
    horizon_from_model_bound,
)


class TestHorizonUtils(unittest.TestCase):
    """Tests for embedding and Lyapunov utilities."""

    def test_embed_series(self):
        series = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        embedded = embed_series(series, dim=3, lag=1)
        self.assertEqual(embedded.shape, (4, 3))
        np.testing.assert_array_equal(
            embedded,
            np.array(
                [
                    [0, 1, 2],
                    [1, 2, 3],
                    [2, 3, 4],
                    [3, 4, 5],
                ],
                dtype=np.float64,
            ),
        )

    def test_build_supervised(self):
        series = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
        x, y = build_supervised(series, dim=2, lag=1, horizon=1)
        self.assertEqual(x.shape, (4, 2))
        np.testing.assert_array_equal(x, np.array([[0, 1], [1, 2], [2, 3], [3, 4]]))
        np.testing.assert_array_equal(y, np.array([2, 3, 4, 5]))

    def test_lyapunov_signs(self):
        chaotic = generate_logistic_map(1200, r=4.0, x0=0.2, warmup=200)
        stable = generate_logistic_map(1200, r=2.5, x0=0.2, warmup=200)
        lyap_c, _ = estimate_lyapunov(
            chaotic,
            dim=3,
            lag=1,
            max_t=20,
            theiler=20,
            fit_start=1,
            fit_end=10,
            dt=1.0,
        )
        lyap_s, _ = estimate_lyapunov(
            stable,
            dim=3,
            lag=1,
            max_t=20,
            theiler=20,
            fit_start=1,
            fit_end=10,
            dt=1.0,
        )
        self.assertGreater(lyap_c, 0.1)
        self.assertLess(lyap_s, 0.05)

    def test_model_bound_monotonic(self):
        h_low = horizon_from_model_bound(0.1, init_err=0.01, delta=0.0, tolerance=0.1)
        h_high = horizon_from_model_bound(0.1, init_err=0.01, delta=0.05, tolerance=0.1)
        self.assertLessEqual(h_high, h_low)

    def test_expansion_quantile(self):
        series = generate_logistic_map(800, r=4.0, x0=0.2, warmup=100)
        lq, ratios = estimate_expansion_quantile(
            series, dim=3, lag=1, quantile=0.9, theiler=5, max_pairs=100, seed=0
        )
        self.assertTrue(lq > 0.0)
        self.assertTrue(ratios.size >= 0)


if __name__ == "__main__":
    unittest.main()
