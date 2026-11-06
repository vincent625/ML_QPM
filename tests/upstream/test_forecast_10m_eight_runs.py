import unittest

import numpy as np
import pandas as pd

from forecast_10m_eight_runs import event_label, numeric_history


class EightRunForecastTests(unittest.TestCase):
    def test_exact_threshold_and_any_lane(self):
        self.assertEqual(event_label(pd.Series([10_000_000, 12_000_000])), 0)
        self.assertEqual(event_label(pd.Series([9_999_999, 30_000_000])), 1)

    def test_missing_lanes_not_negative(self):
        self.assertTrue(np.isnan(event_label(pd.Series([20_000_000, np.nan]))))
        self.assertEqual(event_label(pd.Series([200, np.nan])), 1)

    def test_history_trend_is_per_run(self):
        a = np.asarray([[[5+2*t] for t in range(8)]], dtype=np.float32)
        latest, mean, slope = numeric_history(a)
        np.testing.assert_allclose([latest.item(), mean.item(), slope.item()], [19, 12, 2])

    def test_sparse_trend_and_means(self):
        a = np.asarray([[[np.nan if t % 2 == 0 else 5+2*t] for t in range(8)]], dtype=np.float32)
        _, mean, slope = numeric_history(a)
        np.testing.assert_allclose([mean.item(), slope.item()], [13, 2])
        a[:, 1] = np.nan
        latest, mean, slope = numeric_history(a)
        self.assertEqual(latest.item(), 19)
        self.assertTrue(np.isnan(mean).all() and np.isnan(slope).all())

    def test_all_missing_does_not_become_zero(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            results = numeric_history(np.full((2, 8, 3), np.nan, dtype=np.float32))
        self.assertTrue(all(np.isnan(r).all() for r in results))

    def test_constant_field_has_no_rounding_trend(self):
        rng = np.random.default_rng(123)
        a = np.full((100, 8, 3), [108.1, 1e8, 1e11], dtype=np.float32)
        a[rng.random(a.shape) < .2] = np.nan
        _, _, slopes = numeric_history(a)
        np.testing.assert_allclose(slopes[np.isfinite(slopes)], 0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
