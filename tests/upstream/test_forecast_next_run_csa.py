"""Small regression tests for forecast leakage controls (no production writes)."""
import unittest

import numpy as np
import pandas as pd

from forecast_next_run_csa import FoldFeatures, nullable_any, splits, inner_splits


class ForecastGuards(unittest.TestCase):
    def test_unknown_labels_are_not_negatives(self):
        self.assertTrue(np.isnan(nullable_any(pd.Series([0., np.nan]))))
        self.assertTrue(np.isnan(nullable_any(pd.Series([np.nan]))))
        self.assertEqual(nullable_any(pd.Series([1., np.nan])), 1.)
        self.assertEqual(nullable_any(pd.Series([0., 0.])), 0.)

    def test_preprocessing_does_not_learn_test_values(self):
        rng = np.random.default_rng(8)
        a = rng.normal(size=(60, 3))
        a[::5, 1] = np.nan
        x = pd.DataFrame(a, columns=['signal', 'part_missing', 'other'])
        x['duplicate'] = x.signal
        x['constant'] = 1.
        c = pd.DataFrame({'mode': ['A', 'B']*30})
        y = (x.signal > 0).astype(int).to_numpy()
        pre = FoldFeatures(8).fit(x, c, y)
        medians = pre.median.copy()
        names = pre.names.copy()
        future = x.iloc[:3].copy()*1e9
        pred = pre.transform(future, pd.DataFrame({'mode': ['unknown']*3}))
        self.assertEqual(pred.shape[0], 3)
        self.assertTrue(np.isfinite(pred).all())
        np.testing.assert_array_equal(pre.median, medians)
        np.testing.assert_array_equal(pre.names, names)
        self.assertFalse('constant' in pre.names)
        self.assertLessEqual(sum(n in pre.names for n in ['duplicate', 'signal']), 1)

    def test_temporal_and_instrument_split_purges_unavailable_labels(self):
        dates = pd.date_range('2026-01-01', periods=100, freq='D', tz='UTC')
        f = pd.DataFrame({'start': dates, 'available': dates+pd.Timedelta(days=12),
                          'physical_run_id': [f'r{i}' for i in range(100)],
                          'instrument_id': ['a', 'b']*50})
        for scheme, fold, tr, te, cutoff in splits(f):
            self.assertTrue(f.loc[tr, 'available'].lt(cutoff).all())
            self.assertTrue(f.loc[te, 'start'].ge(cutoff).all())
            self.assertFalse(set(tr) & set(te))
            if scheme == 'unseen_instrument_and_future':
                self.assertTrue(f.loc[tr, 'instrument_id'].ne(fold).all())
                self.assertTrue(f.loc[te, 'instrument_id'].eq(fold).all())
        for tr, va in inner_splits(f, f.index.to_numpy()):
            self.assertTrue(f.loc[tr, 'available'].lt(f.loc[va, 'start'].min()).all())


if __name__ == '__main__':
    unittest.main()
