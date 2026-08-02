import unittest

import numpy as np
import pandas as pd

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


# ----------------------------------------------------------------------------------------------------------------------

def make_data(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    half = n // 2
    x = np.concatenate([
        rng.normal(-3., .5, half),
        rng.normal(3., .5, n - half)
    ])
    y = np.concatenate([
        2. * x[:half] + rng.normal(0, .3, half),
        -2. * x[half:] + rng.normal(0, .3, n - half)
    ])
    label = np.array(['lo'] * half + ['hi'] * (n - half))
    return pd.DataFrame({'x': x, 'y': y, 'label': label})


# ----------------------------------------------------------------------------------------------------------------------

class SampleWeightTest(unittest.TestCase):
    '''``JPT.learn(..., sample_weight=...)`` -- exact integer-weighted learning.'''

    @classmethod
    def setUpClass(cls):
        cls.data = make_data()
        cls.variables = infer_from_dataframe(
            cls.data,
            scale_numeric_types=False
        )

    def _tree(self) -> JPT:
        return JPT(variables=self.variables, min_samples_leaf=.1)

    def test_uniform_weights_reproduce_unweighted_fit(self):
        unweighted = self._tree().learn(self.data)
        weighted = self._tree().learn(
            self.data,
            sample_weight=np.ones(len(self.data))
        )
        self.assertEqual(len(unweighted.leaves), len(weighted.leaves))
        self.assertTrue(np.allclose(
            unweighted.likelihood(self.data),
            weighted.likelihood(self.data)
        ))

    def test_weighted_equals_materialized_resample(self):
        weights = np.random.RandomState(7).multinomial(
            len(self.data),
            np.full(len(self.data), 1. / len(self.data))
        )
        weighted = self._tree().learn(self.data, sample_weight=weights)
        resample = self.data.iloc[
            np.repeat(np.arange(len(self.data)), weights)
        ].reset_index(drop=True)
        resampled = self._tree().learn(resample)
        self.assertEqual(len(weighted.leaves), len(resampled.leaves))
        self.assertTrue(np.allclose(
            weighted.likelihood(self.data),
            resampled.likelihood(self.data)
        ))

    def test_zero_weight_rows_are_ignored(self):
        weights = np.ones(len(self.data), dtype=int)
        weights[self.data.label == 'hi'] = 0
        tree = self._tree().learn(self.data, sample_weight=weights)
        self.assertAlmostEqual(1., tree.infer({'label': 'lo'}), places=10)

    def test_invalid_weights_raise(self):
        for bad in (
                np.full(len(self.data), .5),   # fractional
                np.zeros(len(self.data)),      # all zero
                -np.ones(len(self.data)),      # negative
                np.ones(5),                    # wrong length
        ):
            with self.assertRaises(ValueError):
                self._tree().learn(self.data, sample_weight=bad)


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
