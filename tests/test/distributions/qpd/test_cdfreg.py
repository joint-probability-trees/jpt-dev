from unittest import TestCase

import numpy as np

from jpt.distributions.qpd import QuantileDistribution

from jpt.base.functions import PiecewiseFunction


# ----------------------------------------------------------------------

class QuantileDistributionFitTest(TestCase):

    def test_quantile_dist_linear(self):
        # Arrange
        data = np.array([[1.], [2.]], dtype=np.float64)
        q = QuantileDistribution()

        # Act
        q.fit(
            data,
            np.array([0, 1]),
            0
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,1.000[': '0.0',
                '[1.0,2.0000000000000004[':
                    '1.000x - 1.000',
                '[2.0000000000000004,∞[': '1.0',
            }),
            q.cdf
        )

    def test_quantile_dist_jump(self):
        # Arrange
        data = np.array([[2.]], dtype=np.float64)
        q = QuantileDistribution()

        # Act
        q.fit(data, np.array([0]), 0)

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,2.000[': '0.0',
                '[2.000,∞[': '1.0',
            }),
            q.cdf
        )

    # TODO: finish test implementation
    def test_quantile_dist_jump_first(self):
        # Arrange
        data = np.array([
            [1.],
            [1.],
            [2.],
            [3.]
        ], dtype=np.float64)
        q = QuantileDistribution()

        # Act
        q.fit(data, None, 0)
        print(q.cdf)

    # TODO: finish test implementation
    def test_quantile_dist_jump_last(self):
        # Arrange
        data = np.array([
            [1.],
            [2.],
            [3.],
            [3.]
        ], dtype=np.float64)
        q = QuantileDistribution()

        # Act
        q.fit(data, None, 0)
        print(q.cdf)
