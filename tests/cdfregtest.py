import pyximport
pyximport.install()

import unittest
import numpy as np
from jpt.base.quantiles import QuantileDistribution


class MyTestCase(unittest.TestCase):
    def test_quantile_dist(self):
        data = np.array([[1.], [2.]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0, 1]), 0)
        print(q.cdf.pfmt())
        self.assertEqual(True, False)  # add assertion here

    def test_dist_merge(self):
        data1 = np.array([[1.], [1.1], [1.1], [1.2], [1.4], [1.2], [1.3]], dtype=np.float64)
        data2 = np.array([[5.], [5.1], [5.2], [5.2], [5.2], [5.3], [5.4]], dtype=np.float64)
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q1.fit(data1, np.array(range(data1.shape[0])), 0)
        q2.fit(data2, np.array(range(data2.shape[0])), 0)
        print(QuantileDistribution.merge([q1, q2], [.5, .5]).cdf.pfmt())
        self.assertEqual(True, False)  # add assertion here

    def test_dist_merge_jump_functions(self):
        data1 = np.array([[1.]], dtype=np.float64)
        data2 = np.array([[2.]], dtype=np.float64)
        data3 = np.array([[3.]], dtype=np.float64)
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q3 = QuantileDistribution()
        q1.fit(data1, np.array(range(data1.shape[0])), 0)
        q2.fit(data2, np.array(range(data2.shape[0])), 0)
        q3.fit(data3, np.array(range(data3.shape[0])), 0)
        print(q1.cdf.pfmt())
        print(q2.cdf.pfmt())
        print(q3.cdf.pfmt())
        print('===')
        print(QuantileDistribution.merge([q1, q2, q3], [1/3] * 3).cdf.pfmt())
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
