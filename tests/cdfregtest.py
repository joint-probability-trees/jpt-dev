import pyximport
pyximport.install()

from jpt.base.intervals import ContinuousSet, INC, EXC

import unittest
import numpy as np
from jpt.base.quantiles import QuantileDistribution, PiecewiseFunction, ConstantFunction, LinearFunction, Undefined


class TestCaseMerge(unittest.TestCase):

    def test_quantile_dist_linear(self):
        data = np.array([[1.], [2.]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0, 1]), 0)
        self.assertEqual(PiecewiseFunction.from_dict({
            ']-∞,1.000[': '0.0',
            '[1.000,2.000[': '1.000x - 1.000',
            '[2.000,∞[': '1.0',
            }), q.cdf)  # add assertion here

    def test_quantile_dist_jump(self):
        data = np.array([[2.]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0]), 0)
        self.assertEqual(PiecewiseFunction.from_dict({
            ']-∞,2.000[': '0.0',
            '[2.000,∞[': '1.0',
        }), q.cdf)

    def test_dist_merge(self):
        data1 = np.array([[1.], [1.1], [1.1], [1.2], [1.4], [1.2], [1.3]], dtype=np.float64)
        data2 = np.array([[5.], [5.1], [5.2], [5.2], [5.2], [5.3], [5.4]], dtype=np.float64)
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q1.fit(data1, np.array(range(data1.shape[0])), 0)
        q2.fit(data2, np.array(range(data2.shape[0])), 0)

        self.assertEqual(PiecewiseFunction.from_dict({']-∞,1.000[': 0.0,
                                                      '[1.000,1.200[': '1.667x - 1.667',
                                                      '[1.200,1.400[': '0.833x - 0.667',
                                                      '[1.400,5.000[': '-0.000x + 0.500',
                                                      '[5.000,5.300[': '1.389x - 6.444',
                                                      '[5.300,∞[': '1.0'}),
                         QuantileDistribution.merge([q1, q2], [.5, .5]).cdf)

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
        q = QuantileDistribution.merge([q1, q2, q3], [1/3] * 3)
        print(q.cdf.pfmt())
        print('---')
        print(q.ppf.pfmt())

    def test_merge_jumps(self):
        plf1 = PiecewiseFunction()
        plf1.intervals.append(ContinuousSet.fromstring(']-inf,0.000['))
        plf1.intervals.append(ContinuousSet.fromstring('[0.000, inf['))
        plf1.functions.append(ConstantFunction(0))
        plf1.functions.append(ConstantFunction(1))

        plf2 = PiecewiseFunction()
        plf2.intervals.append(ContinuousSet.fromstring(']-inf,0.5['))
        plf2.intervals.append(ContinuousSet.fromstring('[0.5, inf['))
        plf2.functions.append(ConstantFunction(0))
        plf2.functions.append(ConstantFunction(1))

        merged = QuantileDistribution.merge([QuantileDistribution.from_cdf(plf1),
                                             QuantileDistribution.from_cdf(plf2)])

        result = PiecewiseFunction()
        result.intervals.append(ContinuousSet.fromstring(']-inf,0.0['))
        result.intervals.append(ContinuousSet.fromstring('[0.0, .5['))
        result.intervals.append(ContinuousSet.fromstring('[0.5, inf['))
        result.functions.append(ConstantFunction(0))
        result.functions.append(ConstantFunction(.5))
        result.functions.append(ConstantFunction(1))
        print(result.pfmt())

        self.assertEqual(result, merged.cdf)


class TestCasePPFTransform(unittest.TestCase):

    def test_ppf_transform_jumps_only(self):
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.fromstring(']-inf,1['))
        cdf.intervals.append(ContinuousSet.fromstring('[1, 2['))
        cdf.intervals.append(ContinuousSet.fromstring('[2, 3['))
        cdf.intervals.append(ContinuousSet.fromstring('[3, 4['))
        cdf.intervals.append(ContinuousSet.fromstring('[4, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(ConstantFunction(.25))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(ConstantFunction(.75))
        cdf.functions.append(ConstantFunction(1))
        q = QuantileDistribution.from_cdf(cdf)

        self.assertEqual(q.ppf, PiecewiseFunction.from_dict({
            ']-∞,0.000[': None,
            '[0.,.25[': ConstantFunction(1),
            '[0.25,.5[': ConstantFunction(2),
            '[0.5,.75[': ConstantFunction(3),
            ContinuousSet(.75, np.nextafter(1, 2), INC, EXC): ConstantFunction(4),
            ContinuousSet(np.nextafter(1, 2), np.PINF, INC, EXC): None
        }))

    def test_ppf_transform(self):
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-inf,0.000['))
        cdf.intervals.append(ContinuousSet.parse('[0.000, 1['))
        cdf.intervals.append(ContinuousSet.parse('[1, 2['))
        cdf.intervals.append(ContinuousSet.parse('[2, 3['))
        cdf.intervals.append(ContinuousSet.parse('[3, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(LinearFunction.from_points((0, 0), (1, .5)))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(LinearFunction.from_points((2, .5), (3, 1)))
        cdf.functions.append(ConstantFunction(1))

        q = QuantileDistribution.from_cdf(cdf)
        self.assertEqual(q.ppf, PiecewiseFunction.from_dict({
            ']-∞,0.000[': None,
            '[0.0,.5[': str(LinearFunction.from_points((0, 0), (.5, 1))),
            ContinuousSet(.5, np.nextafter(1, 2), INC, EXC): LinearFunction(2, 1),
            ContinuousSet(np.nextafter(1, 2), np.PINF, INC, EXC): None
        }))


class PLFTest(unittest.TestCase):

    def test_plf_constant_from_dict(self):
        d = {
            ']-∞,1.000[': '0.0',
            '[1.000,2.000[': '0.25',
            '[3.000,4.000[': '0.5',
            '[4.000,5.000[': '0.75',
            '[5.000,∞[': '1.0'
        }

        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-inf,1['))
        cdf.intervals.append(ContinuousSet.parse('[1, 2['))
        cdf.intervals.append(ContinuousSet.parse('[3, 4['))
        cdf.intervals.append(ContinuousSet.parse('[4, 5['))
        cdf.intervals.append(ContinuousSet.parse('[5, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(ConstantFunction(.25))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(ConstantFunction(.75))
        cdf.functions.append(ConstantFunction(1))

        self.assertEqual(cdf, PiecewiseFunction.from_dict(d))

    def test_plf_linear_from_dict(self):
        d = {
            ']-∞,0.000[': 'undef.',
            '[0.000,0.500[': '2.000x',
            '[0.500,1.000[': '2.000x + 1.000',
            '[1.000,∞[': None
        }
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-∞,0.000['))
        cdf.intervals.append(ContinuousSet.parse('[0.000,0.500['))
        cdf.intervals.append(ContinuousSet.parse('[0.500,1.000['))
        cdf.intervals.append(ContinuousSet.parse('[1.000,∞['))
        cdf.functions.append(Undefined())
        cdf.functions.append(LinearFunction(2, 0))
        cdf.functions.append(LinearFunction(2, 1))
        cdf.functions.append(Undefined())

        self.assertEqual(cdf, PiecewiseFunction.from_dict(d))

    def test_plf_mixed_from_dict(self):
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-inf,0.000['))
        cdf.intervals.append(ContinuousSet.parse('[0.000, 1['))
        cdf.intervals.append(ContinuousSet.parse('[1, 2['))
        cdf.intervals.append(ContinuousSet.parse('[2, 3['))
        cdf.intervals.append(ContinuousSet.parse('[3, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(LinearFunction.from_points((0, 0), (1, .5)))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(LinearFunction.from_points((2, .5), (3, 1)))
        cdf.functions.append(ConstantFunction(1))

        self.assertEqual(cdf, PiecewiseFunction.from_dict({
            ']-∞,0.000[': 0,
            '[0.000,1.00[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2.000[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3.000,∞[': 1
        }))


if __name__ == '__main__':
    unittest.main()
