from unittest import TestCase

from matplotlib import pyplot as plt

import unittest
import numpy as np
import numpy.random

from jpt.base.errors import Unsatisfiability

try:
    from jpt.distributions.quantile.quantiles import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.distributions.quantile.quantiles import QuantileDistribution

from jpt.base.intervals import ContinuousSet, INC, EXC
from jpt.base.functions import (PiecewiseFunction, ConstantFunction, LinearFunction)


class TestCaseMerge(unittest.TestCase):

    def test_quantile_dist_linear(self):
        data = np.array([[1.], [2.]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0, 1]), 0)
        self.assertEqual(PiecewiseFunction.from_dict({
            ']-∞,1.000[': '0.0',
            '[1.0,2.0000000000000004[': '1.000x - 1.000',
            '[2.0000000000000004,∞[': '1.0',
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
                                                      '[1.400,5.000[': '0.500',
                                                      '[5.000,5.300[': '1.389x - 6.444',
                                                      '[5.300,5.400[': '0.833x - 3.500',
                                                      '[5.400,∞[': '1.0'}),
                         QuantileDistribution.merge([q1, q2], [.5, .5]).cdf.round())

    def test_dist_merge_singleton(self):
        data1 = np.array([[1.], [1.1], [1.1], [1.2], [1.4], [1.2], [1.3]], dtype=np.float64)
        data2 = np.array([[5.], [5.1], [5.2], [5.2], [5.2], [5.3], [5.4]], dtype=np.float64)
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q1.fit(data1, np.array(range(data1.shape[0])), 0)
        q2.fit(data2, np.array(range(data2.shape[0])), 0)

        self.assertEqual(q2.cdf.round(),
                         QuantileDistribution.merge([q1, q2], [0, 1]).cdf.round())

    def test_dist_merge_throws(self):
        self.assertRaises(ValueError,
                          QuantileDistribution.merge,
                          [1, 2, 3], [0, 0, 1.2])
        self.assertRaises(ValueError,
                          QuantileDistribution.merge,
                          [1, 2, 3], [0, float('nan'), 1])

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
        q = QuantileDistribution.merge([q1, q2, q3], [1/3] * 3)
        self.assertEqual(PiecewiseFunction.from_dict({']-∞,1[': 0,
                                                      '[1,2[': 1/3,
                                                      '[2,3[': 2/3,
                                                      '[3.000,∞[': 1}),
                         q.cdf)

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

        self.assertEqual(result, merged.cdf)

    def test_likelihood_of_fit(self):

        # sample from uniform distribution from [0,1] (likelihood for every sample should be around 1)
        data1 = numpy.random.uniform(0, 1, (1000, ))
        data1 = np.sort(data1).reshape(-1, 1)

        # create quantile distribution
        q1 = QuantileDistribution()
        q1.fit(data1, rows=np.array(range(1000)), col=0)

        # compute likelihood of quantile distributions
        likelihoods = np.array(q1.pdf.multi_eval(data1[:, 0]))

        # no likelihood should be 0
        self.assertTrue(all(likelihoods > 0))

        # start of function should be minimum of data
        self.assertEqual(data1[0], q1.pdf.intervals[1].lowermost())

        # end of function should be maximum of data
        self.assertEqual(data1[-1], q1.pdf.intervals[-2].uppermost())


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


class TestCaseQuantileCrop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = {
            ']-inf,0.[': 0.,
            '[0.,.3[': LinearFunction.from_points((0., 0.), (.3, .25)),
            '[.3,.7[': LinearFunction.from_points((.3, .25), (.7, .75)),
            '[.7,1.[': LinearFunction.from_points((.7, .75), (1., 1.)),
            '[1.,inf[': 1.
        }
        # results in
        #  ]-∞,0.000[      |--> 0.0
        # [0.000,0.300[   |--> 0.833x
        # [0.300,0.700[   |--> 1.250x - 0.125
        # [0.700,1.000[   |--> 0.833x + 0.167
        # [1.000,∞[       |--> 1.0
        cdf = PiecewiseFunction.from_dict(d)
        cls.qdist = QuantileDistribution.from_cdf(cdf)

    # def setUp(self):
    #     print('Setting up test method', self._testMethodName)

    def test_serialization(self):
        self.assertEqual(self.qdist, QuantileDistribution.from_json(self.qdist.to_json()))

    def test_crop_quantiledist_singleslice_inc(self):
        d = {
            ']-inf,.3[': 0.,
            '[.3,.7[': LinearFunction.from_points((.3, 0.), (.7, 1.)),
            '[.7,inf[': 1.
        }

        self.interval = ContinuousSet(.3, .7)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(self.expected, self.actual.cdf)

    def test_crop_quantiledist_singleslice_exc(self):
        d = {
            ']-inf,.3[': 0.,
            ContinuousSet(.3, np.nextafter(0.7, 0.7 - 1), INC, EXC):
                LinearFunction.parse('2.5000000000000013x - 0.7500000000000003'),
            ContinuousSet(np.nextafter(0.7, 0.7 - 1), np.PINF, INC, EXC): 1.
        }
        self.interval = ContinuousSet(.3, .7, INC, EXC)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(self.expected, self.actual.cdf)

    def test_crop_quantiledist_twoslice(self):
        d = {
            ']-inf,.3[': 0.,
            '[.3,.7[': LinearFunction.from_points((.3, .0), (.7, .6666666666666665)),
            '[.7,1.[': LinearFunction.from_points((.7, .6666666666666665), (1., .9999999999999999)),
            '[1.,inf[': 1.
        }

        self.interval = ContinuousSet(.3, 1.)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(self.expected, self.actual.cdf)

    def test_crop_quantiledist_intermediate(self):
        d = {
            ']-inf,.2[': 0.,
            '[.2,.3[': LinearFunction(1.25, -0.25),
            '[.3,.7[': LinearFunction(1.8749999999999998, -0.4374999999999999),
            '[.7,.8[': LinearFunction(1.25, 1.1102230246251565e-16),
            '[.8,∞[': 1.0
        }

        self.interval = ContinuousSet(.2, .8)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(self.expected.round(digits=5), self.actual.cdf.round(digits=5))

    def test_crop_quantiledist_full(self):
        self.interval = ContinuousSet(-1.5, 1.5)
        self.actual = self.qdist.crop(self.interval)
        self.expected = self.qdist.cdf
        self.assertEqual(self.expected.round(digits=5), self.actual.cdf.round(digits=5))

    def test_crop_quantiledist_ident(self):
        self.interval = ContinuousSet(0, 1)
        self.actual = self.qdist.crop(self.interval)
        self.expected = self.qdist.cdf
        self.assertEqual(self.expected.round(digits=5), self.actual.cdf.round(digits=5))

    def test_crop_quantiledist_onepoint(self):
        d = {
            ']-inf,.3[': 0.,
            '[.3,inf[': 1.
        }

        self.interval = ContinuousSet(.3, .3)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(self.expected, self.actual.cdf)

    def test_crop_quantiledist_outside_r(self):
        self.interval = ContinuousSet(1.5, 1.6)
        self.assertRaises(Unsatisfiability, self.qdist.crop, self.interval)

    def test_crop_quantiledist_outside_l(self):
        self.interval = ContinuousSet(-3, -2)
        self.assertRaises(Unsatisfiability, self.qdist.crop, self.interval)

    def plot(self):
        print('Tearing down test method', self._testMethodName)
        x = np.linspace(-2, 2, 100)
        orig = self.qdist.cdf.multi_eval(x)
        if self.actual is not None:
            actual = self.actual.cdf.multi_eval(x)
        if hasattr(self, 'expected'):
            expected = self.expected.multi_eval(x)

        plt.plot(x, orig, label='original CDF')
        if self.actual is not None:
            plt.plot(x, actual, label='actual CDF', marker='*')
        if hasattr(self, 'expected'):
            plt.plot(x, expected, label='expected CDF', marker='+')

        plt.grid()
        plt.legend()
        plt.title(f'{self._testMethodName} - cropping {self.interval}')
        plt.show()


class QuantileTest(TestCase):

    def test_pdf_to_cdf(self):
        '''Convert a PDF into a CDF by piecewise integration'''
        # Arrange
        pdf = PiecewiseFunction.from_dict({
            '(-inf,-2.5)': 0,
            '[-2.5,-1.5)': 1,
            '[-1.5,-.5)': 3,
            '[-.5,.5)': 5,
            '[.5,1.5)': 3,
            '[1.5,2.5)': 1,
            '[2.5,inf)': 0,
        })
        integral = pdf.integrate()
        pdf = pdf.mul(ConstantFunction(1 / integral))

        # Act
        cdf = QuantileDistribution.pdf_to_cdf(pdf)

        # Assert
        self.assertAlmostEqual(
            0,
            cdf.functions[0].value,
            places=12
        )
        self.assertAlmostEqual(
            1,
            cdf.functions[-1].value,
            places=12
        )
        for f in cdf.functions:
            self.assertGreaterEqual(f.m, 0)

    def test_pdf_to_cdf_jump(self):
        # Arrange
        pdf = PiecewiseFunction.from_dict({
            '(-∞,0.0)': 0,
            '[0.0,5e-324)': np.inf,
            '[5e-324,∞)': 0
        })

        # Act
        cdf = QuantileDistribution.pdf_to_cdf(
            pdf,
            dirac_weights=[1]
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '(-inf,0)': 0,
                '[0,inf)': 1
            }),
            cdf
        )

    def test_cdf_to_pdf_simple(self):
        pass

    def test_cdf_to_pdf_jump(self):
        # Arrange
        qdist = QuantileDistribution.from_cdf(
            PiecewiseFunction.zero()
            .overwrite_at(
                '[0,inf)',
                ConstantFunction(1)
            )
        )

        # Act
        pdf = qdist.pdf

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '(-∞,0.0)': 0,
                '[0.0,5e-324)': np.inf,
                '[5e-324,∞)': 0
            }),
            pdf
        )

