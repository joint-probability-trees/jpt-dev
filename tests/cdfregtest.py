import pyximport
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric
from jpt.trees import JPT
from jpt.variables import NumericVariable

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
                                                      '[1.400,5.000[': '0.500',
                                                      '[5.000,5.300[': '1.389x - 6.444',
                                                      '[5.300,5.400[': '0.833x - 3.500',
                                                      '[5.400,∞[': '1.0'}),
                         QuantileDistribution.merge([q1, q2], [.5, .5]).cdf.round())

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


class TestCaseQuantileCrop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)
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

    def setUp(self):
        print('Setting up test method', self._testMethodName)

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
        self.actual = self.qdist.crop(self.interval)
        self.assertIsNone(self.actual)

    def test_crop_quantiledist_outside_l(self):
        self.interval = ContinuousSet(-3, -2)
        self.actual = self.qdist.crop(self.interval)
        self.assertIsNone(self.actual)

    def tearDown(self):
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


class TestCasePosterior(unittest.TestCase):

    @classmethod
    def f(cls, x):
        """The function to predict."""
        return x * np.sin(x)

    @classmethod
    def setUpClass(cls):

        print('Setting up test class', cls.__name__)

        POINTS = 1000
        X = np.atleast_2d(np.random.uniform(-20, 0.0, size=int(POINTS / 2))).T
        X = np.vstack((np.atleast_2d(np.random.uniform(0, 10.0, size=int(POINTS / 2))).T, X))
        X = X.astype(np.float64)
        cls.X = np.array(list(sorted(X)))

        # Observations
        y = TestCasePosterior.f(cls.X).ravel()

        # Add some noise
        dy = 1.5 + .5 * np.random.random(y.shape)
        noise = np.random.normal(0, dy)
        y += noise
        cls.y = y.astype(np.float32)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        xx = np.atleast_2d(np.linspace(-30, 30, 500)).T
        cls.xx = xx.astype(np.float64)

        cls.varx = NumericVariable('x', Numeric)  # , max_std=1)
        cls.vary = NumericVariable('y', Numeric)  # , max_std=1)

        cls.jpt = JPT(variables=[cls.varx, cls.vary], min_samples_leaf=.01)
        cls.jpt.learn(columns=[cls.X.ravel(), cls.y])

    def test_posterior_value_different_q_e(self):
        self.q = [self.varx]
        self.e = {self.vary: 0}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_interval_y_different_q_e(self):
        self.q = [self.varx]
        self.e = {self.vary: ContinuousSet(0, 5)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_interval_x_different_q_e(self):
        self.q = [self.vary]
        self.e = {self.varx: ContinuousSet(-15, -10)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with calculated posterior', f'Posterior P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')

        # Plot the function, the prediction and the 90% confidence interval based on the MSE
        plt.plot(self.xx, TestCasePosterior.f(self.xx), color='black', linestyle=':', linewidth='2', label=r'$f(x) = x\,\sin(x)$')
        plt.plot(self.X, self.y, '.', color='gray', markersize=5, label='Training data')
        for var in self.q:
            if var not in self.posterior: continue
            plt.plot(self.xx, self.posterior[var].pdf.multi_eval(self.xx.ravel().astype(np.float64)), label=f'Posterior P({var.name}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-10, 20)
        plt.xlim(-25, 15)
        plt.legend(loc='upper left')
        plt.title(r'2D Regression Example ($\vartheta=%.2f\%%$)' % (.95 * 100))
        plt.grid()
        plt.show()


if __name__ == '__main__':
    unittest.main()
