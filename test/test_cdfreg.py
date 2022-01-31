import statistics

import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import norm

import unittest
import numpy as np

try:
    from jpt.learning.distributions import Numeric, Gaussian, SymbolicType
    from jpt.trees import JPT
    from jpt.variables import NumericVariable, SymbolicVariable, infer_from_dataframe
    from jpt.base.intervals import ContinuousSet, INC, EXC
    from jpt.base.quantiles import QuantileDistribution, PiecewiseFunction, ConstantFunction, LinearFunction, Undefined
except ModuleNotFoundError:
    import pyximport
    pyximport.install()


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


class TestCasePosteriorNumeric(unittest.TestCase):

    @classmethod
    def f(cls, x):
        """The function to predict."""
        # return x * np.sin(x)
        import math
        return x

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)

        SAMPLES = 200
        gauss1 = Gaussian([-.25, -.25], [[.2, -.07], [-.07, .1]])
        gauss2 = Gaussian([.5, 1], [[.2, .07], [.07, .05]])
        gauss1_data = gauss1.sample(SAMPLES)
        gauss2_data = gauss2.sample(SAMPLES)
        data = np.vstack([gauss1_data, gauss2_data])

        cls.df = DataFrame({'X': data[:, 0], 'Y': data[:, 1], 'Color': ['R'] * SAMPLES + ['B'] * SAMPLES})

        cls.varx = NumericVariable('X', Numeric, precision=.1)
        cls.vary = NumericVariable('Y', Numeric, precision=.1)
        cls.varcolor = SymbolicVariable('Color', SymbolicType('ColorType', ['R', 'B']))

        cls.jpt = JPT(variables=[cls.varx, cls.vary], min_samples_leaf=.01)
        # cls.jpt = JPT(variables=[cls.varx, cls.vary, cls.varcolor], min_samples_leaf=.1)  # TODO use this once symbolic variables are considered in posterior
        cls.jpt.learn(cls.df[['X', 'Y']])
        # cls.jpt.learn(cls.df)  # TODO use this once symbolic variables are considered in posterior

    def test_posterior_numeric_x_given_y_interval(self):
        self.q = [self.varx]
        self.e = {self.vary: ContinuousSet(1, 1.5)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_numeric_y_given_x_interval(self):
        self.q = [self.vary]
        self.e = {self.varx: ContinuousSet(1, 2)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_numeric_x_given_y_value(self):
        self.q = [self.varx]
        self.e = {self.vary: 0}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with calculated posterior', f'Posterior P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        X = np.linspace(-2, 2, 100)
        mean = statistics.mean(self.df['X'])
        sd = statistics.stdev(self.df['X'])
        meanr = statistics.mean(self.df[self.df['Color'] == 'R']['X'])
        sdr = statistics.stdev(self.df[self.df['Color'] == 'R']['X'])
        meanb = statistics.mean(self.df[self.df['Color'] == 'B']['X'])
        sdb = statistics.stdev(self.df[self.df['Color'] == 'B']['X'])

        xr = self.df[self.df['Color'] == 'R']['X']
        xb = self.df[self.df['Color'] == 'B']['X']
        yr = self.df[self.df['Color'] == 'R']['Y']
        yb = self.df[self.df['Color'] == 'B']['Y']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(xr, yr, color='r', marker='.', label='Training data A')
        plt.scatter(xb, yb, color='b', marker='.', label='Training data B')
        plt.plot(sorted(self.df['X']), norm.pdf(sorted(self.df['X']), mean, sd), label='PDF of combined datasets')
        plt.plot(sorted(xr), norm.pdf(sorted(xr), meanr, sdr), label='PDF of dataset A')
        plt.plot(sorted(xb), norm.pdf(sorted(xb), meanb, sdb), label='PDF of dataset B')

        # plot posterior
        for var in self.q:
            if var not in self.posterior: continue
            plt.plot(X, self.posterior[var].cdf.multi_eval(X), label=f'Posterior of combined datasets')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-2, 5)
        plt.xlim(-2, 2)
        plt.legend(loc='upper left')
        plt.title(f'Posterior P({",".join([v.name for v in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')
        plt.grid()
        plt.show()


class TestCasePosteriorSymbolic(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)

        f_csv = '../examples/data/restaurant.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=True, precision=.01, haze=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0--10, 10--30, 30--60, >60
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL (typically target variable)

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)
        cls.jpt.plot(title='Restaurant', filename='Restaurant', directory='TEST', view=False)

    def test_posterior_symbolic_single_candidate_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[8]: 'Thai'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(True, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_single_candidatet_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[8]: 'Italian'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(False, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_evidence_not_in_path_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: True}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(True, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_evidence_not_in_path_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(False, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[1]: True, self.variables[8]: 'French'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertIsNone(self.posterior[self.q[-1]])

    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with calculated posterior', f'Posterior P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')


class TestCasePosteriorSymbolicAndNumeric(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)

        f_csv = '../examples/data/restaurant-mixed.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=True, precision=.01, haze=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)
        cls.jpt.plot(title='Restaurant-Mixed', filename='Restaurant-Mixed', directory='TEST', view=False)

    def test_posterior_mixed_single_candidate_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Thai'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(True, self.posterior.dists[self.q[-1]].expectation())

    def test_posterior_mixed_single_candidatet_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Italian'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(False, self.posterior.dists[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: True}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(True, self.posterior.dists[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual(False, self.posterior.dists[self.q[-1]].expectation())

    def test_posterior_mixed_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[1]: True, self.variables[8]: 'French'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertIsNone(self.posterior.dists[self.q[-1]])

    def test_posterior_mixed_numeric_query(self):
        self.q = [self.variables[9]]
        self.e = {self.variables[8]: 'Burger', self.variables[0]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        print(self.posterior.dists[self.q[-1]].cdf.pfmt())

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        X = np.linspace(-5, 65, 100)
        xr = self.data[(self.data['Food'] == 'Burger') & (self.data['Alternatives'] == False)]['WaitEstimate']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(self.data['WaitEstimate'], [0]*len(self.data), color='b', marker='*', label='All training data')
        plt.scatter(xr, [0]*len(xr), color='r', marker='.', label='Filtered training data')

        # plot posterior
        for var in self.q:
            if var not in self.posterior.dists: continue
            plt.plot(X, self.posterior.dists[var].cdf.multi_eval(np.array([var.domain.values[x] for x in X])), label=f'Posterior of dataset')

        plt.xlabel('$WaitEstimate [min]$')
        plt.ylabel('$f(x)$')
        plt.ylim(-2, 2)
        plt.xlim(-5, 65)
        plt.legend(loc='upper left')
        plt.title(f'Posterior P({",".join([v.name for v in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})'.replace('$', '\$'))
        plt.grid()
        plt.show()

    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with calculated posterior', f'Posterior P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')


class TestCaseExpectation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)

        f_csv = '../examples/data/restaurant-mixed.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=True, precision=.01, haze=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)
        cls.jpt.plot(title='Restaurant-Mixed', filename='Restaurant-Mixed', directory='TEST', view=False)

    def test_expectation_mixed_single_candidate_T(self):
        # [WillWait, Friday]
        self.q = [self.variables[-1], self.variables[2]]
        # {WaitEstimate: [10,30], Food: Thai}
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Thai'}
        self.expectation = self.jpt.expectation(self.q, self.e)
        self.assertEqual([True, False], [e.result for e in self.expectation])

    def test_expectation_mixed_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[1]: True, self.variables[8]: 'French'}
        self.assertRaises(ValueError, self.jpt.expectation, self.q, self.e)

    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with expectation for', f'P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])}) = [{",".join([f"{q.name}: {e.result if e is not None else None}" for q, e in zip(self.q, self.expectation if hasattr(self, "expectation") else [None]*len(self.q))])}]')


class TestCaseInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('Setting up test class', cls.__name__)

        f_csv = '../examples/data/restaurant-mixed.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=True, precision=.01, haze=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)
        cls.jpt.plot(title='Restaurant-Mixed', filename='Restaurant-Mixed', directory='TEST', view=False)

    def test_inference_mixed_single_candidate_T(self):
        self.q = {self.variables[-1]: True, self.variables[2]: False}
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Thai'}
        inf = self.jpt.infer(self.q, self.e)
        print(inf.explain())
        self.assertEqual(True, inf.result)

    def test_inference_mixed_neu(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[-1]: True}
        inf = self.jpt.posterior(self.q, self.e)
        print(inf.explain())
        self.assertEqual(True, inf.result)


    def tearDown(self):
        print('Tearing down test method', self._testMethodName, 'with calculated posterior', f'Posterior P({",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')


if __name__ == '__main__':
    unittest.main()
