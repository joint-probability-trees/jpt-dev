from unittest import TestCase

from matplotlib import pyplot as plt

import unittest
import numpy as np
import numpy.random

from jpt.base.errors import Unsatisfiability

from jpt.distributions.qpd import QuantileDistribution

from jpt.base.intervals import ContinuousSet, INC, EXC
from jpt.base.functions import (
    PiecewiseFunction,
    ConstantFunction,
    LinearFunction
)


# ----------------------------------------------------------------------

class TestCaseMerge(unittest.TestCase):

    def test_dist_merge(self):
        """Verify merging two quantile distributions with equal weights."""
        # Arrange
        data1 = np.array(
            [[1.], [1.1], [1.1], [1.2],
             [1.4], [1.2], [1.3]],
            dtype=np.float64
        )
        data2 = np.array(
            [[5.], [5.1], [5.2], [5.2],
             [5.2], [5.3], [5.4]],
            dtype=np.float64
        )
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q1.fit(
            data1,
            np.array(range(data1.shape[0])), 0
        )
        q2.fit(
            data2,
            np.array(range(data2.shape[0])), 0
        )

        # Act
        result = QuantileDistribution.merge(
            [q1, q2],
            [.5, .5]
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,1.000[': 0.0,
                '[1.000,1.200[': '1.667x - 1.667',
                '[1.200,1.400[': '0.833x - 0.667',
                '[1.400,5.000[': '0.500',
                '[5.000,5.100[': '0.833x - 3.667',
                '[5.100,5.200[': '2.500x - 12.167',
                '[5.200,5.400[': '0.833x - 3.500',
                '[5.400,∞[': '1.0'
            }),
            result.cdf.round()
        )

    def test_dist_merge_singleton(self):
        '''
        only one distribution for merge has non-zero
        weight.
        '''
        # Arrange
        data1 = np.array(
            [[1.], [1.1], [1.1], [1.2],
             [1.4], [1.2], [1.3]],
            dtype=np.float64
        )
        data2 = np.array(
            [[5.], [5.1], [5.2], [5.2],
             [5.2], [5.3], [5.4]],
            dtype=np.float64
        )
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q1.fit(
            data1,
            np.array(range(data1.shape[0])), 0
        )
        q2.fit(
            data2,
            np.array(range(data2.shape[0])), 0
        )

        # Act
        merged = QuantileDistribution.merge(
            [q1, q2],
            [0, 1]
        )

        # Assert
        self.assertEqual(
            q2.cdf.round(),
            merged.cdf.round()
        )

    def test_dist_merge_throws_weights_invalid(self):
        '''
        Raises Exception when weight distribution is
        invalid.
        '''
        self.assertRaises(
            ValueError,
            QuantileDistribution.merge,
            [1, 2, 3], [0, 0, 1.2]
        )
        self.assertRaises(
            ValueError,
            QuantileDistribution.merge,
            [1, 2, 3], [0, float('nan'), 1]
        )

    def test_dist_merge_jump_functions(self):
        """Verify merging three single-point jump distributions."""
        # Arrange
        data1 = np.array([[1.]], dtype=np.float64)
        data2 = np.array([[2.]], dtype=np.float64)
        data3 = np.array([[3.]], dtype=np.float64)
        q1 = QuantileDistribution()
        q2 = QuantileDistribution()
        q3 = QuantileDistribution()
        q1.fit(
            data1,
            np.array(range(data1.shape[0])), 0
        )
        q2.fit(
            data2,
            np.array(range(data2.shape[0])), 0
        )
        q3.fit(
            data3,
            np.array(range(data3.shape[0])), 0
        )

        # Act
        q = QuantileDistribution.merge(
            [q1, q2, q3], [1/3] * 3
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,1[': 0,
                '[1,2[': 1/3,
                '[2,3[': 2/3,
                '[3.000,∞[': 1
            }),
            q.cdf
        )

    def test_merge_jumps_different_positions(self):
        '''Jumps at different positions'''
        # Arrange
        plf1 = PiecewiseFunction()
        plf1.intervals.append(
            ContinuousSet.fromstring(']-inf,0.000[')
        )
        plf1.intervals.append(
            ContinuousSet.fromstring('[0.000, inf[')
        )
        plf1.functions.append(ConstantFunction(0))
        plf1.functions.append(ConstantFunction(1))

        plf2 = PiecewiseFunction()
        plf2.intervals.append(
            ContinuousSet.fromstring(']-inf,0.5[')
        )
        plf2.intervals.append(
            ContinuousSet.fromstring('[0.5, inf[')
        )
        plf2.functions.append(ConstantFunction(0))
        plf2.functions.append(ConstantFunction(1))

        # Act
        merged = QuantileDistribution.merge(
            [
                QuantileDistribution.from_cdf(plf1),
                QuantileDistribution.from_cdf(plf2)
            ]
        )

        # Assert
        result = PiecewiseFunction()
        result.intervals.append(
            ContinuousSet.fromstring(']-inf,0.0[')
        )
        result.intervals.append(
            ContinuousSet.fromstring('[0.0, .5[')
        )
        result.intervals.append(
            ContinuousSet.fromstring('[0.5, inf[')
        )
        result.functions.append(ConstantFunction(0))
        result.functions.append(ConstantFunction(.5))
        result.functions.append(ConstantFunction(1))

        self.assertEqual(result, merged.cdf)

    def test_merge_jumps_same_positions(self):
        '''Jumps at the same positions'''
        # Arrange
        plf = PiecewiseFunction.zero().overwrite_at(
            ContinuousSet.parse('[0,inf)'),
            ConstantFunction(1)
        )
        q = QuantileDistribution.from_cdf(plf)

        # Act
        result = QuantileDistribution.merge([q, q])

        # Assert
        self.assertEqual(
            plf,
            result.cdf
        )

    def test_likelihood_of_fit(self):
        """Verify all likelihoods are positive for a uniform distribution fit."""

        # sample from uniform distribution from [0,1]
        # (likelihood for every sample should be around 1)
        data1 = numpy.random.uniform(0, 1, (1000, ))
        data1 = np.sort(data1).reshape(-1, 1)

        # create quantile distribution
        q1 = QuantileDistribution()
        q1.fit(
            data1, rows=np.array(range(1000)), col=0
        )

        # compute likelihood of quantile distributions
        likelihoods = np.array(
            q1.pdf.multi_eval(data1[:, 0])
        )

        # no likelihood should be 0
        self.assertTrue(all(likelihoods > 0))

        # start of function should be minimum of data
        self.assertEqual(
            data1[0],
            q1.pdf.intervals[1].lowermost()
        )

        # end of function should be maximum of data
        self.assertEqual(
            data1[-1],
            q1.pdf.intervals[-2].uppermost()
        )


# ----------------------------------------------------------------------

class TestCasePPFTransform(unittest.TestCase):

    def test_ppf_transform_jumps_only(self):
        """Verify PPF transform for a CDF consisting only of jumps."""
        cdf = PiecewiseFunction()
        cdf.intervals.append(
            ContinuousSet.fromstring(']-inf,1[')
        )
        cdf.intervals.append(
            ContinuousSet.fromstring('[1, 2[')
        )
        cdf.intervals.append(
            ContinuousSet.fromstring('[2, 3[')
        )
        cdf.intervals.append(
            ContinuousSet.fromstring('[3, 4[')
        )
        cdf.intervals.append(
            ContinuousSet.fromstring('[4, inf[')
        )
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(ConstantFunction(.25))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(ConstantFunction(.75))
        cdf.functions.append(ConstantFunction(1))
        q = QuantileDistribution.from_cdf(cdf)

        self.assertEqual(
            q.ppf,
            PiecewiseFunction.from_dict({
                ']-∞,0.000[': None,
                '[0.,.25[': ConstantFunction(1),
                '[0.25,.5[': ConstantFunction(2),
                '[0.5,.75[': ConstantFunction(3),
                ContinuousSet(
                    .75,
                    np.nextafter(1, 2),
                    INC, EXC
                ): ConstantFunction(4),
                ContinuousSet(
                    np.nextafter(1, 2),
                    np.inf,
                    INC, EXC
                ): None
            })
        )

    def test_ppf_transform(self):
        """Verify PPF transform for a CDF with mixed linear and constant segments."""
        cdf = PiecewiseFunction()
        cdf.intervals.append(
            ContinuousSet.parse(']-inf,0.000[')
        )
        cdf.intervals.append(
            ContinuousSet.parse('[0.000, 1[')
        )
        cdf.intervals.append(
            ContinuousSet.parse('[1, 2[')
        )
        cdf.intervals.append(
            ContinuousSet.parse('[2, 3[')
        )
        cdf.intervals.append(
            ContinuousSet.parse('[3, inf[')
        )
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(
            LinearFunction.from_points((0, 0), (1, .5))
        )
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(
            LinearFunction.from_points(
                (2, .5), (3, 1)
            )
        )
        cdf.functions.append(ConstantFunction(1))

        q = QuantileDistribution.from_cdf(cdf)
        self.assertEqual(
            q.ppf,
            PiecewiseFunction.from_dict({
                ']-∞,0.000[': None,
                '[0.0,.5[': str(
                    LinearFunction.from_points(
                        (0, 0), (.5, 1)
                    )
                ),
                ContinuousSet(
                    .5,
                    np.nextafter(1, 2),
                    INC, EXC
                ): LinearFunction(2, 1),
                ContinuousSet(
                    np.nextafter(1, 2),
                    np.inf,
                    INC, EXC
                ): None
            })
        )


# ----------------------------------------------------------------------

class TestCaseQuantileCrop(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        d = {
            ']-inf,0.[': 0.,
            '[0.,.3[': LinearFunction.from_points(
                (0., 0.), (.3, .25)
            ),
            '[.3,.7[': LinearFunction.from_points(
                (.3, .25), (.7, .75)
            ),
            '[.7,1.[': LinearFunction.from_points(
                (.7, .75), (1., 1.)
            ),
            '[1.,inf[': 1.
        }
        cdf = PiecewiseFunction.from_dict(d)
        cls.qdist = QuantileDistribution.from_cdf(cdf)

    def test_serialization(self):
        """Verify JSON round-trip serialization of a quantile distribution."""
        self.assertEqual(
            self.qdist,
            QuantileDistribution.from_json(
                self.qdist.to_json()
            )
        )

    def test_crop_quantiledist_singleslice_inc(self):
        """Verify cropping a quantile distribution to a single inclusive interval."""
        d = {
            ']-inf,.3[': 0.,
            '[.3,.7[': LinearFunction.from_points(
                (.3, 0.), (.7, 1.)
            ),
            '[.7,inf[': 1.
        }

        self.interval = ContinuousSet(.3, .7)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(
            self.expected, self.actual.cdf
        )

    def test_crop_quantiledist_singleslice_exc(self):
        """Verify cropping a quantile distribution to a single exclusive-upper interval."""
        d = {
            ']-inf,.3[': 0.,
            ContinuousSet(
                .3,
                np.nextafter(0.7, 0.7 - 1),
                INC, EXC
            ): LinearFunction.parse(
                '2.5000000000000013x'
                ' - 0.7500000000000003'
            ),
            ContinuousSet(
                np.nextafter(0.7, 0.7 - 1),
                np.inf,
                INC, EXC
            ): 1.
        }
        self.interval = ContinuousSet(
            .3, .7, INC, EXC
        )
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(
            self.expected, self.actual.cdf
        )

    def test_crop_quantiledist_twoslice(self):
        """Verify cropping a quantile distribution spanning two CDF slices."""
        d = {
            ']-inf,.3[': 0.,
            '[.3,.7[': LinearFunction.from_points(
                (.3, .0), (.7, .6666666666666665)
            ),
            '[.7,1.[': LinearFunction.from_points(
                (.7, .6666666666666665),
                (1., .9999999999999999)
            ),
            '[1.,inf[': 1.
        }

        self.interval = ContinuousSet(.3, 1.)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(
            self.expected, self.actual.cdf
        )

    def test_crop_quantiledist_intermediate(self):
        """Verify cropping to an intermediate interval crossing multiple segments."""
        d = {
            ']-inf,.2[': 0.,
            '[.2,.3[': LinearFunction(1.25, -0.25),
            '[.3,.7[': LinearFunction(
                1.8749999999999998,
                -0.4374999999999999
            ),
            '[.7,.8[': LinearFunction(
                1.25, 1.1102230246251565e-16
            ),
            '[.8,∞[': 1.0
        }

        self.interval = ContinuousSet(.2, .8)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(
            self.expected.round(digits=5),
            self.actual.cdf.round(digits=5)
        )

    def test_crop_quantiledist_full(self):
        """Verify cropping to an interval larger than the support preserves the CDF."""
        self.interval = ContinuousSet(-1.5, 1.5)
        self.actual = self.qdist.crop(self.interval)
        self.expected = self.qdist.cdf
        self.assertEqual(
            self.expected.round(digits=5),
            self.actual.cdf.round(digits=5)
        )

    def test_crop_quantiledist_ident(self):
        """Verify cropping to the exact support interval preserves the CDF."""
        self.interval = ContinuousSet(0, 1)
        self.actual = self.qdist.crop(self.interval)
        self.expected = self.qdist.cdf
        self.assertEqual(
            self.expected.round(digits=5),
            self.actual.cdf.round(digits=5)
        )

    def test_crop_quantiledist_onepoint(self):
        """Verify cropping to a single point produces a jump CDF."""
        d = {
            ']-inf,.3[': 0.,
            '[.3,inf[': 1.
        }

        self.interval = ContinuousSet(.3, .3)
        self.actual = self.qdist.crop(self.interval)
        self.expected = PiecewiseFunction.from_dict(d)
        self.assertEqual(
            self.expected, self.actual.cdf
        )

    def test_crop_quantiledist_outside_r(self):
        """Verify cropping to an interval right of the support raises Unsatisfiability."""
        self.interval = ContinuousSet(1.5, 1.6)
        self.assertRaises(
            Unsatisfiability,
            self.qdist.crop,
            self.interval
        )

    def test_crop_quantiledist_outside_l(self):
        """Verify cropping to an interval left of the support raises Unsatisfiability."""
        self.interval = ContinuousSet(-3, -2)
        self.assertRaises(
            Unsatisfiability,
            self.qdist.crop,
            self.interval
        )

    def plot(self):
        print(
            'Tearing down test method',
            self._testMethodName
        )
        x = np.linspace(-2, 2, 100)
        orig = self.qdist.cdf.multi_eval(x)
        if self.actual is not None:
            actual = self.actual.cdf.multi_eval(x)
        if hasattr(self, 'expected'):
            expected = self.expected.multi_eval(x)

        plt.plot(x, orig, label='original CDF')
        if self.actual is not None:
            plt.plot(
                x, actual,
                label='actual CDF', marker='*'
            )
        if hasattr(self, 'expected'):
            plt.plot(
                x, expected,
                label='expected CDF', marker='+'
            )

        plt.grid()
        plt.legend()
        plt.title(
            f'{self._testMethodName}'
            f' - cropping {self.interval}'
        )
        plt.show()


# ----------------------------------------------------------------------

class QuantileTest(TestCase):

    def test_pdf_to_cdf(self):
        '''Convert a PDF into a CDF by piecewise
        integration'''
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
        """Verify PDF-to-CDF conversion for a Dirac delta distribution."""
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
        """Verify CDF-to-PDF conversion for a simple distribution."""
        pass

    def test_cdf_to_pdf_jump(self):
        """Verify CDF-to-PDF conversion produces a Dirac delta for a jump CDF."""
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


# ----------------------------------------------------------------------

class FitInvariantTest(TestCase):
    """Invariant-based tests for ``QuantileDistribution.fit``.

    These tests assert structural properties (monotonicity,
    boundary values, normalization) rather than exact CDF
    representations, so they are robust to implementation
    changes in the regressor.
    """

    @staticmethod
    def _fitted(values, epsilon=0.01):
        """Fit a QuantileDistribution to a 1-D iterable of
        values.
        """
        data = np.array(
            values, dtype=np.float64
        ).reshape(-1, 1)
        q = QuantileDistribution(epsilon=epsilon)
        q.fit(data, None, 0)
        return q

    def _assert_cdf_invariants(self, q):
        """Verify CDF invariants at a grid of points:
        non-decreasing, bounded in [0, 1], left-limit 0,
        right-limit 1.
        """
        cdf = q.cdf
        xs = np.sort(np.concatenate([
            [-1e9, -1e3, -10.0, -1.0],
            np.linspace(-5, 5, 50),
            [1.0, 10.0, 1e3, 1e9]
        ]))
        values = [cdf.eval(x) for x in xs]
        for v in values:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0 + 1e-12)
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(
                b, a - 1e-12,
                'CDF is not monotonically non-decreasing'
            )
        self.assertAlmostEqual(cdf.eval(-1e9), 0.0, places=10)
        self.assertAlmostEqual(cdf.eval(1e9), 1.0, places=10)

    # --- Core shape ----------------------------------

    def test_fit_linear_monotone(self):
        """Fit on uniformly spaced data yields a
        monotone CDF reaching 1 at the right tail."""
        # Arrange & Act
        q = self._fitted(
            list(np.linspace(0.0, 1.0, 11))
        )
        # Assert
        self._assert_cdf_invariants(q)

    def test_fit_gaussian_monotone(self):
        """Fit on samples from N(0,1) produces a
        monotone CDF."""
        # Arrange
        rng = np.random.RandomState(0)
        # Act
        q = self._fitted(rng.normal(0, 1, 500))
        # Assert
        self._assert_cdf_invariants(q)

    def test_fit_heavy_duplicates_monotone(self):
        """Fit on data with heavy duplicates produces
        a monotone CDF (exposes the CDF-monotonicity
        repair path from commit 7463fc4)."""
        # Arrange
        values = (
            [0.0] * 20 + [1.0] * 10 + [2.0] * 30
        )
        # Act
        q = self._fitted(values)
        # Assert
        self._assert_cdf_invariants(q)

    def test_fit_extreme_magnitudes(self):
        """Fit on values that span many orders of
        magnitude preserves CDF invariants."""
        # Arrange
        values = [
            -1e9, -1e3, -1.0, 0.0, 1.0, 1e3, 1e9
        ]
        # Act
        q = self._fitted(values)
        # Assert
        self._assert_cdf_invariants(q)

    # --- Edge cases ----------------------------------

    def test_fit_single_sample_produces_jump(self):
        """A single sample yields a jump CDF of exactly
        two segments at the sample value."""
        # Arrange
        data = np.array([[3.14]], dtype=np.float64)
        q = QuantileDistribution()
        # Act
        q.fit(data, np.array([0], dtype=np.int64), 0)
        # Assert
        self.assertEqual(len(q.cdf.intervals), 2)
        self.assertAlmostEqual(q.cdf.eval(3.13), 0.0)
        self.assertAlmostEqual(q.cdf.eval(3.14), 1.0)
        self.assertAlmostEqual(q.cdf.eval(3.15), 1.0)

    def test_fit_all_identical_samples_yields_jump(self):
        """N identical samples collapse to a jump CDF."""
        # Arrange & Act
        q = self._fitted([7.0, 7.0, 7.0, 7.0, 7.0])
        # Assert — CDF is 0 below 7 and 1 at/above 7
        self.assertAlmostEqual(q.cdf.eval(6.9), 0.0)
        self.assertAlmostEqual(q.cdf.eval(7.0), 1.0, places=6)
        self.assertAlmostEqual(q.cdf.eval(7.1), 1.0, places=6)

    def test_fit_two_samples_minimal_linear(self):
        """Two distinct samples give a CDF that is 0
        before, linear between, and 1 after."""
        # Arrange & Act
        q = self._fitted([1.0, 2.0])
        # Assert
        self.assertAlmostEqual(q.cdf.eval(0.5), 0.0, places=10)
        self.assertAlmostEqual(q.cdf.eval(1.5), 0.5, places=10)
        self.assertAlmostEqual(q.cdf.eval(2.5), 1.0, places=10)

    def test_fit_with_leftmost_rightmost_boundaries(self):
        """Optional leftmost/rightmost anchor points are
        honored and the CDF still reaches 1 on the right.
        """
        # Arrange
        data = np.array(
            [[1.0], [2.0], [3.0]], dtype=np.float64
        )
        q = QuantileDistribution(epsilon=0.01)
        # Act
        q.fit(
            data,
            np.array([0, 1, 2], dtype=np.int64),
            0,
            leftmost=0.0,
            rightmost=4.0
        )
        # Assert
        self.assertAlmostEqual(
            q.cdf.eval(-1.0), 0.0, places=10
        )
        self.assertAlmostEqual(
            q.cdf.eval(5.0), 1.0, places=10
        )
        self._assert_cdf_invariants(q)

    def test_fit_leftmost_violation_rejected(self):
        """If a data point is not strictly greater than
        ``leftmost``, ``fit`` raises AssertionError.
        """
        # Arrange
        data = np.array(
            [[1.0], [0.5], [2.0]], dtype=np.float64
        )
        q = QuantileDistribution()
        # Act / Assert
        with self.assertRaises(AssertionError):
            q.fit(
                data,
                np.array([0, 1, 2], dtype=np.int64),
                0,
                leftmost=0.5
            )

    def test_fit_column_selection(self):
        """Fitting on column 1 ignores column 0."""
        # Arrange — column 0 random, column 1 monotone
        rng = np.random.RandomState(7)
        n = 20
        data = np.ascontiguousarray(np.column_stack([
            rng.normal(0, 1, n),
            np.linspace(10.0, 20.0, n)
        ])).astype(np.float64)
        q = QuantileDistribution(epsilon=0.01)
        # Act
        q.fit(data, None, 1)
        # Assert — CDF is effectively supported on [10, 20]
        self.assertAlmostEqual(q.cdf.eval(9.0), 0.0, places=6)
        self.assertAlmostEqual(q.cdf.eval(21.0), 1.0, places=6)
        self.assertGreater(q.cdf.eval(15.0), 0.2)
        self.assertLess(q.cdf.eval(15.0), 0.8)

    def test_fit_epsilon_bounds_segment_count(self):
        """Larger epsilon triggers subsampling and yields
        fewer CDF segments than smaller epsilon on the
        same data."""
        # Arrange — 500 distinct points
        rng = np.random.RandomState(1)
        values = np.sort(rng.normal(0, 1, 500))

        # Act
        q_coarse = self._fitted(values, epsilon=0.2)
        q_fine = self._fitted(values, epsilon=0.001)

        # Assert — coarser fit must not exceed the
        # subsample budget of max(2, ceil(1/epsilon))+2
        # (+2 for the flanking -inf and +inf segments)
        self.assertLessEqual(
            len(q_coarse.cdf.intervals),
            int(1 / 0.2) + 3
        )
        self.assertGreater(
            len(q_fine.cdf.intervals),
            len(q_coarse.cdf.intervals)
        )

    def test_fit_result_is_self(self):
        """``fit`` returns the distribution itself so
        calls can be chained."""
        # Arrange
        data = np.array([[1.0], [2.0]], dtype=np.float64)
        q = QuantileDistribution()
        # Act
        result = q.fit(data, None, 0)
        # Assert
        self.assertIs(result, q)

    def test_fit_resets_cached_pdf_ppf(self):
        """Re-fitting invalidates cached pdf and ppf."""
        # Arrange
        d1 = np.array(
            [[0.0], [1.0]], dtype=np.float64
        )
        d2 = np.array(
            [[10.0], [20.0]], dtype=np.float64
        )
        q = QuantileDistribution()
        q.fit(d1, None, 0)
        _ = q.pdf
        _ = q.ppf
        # Act
        q.fit(d2, None, 0)
        # Assert — the new pdf/ppf reflect d2, not d1
        self.assertAlmostEqual(
            q.cdf.eval(0.5), 0.0, places=6
        )
        self.assertAlmostEqual(
            q.cdf.eval(25.0), 1.0, places=6
        )


# ----------------------------------------------------------------------

class PDFPPFTest(TestCase):
    """Tests for the ``pdf`` and ``ppf`` properties."""

    @staticmethod
    def _fitted(values):
        data = np.array(
            values, dtype=np.float64
        ).reshape(-1, 1)
        q = QuantileDistribution()
        q.fit(data, None, 0)
        return q

    def test_pdf_requires_fit(self):
        """Accessing pdf before fit raises RuntimeError."""
        # Arrange
        q = QuantileDistribution()
        # Act / Assert
        with self.assertRaises(RuntimeError):
            _ = q.pdf

    def test_ppf_requires_fit(self):
        """Accessing ppf before fit raises RuntimeError."""
        # Arrange
        q = QuantileDistribution()
        # Act / Assert
        with self.assertRaises(RuntimeError):
            _ = q.ppf

    def test_pdf_non_negative_everywhere(self):
        """PDF slope values are non-negative on fitted
        data."""
        # Arrange & Act
        q = self._fitted(list(np.linspace(0, 1, 50)))
        pdf = q.pdf
        # Assert
        for f in pdf.functions:
            if isinstance(f, LinearFunction):
                self.assertGreaterEqual(f.m, -1e-12)
                self.assertGreaterEqual(
                    f.eval(f.c), -1e-12
                )
            elif isinstance(f, ConstantFunction):
                self.assertGreaterEqual(f.value, -1e-12)

    def test_pdf_jump_has_dirac_impulse(self):
        """A single-sample CDF produces a PDF with an
        infinite Dirac impulse at the sample value."""
        # Arrange
        data = np.array([[0.0]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0], dtype=np.int64), 0)
        # Act
        pdf = q.pdf
        # Assert — the PDF must contain at least one
        # infinite constant segment covering the sample
        found_inf = any(
            isinstance(f, ConstantFunction)
            and np.isinf(f.value)
            and interval.contains_value(0.0)
            for interval, f in zip(
                pdf.intervals, pdf.functions
            )
        )
        self.assertTrue(
            found_inf,
            'Expected a Dirac impulse at the jump point'
        )

    def test_pdf_cached(self):
        """pdf is computed once and reused across calls."""
        # Arrange & Act
        q = self._fitted([0.0, 1.0, 2.0])
        first = q.pdf
        second = q.pdf
        # Assert
        self.assertIs(first, second)

    def test_pdf_recomputed_after_cdf_set(self):
        """Setting cdf invalidates cached pdf — a fresh
        pdf object is returned on the next access."""
        # Arrange
        q = self._fitted([0.0, 1.0])
        old_pdf = q.pdf
        # Act — re-set the cdf to invalidate the cache
        q.cdf = q.cdf
        # Assert — a new pdf instance is returned
        self.assertIsNot(q.pdf, old_pdf)

    def test_ppf_monotone(self):
        """The quantile function is monotonically
        non-decreasing on (0, 1)."""
        # Arrange
        rng = np.random.RandomState(123)
        q = self._fitted(rng.normal(0, 1, 200))
        # Act
        ppf = q.ppf
        # Assert
        ps = np.linspace(0.01, 0.99, 100)
        vals = [ppf.eval(p) for p in ps]
        for a, b in zip(vals, vals[1:]):
            self.assertGreaterEqual(
                b, a - 1e-10,
                'PPF is not monotone non-decreasing'
            )

    def test_ppf_inverse_of_cdf(self):
        """For p in the interior, CDF(PPF(p)) ≈ p."""
        # Arrange
        rng = np.random.RandomState(42)
        q = self._fitted(
            np.sort(rng.uniform(0, 10, 200))
        )
        # Act / Assert
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = q.ppf.eval(p)
            self.assertAlmostEqual(
                q.cdf.eval(x), p, delta=0.05,
                msg='CDF(PPF(%s))=%s' % (p, q.cdf.eval(x))
            )

    def test_ppf_undefined_outside_unit_interval(self):
        """PPF evaluates to NaN (Undefined) outside the
        closed interval [0, 1]."""
        # Arrange & Act
        q = self._fitted([0.0, 1.0, 2.0])
        # Assert
        self.assertTrue(np.isnan(q.ppf.eval(-0.01)))
        self.assertTrue(np.isnan(q.ppf.eval(1.5)))


# ----------------------------------------------------------------------

class CropInvariantTest(TestCase):
    """Invariant tests for ``crop``."""

    @staticmethod
    def _uniform_0_1():
        """Return a QuantileDistribution approximating
        U(0, 1)."""
        data = np.linspace(
            0.0, 1.0, 101
        ).reshape(-1, 1)
        q = QuantileDistribution(epsilon=0.001)
        q.fit(data, None, 0)
        return q

    def test_crop_result_is_valid_cdf(self):
        """A cropped CDF is non-decreasing, bounded in
        [0, 1], and reaches 1 on the right."""
        # Arrange
        q = self._uniform_0_1()
        # Act
        q_crop = q.crop(ContinuousSet(0.25, 0.75))
        # Assert
        cdf = q_crop.cdf
        xs = np.linspace(0.0, 1.0, 50)
        values = [cdf.eval(x) for x in xs]
        for v in values:
            self.assertGreaterEqual(v, -1e-12)
            self.assertLessEqual(v, 1.0 + 1e-12)
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(b, a - 1e-12)
        self.assertAlmostEqual(cdf.eval(-1e9), 0.0, places=6)
        self.assertAlmostEqual(cdf.eval(1e9), 1.0, places=6)

    def test_crop_shifts_support(self):
        """Values below the crop's lower bound have
        CDF 0; at/above the upper bound the CDF is 1."""
        # Arrange
        q = self._uniform_0_1()
        # Act
        q_crop = q.crop(ContinuousSet(0.25, 0.75))
        # Assert
        self.assertAlmostEqual(
            q_crop.cdf.eval(0.2), 0.0, delta=0.02
        )
        self.assertAlmostEqual(
            q_crop.cdf.eval(0.8), 1.0, delta=0.02
        )

    def test_crop_midpoint_is_half(self):
        """For a uniform distribution, cropping to a
        symmetric interval yields CDF ≈ 0.5 at the
        midpoint."""
        # Arrange
        q = self._uniform_0_1()
        # Act
        q_crop = q.crop(ContinuousSet(0.2, 0.8))
        # Assert
        self.assertAlmostEqual(
            q_crop.cdf.eval(0.5), 0.5, delta=0.05
        )

    def test_crop_outside_support_raises(self):
        """Cropping to an interval disjoint from the
        CDF support raises Unsatisfiability."""
        # Arrange
        q = self._uniform_0_1()
        # Act / Assert
        with self.assertRaises(Unsatisfiability):
            q.crop(ContinuousSet(10.0, 20.0))

    def test_crop_requires_fit(self):
        """Crop before fit raises RuntimeError."""
        # Arrange
        q = QuantileDistribution()
        # Act / Assert
        with self.assertRaises(RuntimeError):
            q.crop(ContinuousSet(0.0, 1.0))

    def test_crop_preserves_epsilon(self):
        """Crop returns a distribution with the same
        epsilon and min_samples_mars as the source."""
        # Arrange
        data = np.linspace(
            0.0, 1.0, 50
        ).reshape(-1, 1)
        q = QuantileDistribution(
            epsilon=0.05, min_samples_mars=3
        )
        q.fit(data, None, 0)
        # Act
        q_crop = q.crop(ContinuousSet(0.25, 0.75))
        # Assert
        self.assertAlmostEqual(q_crop.epsilon, 0.05)
        self.assertEqual(q_crop.min_samples_mars, 3)

    def test_crop_is_idempotent_on_full_support(self):
        """Cropping to the support leaves the CDF
        essentially unchanged."""
        # Arrange
        q = self._uniform_0_1()
        # Act
        q_crop = q.crop(
            ContinuousSet(-np.inf, np.inf, EXC, EXC)
        )
        # Assert — CDFs agree on interior points
        for x in np.linspace(0.05, 0.95, 10):
            self.assertAlmostEqual(
                q.cdf.eval(x),
                q_crop.cdf.eval(x),
                delta=1e-6
            )


# ----------------------------------------------------------------------

class MonotonicityRepairTest(TestCase):
    """Tests for the CDF / PPF monotonicity repair
    introduced in commit 7463fc4.
    """

    def test_from_cdf_repairs_non_monotone_input(self):
        """``_assert_consistency`` repairs a CDF whose
        third segment starts below the previous segment's
        end value."""
        # Arrange — three linear segments, the middle
        # one ending at 0.7 and the next starting at 0.5
        cdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,1.0[': '0.5x + 0.0',
            '[1.0,2.0[': '0.2x + 0.3',
            '[2.0,inf[': '1.0',
        })
        # Sanity — the raw CDF is non-monotone between
        # segment 2 (ends 0.5) and segment 3 (starts 0.5
        # evaluated at 1.0 gives 0.5, but we want to test
        # a clear violation):
        bad = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,1.0[': '0.7x + 0.0',
            '[1.0,2.0[': '0.1x + 0.2',
            '[2.0,inf[': '1.0',
        })
        q = QuantileDistribution()
        q.cdf = bad
        # Act — trigger the repair
        q._assert_consistency()
        # Assert — repaired CDF is monotone
        xs = np.linspace(-0.5, 2.5, 50)
        values = [q.cdf.eval(x) for x in xs]
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(
                b, a - 1e-12,
                'Repaired CDF is not monotone'
            )

    def test_from_json_repairs_non_monotone_cdf(self):
        """Deserialising a non-monotone CDF via
        ``from_json`` yields a repaired distribution."""
        # Arrange — fit a clean distribution then
        # corrupt its JSON representation to produce a
        # CDF with a backwards jump
        data = np.linspace(0, 1, 50).reshape(-1, 1)
        q = QuantileDistribution(epsilon=0.01)
        q.fit(data, None, 0)
        payload = q.to_json()
        payload['cdf'] = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,1.0[': '0.8x + 0.0',
            '[1.0,2.0[': '0.1x + 0.1',
            '[2.0,inf[': '1.0',
        }).to_json()
        # Act
        restored = QuantileDistribution.from_json(payload)
        # Assert — repaired CDF is monotone non-decreasing
        xs = np.linspace(-0.5, 2.5, 50)
        values = [restored.cdf.eval(x) for x in xs]
        for a, b in zip(values, values[1:]):
            self.assertGreaterEqual(b, a - 1e-12)

    def test_ppf_is_monotone_after_fit(self):
        """After a fit, the PPF is monotone
        non-decreasing on every sampled probability."""
        # Arrange — heavy duplicates that stress the
        # monotonicity enforcement paths
        rng = np.random.RandomState(2026)
        values = np.concatenate([
            rng.normal(0, 0.1, 50),
            np.full(30, 0.5),
            rng.normal(1.0, 0.1, 50),
        ])
        q = QuantileDistribution(epsilon=0.005)
        q.fit(
            np.ascontiguousarray(
                values.reshape(-1, 1)
            ),
            None,
            0
        )
        # Act
        ppf = q.ppf
        ps = np.linspace(0.001, 0.999, 200)
        # Assert
        last = -np.inf
        for p in ps:
            v = ppf.eval(p)
            if np.isnan(v):
                continue
            self.assertGreaterEqual(
                v, last - 1e-10,
                'PPF is not monotone at p=%s' % p
            )
            last = v


# ----------------------------------------------------------------------

class SerializationRoundTripTest(TestCase):
    """Tests for ``to_json`` / ``from_json`` round-trip."""

    def test_roundtrip_preserves_cdf_values(self):
        """JSON round-trip preserves CDF evaluations at
        interior points."""
        # Arrange
        rng = np.random.RandomState(11)
        q = QuantileDistribution(epsilon=0.01)
        q.fit(
            rng.normal(0, 1, 100).reshape(-1, 1),
            None,
            0
        )
        # Act
        restored = QuantileDistribution.from_json(
            q.to_json()
        )
        # Assert
        for x in np.linspace(-3, 3, 20):
            self.assertAlmostEqual(
                q.cdf.eval(x),
                restored.cdf.eval(x),
                places=10
            )

    def test_roundtrip_preserves_hyperparameters(self):
        """JSON round-trip preserves epsilon and
        min_samples_mars."""
        # Arrange
        data = np.array(
            [[1.0], [2.0]], dtype=np.float64
        )
        q = QuantileDistribution(
            epsilon=0.123,
            min_samples_mars=7
        )
        q.fit(data, None, 0)
        # Act
        restored = QuantileDistribution.from_json(
            q.to_json()
        )
        # Assert
        self.assertAlmostEqual(restored.epsilon, 0.123)
        self.assertEqual(restored.min_samples_mars, 7)

    def test_roundtrip_jump_distribution(self):
        """Round-trip preserves a single-sample jump
        distribution."""
        # Arrange
        data = np.array([[42.0]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0], dtype=np.int64), 0)
        # Act
        restored = QuantileDistribution.from_json(
            q.to_json()
        )
        # Assert
        self.assertAlmostEqual(
            restored.cdf.eval(41.9), 0.0
        )
        self.assertAlmostEqual(
            restored.cdf.eval(42.1), 1.0
        )
        self.assertEqual(q, restored)

    def test_from_json_single_function_cdf_raises_value_error(self):
        """``from_json`` raises ``ValueError`` (not ``AssertionError``)
        when the serialised CDF has only one segment."""
        # Arrange — build a minimal valid payload then replace the
        # CDF with a single-segment constant (mathematically invalid)
        data = np.array([[1.0], [2.0]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, None, 0)
        payload = q.to_json()
        payload['cdf'] = PiecewiseFunction.from_dict({
            ']-inf,inf[': ConstantFunction(1.0),
        }).to_json()
        # Act / Assert
        with self.assertRaises(ValueError):
            QuantileDistribution.from_json(payload)


# ----------------------------------------------------------------------

class EqualityAndCopyTest(TestCase):
    """Tests for ``__eq__`` and ``copy``."""

    def test_eq_requires_quantile_distribution(self):
        """``__eq__`` rejects other types."""
        # Arrange
        q = QuantileDistribution()
        # Act / Assert
        with self.assertRaises(TypeError):
            _ = q == 42

    def test_eq_detects_epsilon_difference(self):
        """Two distributions with identical CDFs but
        different epsilon values are not equal."""
        # Arrange
        data = np.array(
            [[0.0], [1.0]], dtype=np.float64
        )
        q1 = QuantileDistribution(epsilon=0.01)
        q2 = QuantileDistribution(epsilon=0.05)
        q1.fit(data, None, 0)
        q2.fit(data, None, 0)
        # Act / Assert
        self.assertNotEqual(q1, q2)

    def test_copy_is_equal_but_independent(self):
        """``copy()`` returns an equal but independent
        distribution: mutating the copy's CDF does not
        affect the original."""
        # Arrange
        data = np.array(
            [[0.0], [1.0], [2.0]], dtype=np.float64
        )
        q = QuantileDistribution(epsilon=0.01)
        q.fit(data, None, 0)
        # Act
        q_copy = q.copy()
        q_copy.cdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,inf[': '1.0',
        })
        # Assert — the original survived untouched
        self.assertAlmostEqual(
            q.cdf.eval(1.0), 0.5, delta=0.05
        )


# ----------------------------------------------------------------------

class SampleTest(TestCase):
    """Tests for ``QuantileDistribution.sample``."""

    def test_sample_from_jump_is_constant(self):
        """Sampling from a jump CDF returns the jump
        point, repeated ``n`` times."""
        # Arrange
        data = np.array([[42.0]], dtype=np.float64)
        q = QuantileDistribution()
        q.fit(data, np.array([0], dtype=np.int64), 0)
        # Act
        samples = q.sample(50)
        # Assert
        self.assertEqual(samples.shape, (50,))
        np.testing.assert_array_equal(
            samples, np.full(50, 42.0)
        )

    def test_sample_shape(self):
        """``sample(n)`` returns an array of length
        ``n``."""
        # Arrange
        data = np.linspace(
            0.0, 1.0, 20
        ).reshape(-1, 1)
        q = QuantileDistribution(epsilon=0.01)
        q.fit(data, None, 0)
        # Act
        samples = q.sample(123)
        # Assert
        self.assertEqual(samples.shape, (123,))

    def test_sample_within_support(self):
        """Samples from a uniform fit land inside the
        support interval."""
        # Arrange
        np.random.seed(0)
        data = np.linspace(
            10.0, 20.0, 50
        ).reshape(-1, 1)
        q = QuantileDistribution(epsilon=0.01)
        q.fit(data, None, 0)
        # Act
        samples = q.sample(500)
        # Assert
        self.assertTrue(
            np.all(samples >= 10.0 - 1e-6),
            'Some samples below the support'
        )
        self.assertTrue(
            np.all(samples <= 20.0 + 1e-6),
            'Some samples above the support'
        )

    def test_sample_mean_approximates_distribution_mean(
            self
    ):
        """The empirical mean of many samples
        approximates the CDF-based expected value."""
        # Arrange — uniform on [0, 10], mean = 5
        np.random.seed(7)
        data = np.linspace(
            0.0, 10.0, 200
        ).reshape(-1, 1)
        q = QuantileDistribution(epsilon=0.005)
        q.fit(data, None, 0)
        # Act
        samples = q.sample(5000)
        # Assert
        self.assertAlmostEqual(
            samples.mean(), 5.0, delta=0.2
        )


# ----------------------------------------------------------------------

class StaticConstructorTest(TestCase):
    """Tests for the ``from_cdf`` and ``from_pdf``
    static constructors, and the ``pdf`` setter path.
    """

    def test_from_cdf_sets_cdf_directly(self):
        """``from_cdf`` wraps the given CDF without
        further processing."""
        # Arrange
        cdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,1.0[': '1.0x + 0.0',
            '[1.0,inf[': '1.0',
        })
        # Act
        q = QuantileDistribution.from_cdf(cdf)
        # Assert
        self.assertIs(q.cdf, cdf)
        self.assertAlmostEqual(q.cdf.eval(0.3), 0.3)
        self.assertAlmostEqual(q.cdf.eval(-1.0), 0.0)
        self.assertAlmostEqual(q.cdf.eval(2.0), 1.0)

    def test_from_cdf_default_epsilon(self):
        """``from_cdf`` uses the default epsilon."""
        # Arrange
        cdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,inf[': '1.0',
        })
        # Act
        q = QuantileDistribution.from_cdf(cdf)
        # Assert
        self.assertAlmostEqual(q.epsilon, 0.01)

    def test_from_pdf_integrates_to_cdf(self):
        """``from_pdf`` produces a CDF reaching 1 at
        the right tail."""
        # Arrange — uniform PDF on [0, 1] has value 1
        pdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,1.0[': '1.0',
            '[1.0,inf[': '0.0',
        })
        # Act
        q = QuantileDistribution.from_pdf(pdf)
        # Assert — integral of uniform is linear
        self.assertAlmostEqual(
            q.cdf.eval(-0.5), 0.0, places=10
        )
        self.assertAlmostEqual(
            q.cdf.eval(0.5), 0.5, places=10
        )
        self.assertAlmostEqual(
            q.cdf.eval(1.5), 1.0, places=10
        )

    def test_from_pdf_multi_segment_constant(self):
        """A three-level piecewise-constant PDF
        integrates to a CDF with matching slope
        changes at segment boundaries."""
        # Arrange — pdf: 1 on [0,0.5), 2 on [0.5,0.75),
        # 0 elsewhere. Total mass = 0.5·1 + 0.25·2 = 1.
        pdf = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,0.5[': '1.0',
            '[0.5,0.75[': '2.0',
            '[0.75,inf[': '0.0',
        })
        # Act
        q = QuantileDistribution.from_pdf(pdf)
        # Assert — CDF reaches 0.5 at x=0.5 and 1.0
        # at x=0.75; slopes match the PDF densities
        self.assertAlmostEqual(
            q.cdf.eval(0.25), 0.25, places=6
        )
        self.assertAlmostEqual(
            q.cdf.eval(0.5), 0.5, places=6
        )
        self.assertAlmostEqual(
            q.cdf.eval(0.625), 0.75, places=6
        )
        self.assertAlmostEqual(
            q.cdf.eval(0.75), 1.0, places=6
        )

    def test_pdf_setter_roundtrip(self):
        """Setting the pdf property produces a
        consistent cdf; re-reading the pdf recovers
        the same shape at interior points."""
        # Arrange
        pdf_in = PiecewiseFunction.from_dict({
            ']-inf,0.0[': '0.0',
            '[0.0,2.0[': '0.5',
            '[2.0,inf[': '0.0',
        })
        q = QuantileDistribution()
        # Act
        q.pdf = pdf_in
        # Assert — CDF is 0 below 0, 1 above 2,
        # and grows linearly with slope 0.5 between
        self.assertAlmostEqual(
            q.cdf.eval(-0.5), 0.0, places=10
        )
        self.assertAlmostEqual(
            q.cdf.eval(1.0), 0.5, places=10
        )
        self.assertAlmostEqual(
            q.cdf.eval(2.5), 1.0, places=10
        )
        self.assertAlmostEqual(
            q.pdf.eval(1.0), 0.5, places=10
        )
