from unittest import TestCase

import numpy as np

from jpt.distributions.qpd import QuantileDistribution
from jpt.distributions.qpd.cdfreg import CDFRegressor

from jpt.base.functions import PiecewiseFunction


# ----------------------------------------------------------------------

class QuantileDistributionFitTest(TestCase):

    def test_quantile_dist_linear(self):
        """Verify CDF fitting for linearly distributed data."""
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
        """Verify CDF fitting for a single data point producing a jump function."""
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
        """Verify CDF fitting when the first data points are duplicates."""
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
        """Verify CDF fitting when the last data points are duplicates."""
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


# ------------------------------------------------------------------------------

class CDFRegressorBugTest(TestCase):
    """
    Tests that expose known bugs in CDFRegressor.

    Each test is expected to FAIL before the corresponding fix is applied,
    and PASS once the bug is corrected.
    """

    @staticmethod
    def _cdata(xs, ys):
        """Return a C-contiguous 2xN float64 array for CDFRegressor.fit()."""
        return np.ascontiguousarray(
            np.array([xs, ys], dtype=np.float64)
        )

    # ------------------------------------------------------------------

    def test_eps_stored_as_mse_threshold(self):
        """
        CDFRegressor stores eps as its square (the MSE threshold).

        The caller passes an RMSE tolerance; the stored value is the
        corresponding MSE threshold so that the backward-pass comparison
        avoids a square root.  Passing eps=0.1 must store 0.01.
        """
        # Arrange / Act
        reg = CDFRegressor(eps=0.1)

        # Assert
        self.assertAlmostEqual(
            reg.eps,
            0.1 ** 2,
            places=10,
            msg=(
                'eps not stored as MSE (squared): '
                'expected %.4f, got %s' % (0.1 ** 2, reg.eps)
            )
        )

    # ------------------------------------------------------------------

    def test_support_points_not_none_for_single_point(self):
        """
        support_points returns None instead of a list when the fitted
        data has fewer than two breakpoints.

        The bug: `return points.append(data)` returns None because
        list.append() always returns None.
        """
        # Arrange
        reg = CDFRegressor(eps=0.0, delta_min=0.0)
        data = self._cdata([1.0], [1.0])
        reg.fit(data)

        # Act
        result = reg.support_points

        # Assert
        self.assertIsNotNone(
            result,
            'support_points returned None for single-point data'
        )

    # ------------------------------------------------------------------

    def test_verify_no_attribute_error(self):
        """
        verify() raises AttributeError because it references self.cdf,
        which is not an attribute of CDFRegressor.

        After fitting, verify() should be callable without crashing.
        """
        # Arrange
        reg = CDFRegressor(eps=0.01, delta_min=np.nan)
        data = self._cdata([0., 1., 2.], [0., 0.5, 1.])
        reg.fit(data)

        # Act / Assert
        try:
            reg.verify([(0., 0.), (1., 0.5), (2., 1.)])
        except AttributeError as exc:
            self.fail(
                'verify() raised AttributeError: %s' % exc
            )

    # ------------------------------------------------------------------

    def test_verify_passes_for_accurate_fit(self):
        """
        verify() raises AssertionError even for a good fit because the
        assertion direction is inverted: `abs(error) > eps` should be
        `abs(error) < eps`.

        A call to verify() with data that matches the fit within eps
        should not raise any exception.
        """
        # Arrange: perfectly linear data
        reg = CDFRegressor(eps=0.01, delta_min=np.nan)
        data = self._cdata([0., 1., 2.], [0., 0.5, 1.])
        reg.fit(data)

        # Act / Assert: a good fit must not raise AssertionError
        try:
            reg.verify([(0., 0.), (1., 0.5), (2., 1.)])
        except AssertionError:
            self.fail(
                'verify() raised AssertionError for a good fit '
                '(assertion direction is inverted)'
            )

    # ------------------------------------------------------------------

    def test_jump_detected_automatically(self):
        """
        A quantile step that is large relative to the median step must
        be detected as a jump and appear as a near-duplicate x pair in
        support_points: (nextafter(x, x-1), y_before) followed by
        (x, y_after).

        Previously, jump detection was silently disabled when delta_min
        was nan (the default) because `delta - 10*nan` evaluates to nan,
        making the threshold comparison always False.  The new pre-scan
        approach derives the threshold from the data's median step and
        requires no caller-supplied parameter.
        """
        # Arrange: x=[0..4], quantile jumps 0.92 between index 1 and 2;
        # median step is ~0.02, threshold ~0.20 — well below 0.92.
        xs = np.array([0., 1., 2., 3., 4.])
        ys = np.array([0., 0.04, 0.96, 0.98, 1.0])
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=1e-9)

        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert: jump at x=2 must appear as (nextafter(2, 2-1), ...), (2, ...)
        self.assertGreaterEqual(
            len(pts), 2,
            'Expected at least 2 support points, got %d' % len(pts)
        )
        has_jump = any(
            np.nextafter(pts[i + 1, 0], pts[i + 1, 0] - 1.) == pts[i, 0]
            for i in range(len(pts) - 1)
        )
        self.assertTrue(
            has_jump,
            'Jump at x=2 not detected in support_points.\n'
            'support_points:\n%s' % pts
        )
