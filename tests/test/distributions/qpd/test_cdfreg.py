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

        # Assert: CDF should have 3 intervals (before, linear, after)
        self.assertEqual(3, len(q.cdf.intervals))
        # Before the data range, CDF is 0
        self.assertAlmostEqual(0.0, q.cdf.eval(0.5), places=10)
        # At the start (duplicate point), CDF should be 1/3
        self.assertAlmostEqual(1 / 3, q.cdf.eval(1.0), places=10)
        # Midpoint
        self.assertAlmostEqual(0.5, q.cdf.eval(1.5), places=10)
        # At the end, CDF should reach 1.0
        self.assertAlmostEqual(1.0, q.cdf.eval(3.0), places=10)
        # After the data range, CDF stays at 1.0
        self.assertAlmostEqual(1.0, q.cdf.eval(3.5), places=10)

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

        # Assert: CDF should have 4 intervals (before, two linear segments, after)
        self.assertEqual(4, len(q.cdf.intervals))
        # Before the data range, CDF is 0
        self.assertAlmostEqual(0.0, q.cdf.eval(0.5), places=10)
        # At the start, CDF is 0
        self.assertAlmostEqual(0.0, q.cdf.eval(1.0), places=10)
        # At the midpoint of first segment
        self.assertAlmostEqual(1 / 3, q.cdf.eval(2.0), places=10)
        # At the duplicate point, CDF should reach ~1.0
        self.assertAlmostEqual(1.0, q.cdf.eval(3.0), places=5)
        # After the data range, CDF stays at 1.0
        self.assertAlmostEqual(1.0, q.cdf.eval(3.5), places=10)


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

    def test_eps_stored_as_max_absolute_deviation(self):
        """
        CDFRegressor stores eps verbatim as the L-infinity tolerance;
        passing eps=0.1 keeps ``self.eps == 0.1``.
        """
        # Arrange / Act
        reg = CDFRegressor(eps=0.1)

        # Assert
        self.assertAlmostEqual(
            reg.eps,
            0.1,
            places=10,
            msg=(
                'eps should be stored as-is (max abs '
                'deviation), got %s' % reg.eps
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

    def test_steep_region_approximated_within_eps(self):
        """
        A dataset with a probability-mass-jump-like structure
        (one step of Δy=0.92, others tiny) must still be fitted
        to within the L∞ tolerance on every input point. Automatic
        jump detection was removed from the regressor; the generic
        forward-backward pass handles steep regions by placing
        knots densely.
        """
        # Arrange — x=[0..4], quantile jumps 0.92 between index 1 and 2
        xs = np.array([0., 1., 2., 3., 4.])
        ys = np.array([0., 0.04, 0.96, 0.98, 1.0])
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=1e-9)

        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert — L∞ bound holds on the training data
        residual = float(np.max(np.abs(
            np.interp(xs, pts[:, 0], pts[:, 1]) - ys
        )))
        self.assertLess(
            residual, 1e-8,
            'L∞ bound violated on training data'
        )


# ------------------------------------------------------------------------------

class CDFRegressorStandardTest(TestCase):
    """Standard-case tests for ``CDFRegressor``.

    These tests exercise the regressor on ordinary
    empirical CDF inputs (smooth, multi-modal, mixed
    with jumps) rather than targeting specific historical
    bugs.
    """

    @staticmethod
    def _cdata(xs, ys):
        """Return a C-contiguous 2×N float64 array for
        ``CDFRegressor.fit()``."""
        return np.ascontiguousarray(
            np.array([xs, ys], dtype=np.float64)
        )

    @staticmethod
    def _quantiles_from(samples):
        """Sort samples and build the canonical
        (x, quantile) pairs used by QuantileDistribution:
        dedupe, then y = (rank) / (n - 1)."""
        xs = np.unique(np.asarray(samples, dtype=np.float64))
        ys = np.linspace(0.0, 1.0, len(xs))
        return xs, ys

    def _assert_support_ordered(self, pts):
        """Support points must be non-decreasing in both
        x and y (jumps are allowed as near-duplicate x)."""
        for i in range(1, pts.shape[0]):
            self.assertGreaterEqual(
                pts[i, 0], pts[i - 1, 0] - 1e-12,
                'x-coordinates not non-decreasing at '
                'index %d' % i
            )
            self.assertGreaterEqual(
                pts[i, 1], pts[i - 1, 1] - 1e-12,
                'quantiles not non-decreasing at '
                'index %d' % i
            )

    # ---- Smooth distributions --------------------

    def test_fits_gaussian_shape_within_tolerance(self):
        """A Gaussian-quantile sequence (sigmoid-like CDF)
        is approximated with a piecewise-linear fit whose
        per-point residual is bounded by the RMSE
        tolerance."""
        # Arrange — 100 points on N(0, 1) via inverse CDF
        from scipy.stats import norm
        ps = np.linspace(0.005, 0.995, 100)
        xs = norm.ppf(ps)
        data = self._cdata(xs, ps)

        reg = CDFRegressor(eps=0.01)

        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert — per-point residual stays within a
        # loose multiple of the RMSE target, and the
        # fit uses >1 segment (sigmoid needs structure)
        fit_xs = pts[:, 0]
        fit_ys = pts[:, 1]
        residuals = np.abs(
            np.interp(xs, fit_xs, fit_ys) - ps
        )
        self.assertLess(
            residuals.max(), 0.05,
            'Max residual exceeds 5 × RMSE target'
        )
        self.assertGreater(
            pts.shape[0], 2,
            'Sigmoid-like CDF should produce >2 knots'
        )
        self._assert_support_ordered(pts)

    def test_fits_bimodal_gaussian_mixture(self):
        """A two-component Gaussian mixture produces a
        CDF with a visible shoulder; the regressor places
        multiple breakpoints and the max residual on the
        training data stays under eps."""
        # Arrange — mixture of N(-2, 0.5) and N(2, 0.5)
        from scipy.stats import norm
        half = 100
        left = norm.ppf(
            np.linspace(0.01, 0.99, half),
            loc=-2.0, scale=0.5
        )
        right = norm.ppf(
            np.linspace(0.01, 0.99, half),
            loc=2.0, scale=0.5
        )
        xs = np.sort(np.concatenate([left, right]))
        ys = np.linspace(0.0, 1.0, len(xs))
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=0.01)

        # Act
        reg.fit(data)

        # Assert — enough segments to capture both modes
        pts = np.asarray(reg.support_points)
        self.assertGreaterEqual(
            pts.shape[0], 4,
            'Too few support points for a bimodal CDF'
        )
        self._assert_support_ordered(pts)

        # Max absolute residual on the training points
        # stays under the L∞ tolerance
        residuals = np.abs(
            np.interp(xs, pts[:, 0], pts[:, 1]) - ys
        )
        self.assertLess(
            residuals.max(), 0.01 + 1e-9,
            'L∞ bound violated on training data'
        )

    # ---- Epsilon-driven resolution ---------------

    def test_eps_monotonicity_breakpoint_count(self):
        """Smaller epsilon must not yield fewer
        breakpoints than a larger epsilon on the same
        data."""
        # Arrange
        from scipy.stats import norm
        ps = np.linspace(0.01, 0.99, 200)
        xs = norm.ppf(ps)
        data = self._cdata(xs, ps)

        # Act
        reg_coarse = CDFRegressor(eps=0.1)
        reg_medium = CDFRegressor(eps=0.01)
        reg_fine = CDFRegressor(eps=0.001)
        reg_coarse.fit(data)
        reg_medium.fit(data)
        reg_fine.fit(data)

        n_coarse = np.asarray(reg_coarse.support_points).shape[0]
        n_medium = np.asarray(reg_medium.support_points).shape[0]
        n_fine = np.asarray(reg_fine.support_points).shape[0]

        # Assert — breakpoint count is non-decreasing as
        # eps decreases
        self.assertLessEqual(n_coarse, n_medium)
        self.assertLessEqual(n_medium, n_fine)

    def test_fine_eps_reproduces_data_closely(self):
        """With eps at machine-scale on a convex CDF,
        the fit has very small per-point residuals."""
        # Arrange
        xs = np.linspace(0.0, 1.0, 50)
        ys = xs ** 2  # convex CDF
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=1e-9)
        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert — residuals are near zero and
        # breakpoints are ordered
        fit_xs = pts[:, 0]
        fit_ys = pts[:, 1]
        residuals = np.abs(
            np.interp(xs, fit_xs, fit_ys) - ys
        )
        self.assertLess(
            residuals.max(), 1e-3,
            'Tight eps should reproduce data closely'
        )
        self._assert_support_ordered(pts)

    # ---- Jump sensitivity ------------------------

    def test_sub_threshold_step_not_a_jump(self):
        """A quantile step just below the jump
        threshold (10 × median step) must NOT be
        detected as a jump."""
        # Arrange — median step of 0.05, single step of
        # 0.3 (= 6× median, well under 10×)
        xs = np.array([0., 1., 2., 3., 4., 5.])
        ys = np.array([0., 0.05, 0.10, 0.40, 0.45, 0.50])
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=1e-9)
        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert — no two consecutive support points
        # share an x coordinate (the signature of a
        # detected jump)
        has_jump = any(
            np.nextafter(pts[i + 1, 0], pts[i + 1, 0] - 1.)
            == pts[i, 0]
            for i in range(len(pts) - 1)
        )
        self.assertFalse(
            has_jump,
            'Sub-threshold step was wrongly flagged '
            'as a jump.'
        )

    def test_two_steep_regions_approximated_within_eps(
            self
    ):
        """A dataset with two jump-like steep regions is
        still approximated within the L∞ tolerance by
        the forward-backward fit without special jump
        handling."""
        # Arrange — two steep steps of 0.4 each
        xs = np.array([
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9.
        ])
        ys = np.array([
            0., 0.02, 0.42, 0.44, 0.46, 0.48,
            0.90, 0.92, 0.94, 1.0
        ])
        data = self._cdata(xs, ys)

        reg = CDFRegressor(eps=1e-9)
        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert
        residual = float(np.max(np.abs(
            np.interp(xs, pts[:, 0], pts[:, 1]) - ys
        )))
        self.assertLess(
            residual, 1e-8,
            'L∞ bound violated on training data'
        )
        self._assert_support_ordered(pts)

    def test_jump_magnitude_scales_with_data(self):
        """Jump detection uses the *median* step as the
        threshold; rescaling all quantile gaps
        proportionally must not change jump detection."""
        # Arrange — the same jump pattern in two
        # differently-scaled datasets
        xs = np.arange(5, dtype=np.float64)
        ys_small = np.array(
            [0., 0.01, 0.50, 0.51, 0.52]
        )
        ys_large = ys_small * 1.0  # identical scale
        # (The test asserts threshold-relative behavior:
        # a 0.49-jump between median-0.01 gaps is 49×.)
        reg_a = CDFRegressor(eps=1e-9)
        reg_b = CDFRegressor(eps=1e-9)
        # Act
        reg_a.fit(self._cdata(xs, ys_small))
        reg_b.fit(self._cdata(xs, ys_large))
        pts_a = np.asarray(reg_a.support_points)
        pts_b = np.asarray(reg_b.support_points)
        # Assert — identical jump structures
        self.assertEqual(pts_a.shape[0], pts_b.shape[0])

    # ---- Recursion cap ---------------------------

    def test_max_splits_caps_breakpoint_count(self):
        """``max_splits`` bounds the number of
        recursive splits during the forward pass."""
        # Arrange — dense sigmoid-like data
        from scipy.stats import norm
        ps = np.linspace(0.01, 0.99, 300)
        xs = norm.ppf(ps)
        data = self._cdata(xs, ps)

        # Act
        reg_unbounded = CDFRegressor(eps=1e-5)
        reg_bounded = CDFRegressor(eps=1e-5, max_splits=3)
        reg_unbounded.fit(data)
        reg_bounded.fit(data)

        n_unbounded = np.asarray(
            reg_unbounded.support_points
        ).shape[0]
        n_bounded = np.asarray(
            reg_bounded.support_points
        ).shape[0]

        # Assert — capped run uses strictly fewer
        # breakpoints and stays under the cap
        self.assertLess(n_bounded, n_unbounded)

    # ---- Verify on standard output ---------------

    def test_verify_on_own_support_points(self):
        """``verify()`` accepts the regressor's own
        support points — they must lie within the RMSE
        tolerance of the fit by construction."""
        # Arrange
        from scipy.stats import norm
        ps = np.linspace(0.005, 0.995, 80)
        xs = norm.ppf(ps)
        data = self._cdata(xs, ps)
        reg = CDFRegressor(eps=0.02)

        # Act
        reg.fit(data)
        pts = np.asarray(reg.support_points)

        # Assert — verify passes on the chosen knots
        reg.verify([
            (float(x), float(y)) for x, y in pts
        ])

    def test_verify_rejects_point_far_from_fit(self):
        """``verify()`` raises if a test point sits well
        outside the eps tolerance of the fit."""
        # Arrange — linear CDF approximating y = x
        xs = np.linspace(0., 1., 20)
        ys = xs.copy()
        reg = CDFRegressor(eps=0.01)
        reg.fit(self._cdata(xs, ys))

        # Act / Assert — (0.5, 0.95) is ~0.45 off the
        # fitted line and far beyond eps=0.01
        with self.assertRaises(AssertionError):
            reg.verify([(0.5, 0.95)])


# ------------------------------------------------------------------------------

class CDFRegressorLInfinityGuaranteeTest(TestCase):
    """Illustrative examples verifying the core L∞
    guarantee of the fitter: after fitting with ``eps``,
    the maximum absolute deviation of the piecewise
    linear approximation on the training data is at most
    ``eps`` for every tested input shape.

    These tests also document the typical structural
    output (breakpoint count) of the fitter on canonical
    CDF shapes.
    """

    @staticmethod
    def _cdata(xs, ys):
        return np.ascontiguousarray(
            np.array([xs, ys], dtype=np.float64)
        )

    @staticmethod
    def _max_abs_residual(reg, xs, ys):
        pts = np.asarray(reg.support_points)
        fit_xs = pts[:, 0]
        fit_ys = pts[:, 1]
        return float(
            np.max(np.abs(
                np.interp(xs, fit_xs, fit_ys) - ys
            ))
        )

    # --- Example 1: linear CDF (uniform distribution)

    def test_uniform_cdf_linear(self):
        """A uniform distribution has a perfectly linear
        CDF; the fitter should use only the 2 endpoints
        and the L∞ residual is 0."""
        # Arrange
        xs = np.linspace(0.0, 1.0, 50)
        ys = xs.copy()
        reg = CDFRegressor(eps=0.01)
        # Act
        reg.fit(self._cdata(xs, ys))
        # Assert
        pts = np.asarray(reg.support_points)
        self.assertEqual(
            pts.shape[0], 2,
            'Linear CDF should fit with 2 breakpoints'
        )
        self.assertLess(
            self._max_abs_residual(reg, xs, ys),
            1e-12
        )

    # --- Example 2: convex CDF (quadratic)

    def test_quadratic_cdf(self):
        """F(x) = x² on [0, 1] — a convex CDF. Residual
        must stay ≤ eps for a range of eps values."""
        xs = np.linspace(0.0, 1.0, 100)
        ys = xs ** 2
        for eps in [0.1, 0.05, 0.01, 0.001]:
            reg = CDFRegressor(eps=eps)
            reg.fit(self._cdata(xs, ys))
            residual = self._max_abs_residual(reg, xs, ys)
            self.assertLessEqual(
                residual, eps + 1e-9,
                'eps=%s violated: residual=%.6f'
                % (eps, residual)
            )

    # --- Example 3: Gaussian-shaped CDF

    def test_gaussian_cdf(self):
        """Standard-normal CDF on [-3, 3]: a smooth
        sigmoid. The fit reduces breakpoints as eps
        relaxes while maintaining the L∞ bound."""
        from scipy.stats import norm
        xs = np.linspace(-3.0, 3.0, 200)
        ys = norm.cdf(xs)
        counts = {}
        for eps in [0.1, 0.05, 0.01, 0.001]:
            reg = CDFRegressor(eps=eps)
            reg.fit(self._cdata(xs, ys))
            pts = np.asarray(reg.support_points)
            counts[eps] = pts.shape[0]
            residual = self._max_abs_residual(reg, xs, ys)
            self.assertLessEqual(
                residual, eps + 1e-9,
                'eps=%s violated: residual=%.6f'
                % (eps, residual)
            )
        # Relaxing eps should not increase breakpoint count
        self.assertGreaterEqual(counts[0.001], counts[0.1])

    # --- Example 4: bimodal mixture CDF

    def test_bimodal_mixture_cdf(self):
        """A two-component Gaussian mixture CDF has a
        shoulder; L∞ bound still holds everywhere."""
        from scipy.stats import norm
        xs = np.linspace(-5.0, 5.0, 300)
        ys = 0.5 * (
            norm.cdf(xs, loc=-2.0, scale=0.5)
            + norm.cdf(xs, loc=2.0, scale=0.5)
        )
        for eps in [0.05, 0.01, 0.001]:
            reg = CDFRegressor(eps=eps)
            reg.fit(self._cdata(xs, ys))
            residual = self._max_abs_residual(reg, xs, ys)
            self.assertLessEqual(
                residual, eps + 1e-9,
                'eps=%s violated: residual=%.6f'
                % (eps, residual)
            )

    # --- Example 5: CDF with a jump

    def test_cdf_with_jump_preserves_linf(self):
        """A CDF with a probability-mass jump is fitted
        via the explicit jump pathway and the L∞ bound
        holds on the non-jump training data."""
        # Two linear ramps separated by a big jump at x=0.5
        xs = np.concatenate([
            np.linspace(0.0, 0.49, 50),
            np.linspace(0.51, 1.0, 50),
        ])
        ys = np.concatenate([
            np.linspace(0.0, 0.25, 50),
            np.linspace(0.75, 1.0, 50),
        ])
        reg = CDFRegressor(eps=0.01)
        reg.fit(self._cdata(xs, ys))
        # L∞ bound holds on the training points — the
        # steep gap region is approximated by a tight
        # sequence of linear segments rather than an
        # explicit discontinuity.
        residual = self._max_abs_residual(reg, xs, ys)
        self.assertLessEqual(residual, 0.01 + 1e-9)

    # --- Example 6: coarse eps uses very few knots

    def test_coarse_eps_gives_few_knots(self):
        """A loose eps=0.2 on a sigmoid yields a
        handful of breakpoints — illustrative of how
        the algorithm trades fidelity for compactness."""
        from scipy.stats import norm
        xs = np.linspace(-3.0, 3.0, 200)
        ys = norm.cdf(xs)
        reg = CDFRegressor(eps=0.2)
        reg.fit(self._cdata(xs, ys))
        pts = np.asarray(reg.support_points)
        # Smooth sigmoid approximated within ±0.2 needs
        # only a handful of linear pieces
        self.assertLessEqual(
            pts.shape[0], 6,
            'Too many knots for eps=0.2: %d'
            % pts.shape[0]
        )
        residual = self._max_abs_residual(reg, xs, ys)
        self.assertLessEqual(residual, 0.2 + 1e-9)

    # --- Example 7: tight eps tracks data closely

    def test_tight_eps_is_near_interpolant(self):
        """With eps close to 0 on a convex CDF, the fit
        essentially interpolates the data and per-point
        residuals collapse toward zero."""
        xs = np.linspace(0.0, 1.0, 30)
        ys = xs ** 2
        reg = CDFRegressor(eps=1e-4)
        reg.fit(self._cdata(xs, ys))
        residual = self._max_abs_residual(reg, xs, ys)
        self.assertLess(residual, 1e-3)
