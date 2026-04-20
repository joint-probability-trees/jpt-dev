"""Direct tests for ``VWCDFRegressor``.

Exercises the public class API — construction, fit as a chainable
operation, accessor properties before and after fit, and
invariants on the simplified knot set.
"""
import unittest
from unittest import TestCase

import numpy as np
from scipy.stats import norm

from jpt.distributions.qpd.vwcdfreg import VWCDFRegressor


# ----------------------------------------------------------------------

class VWCDFRegressorAPITest(TestCase):
    """Shape and API-level tests. No numerical content beyond
    trivially-verifiable outcomes — tighter numerics live in
    VWCDFRegressorInvariantsTest below.
    """

    def test_init_stores_eps(self):
        """Constructor stores ``eps`` as a readable attribute."""
        # Arrange / Act
        reg = VWCDFRegressor(eps=0.02)
        # Assert
        self.assertAlmostEqual(reg.eps, 0.02)

    def test_default_eps_is_zero(self):
        """Default ``eps`` is 0 (exact interpolation)."""
        # Arrange / Act
        reg = VWCDFRegressor()
        # Assert
        self.assertEqual(reg.eps, 0.0)

    def test_unfit_properties_are_none(self):
        """Before ``fit``, ``fit_xs`` and ``fit_ys`` are None."""
        # Arrange / Act
        reg = VWCDFRegressor(eps=0.01)
        # Assert
        self.assertIsNone(reg.fit_xs)
        self.assertIsNone(reg.fit_ys)

    def test_unfit_support_points_is_empty(self):
        """Before ``fit``, ``support_points`` returns an empty
        ``(0, 2)`` array rather than raising or returning None."""
        # Arrange / Act
        reg = VWCDFRegressor(eps=0.01)
        # Assert
        self.assertEqual(reg.support_points.shape, (0, 2))

    def test_fit_is_chainable(self):
        """``fit`` returns ``self``."""
        # Arrange
        reg = VWCDFRegressor(eps=0.01)
        xs = np.linspace(0, 1, 20).astype(np.float64)
        ys = xs.copy()
        # Act
        returned = reg.fit(xs, ys)
        # Assert
        self.assertIs(returned, reg)

    def test_fit_populates_accessors(self):
        """After ``fit``, all three accessor properties return
        the same knot set in consistent shapes."""
        # Arrange
        reg = VWCDFRegressor(eps=0.05)
        xs = np.linspace(0, 1, 30).astype(np.float64)
        ys = (xs ** 2).astype(np.float64)
        # Act
        reg.fit(xs, ys)
        # Assert
        self.assertIsNotNone(reg.fit_xs)
        self.assertIsNotNone(reg.fit_ys)
        self.assertEqual(
            reg.fit_xs.shape, reg.fit_ys.shape
        )
        self.assertEqual(
            reg.support_points.shape,
            (reg.fit_xs.shape[0], 2)
        )
        np.testing.assert_array_equal(
            reg.support_points[:, 0], reg.fit_xs
        )
        np.testing.assert_array_equal(
            reg.support_points[:, 1], reg.fit_ys
        )

    def test_refit_overwrites_previous_output(self):
        """Calling ``fit`` again replaces the previous output."""
        # Arrange
        reg = VWCDFRegressor(eps=0.01)
        xs1 = np.linspace(0, 1, 10).astype(np.float64)
        ys1 = xs1.copy()
        xs2 = np.linspace(10, 20, 50).astype(np.float64)
        ys2 = (xs2 - 10) / 10
        # Act
        reg.fit(xs1, ys1)
        knots_after_first = reg.fit_xs.copy()
        reg.fit(xs2, ys2)
        # Assert
        self.assertFalse(
            np.array_equal(
                knots_after_first, reg.fit_xs
            )
        )
        self.assertTrue(reg.fit_xs.min() >= 10)
        self.assertTrue(reg.fit_xs.max() <= 20)

    def test_fit_length_mismatch_raises(self):
        """Mismatched ``xs`` and ``ys`` lengths raise
        ``ValueError``."""
        # Arrange
        reg = VWCDFRegressor(eps=0.01)
        xs = np.array([0.0, 1.0], dtype=np.float64)
        ys = np.array([0.0, 0.5, 1.0], dtype=np.float64)
        # Act / Assert
        with self.assertRaises(ValueError):
            reg.fit(xs, ys)


# ----------------------------------------------------------------------

class VWCDFRegressorInvariantsTest(TestCase):
    """Numerical invariants: the fit is within eps, knots are
    ordered, endpoints are always preserved, behaviour degenerates
    gracefully on n <= 2.
    """

    @staticmethod
    def _fit(xs, ys, eps):
        reg = VWCDFRegressor(eps=eps)
        reg.fit(
            np.ascontiguousarray(xs, dtype=np.float64),
            np.ascontiguousarray(ys, dtype=np.float64)
        )
        return reg

    def _max_residual(self, reg, xs, ys):
        return float(np.max(np.abs(
            np.interp(xs, reg.fit_xs, reg.fit_ys) - ys
        )))

    def test_single_point_returns_unchanged(self):
        """``n == 1`` is returned as-is (no simplification
        possible)."""
        # Arrange
        xs = np.array([2.5], dtype=np.float64)
        ys = np.array([0.0], dtype=np.float64)
        # Act
        reg = self._fit(xs, ys, 0.01)
        # Assert
        np.testing.assert_array_equal(reg.fit_xs, xs)
        np.testing.assert_array_equal(reg.fit_ys, ys)

    def test_two_points_returned_unchanged(self):
        """``n == 2`` is returned as-is (nothing to remove)."""
        # Arrange
        xs = np.array([0.0, 1.0], dtype=np.float64)
        ys = np.array([0.0, 1.0], dtype=np.float64)
        # Act
        reg = self._fit(xs, ys, 0.5)
        # Assert
        np.testing.assert_array_equal(reg.fit_xs, xs)
        np.testing.assert_array_equal(reg.fit_ys, ys)

    def test_endpoints_always_preserved(self):
        """First and last input points survive simplification
        regardless of eps."""
        # Arrange
        xs = np.linspace(-5, 5, 200).astype(np.float64)
        ys = norm.cdf(xs).astype(np.float64)
        # Act
        reg = self._fit(xs, ys, 1.0)
        # Assert — even at extreme eps, endpoints must remain
        self.assertAlmostEqual(
            float(reg.fit_xs[0]), xs[0]
        )
        self.assertAlmostEqual(
            float(reg.fit_xs[-1]), xs[-1]
        )

    def test_knots_are_ordered(self):
        """Surviving knots retain the monotone order of the
        input."""
        # Arrange
        rng = np.random.RandomState(0)
        xs = np.sort(rng.normal(0, 1, 200)).astype(np.float64)
        ys = np.linspace(0, 1, 200).astype(np.float64)
        # Act
        reg = self._fit(xs, ys, 0.01)
        # Assert
        self.assertTrue(
            np.all(np.diff(reg.fit_xs) > 0),
            'knot xs are not strictly increasing'
        )
        self.assertTrue(
            np.all(np.diff(reg.fit_ys) >= 0),
            'knot ys are not non-decreasing'
        )

    def test_residual_within_eps_across_shapes(self):
        """For every standard input shape, the max absolute
        residual against the original input is at most
        ``eps``."""
        rng = np.random.RandomState(1)
        cases = [
            (np.linspace(0, 1, 50), np.linspace(0, 1, 50)),
            (np.linspace(-3, 3, 150),
             norm.cdf(np.linspace(-3, 3, 150))),
            (np.linspace(0, 1, 100),
             np.linspace(0, 1, 100) ** 2),
            (np.sort(rng.uniform(0, 10, 300)),
             np.linspace(0, 1, 300)),
        ]
        for eps in [0.1, 0.05, 0.01, 0.001]:
            for xs, ys in cases:
                xs = xs.astype(np.float64)
                ys = ys.astype(np.float64)
                reg = self._fit(xs, ys, eps)
                residual = self._max_residual(
                    reg, xs, ys
                )
                self.assertLessEqual(
                    residual, eps + 1e-12,
                    f'eps={eps} residual={residual:.4f} on '
                    f'shape len={len(xs)}'
                )

    def test_tighter_eps_never_fewer_knots(self):
        """Decreasing ``eps`` cannot increase the *removed*
        knots count (equivalently: cannot produce fewer surviving
        knots)."""
        # Arrange
        rng = np.random.RandomState(2)
        xs = np.sort(rng.normal(0, 1, 200)).astype(np.float64)
        ys = np.linspace(0, 1, 200).astype(np.float64)
        prev_count = 0
        # Act / Assert — walk eps from coarse to fine
        for eps in [0.1, 0.05, 0.01, 0.001]:
            reg = self._fit(xs, ys, eps)
            count = reg.fit_xs.shape[0]
            self.assertGreaterEqual(
                count, prev_count,
                f'tighter eps={eps} gave fewer knots '
                f'({count}) than the previous coarser eps '
                f'({prev_count})'
            )
            prev_count = count

    def test_eps_zero_returns_all_points(self):
        """``eps = 0`` allows no removal: all input points
        survive (interpolating fit)."""
        # Arrange
        xs = np.linspace(0, 1, 20).astype(np.float64)
        ys = (xs ** 2).astype(np.float64)
        # Act
        reg = self._fit(xs, ys, 0.0)
        # Assert
        self.assertEqual(reg.fit_xs.shape[0], 20)
        np.testing.assert_array_equal(reg.fit_xs, xs)
        np.testing.assert_array_equal(reg.fit_ys, ys)


# ----------------------------------------------------------------------

class VWCDFRegressorCythonVsReferenceTest(TestCase):
    """Cross-check the Cython implementation against a short
    Python reference of the same algorithm on random inputs.
    """

    @staticmethod
    def _reference_simplify(xs, ys, eps):
        """Plain-Python Visvalingam-Whyatt with L∞ cost,
        identical to the Cython version. Used only as a
        correctness oracle."""
        import heapq
        xs = np.ascontiguousarray(xs, dtype=np.float64)
        ys = np.ascontiguousarray(ys, dtype=np.float64)
        n = len(xs)
        if n <= 2:
            return xs.copy(), ys.copy()

        left = list(range(n))
        right = list(range(n))
        for i in range(n):
            left[i] = i - 1
            right[i] = i + 1
        alive = [True] * n

        def rcost(i):
            l = left[i]
            r = right[i]
            if l < 0 or r >= n:
                return float('inf')
            xl = xs[l]
            xr = xs[r]
            if xr == xl:
                return float('inf')
            slope = (ys[r] - ys[l]) / (xr - xl)
            yl = ys[l]
            mx = 0.0
            for k in range(l + 1, r):
                err = abs(
                    yl + slope * (xs[k] - xl) - ys[k]
                )
                if err > mx:
                    mx = err
            return mx

        heap = []
        costs = [0.0] * n
        for i in range(1, n - 1):
            costs[i] = rcost(i)
            heapq.heappush(heap, (costs[i], i))

        while heap:
            cost, i = heapq.heappop(heap)
            if not alive[i] or costs[i] != cost:
                continue
            if cost >= eps:
                break
            alive[i] = False
            l = left[i]
            r = right[i]
            if l >= 0:
                right[l] = r
            if r < n:
                left[r] = l
            for nb in (l, r):
                if 0 < nb < n - 1 and alive[nb]:
                    costs[nb] = rcost(nb)
                    heapq.heappush(heap, (costs[nb], nb))

        kept = [i for i in range(n) if alive[i]]
        return (
            np.asarray([xs[i] for i in kept]),
            np.asarray([ys[i] for i in kept])
        )

    def test_matches_python_reference_on_random_inputs(self):
        """Cython output agrees with the Python reference on
        many random monotone inputs."""
        rng = np.random.RandomState(42)
        for trial in range(20):
            n = rng.randint(5, 200)
            # Random monotone CDF-like data
            xs = np.sort(
                rng.uniform(-10, 10, n)
            ).astype(np.float64)
            # Ensure uniqueness
            xs = np.unique(xs)
            n = len(xs)
            if n < 3:
                continue
            ys = np.sort(
                rng.uniform(0, 1, n)
            ).astype(np.float64)
            ys[0] = 0.0
            ys[-1] = 1.0
            eps = float(rng.choice(
                [0.001, 0.01, 0.05, 0.1]
            ))

            reg = VWCDFRegressor(eps=eps)
            reg.fit(xs, ys)
            ref_xs, ref_ys = self._reference_simplify(
                xs, ys, eps
            )

            # Tolerance for lazy-deletion ordering
            # differences: the two implementations may
            # retain/drop marginal ties differently, but the
            # surviving knot counts and ranges should match.
            self.assertEqual(
                len(reg.fit_xs), len(ref_xs),
                f'trial {trial}: knot count mismatch '
                f'(cython={len(reg.fit_xs)} ref={len(ref_xs)})'
            )
            np.testing.assert_allclose(
                reg.fit_xs, ref_xs,
                err_msg=f'trial {trial}: fit_xs differ',
                rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                reg.fit_ys, ref_ys,
                err_msg=f'trial {trial}: fit_ys differ',
                rtol=1e-12, atol=1e-12
            )


# ----------------------------------------------------------------------

class VWCDFRegressorPerformanceTest(TestCase):
    """Performance canaries — lenient upper bounds that catch
    catastrophic regressions without being tied to any specific
    hardware.
    """

    def test_fits_10k_points_under_one_second(self):
        """Fitting 10 000 sorted points completes in under 1 s."""
        import time
        # Arrange
        xs = np.linspace(-5, 5, 10000).astype(np.float64)
        ys = norm.cdf(xs).astype(np.float64)
        # Act
        reg = VWCDFRegressor(eps=0.001)
        t0 = time.perf_counter()
        reg.fit(xs, ys)
        elapsed = time.perf_counter() - t0
        # Assert
        self.assertLess(
            elapsed, 1.0,
            f'fit of 10k points took {elapsed:.3f}s '
            f'(expected < 1.0s)'
        )

    def test_typical_production_fit_under_millisecond(self):
        """The post-subsample typical case
        (n ≈ 1/eps = 100) completes well under 1 ms."""
        import time
        # Arrange
        xs = np.linspace(-3, 3, 100).astype(np.float64)
        ys = norm.cdf(xs).astype(np.float64)
        reg = VWCDFRegressor(eps=0.01)
        # Act — warm up
        reg.fit(xs, ys)
        t0 = time.perf_counter()
        for _ in range(100):
            VWCDFRegressor(eps=0.01).fit(xs, ys)
        elapsed = (
            time.perf_counter() - t0
        ) / 100 * 1000  # ms per fit
        # Assert
        self.assertLess(
            elapsed, 1.0,
            f'typical fit took {elapsed:.3f}ms '
            f'(expected < 1.0ms)'
        )


if __name__ == '__main__':
    unittest.main()
