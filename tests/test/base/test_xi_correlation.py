"""Tests for Chatterjee's xi correlation coefficient.

Tests the mathematical properties of the xi
coefficient: boundary behavior, known functional
relationships, independence, asymmetry, and matrix
computation.
"""
import unittest

import numpy as np

from jpt.base.correlation.xi import (
    xi_correlation,
    xi_correlation_matrix,
)


# ------------------------------------------------------


class TestXiCorrelation(unittest.TestCase):
    """Tests for the xi_correlation function."""

    def test_perfect_monotone_near_one(self):
        """Xi of a perfect monotone function should
        be close to 1.
        """
        # Arrange
        np.random.seed(0)
        n = 5000
        x = np.random.uniform(-5, 5, n)
        y = 2 * x + 3

        # Act
        xi = xi_correlation(x, y)

        # Assert
        self.assertGreater(xi, 0.99)

    def test_perfect_nonmonotone_near_one(self):
        """Xi of a perfect non-monotone function
        (x^2) should be close to 1.
        """
        # Arrange
        np.random.seed(1)
        n = 5000
        x = np.random.uniform(-5, 5, n)
        y = x ** 2

        # Act
        xi = xi_correlation(x, y)

        # Assert
        self.assertGreater(xi, 0.99)

    def test_independence_near_zero(self):
        """Xi of independent variables should be
        close to 0.
        """
        # Arrange
        np.random.seed(2)
        n = 5000
        x = np.random.uniform(-5, 5, n)
        y = np.random.uniform(-5, 5, n)

        # Act
        xi = xi_correlation(x, y)

        # Assert
        self.assertAlmostEqual(xi, 0., delta=0.05)

    def test_asymmetry(self):
        """Xi(X, Y) != Xi(Y, X) for asymmetric
        relationships like many-to-one mappings.
        """
        # Arrange: Y = X^2 is many-to-one,
        # so Y is a function of X but X is not
        # a function of Y.
        np.random.seed(3)
        n = 5000
        x = np.random.uniform(-5, 5, n)
        y = x ** 2

        # Act
        xi_xy = xi_correlation(x, y)
        xi_yx = xi_correlation(y, x)

        # Assert: xi(X, Y) should be much higher
        # than xi(Y, X)
        self.assertGreater(xi_xy, 0.99)
        self.assertLess(xi_yx, xi_xy)

    def test_noisy_relationship_intermediate(self):
        """Xi of a noisy functional relationship
        should be between 0 and 1.
        """
        # Arrange
        np.random.seed(4)
        n = 5000
        x = np.random.uniform(-2, 2, n)
        y = np.sin(x) + np.random.normal(0, 0.5, n)

        # Act
        xi = xi_correlation(x, y)

        # Assert
        self.assertGreater(xi, 0.1)
        self.assertLess(xi, 0.9)

    def test_too_few_samples(self):
        """Xi should return 0 for fewer than 3
        samples.
        """
        # Arrange
        x = np.array([1., 2.])
        y = np.array([3., 4.])

        # Act
        xi = xi_correlation(x, y)

        # Assert
        self.assertEqual(xi, 0.)

    def test_periodic_function_detected(self):
        """Xi should detect periodic dependence
        that Pearson and Spearman miss.
        """
        # Arrange
        np.random.seed(5)
        n = 5000
        x = np.random.uniform(0, 4 * np.pi, n)
        y = np.sin(x)

        # Act
        xi = xi_correlation(x, y)

        # Assert: sin is a function of x,
        # Pearson would give ~0
        self.assertGreater(xi, 0.95)


# ------------------------------------------------------


class TestXiCorrelationMatrix(unittest.TestCase):
    """Tests for the xi_correlation_matrix function.
    """

    def test_shape(self):
        """Output matrix should have shape
        (n_features x n_targets).
        """
        # Arrange
        np.random.seed(10)
        data = np.random.randn(500, 5)
        feat_idx = np.array([0, 1, 2], dtype=np.int64)
        tgt_idx = np.array([3, 4], dtype=np.int64)

        # Act
        result = xi_correlation_matrix(
            data, feat_idx, tgt_idx
        )

        # Assert
        self.assertEqual(result.shape, (3, 2))

    def test_known_structure_recovery(self):
        """Matrix should identify which pairs are
        dependent and which are not.
        """
        # Arrange: col0 -> col2 (quadratic),
        # col1 independent of col2
        np.random.seed(11)
        n = 3000
        col0 = np.random.uniform(-2, 2, n)
        col1 = np.random.uniform(-2, 2, n)
        col2 = col0 ** 2
        data = np.column_stack([col0, col1, col2])
        feat_idx = np.array([0, 1], dtype=np.int64)
        tgt_idx = np.array([2], dtype=np.int64)

        # Act
        result = xi_correlation_matrix(
            data, feat_idx, tgt_idx
        )

        # Assert
        self.assertGreater(
            result[0, 0], 0.9,
            "col0 -> col2 should be strongly "
            "dependent"
        )
        self.assertAlmostEqual(
            result[1, 0], 0., delta=0.05,
            msg="col1 -> col2 should be independent"
        )

    def test_row_indices_subset(self):
        """row_indices should restrict computation
        to the specified rows.
        """
        # Arrange: first 100 rows Y=X, rest Y=-X
        np.random.seed(12)
        n = 1000
        x = np.random.uniform(-2, 2, n)
        y = np.where(
            np.arange(n) < 100, x, -x
        )
        data = np.column_stack([x, y])
        feat_idx = np.array([0], dtype=np.int64)
        tgt_idx = np.array([1], dtype=np.int64)

        # Act: only use first 100 rows
        rows = np.arange(100, dtype=np.int64)
        result_subset = xi_correlation_matrix(
            data, feat_idx, tgt_idx,
            row_indices=rows
        )
        result_full = xi_correlation_matrix(
            data, feat_idx, tgt_idx
        )

        # Assert: subset should show Y=X (high xi),
        # full data mixes Y=X and Y=-X
        self.assertGreater(result_subset[0, 0], 0.9)
        self.assertNotAlmostEqual(
            result_subset[0, 0],
            result_full[0, 0],
            places=1
        )


if __name__ == '__main__':
    unittest.main()
