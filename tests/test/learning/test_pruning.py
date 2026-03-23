"""Tests for xi-correlation-based pruning.

Tests that XiPruningCriterion correctly stops
splitting when functional dependence is no longer
significant, and that it interacts correctly with
the JPT learning pipeline.
"""
import unittest

import numpy as np
from pandas import DataFrame

from jpt.distributions import Numeric
from jpt.learning.pruning import XiPruningCriterion
from jpt.trees import JPT
from jpt.variables import NumericVariable


# ------------------------------------------------------


class TestXiPruningCriterion(unittest.TestCase):
    """Tests for XiPruningCriterion integration
    with JPT learning.
    """

    @classmethod
    def setUpClass(cls):
        """Create shared test data."""
        np.random.seed(42)
        n = 2000
        cls.x1 = np.random.uniform(-2, 2, n)
        cls.x2 = np.random.uniform(-2, 2, n)
        cls.y = (
            cls.x1 ** 2
            + np.random.normal(0, 1.5, n)
        )

        cls.df = DataFrame(
            {'X1': cls.x1, 'X2': cls.x2, 'Y': cls.y}
        )
        cls.vx1 = NumericVariable(
            'X1', Numeric, precision=.1
        )
        cls.vx2 = NumericVariable(
            'X2', Numeric, precision=.1
        )
        cls.vy = NumericVariable(
            'Y', Numeric, precision=.1
        )

    def test_pruning_reduces_tree_size(self):
        """Xi pruning should produce fewer leaves
        than a standard tree on noisy data.
        """
        # Arrange
        tree_std = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            min_samples_leaf=.01
        )
        tree_xi = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            min_samples_leaf=.01
        )

        # Act
        tree_std.learn(self.df)
        tree_xi.learn(
            self.df,
            prune_or_split=XiPruningCriterion(
                alpha=0.05
            )
        )

        # Assert
        self.assertLess(
            len(tree_xi.leaves),
            len(tree_std.leaves)
        )

    def test_stricter_alpha_produces_smaller_tree(
            self
    ):
        """A smaller alpha (stricter test) should
        produce a tree with fewer or equal leaves.
        """
        # Arrange / Act
        trees = {}
        for alpha in [0.10, 0.05, 0.01]:
            tree = JPT(
                variables=[
                    self.vx1, self.vx2, self.vy
                ],
                targets=[self.vy],
                min_samples_leaf=.01
            )
            tree.learn(
                self.df,
                prune_or_split=XiPruningCriterion(
                    alpha=alpha
                )
            )
            trees[alpha] = len(tree.leaves)

        # Assert: stricter alpha => fewer leaves
        self.assertGreaterEqual(
            trees[0.10], trees[0.05]
        )
        self.assertGreaterEqual(
            trees[0.05], trees[0.01]
        )

    def test_pure_noise_yields_single_leaf(self):
        """When Y is independent of all features,
        xi pruning should produce a single leaf
        (no splits justified).
        """
        # Arrange
        np.random.seed(77)
        n = 2000
        df = DataFrame({
            'X1': np.random.uniform(-2, 2, n),
            'X2': np.random.uniform(-2, 2, n),
            'Y': np.random.uniform(-2, 2, n),
        })
        tree = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            min_samples_leaf=.01
        )

        # Act
        tree.learn(
            df,
            prune_or_split=XiPruningCriterion(
                alpha=0.05
            )
        )

        # Assert
        self.assertEqual(
            len(tree.leaves), 1,
            "Pure noise data should yield a single "
            "leaf"
        )

    def test_strong_signal_still_splits(self):
        """When Y is a near-perfect function of X,
        xi pruning should still allow splits.
        """
        # Arrange
        np.random.seed(88)
        n = 2000
        x = np.random.uniform(-2, 2, n)
        df = DataFrame({
            'X1': x,
            'X2': np.random.uniform(-2, 2, n),
            'Y': x ** 2 + np.random.normal(
                0, 0.01, n
            ),
        })
        tree = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            min_samples_leaf=.01
        )

        # Act
        tree.learn(
            df,
            prune_or_split=XiPruningCriterion(
                alpha=0.05
            )
        )

        # Assert
        self.assertGreater(
            len(tree.leaves), 1,
            "Strong signal should produce multiple "
            "leaves"
        )


if __name__ == '__main__':
    unittest.main()
