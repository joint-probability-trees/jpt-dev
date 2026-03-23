"""Tests for dependency discovery strategies.

Tests the DependencyDiscovery ABC, the
XiDependencyDiscovery implementation, and
the integration with JPT learning and
serialization.
"""
import json
import unittest

import numpy as np
from pandas import DataFrame

from jpt.distributions import Numeric
from jpt.learning.dependency.base import (
    DependencyDiscovery,
)
from jpt.learning.dependency.xi import (
    XiDependencyDiscovery,
)
from jpt.trees import JPT
from jpt.variables import NumericVariable


# ------------------------------------------------------


class TestXiDependencyDiscovery(unittest.TestCase):
    """Tests for XiDependencyDiscovery."""

    @classmethod
    def setUpClass(cls):
        """Create shared test data with known
        dependency structure.
        """
        np.random.seed(42)
        cls.n = 3000
        cls.x1 = np.random.uniform(-2, 2, cls.n)
        cls.x2 = np.random.uniform(-2, 2, cls.n)
        cls.y = cls.x1 ** 2 + np.random.normal(
            0, 0.5, cls.n
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
        cls.variables = [cls.vx1, cls.vx2, cls.vy]
        cls.data = np.column_stack(
            [cls.x1, cls.x2, cls.y]
        )

    def test_recovers_known_structure(self):
        """XiDependencyDiscovery should identify
        X1->Y as significant and X2->Y as not.
        """
        # Arrange
        discovery = XiDependencyDiscovery(alpha=0.05)

        # Act
        deps = discovery(
            self.data,
            features=[self.vx1, self.vx2],
            targets=[self.vy],
            variables=self.variables
        )

        # Assert
        self.assertIn(
            self.vy, deps[self.vx1],
            "X1 -> Y should be detected"
        )
        self.assertNotIn(
            self.vy, deps[self.vx2],
            "X2 -> Y should not be detected"
        )

    def test_stricter_alpha_reduces_dependencies(
            self
    ):
        """A smaller alpha should retain fewer
        (or equal) dependencies than a larger one.
        """
        # Arrange
        strict = XiDependencyDiscovery(alpha=0.001)
        lenient = XiDependencyDiscovery(alpha=0.10)

        # Act
        deps_strict = strict(
            self.data,
            features=[self.vx1, self.vx2],
            targets=[self.vy],
            variables=self.variables
        )
        deps_lenient = lenient(
            self.data,
            features=[self.vx1, self.vx2],
            targets=[self.vy],
            variables=self.variables
        )

        # Assert
        n_strict = sum(
            len(d) for d in deps_strict.values()
        )
        n_lenient = sum(
            len(d) for d in deps_lenient.values()
        )
        self.assertLessEqual(n_strict, n_lenient)

    def test_all_independent_yields_empty(self):
        """When all variables are independent, no
        dependencies should be detected.
        """
        # Arrange
        np.random.seed(99)
        data = np.random.randn(2000, 3)
        discovery = XiDependencyDiscovery(alpha=0.05)

        # Act
        deps = discovery(
            data,
            features=[self.vx1, self.vx2],
            targets=[self.vy],
            variables=self.variables
        )

        # Assert
        total = sum(
            len(d) for d in deps.values()
        )
        self.assertEqual(
            total, 0,
            "Independent data should yield no "
            "dependencies"
        )


# ------------------------------------------------------


class TestDependencyDiscoverySerialization(
        unittest.TestCase
):
    """Tests for serialization of dependency
    discovery strategies.
    """

    def test_json_round_trip(self):
        """to_json / from_json should preserve
        the strategy type and parameters.
        """
        # Arrange
        original = XiDependencyDiscovery(alpha=0.01)

        # Act
        data = original.to_json()
        restored = DependencyDiscovery.from_json(data)

        # Assert
        self.assertIsInstance(
            restored, XiDependencyDiscovery
        )
        self.assertEqual(restored.alpha, 0.01)

    def test_json_is_serializable(self):
        """to_json output must be JSON-serializable.
        """
        # Arrange
        discovery = XiDependencyDiscovery(alpha=0.05)

        # Act / Assert: should not raise
        json.dumps(discovery.to_json())

    def test_from_json_none_returns_none(self):
        """from_json(None) should return None."""
        # Act
        result = DependencyDiscovery.from_json(None)

        # Assert
        self.assertIsNone(result)

    def test_from_json_unknown_type_raises(self):
        """from_json with unknown type should raise
        ValueError.
        """
        # Arrange
        data = {'type': 'NonexistentStrategy'}

        # Act / Assert
        with self.assertRaises(ValueError):
            DependencyDiscovery.from_json(data)


# ------------------------------------------------------


class TestDependencyDiscoveryJPTIntegration(
        unittest.TestCase
):
    """Tests for integration of dependency discovery
    with the JPT learning pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """Create shared test data."""
        np.random.seed(42)
        n = 2000
        x1 = np.random.uniform(-2, 2, n)
        x2 = np.random.uniform(-2, 2, n)
        y = x1 ** 2 + np.random.normal(0, 1.5, n)

        cls.df = DataFrame(
            {'X1': x1, 'X2': x2, 'Y': y}
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

    def test_discovery_produces_fewer_leaves(self):
        """A JPT with dependency discovery should
        produce fewer or equal leaves compared to a
        standard JPT, because irrelevant splits are
        suppressed.
        """
        # Arrange
        tree_std = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            min_samples_leaf=.02
        )
        tree_xi = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            dependencies=XiDependencyDiscovery(
                alpha=0.05
            ),
            min_samples_leaf=.02
        )

        # Act
        tree_std.learn(self.df)
        tree_xi.learn(self.df)

        # Assert
        self.assertLessEqual(
            len(tree_xi.leaves),
            len(tree_std.leaves)
        )

    def test_relearn_rediscovers_dependencies(self):
        """Calling learn() twice with different data
        should re-invoke the discovery strategy.
        """
        # Arrange
        tree = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            dependencies=XiDependencyDiscovery(
                alpha=0.05
            ),
            min_samples_leaf=.05
        )

        # Act: first learn with Y = f(X1)
        tree.learn(self.df)
        deps1 = dict(tree.dependencies.items())

        # Second learn with Y = f(X2) instead
        np.random.seed(99)
        n = 2000
        x2 = np.random.uniform(-2, 2, n)
        df2 = DataFrame({
            'X1': np.random.uniform(-2, 2, n),
            'X2': x2,
            'Y': x2 + np.random.normal(0, 0.1, n)
        })
        tree.learn(df2)
        deps2 = dict(tree.dependencies.items())

        # Assert: dependency structure should change
        self.assertNotEqual(
            [v.name for v in deps1[self.vx1]],
            [v.name for v in deps2[self.vx1]]
        )

    def test_jpt_serialization_preserves_discovery(
            self
    ):
        """Serializing and deserializing a JPT with
        a discovery strategy should preserve both
        the resolved dependencies and the strategy.
        """
        # Arrange
        tree = JPT(
            variables=[self.vx1, self.vx2, self.vy],
            targets=[self.vy],
            dependencies=XiDependencyDiscovery(
                alpha=0.05
            ),
            min_samples_leaf=.05
        )
        tree.learn(self.df)

        # Act
        data = tree.to_json()
        restored = JPT.from_json(data)

        # Assert: resolved deps match
        self.assertEqual(
            len(tree.leaves),
            len(restored.leaves)
        )
        # Assert: strategy is preserved
        self.assertIsNotNone(
            restored._dependency_discovery
        )
        self.assertIsInstance(
            restored._dependency_discovery,
            XiDependencyDiscovery
        )
        self.assertEqual(
            restored._dependency_discovery.alpha,
            0.05
        )


if __name__ == '__main__':
    unittest.main()
