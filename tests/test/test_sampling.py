"""
Test cases for sampling algorithms.
"""
import random
import unittest
from unittest import TestCase

import numpy as np

from jpt.base.sampling import RouletteWheelSampler


# ------------------------------------------------------------------------------


class TestRouletteWheelSampler(TestCase):
    """
    Test cases for RouletteWheelSampler functionality.
    """

    def test_constructor_valid_inputs(self):
        """
        Test constructor with valid inputs.
        """
        # Arrange
        elements = ['a', 'b', 'c']
        weights = [0.2, 0.3, 0.5]

        # Act
        sampler = RouletteWheelSampler(elements, weights)

        # Assert
        self.assertEqual(elements, sampler._elements)
        np.testing.assert_array_almost_equal([0.2, 0.5, 1.0], sampler._upperbounds)

    def test_constructor_with_normalization(self):
        """
        Test constructor with weight normalization.
        """
        # Arrange
        elements = ['x', 'y', 'z']
        weights = [2, 3, 5]  # Sum = 10

        # Act
        sampler = RouletteWheelSampler(elements, weights, normalize=True)

        # Assert
        expected_bounds = [0.2, 0.5, 1.0]  # Normalized cumulative weights
        np.testing.assert_array_almost_equal(expected_bounds, sampler._upperbounds)

    def test_constructor_mismatched_lengths(self):
        """
        Test constructor with mismatched element and weight lengths.
        """
        # Arrange
        elements = ['a', 'b']
        weights = [0.3, 0.4, 0.3]  # Different length

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            RouletteWheelSampler(elements, weights)
        
        self.assertIn("must have same lengths", str(context.exception))

    def test_index_valid_values(self):
        """
        Test index method with valid values.
        """
        # Arrange
        elements = ['first', 'second', 'third']
        weights = [0.1, 0.3, 0.6]  # Cumulative: [0.1, 0.4, 1.0]
        sampler = RouletteWheelSampler(elements, weights)

        # Act & Assert
        self.assertEqual(0, sampler.index(0.05))  # Falls in first bucket
        self.assertEqual(0, sampler.index(0.1))   # Edge case
        self.assertEqual(1, sampler.index(0.2))   # Falls in second bucket
        self.assertEqual(1, sampler.index(0.4))   # Edge case
        self.assertEqual(2, sampler.index(0.7))   # Falls in third bucket
        self.assertEqual(2, sampler.index(1.0))   # Edge case

    def test_index_out_of_bounds(self):
        """
        Test index method with out-of-bounds values.
        """
        # Arrange
        elements = ['a', 'b']
        weights = [0.4, 0.6]
        sampler = RouletteWheelSampler(elements, weights)

        # Act & Assert
        with self.assertRaises(IndexError):
            sampler.index(-0.1)  # Below range
        
        with self.assertRaises(IndexError):
            sampler.index(1.1)   # Above range

    def test_getitem_access(self):
        """
        Test element access via __getitem__.
        """
        # Arrange
        elements = ['alpha', 'beta', 'gamma']
        weights = [0.2, 0.3, 0.5]
        sampler = RouletteWheelSampler(elements, weights)

        # Act & Assert
        self.assertEqual('alpha', sampler[0.1])  # First element
        self.assertEqual('beta', sampler[0.3])   # Second element
        self.assertEqual('gamma', sampler[0.8])  # Third element

    def test_sample_single_element(self):
        """
        Test sampling a single element.
        """
        # Arrange
        elements = ['only']
        weights = [1.0]
        sampler = RouletteWheelSampler(elements, weights)

        # Act
        samples = sampler.sample(5)

        # Assert
        self.assertEqual(5, len(samples))
        self.assertTrue(all(s == 'only' for s in samples))

    def test_sample_multiple_elements(self):
        """
        Test sampling multiple elements.
        """
        # Arrange
        elements = ['a', 'b', 'c']
        weights = [0.33, 0.33, 0.34]
        sampler = RouletteWheelSampler(elements, weights)
        random.seed(42)  # For reproducible tests

        # Act
        samples = sampler.sample(100)

        # Assert
        self.assertEqual(100, len(samples))
        self.assertTrue(all(s in elements for s in samples))
        # Check that all elements appear at least once (with high probability)
        unique_samples = set(samples)
        self.assertGreater(len(unique_samples), 1)

    def test_sample_zero_elements(self):
        """
        Test sampling zero elements.
        """
        # Arrange
        elements = ['a', 'b']
        weights = [0.5, 0.5]
        sampler = RouletteWheelSampler(elements, weights)

        # Act
        samples = sampler.sample(0)

        # Assert
        self.assertEqual([], samples)

    def test_samplei_returns_indices(self):
        """
        Test samplei method returns elements with indices.
        """
        # Arrange
        elements = ['x', 'y', 'z']
        weights = [0.4, 0.3, 0.3]
        sampler = RouletteWheelSampler(elements, weights)
        random.seed(123)  # For reproducible tests

        # Act
        samples_with_indices = sampler.samplei(10)

        # Assert
        self.assertEqual(10, len(samples_with_indices))
        for element, index in samples_with_indices:
            self.assertIn(element, elements)
            self.assertEqual(element, elements[index])
            self.assertIn(index, [0, 1, 2])

    def test_weighted_distribution_bias(self):
        """
        Test that higher weights lead to higher selection probability.
        """
        # Arrange
        elements = ['rare', 'common']
        weights = [0.1, 0.9]  # 'common' should be selected much more often
        sampler = RouletteWheelSampler(elements, weights)
        random.seed(456)

        # Act
        samples = sampler.sample(1000)

        # Assert
        rare_count = samples.count('rare')
        common_count = samples.count('common')
        
        # 'common' should appear much more frequently than 'rare'
        self.assertGreater(common_count, rare_count)
        # With 1000 samples, we expect roughly 100 'rare' and 900 'common'
        # Allow some variance but ensure the bias is clear
        self.assertLess(rare_count, 200)  # Should be much less than 200
        self.assertGreater(common_count, 800)  # Should be much more than 800


if __name__ == '__main__':
    unittest.main()