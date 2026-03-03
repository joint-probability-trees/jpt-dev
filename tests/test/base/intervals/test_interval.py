"""
General tests for interval functionality that spans multiple classes.
"""
from unittest import TestCase
import numpy as np

from jpt.base.intervals import (
    ContinuousSet,
    INC,
    EXC,
    UnionSet,
    IntSet,
    Interval
)


class IntervalParsingTest(TestCase):
    """
    Test cases for general interval parsing functionality.
    """

    def test_parse_edge_cases(self):
        """
        Test Interval.parse() for various edge case strings.
        """
        # Arrange & Act & Assert
        test_cases = [
            ("∅", True),        # Empty set
            ("ℤ", False),       # Integers
            ("ℝ", False),       # Reals
            ("{1..5}", False),  # Integer set
            ("[0,1]", False),   # Continuous set
        ]
        
        for string, should_be_empty in test_cases:
            with self.subTest(string=string):
                interval = Interval.parse(string)
                self.assertEqual(interval.isempty(), should_be_empty)


class IntervalOperatorTest(TestCase):
    """
    Test interval operator overloading.
    """

    def test_intersection_operator(self):
        # Arrange
        i1 = ContinuousSet(0, 5)
        i2 = ContinuousSet(3, 8)
        
        # Act
        result = i1 & i2
        
        # Assert
        self.assertEqual(result.lower, 3)
        self.assertEqual(result.upper, 5)

    def test_union_operator(self):
        # Arrange
        i1 = ContinuousSet(0, 5)
        i2 = ContinuousSet(3, 8)
        
        # Act
        result = i1 | i2
        
        # Assert
        self.assertEqual(result.lower, 0)
        self.assertEqual(result.upper, 8)

    def test_difference_operator(self):
        # Arrange
        i1 = ContinuousSet(0, 5)
        i2 = ContinuousSet(3, 8)
        
        # Act
        result = i1 - i2
        
        # Assert
        self.assertEqual(result.lower, 0)
        self.assertEqual(result.upper, 3)


class UtilsTest(TestCase):
    """
    Test utility functions used by intervals.
    """

    def test_chop(self):
        from jpt.base.utils import chop
        
        truth = [
            (0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (1, [2, 3, 4, 5, 6, 7, 8, 9]),
            (2, [3, 4, 5, 6, 7, 8, 9]),
            (3, [4, 5, 6, 7, 8, 9]),
            (4, [5, 6, 7, 8, 9]),
            (5, [6, 7, 8, 9]),
            (6, [7, 8, 9]),
            (7, [8, 9]),
            (8, [9]),
            (9, [])]
        result = []
        for h, t in chop(list(range(10))):
            result.append((h, list(t)))
        self.assertEqual(truth, result)
        self.assertEqual([], list(chop([])))


if __name__ == '__main__':
    import unittest
    unittest.main()