from unittest import TestCase

import numpy as np

from jpt.base.constants import eps
from jpt.base.utils import mapstr, setstr_int, Heap, list2intset, list2set
from jpt.distributions import IntegerType


class UtilsTest(TestCase):

    def test_mapstr(self):
        l = list(range(10))
        self.assertEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], mapstr(l))
        self.assertEqual(['0', '1', '...', '8', '9'], mapstr(l, limit=4))
        self.assertEqual(['0', '9'], mapstr([0, 9], limit=4))
        self.assertEqual(['0', '...'], mapstr([0, 9], limit=1))
        self.assertEqual(['6', '7'], mapstr(['6', '7'], limit=2))

    def test_setstr_int(self):
        # Arrange
        set_ = {0, 1, 2, 4, 6, 7, 9, 10, 11}

        # Act
        str_1 = setstr_int(set_)
        str_2 = setstr_int(set_, sep_inner=', ')

        # Assert
        self.assertEqual('0...2, 4, 6, 7, 9...11', str_1)
        self.assertEqual('0, ..., 2, 4, 6, 7, 9, ..., 11', str_2)

    def test_epsilon(self):
        # Arrange
        x = np.pi

        # Act
        x_plus_eps = x + eps
        x_minus_eps = x - eps

        # Assert
        self.assertEqual(np.nextafter(x, x + 1), x_plus_eps)
        self.assertEqual(np.nextafter(x, x - 1), x_minus_eps)


class VersionTest(TestCase):

    def test_version(self):
        import jpt
        self.assertRegex(jpt.__version__, r'\d\.\d\.\d')


class HeapTest(TestCase):

    def test_iterator(self):
        # Arrange
        h = Heap(data=[5, 4, 8])

        # Act
        result = list(iter(h))

        # Assert
        self.assertEqual([4, 5, 8], result)

    def test_reverse(self):
        # Arrange
        h = Heap(data=[5, 4, 8])

        # Act
        result = list(reversed(h))

        # Assert
        self.assertEqual([8, 5, 4], result)


class ListConversionTest(TestCase):

    def test_list2intset(self):
        # Act
        s = list2intset([2, 4])

        # Assert
        self.assertEqual({2, 3, 4}, s)
        # self.assertRaises(ValueError, list2set, [7, 8])
        # self.assertRaises(ValueError, list2set, [1])
