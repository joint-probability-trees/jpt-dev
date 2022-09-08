from unittest import TestCase

from jpt.base.utils import mapstr, chop


class UtilsTest(TestCase):

    def test_mapstr(self):
        l = list(range(10))
        self.assertEqual(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], mapstr(l))
        self.assertEqual(['0', '1', '...', '8', '9'], mapstr(l, limit=4))
        self.assertEqual(['0', '9'], mapstr([0, 9], limit=4))
        self.assertEqual(['0', '...'], mapstr([0, 9], limit=1))

    def test_chop(self):
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
