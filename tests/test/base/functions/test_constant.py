from unittest import TestCase

import numpy as np
from ddt import ddt, data, unpack

from jpt.base.intervals import ContinuousSet, EMPTY, R
from jpt.base.functions import (
    LinearFunction,
    ConstantFunction,
    Undefined
)


# ----------------------------------------------------------------------

@ddt
class ConstantFunctionTest(TestCase):

    @data(
        (ConstantFunction(1), -1, 1),
        (ConstantFunction(1), 0, 1),
        (ConstantFunction(1), 1, 1),
        (ConstantFunction(-1), -1, -1),
        (ConstantFunction(-1), 0, -1),
        (ConstantFunction(-1), 1, -1),
        (ConstantFunction(0), -1, 0),
        (ConstantFunction(0), 0, 0),
        (ConstantFunction(0), 1, 0),
        (ConstantFunction(0), -np.inf, 0),
        (ConstantFunction(0), np.inf, 0),
        (ConstantFunction(.5), -np.inf, .5),
        (ConstantFunction(.5), np.inf, .5)
    )
    @unpack
    def test_eval(self, f, x, y):
        self.assertEqual(y, f.eval(x))

    @data(
        (ConstantFunction(1), LinearFunction(-1, 0),
         ContinuousSet(-1, -1)),
        (ConstantFunction(-1), ConstantFunction(0), EMPTY),
        (ConstantFunction(0), ConstantFunction(0), R),
        (LinearFunction(1, 1), ConstantFunction(0),
         ContinuousSet(-1, -1))
    )
    @unpack
    def test_intersection(self, f1, f2, v):
        self.assertEqual(v, f1.intersection(f2))

    @data(
        (ConstantFunction(1), 1, ConstantFunction(2)),
        (ConstantFunction(-1), ConstantFunction(1),
         ConstantFunction(0)),
        (ConstantFunction(1), LinearFunction(2, 3),
         LinearFunction(2, 4))
    )
    @unpack
    def test_add(self, f, arg, res):
        self.assertEqual(res, f + arg)

    @data(
        (ConstantFunction(2), 2, ConstantFunction(4)),
        (ConstantFunction(2), ConstantFunction(3),
         ConstantFunction(6)),
        (ConstantFunction(2), LinearFunction(3, 4),
         LinearFunction(6, 8)),
        (ConstantFunction(0), LinearFunction(1, 1),
         ConstantFunction(0))
    )
    @unpack
    def test_mul(self, f, arg, res):
        self.assertEqual(res, f * arg)

    @data(
        (ConstantFunction(0), (0, 1), 0),
        (ConstantFunction(1), (0, 1), 1),
        (ConstantFunction(1), (0, 2), 2),
        (ConstantFunction(-1), (-1, 1), -2),
        (ConstantFunction(0), (-np.inf, np.inf), 0)
    )
    @unpack
    def test_integrate(self, f, x, i):
        self.assertEqual(i, f.integrate(x[0], x[1]))

    @data(
        (ConstantFunction(0), ConstantFunction(0)),
        (ConstantFunction(np.nan), ConstantFunction(np.nan)),
        (ConstantFunction(1), LinearFunction(0, 1)),
        (ConstantFunction(np.nan), LinearFunction(0, np.nan)),
        (ConstantFunction(np.nan), Undefined())
    )
    @unpack
    def test_equal(self, f1, f2):
        self.assertEqual(
            f1,
            f2
        )
        self.assertEqual(
            f2,
            f1
        )

    @data(
        (ConstantFunction(0), ConstantFunction(1)),
        (ConstantFunction(np.nan), ConstantFunction(1)),
        (ConstantFunction(1), LinearFunction(.01, 1)),
        (ConstantFunction(1), LinearFunction(0, 0)),
        (ConstantFunction(1), LinearFunction(0, np.nan)),
        (ConstantFunction(1), LinearFunction(np.nan, 1)),
        (ConstantFunction(1), Undefined())
    )
    @unpack
    def test_unequal(self, f1, f2):
        self.assertNotEqual(
            f1,
            f2
        )
        self.assertNotEqual(
            f2,
            f1
        )
