from unittest import TestCase

import numpy as np
from ddt import ddt, data, unpack

from jpt.base.intervals import ContinuousSet, EMPTY, R
from jpt.base.functions import (
    LinearFunction,
    QuadraticFunction,
    ConstantFunction,
    Undefined
)


# ----------------------------------------------------------------------

@ddt
class LinearFunctionTest(TestCase):

    @data(
        ('3', ConstantFunction(3)),
        ('undef.', Undefined()),
        ('x + 1', LinearFunction(1, 1))
    )
    @unpack
    def test_parsing(self, s, result):
        self.assertEqual(LinearFunction.parse(s), result)

    @data(
        ((0, 0), (1, 1), LinearFunction(1, 0)),
        ((0, 0), (1, 0), ConstantFunction(0)),
        ((0, 1), (1, 2), LinearFunction(1, 1))
    )
    @unpack
    def test_fit(self, p1, p2, truth):
        self.assertEqual(
            LinearFunction.from_points(p1, p2), truth
        )

    @data(
        (LinearFunction(1, 0), 1, 1),
        (LinearFunction(1, 1), 1, 2)
    )
    @unpack
    def test_eval(self, f, x, y):
        self.assertEqual(y, f.eval(x))

    def test_multieval(self):
        # Arrange
        x = np.array([1, 2, 3], dtype=np.float64)
        result_buffer = np.array(x)
        f = LinearFunction(1, 1)

        # Act
        result = f.multi_eval(x)
        result_buffer_ = f.multi_eval(
            x, result=result_buffer
        )

        # Assert
        self.assertEqual([2, 3, 4], list(result))
        self.assertEqual(
            list(result), list(result_buffer)
        )
        self.assertEqual(
            list(result_buffer_), list(result_buffer)
        )

    @data(((0, 0), (0, 0)), ((1, 1), (1, 1)))
    @unpack
    def test_fit_integrity_check(self, p1, p2):
        self.assertRaises(
            ValueError,
            LinearFunction.from_points, p1, p2
        )

    @data(
        (LinearFunction(0, 1), ConstantFunction(1)),
        (LinearFunction(0, 1), LinearFunction(0, 1)),
        (LinearFunction(np.nan, np.nan),
         ConstantFunction(np.nan)),
        (LinearFunction(np.nan, 0),
         ConstantFunction(np.nan)),
        (LinearFunction(0, np.nan),
         ConstantFunction(np.nan)),
    )
    @unpack
    def test_equality(self, f1, f2):
        self.assertEqual(f1, f2)
        self.assertEqual(f2, f1)
        self.assertEqual(f1, f1)
        self.assertEqual(f2, f2)

    @data(
        (LinearFunction(1, 1), ConstantFunction(1)),
        (LinearFunction(1, 1), LinearFunction(1, 0)),
        (LinearFunction(1, 1), LinearFunction(0, 1)),
        (LinearFunction(1, 1), LinearFunction(2, 2)),
        (LinearFunction(np.nan, 1), LinearFunction(2, 2)),
        (LinearFunction(np.nan, np.nan),
         LinearFunction(2, 2)),
    )
    @unpack
    def test_inequality(self, f1, f2):
        self.assertNotEqual(
            f1,
            f2
        )
        self.assertEqual(f1, f1)
        self.assertEqual(f2, f2)


    def test_serialization(self):
        f1 = LinearFunction(1, 1)
        f2 = ConstantFunction(1)
        f3 = LinearFunction(0, 1)
        self.assertEqual(
            f1, LinearFunction.from_json(f1.to_json())
        )
        self.assertEqual(
            f2, LinearFunction.from_json(f2.to_json())
        )
        self.assertEqual(
            f3, LinearFunction.from_json(f3.to_json())
        )

    @data(
        (LinearFunction(1, 0), LinearFunction(-1, 0),
         ContinuousSet(0, 0)),
        (LinearFunction(1, 0), LinearFunction(1, 1), EMPTY),
        (LinearFunction(-1, 1), LinearFunction(-1, 1), R),
        (LinearFunction(1, 1), ConstantFunction(0),
         ContinuousSet(-1, -1))
    )
    @unpack
    def test_intersection(self, f1, f2, v):
        self.assertEqual(v, f1.intersection(f2))

    @data(
        (LinearFunction(1, 1), LinearFunction(1, 0),
         LinearFunction(2, 1)),
        (LinearFunction(1, 1), ConstantFunction(1),
         LinearFunction(1, 2)),
        (LinearFunction(-1, 1), LinearFunction(1, 1),
         ConstantFunction(2)),
        (LinearFunction(-1, 1), 3, LinearFunction(-1, 4)),
    )
    @unpack
    def test_addition(self, f1, f2, v):
        # Act
        sum_ = f1 + f2

        # Assert
        self.assertEqual(
            v,
            sum_
        )

    @data(
        (LinearFunction(1, 2), 3, LinearFunction(3, 6)),
        (LinearFunction(1, 2), ConstantFunction(3),
         LinearFunction(3, 6)),
        (LinearFunction(2, 2), LinearFunction(3, 3),
         QuadraticFunction(6, 12, 6)),
        (LinearFunction(-2, 2), LinearFunction(3, -3),
         QuadraticFunction(-6, 12, -6)),
    )
    @unpack
    def test_multiplication(self, f1, f2, v):
        self.assertEqual(v, f1 * f2)

    @data(
        (LinearFunction(1, 0), (0, 1), .5),
        (LinearFunction(1, 1), (0, 1), 1.5),
        (LinearFunction(1, -1), (0, 2), 0),
        (LinearFunction(1, -1), (0, 1), -.5),
        (LinearFunction(1, 0), (0, np.inf), np.inf),
        (LinearFunction(-1, 0), (0, np.inf), -np.inf),
        (LinearFunction(-1, 0), (-np.inf, np.inf), np.nan)
    )
    @unpack
    def test_integration(self, f, x, i):
        # Act
        result = f.integrate(x[0], x[1])
        if np.isnan(i):
            self.assertTrue(np.isnan(result))
        else:
            self.assertEqual(i, result)

    def test_xshift(self):
        # Arrange
        f = LinearFunction(2, -1)

        # Act
        f_ = f.xshift(5)  # shift f to the left by 5

        # Assert
        self.assertEqual(LinearFunction(2, 9), f_)
