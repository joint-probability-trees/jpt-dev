from unittest import TestCase

import numpy as np
from ddt import ddt, data, unpack

try:
    from jpt.base.functions import __module__
    from jpt.base.intervals import __module__, EMPTY, R
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import (LinearFunction, QuadraticFunction, ConstantFunction, Undefined, Function,
                                    PiecewiseFunction)
    from jpt.base.intervals import ContinuousSet


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class UndefinedFunctionTest(TestCase):

    @data((1,), (-1,), (100,))
    @unpack
    def test_eval(self, x):
        f = Undefined()
        self.assertEqual(np.nan, f.eval(x))


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class ConstantFunctionTest(TestCase):

    @data((ConstantFunction(1), -1, 1),
          (ConstantFunction(1), 0, 1),
          (ConstantFunction(1), 1, 1),
          (ConstantFunction(-1), -1, -1),
          (ConstantFunction(-1), 0, -1),
          (ConstantFunction(-1), 1, -1),
          (ConstantFunction(0), -1, 0),
          (ConstantFunction(0), 0, 0),
          (ConstantFunction(0), 1, 0))
    @unpack
    def test_eval(self, f, x, y):
        self.assertEqual(y, f.eval(x))

    @data((ConstantFunction(1), LinearFunction(-1, 0), ContinuousSet(-1, -1)),
          (ConstantFunction(-1), ConstantFunction(0), EMPTY),
          (ConstantFunction(0), ConstantFunction(0), R),
          (LinearFunction(1, 1), ConstantFunction(0), ContinuousSet(-1, -1)))
    @unpack
    def test_intersection(self, f1, f2, v):
        self.assertEqual(v, f1.intersection(f2))


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class LinearFunctionTest(TestCase):

    @data(('3', ConstantFunction(3)),
          ('undef.', Undefined(0)),
          ('x + 1', LinearFunction(1, 1)))
    @unpack
    def test_parsing(self, s, result):
        self.assertEqual(LinearFunction.parse(s), result)

    @data(((0, 0), (1, 1), LinearFunction(1, 0)),
          ((0, 0), (1, 0), ConstantFunction(0)),
          ((0, 1), (1, 2), LinearFunction(1, 1)))
    @unpack
    def test_fit(self, p1, p2, truth):
        self.assertEqual(LinearFunction.from_points(p1, p2), truth)

    @data((LinearFunction(1, 0), 1, 1),
          (LinearFunction(1, 1), 1, 2))
    @unpack
    def test_eval(self, f, x, y):
        self.assertEqual(y, f.eval(x))

    @data(((0, 0), (0, 0)), ((1, 1), (1, 1)))
    @unpack
    def test_fit_integrity_check(self, p1, p2):
        self.assertRaises(ValueError, LinearFunction.from_points, p1, p2)

    def test_equality(self):
        f1 = LinearFunction(1, 1)
        f2 = ConstantFunction(1)
        f3 = LinearFunction(0, 1)

        self.assertTrue(f1 == LinearFunction(1, 1))
        self.assertTrue(f2 == ConstantFunction(1))
        self.assertTrue(f3 == LinearFunction(0, 1))
        self.assertEqual(f2, f3)

    def test_serialization(self):
        f1 = LinearFunction(1, 1)
        f2 = ConstantFunction(1)
        f3 = LinearFunction(0, 1)
        self.assertEqual(f1, LinearFunction.from_json(f1.to_json()))
        self.assertEqual(f2, LinearFunction.from_json(f2.to_json()))
        self.assertEqual(f3, LinearFunction.from_json(f3.to_json()))

    @data((LinearFunction(1, 0), LinearFunction(-1, 0), ContinuousSet(0, 0)),
          (LinearFunction(1, 0), LinearFunction(1, 1), EMPTY),
          (LinearFunction(-1, 1), LinearFunction(-1, 1), R),
          (LinearFunction(1, 1), ConstantFunction(0), ContinuousSet(-1, -1)))
    @unpack
    def test_intersection(self, f1, f2, v):
        self.assertEqual(v, f1.intersection(f2))

    @data((LinearFunction(1, 1), LinearFunction(1, 0), LinearFunction(2, 1)),
          (LinearFunction(1, 1), ConstantFunction(1), LinearFunction(1, 2)),
          (LinearFunction(-1, 1), LinearFunction(1, 1), ConstantFunction(2)),
          (LinearFunction(-1, 1), 3, LinearFunction(-1, 4)),
          )
    @unpack
    def test_addition(self, f1, f2, v):
        self.assertEqual(v, f1 + f2)

    @data((LinearFunction(1, 2), 3, LinearFunction(3, 6)),
          (LinearFunction(1, 2), ConstantFunction(3), LinearFunction(3, 6)),
          (LinearFunction(2, 2), LinearFunction(3, 3), QuadraticFunction(6, 12, 6)),
          (LinearFunction(-2, 2), LinearFunction(3, -3), QuadraticFunction(-6, 12, -6)),
          )
    @unpack
    def test_multiplication(self, f1, f2, v):
        self.assertEqual(v, f1 * f2)


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class QuadraticFunctionTest(TestCase):

    @data((QuadraticFunction(1, 1, 1), 1, 3),
          (QuadraticFunction(1, 1, 1), 2, 7))
    @unpack
    def test_eval(self, f, x, y):
        self.assertEqual(y, f.eval(x))

    @data(((0, 0), (1, 1), (2, 4), QuadraticFunction(1, 0, 0)))
    @unpack
    def test_fit(self, p1: tuple, p2: tuple, p3: tuple, truth: QuadraticFunction):
        self.assertEqual(QuadraticFunction.from_points(p1, p2, p3), truth)

    @data(((0, 0), (0, 0), (1, 2)),
          ((1, 2), (0, 0), (1, 2)),
          ((1, 2), (0, 0), (0, 0)),)
    @unpack
    def test_fit_integrity_check(self, p1: tuple, p2: tuple, p3: tuple):
        self.assertRaises(ValueError, QuadraticFunction.from_points, p1, p2, p3)

    def test_serialization(self):
        f = QuadraticFunction(1, 2, 3)
        self.assertEqual(f, QuadraticFunction.from_json(f.to_json()))

    @data((QuadraticFunction(1, 0, 0), 0),
          (QuadraticFunction(1, 0, 1), 0),
          (QuadraticFunction(1, 0, 0), 0),
          (QuadraticFunction(-2, 8, -5), 2),
          (QuadraticFunction(3, 6, 7), -1))
    @unpack
    def test_argvertex(self, f: QuadraticFunction, xmax: float):
        self.assertEqual(xmax, f.argvertex())

    @data((QuadraticFunction(1, 2, 3), QuadraticFunction(1, 2, 3)),
          (QuadraticFunction(0, 0, 1), ConstantFunction(1)),
          (QuadraticFunction(0, 1, 0), LinearFunction(1, 0)),
          (QuadraticFunction(0, 1, 2), LinearFunction(1, 2)),
          (QuadraticFunction(3, 0, 0), QuadraticFunction(3, 0, 0)))
    @unpack
    def test_simplify(self, f1: QuadraticFunction, f2: Function):
        self.assertEqual(f1.simplify(), f2)


# ----------------------------------------------------------------------------------------------------------------------

class PLFTest(TestCase):

    def test_plf_constant_from_dict(self):
        d = {
            ']-∞,1.000[': '0.0',
            '[1.000,2.000[': '0.25',
            '[3.000,4.000[': '0.5',
            '[4.000,5.000[': '0.75',
            '[5.000,∞[': '1.0'
        }

        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-inf,1['))
        cdf.intervals.append(ContinuousSet.parse('[1, 2['))
        cdf.intervals.append(ContinuousSet.parse('[3, 4['))
        cdf.intervals.append(ContinuousSet.parse('[4, 5['))
        cdf.intervals.append(ContinuousSet.parse('[5, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(ConstantFunction(.25))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(ConstantFunction(.75))
        cdf.functions.append(ConstantFunction(1))

        self.assertEqual(cdf, PiecewiseFunction.from_dict(d))

    def test_plf_linear_from_dict(self):
        d = {
            ']-∞,0.000[': 'undef.',
            '[0.000,0.500[': '2.000x',
            '[0.500,1.000[': '2.000x + 1.000',
            '[1.000,∞[': None
        }
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-∞,0.000['))
        cdf.intervals.append(ContinuousSet.parse('[0.000,0.500['))
        cdf.intervals.append(ContinuousSet.parse('[0.500,1.000['))
        cdf.intervals.append(ContinuousSet.parse('[1.000,∞['))
        cdf.functions.append(Undefined())
        cdf.functions.append(LinearFunction(2, 0))
        cdf.functions.append(LinearFunction(2, 1))
        cdf.functions.append(Undefined())

        self.assertEqual(cdf, PiecewiseFunction.from_dict(d))

    def test_plf_mixed_from_dict(self):
        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet.parse(']-inf,0.000['))
        cdf.intervals.append(ContinuousSet.parse('[0.000, 1['))
        cdf.intervals.append(ContinuousSet.parse('[1, 2['))
        cdf.intervals.append(ContinuousSet.parse('[2, 3['))
        cdf.intervals.append(ContinuousSet.parse('[3, inf['))
        cdf.functions.append(ConstantFunction(0))
        cdf.functions.append(LinearFunction.from_points((0, 0), (1, .5)))
        cdf.functions.append(ConstantFunction(.5))
        cdf.functions.append(LinearFunction.from_points((2, .5), (3, 1)))
        cdf.functions.append(ConstantFunction(1))

        self.assertEqual(cdf, PiecewiseFunction.from_dict({
            ']-∞,0.000[': 0,
            '[0.000,1.00[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2.000[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3.000,∞[': 1
        }))

    def test_serialization(self):
        plf = PiecewiseFunction.from_dict({
            ']-∞,0.000[': 0,
            '[0.000,1.00[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2.000[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3.000,∞[': 1
        })
        self.assertEqual(plf, PiecewiseFunction.from_json(plf.to_json()))

