from unittest import TestCase

import numpy as np
from ddt import ddt, data, unpack
from dnutils.tools import ifstr

from jpt.base.constants import eps
from utils import gaussian_numeric

try:
    from jpt.base.functions import __module__
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import (
        LinearFunction,
        QuadraticFunction,
        ConstantFunction,
        Undefined,
        Function,
        PiecewiseFunction,
        PLFApproximator
    )
    from jpt.base.intervals import ContinuousSet, EMPTY, R, EXC, INC, RealSet


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class UndefinedFunctionTest(TestCase):

    @data((1,), (-1,), (100,))
    @unpack
    def test_eval(self, x):
        f = Undefined()
        self.assertTrue(np.isnan(f.eval(x)))


# ----------------------------------------------------------------------------------------------------------------------

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
        (ConstantFunction(0), np.NINF, 0),
        (ConstantFunction(0), np.PINF, 0),
        (ConstantFunction(.5), np.NINF, .5),
        (ConstantFunction(.5), np.PINF, .5)
    )
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

    @data((ConstantFunction(1), 1, ConstantFunction(2)),
          (ConstantFunction(-1), ConstantFunction(1), ConstantFunction(0)),
          (ConstantFunction(1), LinearFunction(2, 3), LinearFunction(2, 4)))
    @unpack
    def test_add(self, f, arg, res):
        self.assertEqual(res, f + arg)

    @data((ConstantFunction(2), 2, ConstantFunction(4)),
          (ConstantFunction(2), ConstantFunction(3), ConstantFunction(6)),
          (ConstantFunction(2), LinearFunction(3, 4), LinearFunction(6, 8)),
          (ConstantFunction(0), LinearFunction(1, 1), ConstantFunction(0)))
    @unpack
    def test_mul(self, f, arg, res):
        self.assertEqual(res, f * arg)

    @data(
        (ConstantFunction(0), (0, 1), 0),
        (ConstantFunction(1), (0, 1), 1),
        (ConstantFunction(1), (0, 2), 2),
        (ConstantFunction(-1), (-1, 1), -2),
        (ConstantFunction(0), (np.NINF, np.PINF), 0)
    )
    @unpack
    def test_integrate(self, f, x, i):
        self.assertEqual(i, f.integrate(x[0], x[1]))


# ----------------------------------------------------------------------------------------------------------------------

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
        self.assertEqual(LinearFunction.from_points(p1, p2), truth)

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
        result_buffer_ = f.multi_eval(x, result=result_buffer)

        # Assert
        self.assertEqual([2, 3, 4], list(result))
        self.assertEqual(list(result), list(result_buffer))
        self.assertEqual(list(result_buffer_), list(result_buffer))

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

    @data(
        (LinearFunction(1, 0), (0, 1), .5),
        (LinearFunction(1, 1), (0, 1), 1.5),
        (LinearFunction(1, -1), (0, 2), 0),
        (LinearFunction(1, -1), (0, 1), -.5),
        (LinearFunction(1, 0), (0, np.PINF), np.PINF),
        (LinearFunction(-1, 0), (0, np.PINF), np.NINF),
        (LinearFunction(-1, 0), (np.NINF, np.PINF), np.nan)
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

    @data((QuadraticFunction(2, 3, 4), 2, QuadraticFunction(4, 6, 8)),
          (QuadraticFunction(2, 3, 4), ConstantFunction(-2), QuadraticFunction(-4, -6, -8)),
          (QuadraticFunction(2, 3, 4), 0, ConstantFunction(0)))
    @unpack
    def test_mul(self, f, a, r):
        self.assertEqual(r, f * a)

    @data((QuadraticFunction(2, 3, 4), 2, QuadraticFunction(2, 3, 6)),
          (QuadraticFunction(2, 3, 4), ConstantFunction(-2), QuadraticFunction(2, 3, 2)),
          (QuadraticFunction(2, 3, 4), 0, QuadraticFunction(2, 3, 4)),
          (QuadraticFunction(2, 3, 4), LinearFunction(2, 3), QuadraticFunction(2, 5, 7)))
    @unpack
    def test_add(self, f, a, r):
        self.assertEqual(r, f + a)

    def test_roots_2_solutions(self):
        # Arrange
        f = QuadraticFunction(2, -8, 6)

        # Act
        roots = f.roots()

        # Assert
        self.assertEqual(
            [1, 3],
            list(roots)
        )

    def test_roots_1_solution(self):
        # Arrange
        f = QuadraticFunction(2, -8, 8)

        # Act
        roots = f.roots()

        # Assert
        self.assertEqual(
            [2],
            list(roots)
        )

    def test_roots_no_solution(self):
        # Arrange
        f = QuadraticFunction(2, -8, 11)

        # Act
        roots = f.roots()

        # Assert
        self.assertEqual(
            [],
            list(roots)
        )

# ----------------------------------------------------------------------------------------------------------------------

@ddt
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

    def test_mul_constant(self):
        plf = PiecewiseFunction.from_dict({
            ']-∞,0.000[': 0,
            '[0.000,1.00[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2.000[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3.000,∞[': 1
        })
        plf_res = PiecewiseFunction.from_dict({
            ']-∞,0.000[': 0,
            '[0.000,1.00[': LinearFunction.from_points((0, 0), (1, .25)),
            '[1.,2.000[': .25,
            '[2,3[': LinearFunction.from_points((2, .25), (3, .5)),
            '[3.000,∞[': .5
        })
        self.assertEqual(plf_res, plf * .5)

    def test_add_const(self):
        plf1 = PiecewiseFunction.from_dict({
            ']-∞,0[': 0,
            '[0,1[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3,∞[': 1
        })
        f = ConstantFunction(.5)
        res = PiecewiseFunction.from_dict({
            ']-∞,0[': 0.5,
            '[0,1[': '0.5x + .5',
            '[1.,2[': '1.',
            '[2,3[': '.5x',
            '[3,∞[': 1.5
        })
        self.assertEqual(res, plf1 + f)

    def test_add_linear(self):
        plf1 = PiecewiseFunction.from_dict({
            ']-∞,0[': 0,
            '[0,1[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3,∞[': 1
        })
        f = LinearFunction(2.5, 3.5)
        res = PiecewiseFunction.from_dict({
            ']-∞,0[': '2.5x + 3.5',
            '[0,1[': '3.0x + 3.5',
            '[1,2[': '2.5x + 4',
            '[2,3[': '3x + 3',
            '[3,∞[': '2.5x + 4.5'
        })
        self.assertEqual(res, plf1 + f)

    def test_add_plf(self):
        plf1 = PiecewiseFunction.from_dict({
            ']-∞,0[': 0,
            '[0,1[': str(LinearFunction.from_points((0, 0), (1, .5))),
            '[1.,2[': '.5',
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3,∞[': 1
        })
        plf2 = PiecewiseFunction.from_dict({
            ']-∞,-1[': 0,
            '[-1,3[': LinearFunction(.5, 4),
            '[3,∞[': 1
        })
        res = PiecewiseFunction.from_dict({
            ']-∞,-1.0[': 0,
            '[-1.0,0.0[': '0.5x + 4.0',
            '[0,1[': '1.0x + 4.0',
            '[1.,2[': '0.5x + 4.5',
            '[2,3[': '1.0x + 3.5',
            '[3,∞[': 2
        })
        self.assertEqual(res, plf1 + plf2)

    def test_mul_const_const(self):
        # Arrange
        plf = PiecewiseFunction.zero().overwrite_at(
            '[0,1)', ConstantFunction(1)
        )
        # Act
        product = plf.mul(ConstantFunction(2))
        # Assert
        self.assertEqual(
            PiecewiseFunction.zero().overwrite_at(
                '[0,1)', ConstantFunction(2)
            ),
            product
        )

    def test_mul_const_linear(self):
        # Arrange
        plf = PiecewiseFunction.zero().overwrite_at(
            '[-1,1)', ConstantFunction(1)
        )
        # Act
        product = plf.mul(LinearFunction(1, 0))
        # Assert
        self.assertEqual(
            PiecewiseFunction.zero().overwrite_at(
                '[-1,1)', LinearFunction(1, 0)
            ),
            product
        )

    def test_mul_plf_constant(self):
        # Arrange
        plf1 = PiecewiseFunction.from_dict({
            '[-2,-1)': 1,
            '[1,inf)': 2
        })
        plf2 = PiecewiseFunction.from_dict({
            '[-1.5,1.5)': -.5
        })
        # Act
        product = plf1 * plf2
        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[-2,-1.5)': Undefined(),
                '[-1.5,-1)': -.5,
                '[-1,1)': Undefined(),
                '[1,1.5)': -1,
                '[1.5,inf)': Undefined()
            }),
            product
        )

    def test_mul_plf_mixed(self):
        plf1 = PiecewiseFunction.from_dict({
            ']-∞,0[': 0,
            '[0,1[': LinearFunction.from_points((0, 0), (1, .5)),
            '[1,2[': .5,
            '[2,3[': LinearFunction.from_points((2, .5), (3, 1)),
            '[3,∞[': 1
        })
        plf2 = PiecewiseFunction.from_dict({
            ']-∞,-1[': 0,
            '[-1,3[': LinearFunction(.5, 4),
            '[3,∞[': 1
        })
        res = PiecewiseFunction.from_dict({
            ']-∞,0[': 0,
            '[0,1[': QuadraticFunction(.25, 2, 0),
            '[1.,2[': LinearFunction(.25, 2),
            '[2,3[': QuadraticFunction(.25, 1.75, -2),
            '[3,∞[': 1
        })
        self.assertEqual(res, plf1 * plf2)

    def test_from_points(self):
        # Arrange
        points = [(0, 0), (1, 1), (2, -2), (3, 3)]

        # Act
        plf = PiecewiseFunction.from_points(points)

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[0,1[': 'x',
                '[1,2[': '-3x+4',
                ContinuousSet(2, 3 + eps, INC, EXC): '5x-12'
            }),
            plf
        )

    @data(
        (
            PiecewiseFunction.zero().overwrite({
                ContinuousSet(0, 0 + eps, INC, EXC): 1.23,
            }),
            1.23
        ),
        (
            PiecewiseFunction.zero(),
            False
        )

    )
    @unpack
    def test_is_impulse(self, plf, truth):
        # Act
        impulse = plf.is_impulse()
        self.assertEqual(truth, impulse)

    def test_min_max(self):
        # Arrange
        plf1 = PiecewiseFunction.from_dict({
                '[0,1[': 'x',
                '[1,2[': '-3x+4',
                '[2,4]': '5x-12'
            })
        plf2 = PiecewiseFunction.from_dict({
                '[0,5]': 'x',
            })

        # Act
        plf_min = PiecewiseFunction.min(plf1, plf2)
        plf_max = PiecewiseFunction.max(plf1, plf2)

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[0.0,1.0[': '1.0x',
                '[1.0,2.0[': '-3.0x + 4.0',
                '[2.0,3.0[': '5.0x - 12.0',
                '[3.0,4.0]': '1.0x'
            }),
            plf_min
        )
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[0.0,3.0[': '1.0x',
                '[3.0,4.0]': '5.0x - 12.0'
            }),
            plf_max
        )

    def test_integral(self):
        # Arrange
        plf1 = PiecewiseFunction.from_dict({
            '[0,1[': 'x',
            '[1,2[': '-3x+4',
            '[2,4]': '5x-12'
        })

        # Act
        area = plf1.integrate(ContinuousSet(.5, 1.5))

        # Assert
        self.assertAlmostEqual(.5, area, places=6)

    def test_jaccard(self):
        # Arrange
        plf1 = PiecewiseFunction.from_dict({
            '[0,1[': 'x',
            '[1,2[': '-3x+4',
            '[2,4]': '5x-12'
        })
        plf2 = PiecewiseFunction.from_dict({
            '[0,5]': 'x',
        })

        plf3 = PiecewiseFunction.from_points(
            [(0, 0), (.5, 1), (1, 0), (10, 0)]
        )

        plf4 = PiecewiseFunction.from_points(
            [(0, 0), (2, 0), (2.5, 1), (3, 0)]
        )

        # Act
        sim_normal = PiecewiseFunction.jaccard_similarity(plf1, plf2)
        sim_symmetric = PiecewiseFunction.jaccard_similarity(plf2, plf1)
        sim_reflexive = PiecewiseFunction.jaccard_similarity(plf1, plf1)
        sim_disjoint = PiecewiseFunction.jaccard_similarity(plf3, plf4)

        # Assert
        self.assertAlmostEqual(.4, sim_normal, places=7)
        self.assertAlmostEqual(.4, sim_symmetric, places=7)
        self.assertEqual(1, sim_reflexive)
        self.assertEqual(0, sim_disjoint)

    def test_overwrite_1(self):
        # Arrange
        result = PiecewiseFunction.from_dict({R: 0})

        # Act
        result = result.overwrite_at(
            ContinuousSet(0, 1),
            ConstantFunction(1)
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-inf,0[': 0,
                ContinuousSet(0, 1 + eps, INC, EXC): 1,
                ContinuousSet(1 + eps, np.inf, INC, EXC): 0
            }),
            result
        )

        # Act
        result = result.overwrite_at(
            ContinuousSet(.5, 1.5),
            ConstantFunction(2)
        )

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-inf,0[': 0,
                ContinuousSet(0, .5, INC, EXC): 1,
                ContinuousSet(.5, 1.5 + eps, INC, EXC): 2,
                ContinuousSet(1.5 + eps, np.inf, INC, EXC): 0
            }),
            result
        )

    def test_overwrite_2(self):
        # Arrange
        result = PiecewiseFunction.from_dict({
            ContinuousSet(-2, -1, INC, EXC): 1,
            ContinuousSet(1, 2, INC, EXC): 1
        })

        # Act
        result = result.overwrite_at(ContinuousSet(-.5, .5, INC, EXC), ConstantFunction(2))

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ContinuousSet(-2, -1, INC, EXC): 1,
                ContinuousSet(-.5, .5, INC, EXC): ConstantFunction(2),
                ContinuousSet(1, 2, INC, EXC): 1
            }),
            result
        )

    def test_xshift(self):
        # Arrange
        plf = PiecewiseFunction.from_dict({
             R: 0
        })
        plf2 = PiecewiseFunction.from_points([
                (-2, 0),
                (-1, 1),
                (1, 1),
                (2 - eps, 0)
        ])
        for i, f in plf2:
            plf = plf.overwrite_at(i, f)

        # Act
        result = plf.xshift(3)

        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,-5.0[': 0,
                '[-5.0,-4.0[': 'x+5',
                '[-4.0,-2.0[': 1,
                '[-2.0,-1.0[': LinearFunction(-1 - eps, - 1 - eps - eps - eps - eps),
                '[-1.0,∞[': 0,
            }),
            result
        )

    def test_xmirror_simple(self):
        plf = PiecewiseFunction.zero().overwrite_at(
            ContinuousSet(-1, 1 + eps, INC, EXC),
            ConstantFunction(1)
        )
        mirror = plf.xmirror()
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ']-∞,-1.0[': 0,
                ContinuousSet(-1, 1+eps, INC, EXC): 1,
                ContinuousSet(1+eps, np.PINF, INC,  EXC): 0
            }),
            mirror
        )

    def test_xmirror(self):
        # Arrange
        plf = PiecewiseFunction.from_points(
            [(1, 0), (2, 1), (3, 0)]
        )
        # Act
        mirror = plf.xmirror()
        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                ContinuousSet(-3, -2 + eps, INC, EXC): 'x+3',
                ContinuousSet(-2 + eps, -1 + eps, INC, EXC): '-1x-1',
            }),
            mirror
        )

    def test_xmirror_symmetry(self):
        '''xmirror() must maintain a function's symmetry at x=0'''
        plf = PiecewiseFunction.zero()
        for i, f in PiecewiseFunction.from_points(
            [(-1, 0), (0, 1), (1, 0)]
        ):
            plf = plf.overwrite_at(i, f)
        # Act
        mirror = plf.xmirror().round(64)
        self.assertEqual(plf, mirror)

    def test_boundaries(self):
        # Symmatric functions have identical boundaries
        plf = PiecewiseFunction.zero().overwrite_at(
            ContinuousSet(-1, 1, INC, EXC),
            ConstantFunction(1)
        )
        self.assertEqual([-1, 1], list(plf.boundaries()))

    def test_drop_undef(self):
        # Arrange
        plf = PiecewiseFunction.from_dict({
            '[-inf,0)': 0,
            '[0,1)': Undefined(),
            '[1,inf)': 0
        })
        # Act
        result = plf.drop_undef()
        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[-inf,0)': 0,
                '[1,inf)': 0
            }),
            result
        )

    @data(
        (
                PiecewiseFunction.from_dict({
                    R: 0
                }).overwrite_at(
                    ContinuousSet(-1, 1, INC, EXC), ConstantFunction(1)
                ),
                PiecewiseFunction.from_dict({
                    R: 0
                }).overwrite_at(
                    ContinuousSet(-2, 2, INC, EXC), ConstantFunction(.5)
                ),
                PiecewiseFunction.from_dict({
                    ContinuousSet(np.NINF, -3, EXC, EXC): 0,
                    ContinuousSet(-3, -1, INC, EXC): LinearFunction(.5, 1.5),
                    ContinuousSet(-1, 1, INC, EXC): 1,
                    ContinuousSet(1, 3, INC, EXC): LinearFunction(-.5, 1.5),
                    ContinuousSet(3, np.PINF, INC, EXC): 0,
                })
        ),
        (
                PiecewiseFunction.from_dict({
                    R: 0
                }).overwrite_at(
                    ContinuousSet(-1, 1, INC, EXC), ConstantFunction(.5)
                ),
                PiecewiseFunction.from_dict({
                    R: 0
                }).overwrite_at(
                    ContinuousSet(-1, 1, INC, EXC), ConstantFunction(.5)
                ),
                PiecewiseFunction.from_dict({
                    '(-∞,-2.0)': 0.0,
                    '[-2.0,0.0)': '.25x + .5',
                    ContinuousSet(0, 2, INC, EXC): '-.25x + .5',
                    ContinuousSet(2, np.inf, INC, EXC): 0
                })
        )
    )
    @unpack
    def test_convolution(self, plf1, plf2, truth):
        # Act
        result = plf1.convolution(plf2)
        # Assert
        self.assertEqual(truth.round(8), result.round(12))

    def test_rectify(self):
        # Arrange
        plf = PiecewiseFunction.from_dict({
            '[0,1)': '3x+2',
            '[1,2)': '-1x',
            '[2,inf)': 1
        })
        # Act
        result = plf.rectify()
        # Assert
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '[0,1)': 3.5,
                '[1,2)': -1.5,
                '[2,inf)': 1
            }),
            result
        )

    def test_rectify_error(self):
        # Arrange
        plf1 = PiecewiseFunction.from_dict({
            R: 'x+1'
        })
        plf2 = PiecewiseFunction.from_dict({
            R: '0x+.5'
        })
        # Act & Assert
        self.assertRaises(ValueError, plf1.rectify)
        self.assertEqual(
            PiecewiseFunction.from_dict({
                R: .5
            }),
            plf2.rectify()
        )


    @data(
        (
            PiecewiseFunction.zero().overwrite({
                '[-2,-1)': 1,
                '[1,2)': 1
            }),
            1,
            RealSet(['[-2.0,-1.0)', '[1.0,2.0)']),
        ),
        (
            PiecewiseFunction.zero().overwrite({
                '[0,1)': '1x',
                '[1,2)': '-1x+2'
            }),
            1,
            '[1,1]'
        ),
        (
            PiecewiseFunction.zero().overwrite({
                '[-2,-1)': 1,
                '[1,2)': 1.5
            }),
            1.5,
            '[1,2)',
        ),
        (
            PiecewiseFunction.from_dict({
                R: '1x',
            }),
            np.inf,
            ContinuousSet(np.inf, np.inf),
        ),
    )
    @unpack
    def test_maximize(self, f, f_max, f_argmax):
        # Arrange
        f_argmax = ifstr(f_argmax, ContinuousSet.parse)
        # Act
        argmax, max_ = f.maximize()
        # Assert
        self.assertEqual(f_argmax, argmax)
        self.assertEqual(f_max, max_)

    def test_approximate(self):
        plf = PiecewiseFunction.zero().overwrite({
            '[0,.25[': .1,
            '[.25,.5[': .5,
            '[.5,.75[': .7
        })
        self.assertEqual(
            5,
            len(plf)
        )
        self.assertEqual(
            3,
            len(
                plf.approximate(
                    n_segments=3,
                    replace_by=ConstantFunction
                )
            )
        )


# ----------------------------------------------------------------------------------------------------------------------

class PLFApproximatorTest(TestCase):

    def test_approximation_linear_k(self):
        for k in range(10, 2, -1):
            # Arrange
            plf: PiecewiseFunction = gaussian_numeric().cdf
            approximator = PLFApproximator(
                plf
            )
            # Act
            approx = approximator.run(k=k)
            # Assert
            self.assertGreater(len(plf), k)
            self.assertEqual(k, len(approx))

    def test_approximation_constant_k(self):
        for k in range(10, 2, -1):
            # Arrange
            plf: PiecewiseFunction = gaussian_numeric().pdf
            approximator = PLFApproximator(
                plf,
                replace_by=ConstantFunction
            )
            # Act
            approx = approximator.run(k=k)
            # Assert
            self.assertGreater(len(plf), k)
            self.assertEqual(k, len(approx))

    def test_approximation_constant_error(self):
        # Arrange
        plf: PiecewiseFunction = gaussian_numeric().pdf
        approximator = PLFApproximator(
            plf,
            replace_by=ConstantFunction
        )
        # Act
        approx = approximator.run(error_max=.2)
        # Assert
        self.assertGreater(len(plf), len(approx))

    def test_invalid(self):
        approximator = PLFApproximator(
            None
        )
        self.assertRaises(
            ValueError,
            approximator.run,
            k=2
        )

