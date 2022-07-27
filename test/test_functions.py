from unittest import TestCase

from ddt import ddt, data, unpack

try:
    from jpt.base.functions import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import LinearFunction, QuadraticFunction, ConstantFunction, Undefined, Function


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


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class QuadraticFunctionTest(TestCase):

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

