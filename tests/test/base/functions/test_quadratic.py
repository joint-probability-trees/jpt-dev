from unittest import TestCase

from ddt import ddt, data, unpack

from jpt.base.functions import (
    LinearFunction,
    QuadraticFunction,
    ConstantFunction,
    Function
)


# ----------------------------------------------------------------------

@ddt
class QuadraticFunctionTest(TestCase):

    @data(
        (QuadraticFunction(1, 1, 1), 1, 3),
        (QuadraticFunction(1, 1, 1), 2, 7)
    )
    @unpack
    def test_eval(self, f, x, y):
        """Verify quadratic function evaluation at a given point."""
        self.assertEqual(y, f.eval(x))

    @data(
        ((0, 0), (1, 1), (2, 4),
         QuadraticFunction(1, 0, 0))
    )
    @unpack
    def test_fit(
            self,
            p1: tuple,
            p2: tuple,
            p3: tuple,
            truth: QuadraticFunction
    ):
        """Verify fitting a quadratic function from three points."""
        self.assertEqual(
            QuadraticFunction.from_points(p1, p2, p3), truth
        )

    @data(
        ((0, 0), (0, 0), (1, 2)),
        ((1, 2), (0, 0), (1, 2)),
        ((1, 2), (0, 0), (0, 0)),
    )
    @unpack
    def test_fit_integrity_check(
            self,
            p1: tuple,
            p2: tuple,
            p3: tuple
    ):
        """Verify fitting from collinear or duplicate points raises ValueError."""
        self.assertRaises(
            ValueError,
            QuadraticFunction.from_points, p1, p2, p3
        )

    def test_serialization(self):
        """Verify JSON round-trip serialization of quadratic functions."""
        f = QuadraticFunction(1, 2, 3)
        self.assertEqual(
            f, QuadraticFunction.from_json(f.to_json())
        )

    @data(
        (QuadraticFunction(1, 0, 0), 0),
        (QuadraticFunction(1, 0, 1), 0),
        (QuadraticFunction(1, 0, 0), 0),
        (QuadraticFunction(-2, 8, -5), 2),
        (QuadraticFunction(3, 6, 7), -1)
    )
    @unpack
    def test_argvertex(
            self,
            f: QuadraticFunction,
            xmax: float
    ):
        """Verify computation of the x-coordinate of the vertex."""
        self.assertEqual(xmax, f.argvertex())

    @data(
        (QuadraticFunction(1, 2, 3),
         QuadraticFunction(1, 2, 3)),
        (QuadraticFunction(0, 0, 1), ConstantFunction(1)),
        (QuadraticFunction(0, 1, 0), LinearFunction(1, 0)),
        (QuadraticFunction(0, 1, 2), LinearFunction(1, 2)),
        (QuadraticFunction(3, 0, 0),
         QuadraticFunction(3, 0, 0))
    )
    @unpack
    def test_simplify(
            self,
            f1: QuadraticFunction,
            f2: Function
    ):
        """Verify simplification of degenerate quadratics to lower-order functions."""
        self.assertEqual(f1.simplify(), f2)

    @data(
        (QuadraticFunction(2, 3, 4), 2,
         QuadraticFunction(4, 6, 8)),
        (QuadraticFunction(2, 3, 4), ConstantFunction(-2),
         QuadraticFunction(-4, -6, -8)),
        (QuadraticFunction(2, 3, 4), 0, ConstantFunction(0))
    )
    @unpack
    def test_mul(self, f, a, r):
        """Verify scalar and constant multiplication of quadratic functions."""
        self.assertEqual(r, f * a)

    @data(
        (QuadraticFunction(2, 3, 4), 2,
         QuadraticFunction(2, 3, 6)),
        (QuadraticFunction(2, 3, 4), ConstantFunction(-2),
         QuadraticFunction(2, 3, 2)),
        (QuadraticFunction(2, 3, 4), 0,
         QuadraticFunction(2, 3, 4)),
        (QuadraticFunction(2, 3, 4), LinearFunction(2, 3),
         QuadraticFunction(2, 5, 7))
    )
    @unpack
    def test_add(self, f, a, r):
        """Verify addition of quadratic functions with scalars and functions."""
        self.assertEqual(r, f + a)

    def test_roots_2_solutions(self):
        """Verify root finding for a quadratic with two distinct roots."""
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
        """Verify root finding for a quadratic with one repeated root."""
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
        """Verify root finding returns empty list when no real roots exist."""
        # Arrange
        f = QuadraticFunction(2, -8, 11)

        # Act
        roots = f.roots()

        # Assert
        self.assertEqual(
            [],
            list(roots)
        )

    @data(
        ((3, 1, 4), QuadraticFunction(3, -6, 7)),
        ((-2, 2, 3), QuadraticFunction(-2, 8, -5))
    )
    @unpack
    def test_vertexform(self, params, result):
        """Verify construction of a quadratic from vertex form parameters."""
        # Act
        vertex = QuadraticFunction.from_vertexform(*params)

        # Assert
        self.assertEqual(
            result,
            vertex
        )
