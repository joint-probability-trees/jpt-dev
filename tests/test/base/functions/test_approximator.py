from unittest import TestCase

import numpy as np

from jpt.base.intervals import ContinuousSet, INC, EXC

from jpt.base.functions import (
    LinearFunction,
    ConstantFunction,
    PiecewiseFunction,
    PLFApproximator
)


# ----------------------------------------------------------------------

class PLFApproximatorTest(TestCase):

    def test_approximation_linear_k(self):
        """Reduce linear PLF to exactly k segments."""
        for k in range(5, 2, -1):
            # Arrange
            plf: PiecewiseFunction = (
                PiecewiseFunction.zero()
                + PiecewiseFunction.from_points(
                    [
                        (1, 0), (2, .1), (3, .2),
                        (4, .6), (5, .8), (6, 1)
                    ]
                )
            )

            approximator = PLFApproximator(
                plf
            )
            # Act
            approx = approximator.run(k=k)
            # Assert
            self.assertGreater(len(plf), k)
            self.assertEqual(k, len(approx))

    def test_approximation_constant_k(self):
        """Reduce constant PLF to exactly k segments."""
        for k in range(5, 2, -1):
            # Arrange
            plf: PiecewiseFunction = (
                PiecewiseFunction
                .zero()
                .overwrite({
                    '[1,2)': .1,
                    '[2,3)': .2,
                    '[3,4)': .3,
                    '[4,5)': .2,
                    '[5,6)': .2
                })
            )

            approximator = PLFApproximator(
                plf,
                replace_by=ConstantFunction
            )

            # Act
            approx = approximator.run(k=k)

            # Assert
            self.assertGreater(
                len(plf),
                k
            )
            self.assertEqual(
                k,
                len(approx)
            )

    def test_approximation_constant_error(self):
        """Merge segments within error bound."""
        # Arrange
        plf: PiecewiseFunction = (
            PiecewiseFunction
            .zero()
            .overwrite({
                '[1,2)': .1,
                '[2,3)': .2,
                '[3,4)': .3,
                '[4,5)': .2,
                '[5,6)': .2
            })
        )
        approximator = PLFApproximator(
            plf,
            replace_by=ConstantFunction
        )
        # Act
        approx = approximator.run(error_max=.1)

        # Assert
        self.assertEqual(
            PiecewiseFunction.zero().overwrite_at(
                ContinuousSet(1, 6, INC, EXC),
                ConstantFunction(0.19999999999999993)
            ),
            approx
        )

    def test_invalid(self):
        """Reject None as input PLF."""
        self.assertRaises(
            TypeError,
            PLFApproximator,
            None
        )

    def test_jumps_linear_function(self):
        '''Jumps at the same positions'''
        # Arrange
        plf = PiecewiseFunction.zero().overwrite({
            '[0.0,5e-324)': LinearFunction(
                np.inf, np.nan
            ),
        })

        approx = PLFApproximator(plf, LinearFunction)

        # Act
        result = approx.run(error_max=.1)

        # Assert
        self.assertEqual(
            plf,
            result
        )

    def test_jumps_constant_function(self):
        """Preserve impulse segments in constant PLF."""
        # Arrange
        plf = PiecewiseFunction.from_dict({
            '(-∞,0.0)': 0,
            '[0.0,5e-324)': np.inf,
            '[5e-324,∞)': 0
        })
        approx = PLFApproximator(plf, ConstantFunction)

        # Act
        result = approx.run(error_max=.2)

        # Assert
        self.assertEqual(
            plf,
            result
        )
