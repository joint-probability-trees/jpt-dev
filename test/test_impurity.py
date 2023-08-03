import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt.distributions import SymbolicType, Bool, Numeric
from jpt.trees import JPT
from jpt.variables import SymbolicVariable, NumericVariable, infer_from_dataframe

try:
    from jpt.learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.learning.impurity import (
        Impurity,
        _sum_at,
        _sq_sum_at,
        _variances,
        _compute_var_improvements
    )


class ImpurityTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))

        # replace na with "None"
        cls.data = cls.data.fillna("None")

        # declare variable types
        PatronsType = SymbolicType('Patrons', ['Some', 'Full', 'None'])
        PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
        FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
        WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

        # create variables
        cls.al = SymbolicVariable('Alternatives', Bool)
        cls.ba = SymbolicVariable('Bar', Bool)
        cls.fr = SymbolicVariable('Friday', Bool)
        cls.hu = SymbolicVariable('Hungry', Bool)
        cls.pa = SymbolicVariable('Patrons', PatronsType)
        cls.pr = SymbolicVariable('Price', PriceType)
        cls.ra = SymbolicVariable('Rain', Bool)
        cls.re = SymbolicVariable('Reservation', Bool)
        cls.fo = SymbolicVariable('Food', FoodType)
        cls.we = SymbolicVariable('WaitEstimate', WaitEstType)
        cls.wa = SymbolicVariable('WillWait', Bool)

        cls.variables = [cls.al, cls.ba, cls.fr, cls.hu, cls.pa, cls.pr, cls.ra, cls.re, cls.fo, cls.we, cls.wa]

    def test_symbolic(self):
        jpt = JPT(
            variables=self.variables,
            targets=[self.wa]
        )
        data = jpt._preprocess_data(self.data)
        impurity = Impurity(jpt)
        impurity.min_samples_leaf = max(1, jpt.min_samples_leaf)
        impurity.setup(data, np.array(list(range(data.shape[0]))))
        impurity.compute_best_split(0, data.shape[0])

        self.assertNotEqual(impurity.best_var, -1)
        self.assertIs(self.variables[impurity.best_var], ImpurityTest.pa)
        self.assertEqual([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 2, 2],
                         list(np.asarray(impurity.feat, dtype=np.int32)))
        self.assertEqual({0, 2, 5, 7}, set(impurity.indices[:4]))
        self.assertEqual({1, 3, 4, 8, 9, 11}, set(impurity.indices[4:10]))
        self.assertEqual({6, 10}, set(impurity.indices[10:]))

    def test_col_is_constant(self):
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric), NumericVariable('x2', domain=Numeric)])
        impurity = Impurity(jpt)
        impurity.min_samples_leaf = max(1, jpt.min_samples_leaf)

        data = np.array([[1, 0, np.nan], [1, 1, 0]], dtype=np.float64)
        impurity.setup(data, np.array(list(range(data.shape[0]))))
        self.assertTrue(impurity._col_is_constant(0, 2, 0))
        self.assertFalse(impurity._col_is_constant(0, 2, 1))
        self.assertEqual(-1, impurity._col_is_constant(0, 2, 2))
        self.assertTrue(impurity._col_is_constant(1, 2, 0))
        self.assertTrue(impurity._col_is_constant(1, 2, 1))
        self.assertTrue(impurity._col_is_constant(1, 2, 2))

    def test_has_numeric_vars(self):
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric), NumericVariable('x2', domain=Numeric)])
        impurity = Impurity(jpt)
        self.assertTrue(impurity.has_numeric_vars_())
        self.assertTrue(impurity.has_numeric_vars_(0))
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric)])
        impurity = Impurity(jpt)
        self.assertTrue(impurity.has_numeric_vars_())
        self.assertFalse(impurity.has_numeric_vars_(0))


# ----------------------------------------------------------------------------------------------------------------------

class SumAtTest(TestCase):
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)

    def test_full_spec(self):
        # Arrange
        cols = np.array([0, 1, 2], dtype=np.int64)
        rows = np.array([0, 1, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [12, 15, 18],
            list(result)
        )

    def test_partial_rows(self):
        # Arrange
        rows = np.array([0, 2], dtype=np.int64)
        cols = np.array([0, 1, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [8, 10, 12],
            list(result)
        )

    def test_partial_cols(self):
        # Arrange
        rows = np.array([0, 1, 2], dtype=np.int64)
        cols = np.array([0, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [12, 18],
            list(result)
        )

    def test_partial(self):
        # Arrange
        rows = np.array([0, 2], dtype=np.int64)
        cols = np.array([0, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [8, 12],
            list(result)
        )


# ----------------------------------------------------------------------------------------------------------------------

class SqSumAt(TestCase):
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)

    def test_full_spec(self):
        # Arrange
        cols = np.array([0, 1, 2], dtype=np.int64)
        rows = np.array([0, 1, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sq_sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [66, 93, 126],
            list(result)
        )

    def test_partial_rows(self):
        # Arrange
        rows = np.array([0, 2], dtype=np.int64)
        cols = np.array([0, 1, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sq_sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [50, 68, 90],
            list(result)
        )

    def test_partial_cols(self):
        # Arrange
        rows = np.array([0, 1, 2], dtype=np.int64)
        cols = np.array([0, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sq_sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [66, 126],
            list(result)
        )

    def test_partial(self):
        # Arrange
        rows = np.array([0, 2], dtype=np.int64)
        cols = np.array([0, 2], dtype=np.int64)
        result = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _sq_sum_at(
            self.data,
            rows,
            cols,
            result
        )
        # Assert
        self.assertEqual(
            [50, 90],
            list(result)
        )


# ----------------------------------------------------------------------------------------------------------------------

class VariancesTest(TestCase):
    data = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ], dtype=np.float64)

    def test_variances(self):
        # Arrange
        rows = np.array([0, 1, 2], dtype=np.int64)
        cols = np.array([0, 1, 2], dtype=np.int64)
        sq_sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sq_sum_at(self.data, rows, cols, sq_sums)
        sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sum_at(self.data, rows, cols, sums)
        variances = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _variances(sq_sums, sums, rows.shape[0], variances)
        # Assert
        self.assertEqual(
            list(np.var(self.data, axis=0, ddof=0)),
            list(np.asarray(variances))
        )

    def test_scalar(self):
        data = np.array([
            [1]
        ], dtype=np.float64)

        # Arrange
        rows = np.array([0], dtype=np.int64)
        cols = np.array([0], dtype=np.int64)
        sq_sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sq_sum_at(data, rows, cols, sq_sums)
        sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sum_at(data, rows, cols, sums)
        variances = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        # Act
        _variances(sq_sums, sums, rows.shape[0], variances)
        # Assert
        self.assertEqual(
            list([0]),
            list(np.asarray(variances))
        )


# ----------------------------------------------------------------------------------------------------------------------

@unittest.skip
class VarianceImprovementTest(TestCase):
    data = np.array(
            [
                [1, 0, 0, 0, 1],
                [2, 1, 0, 1, 0],
                [0, 1, 0, 1, 2],
                [1, 2, 0, 2, 1],
                [0, 1, 1, 1, 1],
                [2, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 2, 1, 1, 1],
                [1, 2, 2, 0, 1],
                [0, 1, 2, 1, 0],
                [2, 1, 2, 1, 2],
                [1, 0, 2, 2, 1],
            ],
            dtype=np.float64
    )

    def compute_variances(self, arr: np.ndarray):
        rows = np.array(range(arr.shape[0]), dtype=np.int64)
        cols = np.array(range(arr.shape[1]), dtype=np.int64)
        sq_sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sq_sum_at(self.data, rows, cols, sq_sums)
        sums = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _sum_at(self.data, rows, cols, sums)
        variances = np.ndarray(
            shape=cols.shape[0],
            dtype=np.float64
        )
        _variances(sq_sums, sums, rows.shape[0], variances)
        return variances

    def compute_var_left_right(self, arr, i):
        variances_left = self.compute_variances(arr[:i, :])
        variances_right = self.compute_variances(arr[i:, :])
        return variances_left, variances_right

    def test_compute_best_split(self):
        # Arrange
        df = pd.DataFrame(
            data=self.data,
            columns=['xi', 'yi', 'a', 'xo', 'yo'],
            dtype=np.int_
        )

        vars = infer_from_dataframe(df)
        t = JPT(
            variables=vars,
            targets=vars[3:]
        )

        _data = t._preprocess_data(data=df)
        indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        indices[0] = 0
        np.cumsum(indices, out=indices)

        impurity = Impurity(t)
        impurity.setup(_data, indices)
        impurity.min_samples_leaf = 1

        # Act
        max_gain = impurity.compute_best_split(0, _data.shape[0])

        # Assert
        print('maxgain:', max_gain)

    def test_var_improvement(self):
        # Arrange
        data = np.ascontiguousarray(self.data[:, -2:], dtype=np.float64)

        variances_total = self.compute_variances(data)
        print('variances total:', variances_total)

        # Act
        improvements = [
            _compute_var_improvements(
                variances_total,
                *self.compute_var_left_right(data, i),
                float(i),
                float(data.shape[0] - i)
            ) for i in range(1, data.shape[0])]

        # Assert
        print('improvements:', improvements)
