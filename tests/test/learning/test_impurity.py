import os
import unittest
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt.distributions import SymbolicType, Bool, Numeric
from jpt.learning.preprocessing import preprocess_data
from jpt.trees import JPT
from jpt.variables import (
    SymbolicVariable,
    NumericVariable,
    VariableMap,
    infer_from_dataframe,
)

from jpt.learning.impurity.impurity import (
    Impurity,
    _sum_at,
    _sq_sum_at,
    _variances,
    _compute_var_improvements
)
from test.testutils import EXAMPLES_DATA


class ImpuritySymbolsMisalignmentTest(TestCase):
    """Expose bug: ``get_size_of_symbolic_variables_domain_
    from_tree`` returns domain sizes for *all* symbolic
    variables, but the Gini loop indexes into this array
    using the symbolic-*target* counter.  When a symbolic
    feature with a different domain size precedes the
    symbolic target in the variable list, the Gini
    computation uses the wrong domain size."""

    def test_symbols_aligned_with_symbolic_targets(self):
        """Verify that the domain-size array returned by
        ``get_size_of_symbolic_variables_domain_from_tree``
        contains only the domain sizes of symbolic *target*
        variables, not all symbolic variables."""
        # Arrange
        #   - A is a symbolic *feature* (3 values)
        #   - B is a symbolic *target* (2 values)
        # A comes first in the variable list, so if the
        # method collects ALL symbolic vars, the array is
        # [3, 2] (length 2) while the Gini loop expects
        # an array of length 1 containing just [2].
        FeatureType = SymbolicType(
            'FeatureType',
            labels=['x', 'y', 'z'],
        )
        TargetType = SymbolicType(
            'TargetType',
            labels=['p', 'q'],
        )
        A = SymbolicVariable('feature', FeatureType)
        B = SymbolicVariable('target', TargetType)

        jpt = JPT(
            variables=[A, B],
            targets=[B],
        )

        # Act
        symbols = (
            Impurity
            .get_size_of_symbolic_variables_domain_from_tree(
                jpt
            )
        )
        n_sym_targets = len([
            v for v in jpt.variables
            if v.symbolic and v in jpt.targets
        ])

        # Assert — there is exactly one symbolic target (B)
        # with domain size 2.  The array must have exactly
        # n_sym_targets entries, each corresponding to a
        # symbolic target's domain size.
        self.assertEqual(
            n_sym_targets,
            len(symbols),
            'symbols array length should equal the number '
            'of symbolic targets (%d), not all symbolic '
            'variables (%d)' % (
                n_sym_targets,
                len(symbols),
            )
        )
        self.assertEqual(
            2,
            symbols[0],
            'symbols[0] should be 2 (target domain size), '
            'not 3 (feature domain size)'
        )


# ----------------------------------------------------------------------

class MaxVarianceEarlyReturnTest(TestCase):
    """Expose bug: ``check_max_variances`` causes
    ``compute_best_split`` to return ``-inf`` (no split)
    when all numeric variances are below their ``max_std``
    limits, even if symbolic targets still have high
    impurity that would benefit from splitting."""

    def test_symbolic_split_not_blocked_by_max_std(self):
        """Verify that a tree with a numeric variable
        whose variance is below ``max_std`` still splits
        to resolve a symbolic target."""
        # Arrange
        #   x is numeric, nearly constant → variance ≈ 0
        #   y is symbolic with a clear split on x
        # With max_std set on x, the early return in
        # compute_best_split fires and prevents any split,
        # even though y needs splitting.
        n = 100
        x = np.concatenate([
            np.full(n // 2, 1.0),
            np.full(n // 2, 2.0),
        ])
        y = np.array(
            ['a'] * (n // 2) + ['b'] * (n // 2)
        )
        df = pd.DataFrame({'x': x, 'y': y})

        YType = SymbolicType('YType', ['a', 'b'])
        xvar = NumericVariable('x', max_std=100.0)
        yvar = SymbolicVariable('y', YType)

        jpt = JPT(
            variables=[xvar, yvar],
        )
        jpt.fit(df)

        # Act
        n_leaves = len(jpt.leaves)

        # Assert — the tree must have split at least once
        # to separate 'a' from 'b'.  Without the bug, the
        # symbolic impurity would drive the split.  With
        # the bug, the early return from max_std prevents
        # it and the tree has exactly 1 leaf.
        self.assertGreater(
            n_leaves,
            1,
            'Tree should have split to resolve the '
            'symbolic target, but max_std early return '
            'in compute_best_split blocked all splits '
            '(got %d leaf)' % n_leaves,
        )


# ----------------------------------------------------------------------

class DependencyStatsResetTest(TestCase):
    """Expose bug: when custom ``dependencies`` exclude a
    numeric target from a symbolic feature, the right-side
    sums/sq_sums for that target are not reset between
    symbolic value groups, leading to incorrect variance
    computations."""

    def test_nondependent_target_variance_improvement(
            self
    ):
        """Verify that excluding a numeric target from a
        symbolic feature's dependencies produces the same
        gain as a tree without that target entirely."""
        # Arrange
        #   A (symbolic feature, 3 values)
        #   B (numeric target) — depends on A
        #   C (numeric target) — does NOT depend on A
        #
        # When C is excluded from A's dependencies, its
        # variance improvement for splits on A must be 0.
        # The resulting gain should match a tree that has
        # only B as target (no C).
        #
        # With the bug, C contributes a spurious non-zero
        # improvement because its right-side sums are
        # stale after the symbolic value group reset.
        AT = SymbolicType('AT', ['x', 'y', 'z'])
        A = SymbolicVariable('A', AT)
        B = NumericVariable('B')
        C = NumericVariable('C')

        df = pd.DataFrame({
            'A': ['x', 'x', 'y', 'y', 'z', 'z'],
            'B': [1., 2., 5., 6., 9., 10.],
            'C': [10., 20., 15., 25., 12., 22.],
        })

        # Tree 1: C excluded from A's deps
        deps_excluded = VariableMap({
            A: [B],
            B: [B, C],
            C: [B, C],
        })
        jpt_excluded = JPT(
            variables=[A, B, C],
            targets=[B, C],
            features=[A, B, C],
            dependencies=deps_excluded,
        )
        data = preprocess_data(jpt_excluded, df)
        imp = Impurity.from_tree(jpt_excluded)
        imp.min_samples_leaf = 1
        imp.setup(
            data.values,
            np.arange(
                data.shape[0],
                dtype=np.int64,
            ),
        )
        gain_excluded = imp.compute_best_split(
            0, data.shape[0]
        )

        # Tree 2: only B as target (no C at all)
        jpt_b_only = JPT(
            variables=[A, B],
            targets=[B],
            features=[A, B],
        )
        data2 = preprocess_data(
            jpt_b_only,
            df[['A', 'B']],
        )
        imp2 = Impurity.from_tree(jpt_b_only)
        imp2.min_samples_leaf = 1
        imp2.setup(
            data2.values,
            np.arange(
                data2.shape[0],
                dtype=np.int64,
            ),
        )
        gain_b_only = imp2.compute_best_split(
            0, data2.shape[0]
        )

        # Assert — excluding C from A's deps should
        # give the same gain as having no C at all,
        # since C contributes zero improvement for A.
        self.assertAlmostEqual(
            gain_excluded,
            gain_b_only,
            places=10,
            msg='Gain with C excluded from A deps '
                '(%.6f) should equal gain with B-only '
                'tree (%.6f)' % (
                    gain_excluded,
                    gain_b_only,
                ),
        )


# ----------------------------------------------------------------------

class ImpurityTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(os.path.join(EXAMPLES_DATA, 'restaurant.csv'))

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
        """Verify best split selection on symbolic restaurant data."""
        jpt = JPT(
            variables=self.variables,
            targets=[self.wa]
        )
        data = preprocess_data(jpt, self.data)
        impurity = Impurity.from_tree(jpt)
        impurity.min_samples_leaf = max(1, jpt.min_samples_leaf)
        impurity.setup(
            data.values,
            np.array(list(range(data.shape[0])))
        )
        impurity.compute_best_split(0, data.shape[0])

        self.assertNotEqual(impurity.best_var, -1)
        self.assertIs(self.variables[impurity.best_var], ImpurityTest.pa)
        self.assertEqual([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 2, 2],
                         list(np.asarray(impurity.feat, dtype=np.int32)))
        self.assertEqual({0, 2, 5, 7}, set(impurity.indices[:4]))
        self.assertEqual({1, 3, 4, 8, 9, 11}, set(impurity.indices[4:10]))
        self.assertEqual({6, 10}, set(impurity.indices[10:]))

    def test_col_is_constant(self):
        """Verify constant column detection including NaN handling."""
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric), NumericVariable('x2', domain=Numeric)])
        impurity = Impurity.from_tree(jpt)
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
        """Verify detection of numeric variables in impurity object."""
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric), NumericVariable('x2', domain=Numeric)])
        impurity = Impurity.from_tree(jpt)
        self.assertTrue(impurity.has_numeric_vars_())
        self.assertTrue(impurity.has_numeric_vars_(0))
        jpt = JPT(variables=[NumericVariable('x1', domain=Numeric)])
        impurity = Impurity.from_tree(jpt)
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
        """Verify column sums over all rows and columns."""
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
        """Verify column sums over a subset of rows."""
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
        """Verify sums over a subset of columns."""
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
        """Verify sums over subsets of both rows and columns."""
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
        """Verify squared column sums over all rows and columns."""
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
        """Verify squared column sums over a subset of rows."""
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
        """Verify squared sums over a subset of columns."""
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
        """Verify squared sums over subsets of both rows and columns."""
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
        """Verify computed variances match numpy reference values."""
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
        """Verify variance of a single scalar value is zero."""
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
        """Verify best split computation on mixed variable data."""
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

        impurity = Impurity.from_tree(t)
        impurity.setup(_data, indices)
        impurity.min_samples_leaf = 1

        # Act
        max_gain = impurity.compute_best_split(0, _data.shape[0])

        # Assert
        print('maxgain:', max_gain)

    def test_var_improvement(self):
        """Verify variance improvement computation for all split positions."""
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
