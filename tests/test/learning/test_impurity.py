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


# ----------------------------------------------------------------------

class SplitValidationTest(TestCase):
    """Tests for the split validation feature at the Impurity level.

    These tests directly construct Impurity objects with C-contiguous
    data arrays to avoid environment-specific pandas contiguity issues.
    """

    @staticmethod
    def _make_numeric_impurity(n_features, n_targets):
        """Helper to create an Impurity for numeric-only variables.

        Returns (impurity, feature_indices, target_indices) where
        variables are [feat0, feat1, ..., tgt0, tgt1, ...].
        """
        feat_vars = [NumericVariable(f'f{i}') for i in range(n_features)]
        tgt_vars = [NumericVariable(f't{i}') for i in range(n_targets)]
        variables = feat_vars + tgt_vars
        jpt = JPT(variables=variables, targets=tgt_vars)
        impurity = Impurity.from_tree(jpt)
        impurity.min_samples_leaf = 1
        return impurity

    @staticmethod
    def _make_symbolic_impurity():
        """Helper to create an Impurity with 1 numeric feature
        and 1 symbolic target (Bool, 2 values)."""
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(variables=[xvar, yvar], targets=[yvar])
        impurity = Impurity.from_tree(jpt)
        impurity.min_samples_leaf = 1
        return impurity

    def test_no_mask_same_as_default(self):
        """compute_best_split with mask=None gives the same result
        as without a mask."""
        # 10 samples: feature x, target y with clear split at x=5
        data = np.array([
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
            [5.0, 14.0],
            [6.0, 50.0],
            [7.0, 51.0],
            [8.0, 52.0],
            [9.0, 53.0],
            [10.0, 54.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)

        imp1 = self._make_numeric_impurity(1, 1)
        imp1.setup(data, indices.copy())
        gain1 = imp1.compute_best_split(0, 10)

        imp2 = self._make_numeric_impurity(1, 1)
        imp2.setup(data, indices.copy(), None, 0)
        gain2 = imp2.compute_best_split(0, 10)

        self.assertAlmostEqual(gain1, gain2, places=10)
        self.assertEqual(imp1.best_var, imp2.best_var)

    def test_all_training_mask_same_as_no_mask(self):
        """compute_best_split with all-True mask gives the same
        result as without a mask."""
        data = np.array([
            [1.0, 10.0],
            [2.0, 11.0],
            [3.0, 12.0],
            [4.0, 13.0],
            [5.0, 14.0],
            [6.0, 50.0],
            [7.0, 51.0],
            [8.0, 52.0],
            [9.0, 53.0],
            [10.0, 54.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        mask = np.ones(10, dtype=np.uint8)

        imp1 = self._make_numeric_impurity(1, 1)
        imp1.setup(data, indices.copy())
        gain1 = imp1.compute_best_split(0, 10)

        imp2 = self._make_numeric_impurity(1, 1)
        imp2.setup(data, indices.copy(), mask, 0)
        gain2 = imp2.compute_best_split(0, 10)

        self.assertAlmostEqual(gain1, gain2, places=10)
        self.assertEqual(imp1.best_var, imp2.best_var)
        self.assertEqual(imp1.best_split_pos, imp2.best_split_pos)

    def test_mask_skips_eval_as_candidates(self):
        """Evaluation samples should not be used as candidate split points
        but the impurity should still find a good split."""
        # Clear numeric split: low values vs high values
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
            [9.0, 100.0],
            [10.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        # Only even-indexed samples are training
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        imp = self._make_numeric_impurity(1, 1)
        imp.setup(data, indices.copy(), mask, 0)  # SV_BOTH
        gain = imp.compute_best_split(0, 10)

        # Should still find a split with positive gain
        self.assertGreater(gain, 0, 'Should find a split with positive gain')
        self.assertEqual(imp.best_var, 0, 'Should split on feature 0')

    def test_mask_eval_targets_contribute_to_impurity(self):
        """In SV_BOTH mode, evaluation targets contribute to impurity.
        A split that separates training features well but not evaluation
        targets should have lower gain than one that separates all targets."""
        # Scenario: feature x has two clusters
        # Training samples (mask=1) are at x=1..5 and x=6..10
        # Targets: all consistent (low target for x<5.5, high for x>5.5)
        data = np.array([
            [1.0, 0.0],   # train
            [2.0, 0.0],   # eval
            [3.0, 0.0],   # train
            [4.0, 0.0],   # eval
            [5.0, 0.0],   # train
            [6.0, 100.0], # eval
            [7.0, 100.0], # train
            [8.0, 100.0], # eval
            [9.0, 100.0], # train
            [10.0, 100.0],# eval
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        imp = self._make_numeric_impurity(1, 1)
        imp.setup(data, indices.copy(), mask, 0)  # SV_BOTH
        gain = imp.compute_best_split(0, 10)

        # The gain should be high since all targets (train + eval) agree
        self.assertGreater(gain, 0.5,
                           'Gain should be high when all targets agree with the split')

    def test_symbolic_target_with_mask(self):
        """Split validation works with symbolic targets."""
        # x is numeric feature, y is symbolic (0 or 1)
        # Clear split: x < 5 -> y=0, x >= 5 -> y=1
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 1.0],
            [7.0, 1.0],
            [8.0, 1.0],
            [9.0, 1.0],
            [10.0, 1.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        imp = self._make_symbolic_impurity()
        imp.setup(data, indices.copy(), mask, 0)  # SV_BOTH
        gain = imp.compute_best_split(0, 10)

        self.assertGreater(gain, 0, 'Should split on symbolic target with mask')
        self.assertEqual(imp.best_var, 0, 'Should split on feature x')

    def test_mode_training_only(self):
        """SV_TRAINING mode only uses training targets for impurity."""
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
            [9.0, 100.0],
            [10.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        imp = self._make_numeric_impurity(1, 1)
        imp.setup(data, indices.copy(), mask, 1)  # SV_TRAINING
        gain = imp.compute_best_split(0, 10)

        self.assertGreater(gain, 0, 'Should find a split in training-only mode')

    def test_mode_evaluation_only(self):
        """SV_EVALUATION mode only uses evaluation targets for impurity."""
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
            [9.0, 100.0],
            [10.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)

        imp = self._make_numeric_impurity(1, 1)
        imp.setup(data, indices.copy(), mask, 2)  # SV_EVALUATION
        gain = imp.compute_best_split(0, 10)

        self.assertGreater(gain, 0, 'Should find a split in evaluation-only mode')

    def test_input_validation_mask_length(self):
        """Verify that mismatched mask length raises ValueError via C45Algorithm."""
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(variables=[xvar, yvar], targets=[yvar])

        data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [True, False, True]
        })
        mask = np.array([1, 0], dtype=np.uint8)  # wrong length

        with self.assertRaises(ValueError):
            jpt.fit(data, multicore=0, split_validation_mask=mask)

    def test_input_validation_no_training(self):
        """Verify that all-zero mask raises ValueError."""
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(variables=[xvar, yvar], targets=[yvar])

        data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [True, False, True]
        })
        mask = np.zeros(3, dtype=np.uint8)

        with self.assertRaises(ValueError):
            jpt.fit(data, multicore=0, split_validation_mask=mask)

    def test_input_validation_invalid_mode(self):
        """Verify that invalid mode string raises ValueError."""
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(variables=[xvar, yvar], targets=[yvar])

        data = pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [True, False, True]
        })
        mask = np.ones(3, dtype=np.uint8)

        with self.assertRaises(ValueError):
            jpt.fit(data, multicore=0,
                    split_validation_mask=mask,
                    split_validation_mode='invalid')


# ----------------------------------------------------------------------

class SplitValidationEndToEndTest(TestCase):
    """End-to-end tests for the split validation feature
    via ``JPT.fit()`` and subsequent inference."""

    @staticmethod
    def _make_separable_data(n=200):
        """Create a DataFrame with a clear split:
        x < 0.5 => y = False, x >= 0.5 => y = True.

        Returns (data, mask) where mask marks every
        other row as evaluation.
        """
        rng = np.random.RandomState(42)
        x = rng.uniform(0, 1, n)
        y = x >= 0.5
        data = pd.DataFrame({'x': x, 'y': y})
        mask = np.zeros(n, dtype=np.uint8)
        mask[::2] = 1  # even indices are training
        return data, mask

    def test_fit_with_mask_both_mode(self):
        """JPT.fit with split_validation_mask in 'both'
        mode produces a tree with leaves."""
        # Arrange
        data, mask = self._make_separable_data()
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='both'
        )

        # Assert
        self.assertGreater(
            len(jpt.leaves), 0,
            'Tree should have at least one leaf'
        )

    def test_fit_with_mask_training_mode(self):
        """JPT.fit with split_validation_mask in
        'training' mode produces a valid tree."""
        # Arrange
        data, mask = self._make_separable_data()
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='training'
        )

        # Assert
        self.assertGreater(
            len(jpt.leaves), 0,
            'Tree should have at least one leaf'
        )

    def test_fit_with_mask_evaluation_mode(self):
        """JPT.fit with split_validation_mask in
        'evaluation' mode produces a valid tree."""
        # Arrange
        data, mask = self._make_separable_data()
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )

        # Assert
        self.assertGreater(
            len(jpt.leaves), 0,
            'Tree should have at least one leaf'
        )

    def test_fit_predict_roundtrip(self):
        """A tree trained with split validation can
        still perform posterior inference."""
        # Arrange
        data, mask = self._make_separable_data(300)
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='both'
        )

        # Act
        result = jpt.posterior(
            evidence={xvar: 0.1}
        )

        # Assert — x=0.1 is in the False region
        p_false = result[yvar].p({False})
        self.assertGreater(
            p_false, 0.5,
            'P(y=False | x=0.1) should be > 0.5'
        )

    def test_no_mask_and_mask_both_produce_trees(self):
        """Trees trained with and without a mask both
        produce valid models on the same data."""
        # Arrange
        data, mask = self._make_separable_data()
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)

        jpt_plain = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )
        jpt_masked = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )

        # Act
        jpt_plain.fit(data, multicore=0)
        jpt_masked.fit(
            data,
            multicore=0,
            split_validation_mask=mask
        )

        # Assert — both should have leaves
        self.assertGreater(len(jpt_plain.leaves), 0)
        self.assertGreater(len(jpt_masked.leaves), 0)

    def test_serialization_roundtrip(self):
        """A tree trained with split validation
        survives JSON serialization."""
        # Arrange
        data, mask = self._make_separable_data()
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask
        )

        # Act
        json_data = jpt.to_json()
        jpt2 = JPT.from_json(json_data)

        # Assert
        self.assertEqual(
            len(jpt.leaves), len(jpt2.leaves)
        )
        self.assertEqual(
            len(jpt.innernodes), len(jpt2.innernodes)
        )
        self.assertEqual(
            jpt.min_samples_leaf,
            jpt2.min_samples_leaf
        )
        self.assertEqual(
            jpt.min_eval_samples,
            jpt2.min_eval_samples
        )

    def test_numeric_target_with_mask(self):
        """Split validation works with numeric targets."""
        # Arrange — x < 0.5 => y ~ 0, x >= 0.5 => y ~ 100
        rng = np.random.RandomState(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = np.where(x < 0.5, rng.normal(0, 1, n),
                     rng.normal(100, 1, n))
        data = pd.DataFrame({'x': x, 'y': y})
        mask = np.zeros(n, dtype=np.uint8)
        mask[::2] = 1

        xvar = NumericVariable('x')
        yvar = NumericVariable('y')
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )

        # Assert
        self.assertGreater(
            len(jpt.leaves), 1,
            'Should split on a clear numeric pattern'
        )


# ----------------------------------------------------------------------

class MinEvalSamplesImpurityTest(TestCase):
    """Unit tests for the ``min_eval_samples`` threshold
    at the Impurity level."""

    @staticmethod
    def _make_numeric_impurity(
            n_features,
            n_targets,
            min_eval_samples=0
    ):
        """Create an Impurity for numeric-only variables
        with a given ``min_eval_samples``.
        """
        feat_vars = [
            NumericVariable(f'f{i}')
            for i in range(n_features)
        ]
        tgt_vars = [
            NumericVariable(f't{i}')
            for i in range(n_targets)
        ]
        variables = feat_vars + tgt_vars
        jpt = JPT(
            variables=variables,
            targets=tgt_vars,
            min_eval_samples=min_eval_samples
        )
        impurity = Impurity.from_tree(jpt)
        impurity.min_samples_leaf = 1
        return impurity

    def test_disabled_by_default(self):
        """min_eval_samples=0 does not reject any
        splits."""
        # Arrange — clear split with 1 eval sample
        # per side
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(6, dtype=np.int64)
        # train: 0, 2, 3, 5; eval: 1, 4
        mask = np.array(
            [1, 0, 1, 1, 0, 1], dtype=np.uint8
        )

        imp = self._make_numeric_impurity(1, 1, 0)
        imp.setup(data, indices.copy(), mask, 2)
        gain = imp.compute_best_split(0, 6)

        # Assert — split should be accepted
        self.assertGreater(gain, 0)

    def test_rejects_split_with_too_few_eval(self):
        """A split where one child has fewer eval
        samples than the threshold is rejected."""
        # Arrange — 6 samples, eval at indices 1 and 4
        # With a split at ~4.5, left has 1 eval, right
        # has 1 eval. Requiring 2 should reject.
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(6, dtype=np.int64)
        mask = np.array(
            [1, 0, 1, 1, 0, 1], dtype=np.uint8
        )

        imp = self._make_numeric_impurity(1, 1, 2)
        imp.setup(data, indices.copy(), mask, 2)
        gain = imp.compute_best_split(0, 6)

        # Assert — all candidate splits leave < 2
        # eval samples on at least one side
        self.assertLessEqual(
            gain, 0,
            'Split should be rejected when eval '
            'samples per child < min_eval_samples'
        )

    def test_accepts_split_with_enough_eval(self):
        """A split where both children have enough
        eval samples is accepted."""
        # Arrange — 10 samples, 4 eval, split in the
        # middle gives 2 eval per side
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
            [9.0, 100.0],
            [10.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(10, dtype=np.int64)
        # eval at indices 1, 3, 6, 8
        mask = np.array(
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            dtype=np.uint8
        )

        imp = self._make_numeric_impurity(1, 1, 2)
        imp.setup(data, indices.copy(), mask, 2)
        gain = imp.compute_best_split(0, 10)

        # Assert — best split gives >= 2 eval per side
        self.assertGreater(
            gain, 0,
            'Split should be accepted when each '
            'child has >= min_eval_samples eval rows'
        )

    def test_only_active_in_evaluation_mode(self):
        """min_eval_samples has no effect in 'both'
        or 'training' modes."""
        # Arrange — same data where evaluation mode
        # would reject
        data = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [6.0, 100.0],
            [7.0, 100.0],
            [8.0, 100.0],
        ], dtype=np.float64)
        indices = np.arange(6, dtype=np.int64)
        mask = np.array(
            [1, 0, 1, 1, 0, 1], dtype=np.uint8
        )

        # Act — SV_BOTH (mode 0) with high threshold
        imp_both = self._make_numeric_impurity(1, 1, 5)
        imp_both.setup(
            data, indices.copy(), mask, 0
        )
        gain_both = imp_both.compute_best_split(0, 6)

        # Act — SV_TRAINING (mode 1) with high threshold
        imp_train = self._make_numeric_impurity(1, 1, 5)
        imp_train.setup(
            data, indices.copy(), mask, 1
        )
        gain_train = imp_train.compute_best_split(0, 6)

        # Assert — both should still find splits
        self.assertGreater(
            gain_both, 0,
            'min_eval_samples should not affect '
            'SV_BOTH mode'
        )
        self.assertGreater(
            gain_train, 0,
            'min_eval_samples should not affect '
            'SV_TRAINING mode'
        )


# ----------------------------------------------------------------------

class MinEvalSamplesEndToEndTest(TestCase):
    """End-to-end tests for ``min_eval_samples``
    via ``JPT.fit()``."""

    def test_fit_with_int_min_eval_samples(self):
        """JPT with int min_eval_samples produces a
        valid tree."""
        # Arrange
        rng = np.random.RandomState(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = x >= 0.5
        data = pd.DataFrame({'x': x, 'y': y})
        mask = np.zeros(n, dtype=np.uint8)
        mask[::2] = 1

        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5,
            min_eval_samples=3
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )

        # Assert
        self.assertGreater(len(jpt.leaves), 0)

    def test_fit_with_float_min_eval_samples(self):
        """JPT with float min_eval_samples resolves
        the fraction against total rows."""
        # Arrange
        rng = np.random.RandomState(42)
        n = 200
        x = rng.uniform(0, 1, n)
        y = x >= 0.5
        data = pd.DataFrame({'x': x, 'y': y})
        mask = np.zeros(n, dtype=np.uint8)
        mask[::2] = 1

        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5,
            min_eval_samples=0.05  # 5% of 200 = 10
        )

        # Act
        jpt.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )

        # Assert
        self.assertGreater(len(jpt.leaves), 0)

    def test_high_threshold_limits_splits(self):
        """A very high min_eval_samples should result in
        fewer leaves than a low one, because many splits
        are rejected for insufficient eval samples."""
        # Arrange
        rng = np.random.RandomState(42)
        n = 400
        x = rng.uniform(0, 1, n)
        y = np.where(x < 0.25, 0.0,
                     np.where(x < 0.5, 1.0,
                              np.where(x < 0.75, 2.0,
                                       3.0)))
        data = pd.DataFrame({'x': x, 'y': y})
        mask = np.zeros(n, dtype=np.uint8)
        mask[::2] = 1

        xvar = NumericVariable('x')
        yvar = NumericVariable('y')

        jpt_low = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5,
            min_eval_samples=2
        )
        jpt_high = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_samples_leaf=5,
            min_eval_samples=80
        )

        # Act
        jpt_low.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )
        jpt_high.fit(
            data,
            multicore=0,
            split_validation_mask=mask,
            split_validation_mode='evaluation'
        )

        # Assert — high threshold should constrain tree
        self.assertGreaterEqual(
            len(jpt_low.leaves),
            len(jpt_high.leaves),
            'Higher min_eval_samples should result '
            'in same or fewer leaves'
        )

    def test_serialization_preserves_min_eval_samples(
            self
    ):
        """min_eval_samples survives JSON
        serialization."""
        # Arrange
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar],
            min_eval_samples=42
        )

        # Act
        json_data = jpt.to_json()
        jpt2 = JPT.from_json(json_data)

        # Assert
        self.assertEqual(jpt2.min_eval_samples, 42)

    def test_default_is_zero(self):
        """min_eval_samples defaults to 0."""
        # Arrange & Act
        xvar = NumericVariable('x')
        yvar = NumericVariable('y')
        jpt = JPT(variables=[xvar, yvar])

        # Assert
        self.assertEqual(jpt.min_eval_samples, 0)

    def test_from_json_missing_key_defaults_zero(self):
        """Deserializing a JSON dict without
        min_eval_samples defaults to 0."""
        # Arrange
        xvar = NumericVariable('x')
        yvar = SymbolicVariable('y', Bool)
        jpt = JPT(
            variables=[xvar, yvar],
            targets=[yvar]
        )
        json_data = jpt.to_json()
        del json_data['min_eval_samples']

        # Act
        jpt2 = JPT.from_json(json_data)

        # Assert
        self.assertEqual(jpt2.min_eval_samples, 0)
