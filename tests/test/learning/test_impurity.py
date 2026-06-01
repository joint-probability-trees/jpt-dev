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
    _compute_var_improvements,
    _compute_gini_improvement,
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

class VarianceImprovementTest(TestCase):
    """Pin the new ``compute_var_improvements`` semantics:
    mean over dependent active targets, bounded in [0, 1]."""

    @staticmethod
    def _dep_row(indices):
        """Pack ``indices`` followed by a -1 sentinel into a contiguous
        SIZE_t row, matching the on-disk dependency-matrix layout."""
        return np.array(list(indices) + [-1], dtype=np.int64)

    def _improvement(self, var_total, var_left, var_right, n_l, n_r, dep):
        return _compute_var_improvements(
            np.ascontiguousarray(var_total, dtype=np.float64),
            np.ascontiguousarray(var_left, dtype=np.float64),
            np.ascontiguousarray(var_right, dtype=np.float64),
            int(n_l),
            int(n_r),
            self._dep_row(dep),
        )

    def test_single_target_reduction_unchanged(self):
        """One dependent target with a perfect split scores 1.0."""
        # Parent var = 0.25, perfect split → both child vars = 0
        result = self._improvement(
            var_total=[0.25],
            var_left=[0.0],
            var_right=[0.0],
            n_l=2,
            n_r=2,
            dep=[0],
        )
        self.assertAlmostEqual(1.0, result, places=10)

    def test_partial_reduction_unchanged(self):
        """One dependent target with a 50% variance reduction scores 0.5."""
        # left+right weighted variance = 0.5 * var_total
        result = self._improvement(
            var_total=[1.0],
            var_left=[0.5],
            var_right=[0.5],
            n_l=5,
            n_r=5,
            dep=[0],
        )
        self.assertAlmostEqual(0.5, result, places=10)

    def test_mean_over_three_targets(self):
        """Three dependent targets, only first reduces fully → mean is 1/3.

        This is the regression the plan calls out — the old
        factorial-weighted update returned 2.0 for the same inputs.
        """
        # Targets 1 and 2 are unchanged by the split (vars stay at parent).
        result = self._improvement(
            var_total=[1.0, 1.0, 1.0],
            var_left=[0.0, 1.0, 1.0],
            var_right=[0.0, 1.0, 1.0],
            n_l=5,
            n_r=5,
            dep=[0, 1, 2],
        )
        self.assertAlmostEqual(1.0 / 3.0, result, places=10)
        self.assertLessEqual(result, 1.0)

    def test_permutation_invariance(self):
        """The mean does not depend on the order targets appear in
        the dependency row."""
        var_total = [1.0, 0.5, 2.0]
        var_left = [0.2, 0.1, 1.0]
        var_right = [0.4, 0.3, 1.5]
        baseline = self._improvement(
            var_total, var_left, var_right, 4, 6, [0, 1, 2]
        )
        for perm in [(2, 0, 1), (1, 2, 0), (2, 1, 0)]:
            permuted = self._improvement(
                var_total, var_left, var_right, 4, 6, list(perm)
            )
            self.assertAlmostEqual(
                baseline, permuted, places=10,
                msg='Result must be invariant under target permutation '
                    '(perm %r differs by %r)' % (perm, permuted - baseline),
            )

    def test_result_bounded_above_by_one(self):
        """For any valid weighted-mean partition, the score stays ≤ 1."""
        rng = np.random.RandomState(1)
        for _ in range(50):
            n = rng.randint(1, 5)
            n_l = int(rng.randint(1, 10))
            n_r = int(rng.randint(1, 10))
            var_total = rng.uniform(0.1, 5.0, n)
            # Pick child variances that satisfy the law of total variance
            # (so the *weighted* combination cannot exceed the parent).
            var_left = rng.uniform(0, 1, n) * var_total
            var_right = rng.uniform(0, 1, n) * var_total
            dep = list(range(n))
            result = self._improvement(
                var_total, var_left, var_right, n_l, n_r, dep
            )
            self.assertLessEqual(result, 1.0 + 1e-9)
            self.assertGreaterEqual(result, -1e-9)

    def test_negative_zero_parent_variance_skipped(self):
        """Float cancellation in ``variances`` can produce a tiny
        negative parent variance for near-constant numeric columns;
        the helper must skip those targets (treat as un-reducible)
        instead of dividing through and blowing up the running mean.
        """
        # Target 0 is a near-constant numeric (numerically tiny-negative
        # variance), target 1 has reducible variance. Without the
        # ``<= 0`` skip, target 0 would dominate with a finite-but-huge
        # contribution.
        result = self._improvement(
            var_total=[-1e-18, 1.0],
            var_left=[0.0, 0.5],
            var_right=[0.0, 0.5],
            n_l=4,
            n_r=4,
            dep=[0, 1],
        )
        self.assertTrue(np.isfinite(result))
        # Only target 1 contributes, with a 0.5 reduction.
        self.assertAlmostEqual(0.5, result, places=10)


# ----------------------------------------------------------------------

class DependentAggregatorTest(TestCase):
    """The aggregator divides by the number of *dependent* targets,
    not the total. Excluding non-dependents from the row leaves the
    per-target scores untouched and the mean tighter."""

    def test_excluded_targets_do_not_affect_mean(self):
        # Two targets; only the first is in the dependency row.
        var_total = np.array([1.0, 1.0], dtype=np.float64)
        var_left = np.array([0.0, 0.7], dtype=np.float64)
        var_right = np.array([0.0, 0.7], dtype=np.float64)

        dep_one = np.array([0, -1], dtype=np.int64)
        dep_both = np.array([0, 1, -1], dtype=np.int64)

        score_one = _compute_var_improvements(
            var_total, var_left, var_right, 4, 4, dep_one
        )
        score_both = _compute_var_improvements(
            var_total, var_left, var_right, 4, 4, dep_both
        )

        # Restricted to the dependent target, the split is perfect.
        self.assertAlmostEqual(1.0, score_one, places=10)
        # With both, the mean halves: (1.0 + 0.3) / 2 = 0.65.
        self.assertAlmostEqual(0.65, score_both, places=10)

    def test_zero_variance_targets_skipped(self):
        # Target 1 has zero parent variance — must not pollute the
        # divisor; only target 0 contributes.
        var_total = np.array([1.0, 0.0], dtype=np.float64)
        var_left = np.array([0.5, 0.0], dtype=np.float64)
        var_right = np.array([0.5, 0.0], dtype=np.float64)
        dep = np.array([0, 1, -1], dtype=np.int64)

        score = _compute_var_improvements(
            var_total, var_left, var_right, 3, 3, dep
        )
        self.assertAlmostEqual(0.5, score, places=10)


# ----------------------------------------------------------------------

class GiniImpurityRawTest(TestCase):
    """Pin the new ``gini_impurity`` output: raw ΣP² − 1 per target,
    no per-partition normaliser and no in-place inversion flip."""

    @staticmethod
    def _make_impurity(domain_size, invert=False):
        """Build an Impurity over one symbolic target with the given
        domain size."""
        labels = ['v%d' % i for i in range(domain_size)]
        ST = SymbolicType('ST_%d' % domain_size, labels=labels)
        x = NumericVariable('x')
        y = SymbolicVariable('y', ST, invert_impurity=invert)
        jpt = JPT(variables=[x, y], targets=[y], features=[x, y])
        return Impurity.from_tree(jpt)

    def test_balanced_two_class(self):
        """Counts ``[3, 3]`` (6 samples) → raw gini = 9/36 + 9/36 − 1 = −0.5."""
        imp = self._make_impurity(2)
        counts = np.array([[3], [3]], dtype=np.int64)
        result = np.zeros(1, dtype=np.float64)
        imp._gini_impurity(counts, 6, result)
        self.assertAlmostEqual(-0.5, result[0], places=10)

    def test_balanced_three_class(self):
        """Counts ``[2, 2, 2]`` → raw gini = 3·(4/36) − 1 = −2/3."""
        imp = self._make_impurity(3)
        counts = np.array([[2], [2], [2]], dtype=np.int64)
        result = np.zeros(1, dtype=np.float64)
        imp._gini_impurity(counts, 6, result)
        self.assertAlmostEqual(-2.0 / 3.0, result[0], places=10)

    def test_pure_partition_is_zero(self):
        """A single non-empty class → raw gini collapses to 0 via the
        ``n_local <= 1`` short-circuit."""
        imp = self._make_impurity(3)
        counts = np.array([[5], [0], [0]], dtype=np.int64)
        result = np.zeros(1, dtype=np.float64)
        imp._gini_impurity(counts, 5, result)
        self.assertEqual(0.0, result[0])

    def test_inversion_no_longer_applied_inplace(self):
        """With ``invert_impurity=True``, the raw output is still the
        un-flipped ΣP² − 1 — inversion now lives in the caller, not
        in ``gini_impurity`` itself."""
        imp = self._make_impurity(2, invert=True)
        counts = np.array([[3], [3]], dtype=np.int64)
        result = np.zeros(1, dtype=np.float64)
        imp._gini_impurity(counts, 6, result)
        # Old behaviour would have produced 1 − (8/9 · −0.5) ≈ 1.444;
        # new behaviour must keep the raw ≤ 0 value.
        self.assertAlmostEqual(-0.5, result[0], places=10)

    def test_no_per_partition_normaliser(self):
        """Re-introducing the old ``1/(1/n_local − 1)`` normaliser
        would scale the raw output by ``-2`` for any 2-class partition
        with both classes present. Confirm the output equals raw
        ΣP² − 1, not the rescaled form."""
        imp = self._make_impurity(4)
        counts = np.array([[2], [3], [0], [0]], dtype=np.int64)
        result = np.zeros(1, dtype=np.float64)
        imp._gini_impurity(counts, 5, result)
        # ΣP² = (4 + 9)/25 = 0.52 → raw = -0.48; old normaliser would
        # divide by (1/2 − 1) = −0.5 giving +0.96.
        self.assertAlmostEqual(-0.48, result[0], places=10)


# ----------------------------------------------------------------------

class DependencyMatrixSelfExclusionTest(TestCase):
    """Item 2's self-exclusion replaces the old ``skip_idx`` plumbing:
    a variable that is both feature and target must not appear in its
    own dependency rows."""

    def test_numeric_and_symbolic_rows_exclude_self(self):
        # Mixed tree: A symbolic + B numeric + C symbolic, all are
        # features and all are targets, with the default fully-connected
        # dependency map.
        AT = SymbolicType('AT_excl', ['a0', 'a1'])
        CT = SymbolicType('CT_excl', ['c0', 'c1', 'c2'])
        A = SymbolicVariable('A', AT)
        B = NumericVariable('B')
        C = SymbolicVariable('C', CT)
        jpt = JPT(variables=[A, B, C], targets=[A, B, C], features=[A, B, C])
        imp = Impurity.from_tree(jpt)

        num_mat = np.asarray(imp.numeric_dependency_matrix)
        sym_mat = np.asarray(imp.symbolic_dependency_matrix)

        # Helper: indices in the row before the -1 sentinel.
        def deps(row):
            row = list(row)
            if -1 in row:
                row = row[: row.index(-1)]
            return set(row)

        # A's numeric dep row covers [B_num_idx] (only numeric target);
        # A is symbolic so no self-collision in numeric_dependency_matrix
        # is possible — but the symbolic row must omit A's own slot.
        a_var_idx = jpt.variables.index(A)
        b_var_idx = jpt.variables.index(B)
        c_var_idx = jpt.variables.index(C)

        # numeric_vars / symbolic_vars store variable indices in
        # jpt.variables. Find each target's local index.
        a_sym_local = 0  # A is the first symbolic target
        c_sym_local = 1
        b_num_local = 0

        # A row: numeric dep = {B}, symbolic dep = {C} (A excluded).
        self.assertEqual({b_num_local}, deps(num_mat[a_var_idx]))
        self.assertEqual({c_sym_local}, deps(sym_mat[a_var_idx]))
        self.assertNotIn(a_sym_local, deps(sym_mat[a_var_idx]))

        # B row: numeric dep = {} (B excluded as self), symbolic = {A, C}.
        self.assertEqual(set(), deps(num_mat[b_var_idx]))
        self.assertEqual({a_sym_local, c_sym_local}, deps(sym_mat[b_var_idx]))
        self.assertNotIn(b_num_local, deps(num_mat[b_var_idx]))

        # C row: numeric dep = {B}, symbolic dep = {A} (C excluded).
        self.assertEqual({b_num_local}, deps(num_mat[c_var_idx]))
        self.assertEqual({a_sym_local}, deps(sym_mat[c_var_idx]))
        self.assertNotIn(c_sym_local, deps(sym_mat[c_var_idx]))


# ----------------------------------------------------------------------

class GiniImprovementTest(TestCase):
    """Behaviour of the new ``compute_gini_improvement`` helper."""

    @staticmethod
    def _raw_gini(counts):
        """Raw ΣP² − 1 for a histogram."""
        counts = np.asarray(counts, dtype=np.float64)
        n = counts.sum()
        return float((counts * counts).sum() / (n * n) - 1.0)

    def test_three_equal_partitions(self):
        """Parent split into three identical children → no reduction."""
        g_p = self._raw_gini([4, 4, 4])  # parent 12
        g_l = self._raw_gini([2, 2, 2])
        g_r = self._raw_gini([2, 2, 2])
        result = _compute_gini_improvement(g_p, g_l, g_r, 6, 6)
        self.assertAlmostEqual(0.0, result, places=10)

    def test_perfect_split(self):
        """Pure children → full reduction = 1.0."""
        g_p = self._raw_gini([5, 5])
        g_l = self._raw_gini([5, 0])
        g_r = self._raw_gini([0, 5])
        result = _compute_gini_improvement(g_p, g_l, g_r, 5, 5)
        self.assertAlmostEqual(1.0, result, places=10)

    def test_worked_example_50_10_10_10(self):
        """Parent 50/10/10/10 → left 50/10, right 10/10 → ≈ 0.407.

        Today's per-partition normaliser would emit ≈ 0.111 here.
        """
        g_p = self._raw_gini([50, 10, 10, 10])
        g_l = self._raw_gini([50, 10, 0, 0])
        g_r = self._raw_gini([0, 0, 10, 10])
        result = _compute_gini_improvement(g_p, g_l, g_r, 60, 20)
        self.assertAlmostEqual(0.4074, result, places=3)

    def test_random_grid_in_unit_interval(self):
        """For arbitrary histograms with weighted-average law of total
        variance, the score stays inside ``[0, 1]``."""
        rng = np.random.RandomState(7)
        for _ in range(100):
            k = rng.randint(2, 6)
            n_l = int(rng.randint(2, 50))
            n_r = int(rng.randint(2, 50))
            left = rng.multinomial(n_l, np.full(k, 1.0 / k))
            right = rng.multinomial(n_r, np.full(k, 1.0 / k))
            parent = left + right
            if parent.sum() == 0:
                continue
            g_p = self._raw_gini(parent)
            if g_p == 0:
                continue
            g_l = self._raw_gini(left) if left.sum() > 0 else 0.0
            g_r = self._raw_gini(right) if right.sum() > 0 else 0.0
            result = _compute_gini_improvement(g_p, g_l, g_r, n_l, n_r)
            self.assertLessEqual(result, 1.0 + 1e-9)
            self.assertGreaterEqual(result, -1e-9)


# ----------------------------------------------------------------------

class InvertedImpurityBoundedTest(TestCase):
    """Inversion now applies to the per-target improvement, not the
    raw gini, so the bounded ``[0, 1]`` invariant survives."""

    def test_pure_children_with_inverted_target(self):
        """Parent ``G_p = 0.9`` (≈ −0.5 raw), both children pure →
        ``score = 1 − 1 = 0``, never the −9 the old code produced."""
        # Construct a tree whose only target is symbolic with
        # ``invert_impurity=True``. Feed a clean two-class split.
        InvType = SymbolicType('Inv', labels=['a', 'b'])
        x = NumericVariable('x')
        y = SymbolicVariable('y', InvType, invert_impurity=True)
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y': ['a', 'a', 'a', 'b', 'b', 'b'],
        })
        jpt = JPT(variables=[x, y], targets=[y], features=[x, y])
        data = preprocess_data(jpt, df)
        imp = Impurity.from_tree(jpt)
        imp.min_samples_leaf = 1
        imp.setup(
            data.values,
            np.arange(data.shape[0], dtype=np.int64),
        )
        gain = imp.compute_best_split(0, data.shape[0])

        # Inverted target + a discriminative split = no preference for
        # the split (score 0 after inversion). The numeric features can
        # still drive a split, but for this all-symbolic-target tree
        # the best gain must not be the runaway negative the old code
        # produced.
        self.assertGreaterEqual(gain, -1e-9)
        self.assertLessEqual(gain, 1.0 + 1e-9)


# ----------------------------------------------------------------------

class SymmetricModalityWeightTest(TestCase):
    """The per-variable ``w_num_local`` collapses to the single
    surviving modality when the other has no reducible impurity."""

    def test_all_symbolic_pure_numeric_perfect_split(self):
        """One numeric target with a perfect split + one already-pure
        symbolic target → impurity improvement should be ≈ 1.0
        (not the old half-weight 0.5)."""
        Y2T = SymbolicType('Y2T', labels=['a'])  # single value → pure
        x = NumericVariable('x')
        y1 = NumericVariable('y1')
        # Symbolic target whose only legal value is 'a' — Patrons-style
        # degenerate target. The symbolic side has nothing to learn.
        y2 = SymbolicVariable('y2', Y2T)
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y1': [0.0, 0.0, 0.0, 100.0, 100.0, 100.0],
            'y2': ['a'] * 6,
        })
        jpt = JPT(
            variables=[x, y1, y2],
            targets=[y1, y2],
            features=[x],
        )
        data = preprocess_data(jpt, df)
        imp = Impurity.from_tree(jpt)
        imp.min_samples_leaf = 1
        imp.setup(
            data.values,
            np.arange(data.shape[0], dtype=np.int64),
        )
        gain = imp.compute_best_split(0, data.shape[0])

        # Numeric side perfect split → 1.0 mean reduction. Symbolic
        # side is pure → contributes nothing. With dynamic
        # ``w_num_local`` the active count of sym targets is 0, so
        # the weight collapses to the numeric side and we get 1.0.
        self.assertAlmostEqual(1.0, gain, places=6)

    def test_all_numeric_pure_symbolic_perfect_split(self):
        """One numeric target that is *already constant* + one
        symbolic target with a perfect split → the symbolic side
        carries the whole score."""
        YBoolT = SymbolicType('YBoolT', labels=['a', 'b'])
        x = NumericVariable('x')
        y1 = NumericVariable('y1')   # constant → variance 0 → pure
        y2 = SymbolicVariable('y2', YBoolT)
        df = pd.DataFrame({
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y1': [7.0] * 6,
            'y2': ['a', 'a', 'a', 'b', 'b', 'b'],
        })
        jpt = JPT(
            variables=[x, y1, y2],
            targets=[y1, y2],
            features=[x],
        )
        data = preprocess_data(jpt, df)
        imp = Impurity.from_tree(jpt)
        imp.min_samples_leaf = 1
        imp.setup(
            data.values,
            np.arange(data.shape[0], dtype=np.int64),
        )
        gain = imp.compute_best_split(0, data.shape[0])

        # Numeric pure → contributes 0 with active count 0. Symbolic
        # side fully separates → 1.0.
        self.assertAlmostEqual(1.0, gain, places=6)

    def test_mixed_modality_combines_as_unweighted_mean(self):
        """One numeric + one symbolic active target → the combined
        score must equal ``0.5·num_score + 0.5·sym_score``.

        Pins the symmetric reformulation: the per-modality means are
        themselves weighted by the per-variable active counts, not by
        the tree-wide n_num_vars / n_vars ratio the old aggregator
        used. With one active target per modality, both weights are
        ½ and the combined score is the unweighted per-target mean.
        """
        YBoolT = SymbolicType('YBoolT_mixed', labels=['a', 'b'])
        x = NumericVariable('x')
        y_num = NumericVariable('y_num')
        y_sym = SymbolicVariable('y_sym', YBoolT)
        # 6 rows, balanced a/b in y_sym; y_num jumps at the same
        # midpoint y_sym mixes around. The best split is at x=3.5:
        #   num_score = 1.0 (var-perfect)
        #   sym_score = 1/9 (parent {a:3, b:3} → g_p=−0.5;
        #                   left {a:2, b:1}, right {a:1, b:2} →
        #                   each child g = −4/9;
        #                   imp_orig = (−0.5 − 0.5·−4/9 − 0.5·−4/9)
        #                              / −0.5 = (1/18)/(0.5) = 1/9)
        # Both modalities contribute one active target → combined
        # = 0.5·1.0 + 0.5·(1/9) = 5/9.
        df = pd.DataFrame({
            'x':     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'y_num': [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            'y_sym': ['a', 'a', 'b', 'a', 'b', 'b'],
        })
        jpt = JPT(
            variables=[x, y_num, y_sym],
            targets=[y_num, y_sym],
            features=[x],
        )
        data = preprocess_data(jpt, df)
        imp = Impurity.from_tree(jpt)
        imp.min_samples_leaf = 1
        imp.setup(
            data.values,
            np.arange(data.shape[0], dtype=np.int64),
        )
        gain = imp.compute_best_split(0, data.shape[0])
        self.assertAlmostEqual(5.0 / 9.0, gain, places=6)


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
