import pickle
from unittest import TestCase

from jpt.base.intervals import IntSet, UnionSet
from jpt.base.utils import format_path
from jpt.distributions.univariate import IntegerType
from jpt.variables import IntegerVariable, Variable


# ----------------------------------------------------------------------

class IntegerVariableTest(TestCase):

    dice = IntegerType('Dice', lmin=1, lmax=400)

    def test_hash(self):
        """Verify equal integer variables produce equal hashes."""
        # Arrange
        toss1 = IntegerVariable('Toss', domain=self.dice)
        toss2 = IntegerVariable('Toss', domain=self.dice)
        baz = IntegerVariable('baz', domain=self.dice)

        # Act
        print(hasattr(toss1, '__hash__'), toss1.__hash__)
        hash_1 = hash(toss1)
        hash_2 = hash(toss2)
        hash_3 = hash(baz)

        # Assert
        self.assertEqual(hash_1, hash_2)
        self.assertNotEqual(hash_2, hash_3)

    def test_serialization(self):
        """Verify JSON round-trip serialization of IntegerVariable."""
        toss = IntegerVariable('Toss', domain=self.dice)
        self.assertEqual(toss, Variable.from_json(toss.to_json()))

    def test_pickle(self):
        """Verify pickle round-trip serialization of IntegerVariable."""
        toss = IntegerVariable('Toss', domain=self.dice)
        self.assertEqual(toss, pickle.loads(pickle.dumps(toss)))

    def test_string_representation(self):
        """Verify str() rendering for scalars, IntSets, and UnionSets."""
        toss = IntegerVariable('Toss', domain=self.dice)

        # Scalar
        self.assertEqual('Toss = 3', toss.str(3, fmt='set'))
        self.assertEqual('Toss = 3', toss.str(3, fmt='logic'))

        # IntSet — both formats
        intset = IntSet(2, 5)
        self.assertEqual(f'Toss ∈ {intset}', toss.str(intset, fmt='set'))
        self.assertEqual('2 ≤ Toss ≤ 5', toss.str(intset, fmt='logic'))

        # Singleton IntSet collapses to `name = value` in logic mode
        self.assertEqual('Toss = 3', toss.str(IntSet(3, 3), fmt='logic'))

        # UnionSet — the reproduction case from the bug report
        us = UnionSet([IntSet(0, 32), IntSet(127, 365)])
        set_str = toss.str(us, fmt='set')
        self.assertEqual(f'Toss ∈ {us}', set_str)
        self.assertIn('∪', set_str)
        self.assertEqual(
            '0 ≤ Toss ≤ 32 ∨ 127 ≤ Toss ≤ 365',
            toss.str(us, fmt='logic'),
        )

        # Empty UnionSet renders as `name ∈ ∅`
        self.assertEqual('Toss ∈ ∅', toss.str(UnionSet([]), fmt='set'))

        # Unknown fmt raises ValueError (matches NumericVariable.str behaviour)
        self.assertRaises(ValueError, toss.str, us, fmt='nonsense')

    def test_format_path_with_unionset(self):
        """``format_path`` must not raise on a UnionSet for an integer var."""
        # Regression test for the Unsatisfiability-masked-by-TypeError bug:
        # ``JPT.posterior`` calls ``format_path(evidence, fmt='logic')`` to build
        # the ``Unsatisfiability`` message.  Before the fix, this raised
        # ``TypeError`` on any ``UnionSet`` evidence for an integer variable,
        # masking the real ``Unsatisfiability``.
        toss = IntegerVariable('Toss', domain=self.dice)
        path = {toss: UnionSet([IntSet(0, 32), IntSet(127, 365)])}
        rendered = format_path(path)
        self.assertIn('Toss', rendered)
        self.assertIn(' ∨ ', rendered)
