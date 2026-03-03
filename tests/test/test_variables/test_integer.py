import pickle
from unittest import TestCase

from jpt.distributions.univariate import IntegerType
from jpt.variables import IntegerVariable, Variable


# ----------------------------------------------------------------------

class IntegerVariableTest(TestCase):

    dice = IntegerType('Dice', lmin=1, lmax=6)

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
