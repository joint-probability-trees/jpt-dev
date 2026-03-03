from unittest import TestCase

from jpt.distributions import SymbolicType
from jpt.variables import SymbolicVariable


# ----------------------------------------------------------------------

class SymbolicVariableTest(TestCase):

    def test_impurity_inversion(self):
        symbolicType = SymbolicType('BlaType', labels=['a', 'b', 'c'])
        v = SymbolicVariable('var', domain=symbolicType, invert_impurity=True)
        self.assertTrue(v.invert_impurity)

    def test_hash(self):
        # Arrange
        x1 = SymbolicVariable('x', domain=SymbolicType('BOOL', ['T', 'F']))
        x2 = SymbolicVariable('x', domain=SymbolicType('BOOL', ['T', 'F']))

        # Act
        hash_1 = hash(x1)
        hash_2 = hash(x2)

        # Assert
        self.assertEqual(
            hash_1,
            hash_2
        )
