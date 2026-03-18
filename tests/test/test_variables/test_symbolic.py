import json
from unittest import TestCase

from jpt.distributions import SymbolicType
from jpt.variables import SymbolicVariable, Variable


# ----------------------------------------------------------------------

class SymbolicVariableTest(TestCase):

    def test_impurity_inversion(self):
        """Verify that invert_impurity flag is set correctly."""
        symbolicType = SymbolicType(
            'BlaType',
            labels=['a', 'b', 'c']
        )
        v = SymbolicVariable(
            'var',
            domain=symbolicType,
            invert_impurity=True
        )
        self.assertTrue(v.invert_impurity)

    def test_invert_impurity_survives_json_roundtrip(self):
        """Verify that invert_impurity is preserved after
        JSON serialization and deserialization."""
        # Arrange
        domain = SymbolicType(
            'Color',
            labels=['red', 'green', 'blue']
        )
        original = SymbolicVariable(
            'color',
            domain,
            invert_impurity=True,
        )

        # Act
        data = json.loads(json.dumps(original.to_json()))
        restored = Variable.from_json(data)

        # Assert
        self.assertTrue(
            restored.invert_impurity,
            'invert_impurity should be True after JSON '
            'round-trip but was reset to False'
        )
        self.assertEqual(original, restored)

    def test_hash(self):
        """Verify equal symbolic variables produce equal hashes."""
        # Arrange
        x1 = SymbolicVariable(
            'x',
            domain=SymbolicType('BOOL', ['T', 'F'])
        )
        x2 = SymbolicVariable(
            'x',
            domain=SymbolicType('BOOL', ['T', 'F'])
        )

        # Act
        hash_1 = hash(x1)
        hash_2 = hash(x2)

        # Assert
        self.assertEqual(
            hash_1,
            hash_2
        )
