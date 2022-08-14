import json
import pickle
from types import GeneratorType
from unittest import TestCase

from jpt.distributions import Bool, Numeric
from jpt.variables import VariableMap, NumericVariable, SymbolicVariable, Variable


class VariableMapTest(TestCase):
    '''
    Test the basic functionality of the ``VariableMap`` class.
    '''

    TEST_DATA = [NumericVariable('A'),
                 NumericVariable('B'),
                 SymbolicVariable('C', Bool)]

    def test_set_and_get(self):
        '''Basic set and get functionality'''
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'

        self.assertEqual('foo', varmap[A])
        self.assertEqual('foo', varmap['A'])

        self.assertEqual('bar', varmap[B])
        self.assertEqual('bar', varmap['B'])

        self.assertEqual('baz', varmap[C])
        self.assertEqual('baz', varmap['C'])

    def test_hash(self):
        '''Custom has value calculation.'''
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'
        varmap2 = VariableMap()
        varmap2[A] = 'foo'
        varmap2[B] = 'bar'
        varmap2[C] = 'baz'
        self.assertEqual(hash(varmap), hash(varmap2))
        varmap2[C] = 'ba'
        self.assertNotEqual(hash(varmap), hash(varmap2))

    def test_iadd_isub_operators(self):
        A, B, C = VariableMapTest.TEST_DATA
        D = SymbolicVariable('D', domain=None)
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'
        varmap2 = VariableMap()
        varmap2[A] = 'foooob'
        varmap2[D] = 'daz'
        varmap += varmap2
        self.assertEqual(VariableMap([(A, 'foooob'), (B, 'bar'), (C, 'baz'), (D, 'daz')]),
                         varmap)
        self.assertRaises(TypeError, varmap.__iadd__, 'bla')
        varmap -= 'A'
        self.assertEqual(VariableMap([(B, 'bar'), (C, 'baz'), (D, 'daz')]), varmap)
        varmap -= varmap2
        self.assertEqual(VariableMap([(B, 'bar'), (C, 'baz')]), varmap)

    def test_iteration(self):
        '''Iteration over map elements'''
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'

        self.assertEqual([(A, 'foo'), (B, 'bar'), (C, 'baz')], list(varmap.items()))
        self.assertIsInstance(varmap.items(), GeneratorType)

        self.assertEqual([A, B, C], list(varmap.keys()))
        self.assertIsInstance(varmap.keys(), GeneratorType)

        self.assertEqual(['foo', 'bar', 'baz'], list(varmap.values()))
        self.assertIsInstance(varmap.values(), GeneratorType)

    def test_removal_containment(self):
        '''Removal and containment of map elements'''
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'

        self.assertTrue('A' in varmap)
        self.assertTrue(A in varmap)

        self.assertTrue('B' in varmap)
        self.assertTrue(B in varmap)

        self.assertTrue('C' in varmap)
        self.assertTrue(C in varmap)

        # Remove elements
        del varmap[A]
        self.assertFalse('A' in varmap)
        self.assertFalse(A in varmap)

        del varmap['B']
        self.assertFalse('B' in varmap)
        self.assertFalse(B in varmap)

    def test_serialization(self):
        '''(De)serialization of a VariableMap'''
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 'foo'
        varmap[B] = 'bar'
        varmap[C] = 'baz'
        self.assertEqual(varmap, VariableMap.from_json([A, B, C], json.loads(json.dumps(varmap.to_json()))))


# ----------------------------------------------------------------------------------------------------------------------

class VariableTest(TestCase):
    '''Test basic functionality of Variable classes.'''

    TEST_DATA = [NumericVariable('A'),
                 NumericVariable('B'),
                 SymbolicVariable('C', Bool)]

    def test_hash(self):
        '''Custom has value calculation.'''
        h1 = hash(NumericVariable('bar'))
        h2 = hash(SymbolicVariable('baz', domain=Bool))
        h3 = hash(NumericVariable('bar'))
        self.assertEqual(h1, h3)
        self.assertNotEqual(h1, h2)

    def test_serialization(self):
        '''Test (de)serialization of Variable classes'''
        A, B, C = VariableTest.TEST_DATA
        self.assertEqual(A, Variable.from_json(json.loads(json.dumps(A.to_json()))))
        self.assertEqual(B, Variable.from_json(json.loads(json.dumps(B.to_json()))))
        self.assertEqual(C, Variable.from_json(json.loads(json.dumps(C.to_json()))))

    def test_pickle(self):
        '''Test (de)serialization of Variable classes'''
        A, B, C = VariableTest.TEST_DATA
        self.assertEqual(A, pickle.loads(pickle.dumps(A)))
        self.assertEqual(B, pickle.loads(pickle.dumps(B)))
        self.assertEqual(C, pickle.loads(pickle.dumps(C)))

    def test_string_representation(self):
        A, B, C = VariableTest.TEST_DATA
        self.assertEqual('C = True', C.str(True, fmt='logic'))
        self.assertEqual('C = True', C.str(True, fmt='set'))
        self.assertIn(C.str({True, False}, fmt='logic'),
                      ['C = False v C = True', 'C = True v C = False'])
        self.assertIn(C.str({True, False}, fmt='set'),
                      ['C ∈ {False, True}', 'C ∈ {True, False}'])
