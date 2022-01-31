from types import GeneratorType
from unittest import TestCase

try:
    from jpt.learning.distributions import Bool
    from jpt.variables import VariableMap, NumericVariable, SymbolicVariable
except ModuleNotFoundError:
    import pyximport
    pyximport.install()


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
