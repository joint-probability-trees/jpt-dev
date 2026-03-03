import json
from types import GeneratorType
from unittest import TestCase

from jpt.distributions import Bool, Distribution
from jpt.distributions import NumericType, SymbolicType
from jpt.base.intervals import ContinuousSet
from jpt.variables import (
    VariableMap,
    NumericVariable,
    SymbolicVariable,
    LabelAssignment,
    ValueAssignment
)


# ----------------------------------------------------------------------

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

    def test__set_and_get_with_variables(self):
        varmap = VariableMap(variables=VariableMapTest.TEST_DATA)
        varmap['A'] = 'foo'
        varmap['B'] = 'bar'
        varmap['C'] = 'baz'

        self.assertEqual('foo', varmap[VariableMapTest.TEST_DATA[0]])
        self.assertEqual('foo', varmap['A'])

        self.assertEqual('bar', varmap[VariableMapTest.TEST_DATA[1]])
        self.assertEqual('bar', varmap['B'])

        self.assertEqual('baz', varmap[VariableMapTest.TEST_DATA[2]])
        self.assertEqual('baz', varmap['C'])

    def test_raises(self):
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        self.assertRaises(ValueError, varmap.__setitem__, 'C', True)

    def test_equality(self):
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 1
        varmap[C] = 'blub'
        self.assertEqual(varmap, varmap)

    def test_copy(self):
        A, B, C = VariableMapTest.TEST_DATA
        varmap = VariableMap()
        varmap[A] = 1
        varmap[C] = 'blub'
        self.assertEqual(varmap, varmap.copy())

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



# ----------------------------------------------------------------------

class LabelValueAssignmentTest(TestCase):

    def test_label_assignment(self):
        A, B, C = VariableMapTest.TEST_DATA
        a = LabelAssignment()
        self.assertRaises(ValueError, a.__setitem__, 'C', True)
        self.assertRaises(TypeError, a.__setitem__, C, 'Bla')
        self.assertRaises(TypeError, a.__setitem__, A, 'blub')
        a[A] = ContinuousSet(0, 1)
        self.assertEqual(ContinuousSet(0, 1), a['A'])

    def test_value_assignment(self):
        A, B, C = VariableMapTest.TEST_DATA
        dom = SymbolicType('TestType', labels=['zero', 'one', 'two'])
        D = SymbolicVariable('D', domain=dom)
        a = ValueAssignment()
        self.assertRaises(TypeError, a.__setitem__, C, 'Bla')
        self.assertRaises(TypeError, a.__setitem__, D, 'zero')
        self.assertRaises(TypeError, a.__setitem__, D, 'one')
        a[D] = 0
        self.assertEqual(0, a['D'])

    def test_conversion(self):
        A, B, C = VariableMapTest.TEST_DATA
        dom = SymbolicType('TestType', labels=['zero', 'one', 'two'])
        D = SymbolicVariable('D', domain=dom)
        l = LabelAssignment([(A, ContinuousSet(0, 1)),
                             (D, 'one')])
        self.assertEqual(l, l.value_assignment().label_assignment())
        v = l.value_assignment()
        self.assertIsInstance(v, ValueAssignment)
        self.assertEqual(v['A'], ContinuousSet(0, 1))
        self.assertEqual(v['D'], 1)
        l_ = v.label_assignment()
        self.assertIsInstance(l_, LabelAssignment)
        self.assertEqual(l_['A'], ContinuousSet(0, 1))
        self.assertEqual(l_['D'], 'one')

    def test_serialization(self):
        """Test the serialization to json. This is special since sets cannot be serialized to json."""
        dom = SymbolicType('TestType', labels=['zero', 'one', 'two'])
        D = SymbolicVariable('D', domain=dom)
        a = LabelAssignment()
        a[D] = {'zero', 'one'}
        a_json = a.to_json()
        a_json["D"] = set(a_json["D"])
        solution = {"D": set(['zero', 'one'])}
        self.assertEqual(a_json, solution)
