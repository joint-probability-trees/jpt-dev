import json
import pickle
from types import GeneratorType
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt import NumericType, SymbolicType
from jpt.base.intervals import ContinuousSet
from jpt.distributions import Bool, Numeric, Distribution
from jpt.variables import VariableMap, NumericVariable, SymbolicVariable, Variable, infer_from_dataframe, \
    LabelAssignment, ValueAssignment


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


# ----------------------------------------------------------------------------------------------------------------------

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
        self.assertEqual('A = 2.0 v A = 3.0', A.str({2, 3}, fmt='logic'))
        self.assertEqual('A ∈ {2.0} ∪ {3.0}', A.str({2, 3}, fmt='set'))
        self.assertEqual('2.000 ≤ A ≤ 4.000', A.str({(2, 3), (3, 4)}, fmt='logic'))
        self.assertEqual('A ∈ [2.0,4.0]', A.str({(2, 3), (3, 4)}, fmt='set'))


class DuplicateDomainTest(TestCase):
    '''Test domain functionality of Variable classes.'''

    TEST_DATA = [NumericVariable('A'),
                 NumericVariable('B'),
                 NumericVariable('B'),
                 SymbolicVariable('C', Bool)]

    data1 = {'A': ['one', 'two', 'three', 'four'],
             'B': [68, 74, 77, 78],
             'C': [84, 56, 73, 69],
             'D': [78, 88, 82, 87]}
    DF1 = pd.DataFrame(data1)

    data2 = {'A': ['three', 'six', 'seven', 'four'],
             'B': [5, 4, 3, 2],
             'C': [9, 8, 5, 2],
             'E': [7, 8, 5, 1]}
    DF2 = pd.DataFrame(data2)

    def test_duplicate_dom_symbolic_raise_err(self):
        '''Raise exception when generating duplicate symbolic domains.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)[0].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2)[0].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        self.assertRaises(TypeError, Distribution.type_from_json, v2['domain'])

    def test_duplicate_dom_numeric_raise_err(self):
        '''Raise exception when generating duplicate numeric domains.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)[1].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2)[1].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        self.assertRaises(TypeError, Distribution.type_from_json, v2['domain'])

    def test_duplicate_dom_symbolic_unique(self):
        '''Raise exception when generating duplicate symbolic domains.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[0].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2, unique_domain_names=True)[0].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        t2_ = Distribution.type_from_json(v2['domain'])
        self.assertNotEqual(t1_, t2_)
        self.assertFalse(t1_.equiv(t2_))

    def test_duplicate_dom_numeric_unique(self):
        '''Types get different names when using flag unique_domain_names.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[1].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2, unique_domain_names=True)[1].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        t2_ = Distribution.type_from_json(v2['domain'])
        self.assertNotEqual(t1_, t2_)
        self.assertFalse(t1_.equiv(t2_))

    def test_duplicate_dom_symbolic_unique_identical_data(self):
        '''Even with identical data source, resulting types are neither equal nor equivalent.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[0].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[0].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        t2_ = Distribution.type_from_json(v2['domain'])
        self.assertNotEqual(t1_, t2_)
        self.assertFalse(t1_.equiv(t2_))

    def test_duplicate_dom_numeric_unique_identical_data(self):
        '''Even with identical data source, resulting types are neither equal nor equivalent.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[1].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF1, unique_domain_names=True)[1].to_json()
        t1_ = Distribution.type_from_json(v1['domain'])
        t2_ = Distribution.type_from_json(v2['domain'])
        self.assertNotEqual(t1_, t2_)
        self.assertFalse(t1_.equiv(t2_))

    def test_duplicate_dom_symbolic_excluded_columns(self):
        '''User-created type is used in infer_from_dataframe when setting excluded_columns.'''
        atype = SymbolicType('A_TYPE', ['one', 'two', 'three', 'four'])
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, excluded_columns={'A': atype})
        self.assertEqual(atype, v1[0].domain)
        self.assertTrue(atype.equiv(v1[0].domain))

    def test_duplicate_dom_numeric_excluded_columns(self):
        '''User-created type is used in infer_from_dataframe when setting excluded_columns.'''
        btype = NumericType('B_TYPE', np.array([68, 74, 77, 78]))
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, excluded_columns={'B': btype})
        self.assertEqual(btype, v1[1].domain)
        self.assertTrue(btype.equiv(v1[1].domain))

    def test_duplicate_dom_symbolic_not_excluded_columns(self):
        '''User-created type is not used in infer_from_dataframe and therefore not equal but equivalent.'''
        atype = SymbolicType('A_TYPE', ['one', 'two', 'three', 'four'])
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)
        self.assertNotEqual(atype, v1[0].domain)
        self.assertTrue(atype.equiv(Distribution.type_from_json(v1[0].domain.to_json())))

    def test_duplicate_dom_numeric_not_excluded_columns(self):
        '''User-created type is not used in infer_from_dataframe and therefore not equal but equivalent.'''
        btype = NumericType('B_TYPE', np.array([68, 74, 77, 78]))
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)
        self.assertNotEqual(btype, v1[1].domain)
        self.assertTrue(btype.equiv(Distribution.type_from_json(v1[1].domain.to_json())))
