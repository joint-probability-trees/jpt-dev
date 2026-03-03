import json
import pickle
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt.distributions import (
    NumericType,
    SymbolicType,
    Bool,
    Distribution
)
from jpt.distributions.univariate import IntegerType
from jpt.variables import (
    NumericVariable,
    SymbolicVariable,
    Variable,
    infer_from_dataframe
)


# ----------------------------------------------------------------------

class VariableTest(TestCase):
    '''Test basic functionality of Variable classes.'''

    TEST_DATA = [
        NumericVariable('A'),
        NumericVariable('B'),
        SymbolicVariable('C', Bool)
    ]

    def test_hash(self):
        '''Custom hash value calculation.'''
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
        self.assertIn(
            C.str({True, False}, fmt='logic'),
            {'C = False ∨ C = True', 'C = True ∨ C = False'}
        )
        self.assertIn(C.str({True, False}, fmt='set'),
                      ['C ∈ {False, True}', 'C ∈ {True, False}'])
        self.assertEqual('A = 2.0 ∨ A = 3.0', A.str({2, 3}, fmt='logic'))
        self.assertEqual('A ∈ {2.0} ∪ {3.0}', A.str({2, 3}, fmt='set'))
        self.assertEqual('2.000 ≤ A ≤ 4.000', A.str({(2, 3), (3, 4)}, fmt='logic'))
        self.assertEqual('A ∈ [2.0,4.0]', A.str({(2, 3), (3, 4)}, fmt='set'))



# ----------------------------------------------------------------------

class NumericVariableTest(TestCase):

    def test_hash(self):
        # Arrange
        x1 = NumericVariable('x')
        x2 = NumericVariable('x')

        # Act
        hash_1 = hash(x1)
        hash_2 = hash(x2)

        # Assert
        self.assertEqual(
            hash_1,
            hash_2
        )



# ----------------------------------------------------------------------

class DuplicateDomainTest(TestCase):
    '''Test domain functionality of Variable classes.'''

    TEST_DATA = [
        NumericVariable('A'),
        NumericVariable('B'),
        NumericVariable('B'),
        SymbolicVariable('C', Bool)
    ]

    data1 = {
        'A': ['one', 'two', 'three', 'four'],
        'B': [68., 74., 77., 78.],
        'C': [84., 56., 73., 69.],
        'D': [78., 88., 82., 87.]
    }
    DF1 = pd.DataFrame(data1)

    data2 = {
        'A': ['three', 'six', 'seven', 'four'],
        'B': [5., 4., 3., 2.],
        'C': [9., 8., 5., 2.],
        'E': [7., 8., 5., 1.]
    }
    DF2 = pd.DataFrame(data2)

    def test_duplicate_dom_symbolic_not_raise_err(self):
        '''Raise exception when generating duplicate symbolic domains.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)[0].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2)[0].to_json()
        t1 = Distribution.type_from_json(v1['domain'])
        t2 = Distribution.type_from_json(v2['domain'])
        self.assertEqual(t1.__name__, t2.__name__)
        self.assertNotEqual(t1.labels, t2.labels)

    def test_duplicate_dom_numeric_not_raise_err(self):
        '''Raise exception when generating duplicate numeric domains.'''
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)[1].to_json()
        v2 = infer_from_dataframe(DuplicateDomainTest.DF2)[1].to_json()
        t1 = Distribution.type_from_json(v1['domain'])
        t2 = Distribution.type_from_json(v2['domain'])
        self.assertEqual(t1.__name__, t2.__name__)
        self.assertNotEqual(id(t1), id(t2))

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
        atype = SymbolicType('A_TYPE_S', ['one', 'two', 'three', 'four'])
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, excluded_columns={'A': atype})
        self.assertEqual(atype, v1[0].domain)
        self.assertTrue(atype.equiv(v1[0].domain))

    def test_duplicate_dom_numeric_excluded_columns(self):
        '''User-created type is used in infer_from_dataframe when setting excluded_columns.'''
        btype = NumericType('B_TYPE_N', np.array([68, 74, 77, 78], dtype=np.float64))
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1, excluded_columns={'B': btype})
        self.assertEqual(btype, v1[1].domain)
        self.assertTrue(btype.equiv(v1[1].domain))

    def test_duplicate_dom_symbolic_not_excluded_columns(self):
        '''User-created type is not used in infer_from_dataframe and therefore not equal but equivalent.'''
        atype = SymbolicType('A_TYPE_S', ['one', 'two', 'three', 'four'])
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)
        self.assertNotEqual(atype, v1[0].domain)
        self.assertTrue(atype.equiv(Distribution.from_json(v1[0].domain.to_json())))

    def test_duplicate_dom_numeric_not_excluded_columns(self):
        '''User-created type is not used in infer_from_dataframe and therefore not equal but equivalent.'''
        btype = NumericType('B_TYPE_N', np.array([68, 74, 77, 78], dtype=np.float64))
        v1 = infer_from_dataframe(DuplicateDomainTest.DF1)
        self.assertNotEqual(btype, v1[1].domain)
        print(btype, v1[1].domain)
        self.assertTrue(btype.equiv(Distribution.from_json(v1[1].domain.to_json())))
