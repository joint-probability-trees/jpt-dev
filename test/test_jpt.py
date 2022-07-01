import json
import pickle
from pprint import pprint
from unittest import TestCase

from jpt.trees import JPT
from jpt.variables import NumericVariable, VariableMap
import numpy as np

class JPTTest(TestCase):

    def setUp(self) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            self.data = pickle.load(f)

    def test_hyperparameter_serialization(self):
        '''Serialization with complete hyperparameters without training'''
        x = NumericVariable('X')
        y = NumericVariable('Y')
        variable_dependencies = VariableMap([(x, [x, y]), (y, [x])])
        jpt = JPT(variables=[x, y],
                  targets=[x, y],
                  min_samples_leaf=.1,
                  min_impurity_improvement=0.1,
                  max_leaves=100,
                  max_depth=10,
                  variable_dependencies=variable_dependencies)

        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

    def test_serialization(self):
        '''(de)serialization of JPTs with training'''
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))

        # pprint(jpt.to_json())

        self.assertIsNone(jpt.root.parent)
        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

    def test_pickle(self):
        '''(de)serialization of JPTs using pickle'''
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        jpt_ = pickle.loads(pickle.dumps(jpt))
        self.assertEqual(jpt, jpt_)

    def test_likelihood(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        probs = jpt.likelihood(self.data.reshape(-1, 1))

    def test_conditional_jpt(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')
        jpt = JPT(variables=[x, y],
                  min_samples_leaf=.05,)
        jpt.learn(self.data.reshape(-1, 2))
        ct = jpt.conditional_jpt(VariableMap(zip([x], [0.5])), keep_evidence=True)
