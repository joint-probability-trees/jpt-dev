import json
import os
import pickle
from unittest import TestCase

import pandas as pd
from dnutils import out

from jpt.base.errors import Unsatisfiability
from jpt.trees import JPT
from jpt.variables import NumericVariable, VariableMap, infer_from_dataframe


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

    def learn(self):
        trees = []
        for _ in range(1000):
            out(_)
            var = NumericVariable('X')
            jpt = JPT([var], min_samples_leaf=.1)
            jpt.learn(self.data.reshape(-1, 1))
            trees.append(jpt)
        return trees

    def test_likelihood(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        probs = jpt.likelihood(self.data.reshape(-1, 1))
        #TODO: add test condition

    def test_unsatisfiability(self):
        df = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))
        jpt = JPT(variables=infer_from_dataframe(df), targets=['WillWait'], min_samples_leaf=1)
        jpt.fit(df)
        self.assertRaises(Unsatisfiability,
                          jpt.posterior,
                          evidence={'WillWait': False, 'Patrons': 'Some'},
                          fail_on_unsatisfiability=True)
        self.assertIsNone(jpt.posterior(evidence={'WillWait': False, 'Patrons': 'Some'},
                                        fail_on_unsatisfiability=False))

        try:
            jpt.posterior(evidence={'WillWait': False, 'Patrons': 'Some'},
                          report_inconsistencies=True)
        except Unsatisfiability as e:
            self.assertEqual({VariableMap([(jpt.varnames['WillWait'], {False})]): 1},
                             e.reasons)
        else:
            raise RuntimeError('jpt.posterior did not raise Unsatisfiability.')
