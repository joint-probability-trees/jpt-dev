import gc
import json
import pickle
from unittest import TestCase

from dnutils import out

from jpt.trees import JPT, SumJPT, ProductJPT
from jpt.variables import NumericVariable, VariableMap
from jpt.base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
import matplotlib.pyplot as plt

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

    def test_conditional_jpt_hard_evidence(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')
        jpt = JPT(variables=[x, y],
                  min_samples_leaf=.05,)
        jpt.learn(self.data.reshape(-1, 2))
        evidence = VariableMap()
        evidence[x] = 0.5
        ct = jpt.conditional_jpt(evidence, keep_evidence=True)
        self.assertEqual(len(ct.leaves), 2)

    def test_conditional_jpt_soft_evidence(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')
        jpt = JPT(variables=[x, y],
                  min_samples_leaf=.05, )
        evidence = VariableMap()
        evidence[y] = ContinuousSet(0.2, 0.5)
        jpt.learn(self.data.reshape(-1, 2))

        ct = jpt.conditional_jpt(evidence, keep_evidence=True)
        r = jpt.expectation([x], evidence)
        r_ = ct.expectation([x], VariableMap())
        self.assertAlmostEqual(r[x].result, r_[x].result, delta=0.01)

    def test_marginal(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')

        evidence = VariableMap()
        evidence[y] = ContinuousSet(0.2, 0.5)

        jpt = JPT(variables=[x, y],
                  min_samples_leaf=.05, )
        jpt.learn(self.data.reshape(-1, 2))

        mt = jpt.marginal_jpt([x])

        self.assertEqual(len(mt.leaves), 1)

    def test_jpt_like(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')

        evidence = VariableMap()
        evidence[y] = ContinuousSet(0.2, 0.5)

        jpt = JPT(variables=[x, y],
                  min_samples_leaf=.05, )
        jpt.learn(self.data.reshape(-1, 2))

        result = jpt.independent_marginals([x, y], evidence)

        sjpt = SumJPT([x,y], [jpt, jpt])
        s_result = sjpt.independent_marginals([x, y], evidence)

        for v in [x, y]:
            self.assertAlmostEqual(result.distributions[v].kl_divergence(s_result.distributions[v]), 0)

