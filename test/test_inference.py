import os
import pickle
import unittest

import numpy as np

from jpt import SymbolicVariable, JPT, NumericVariable
from jpt.base.intervals import ContinuousSet, RealSet, EXC, INC
from jpt.distributions import Bool


class JPTInferenceSymbolic(unittest.TestCase):

    data = None
    jpt = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.E = SymbolicVariable('Earthquake', Bool)  # .02
        cls.B = SymbolicVariable('Burglary', Bool)  # Bool(.01)
        cls.A = SymbolicVariable('Alarm', Bool)
        cls.M = SymbolicVariable('MaryCalls', Bool)
        cls.J = SymbolicVariable('JohnCalls', Bool)

        with open(os.path.join('..', 'examples', 'data', 'alarm.pkl'), 'rb') as f:
            cls.data = np.array(pickle.load(f))

        cls.jpt = JPT(variables=[cls.E, cls.B, cls.A, cls.M, cls.J],
                      min_impurity_improvement=0).learn(rows=cls.data)
        cls.jpt.learn(rows=cls.data)

    def test_infer_alarm_given_mary(self):
        q = {'Alarm': True}
        e = {'MaryCalls': True}
        res = JPTInferenceSymbolic.jpt.infer(q, e)
        self.assertAlmostEqual(0.950593, res, places=5)

    def test_infer_alarm(self):
        q = {'Alarm': True}
        e = {}
        res = JPTInferenceSymbolic.jpt.infer(q, e)
        self.assertAlmostEqual(0.210199, res, places=5)

    def test_infer_alarm_evidence_disjunction_symbolic(self):
        q = {'Alarm': True}
        # This tautological evidence must result in the same posterior as the empty evidence
        e = {'MaryCalls': {True, False}}
        res = JPTInferenceSymbolic.jpt.infer(q, e)
        self.assertAlmostEqual(0.210199, res, places=5)

    def test_likelihood_discrete(self):
        probs = JPTInferenceSymbolic.jpt.likelihood(self.data)
        self.assertGreater(sum(np.log(probs)), -np.inf)


class JPTInferenceNumeric(unittest.TestCase):

    def setUp(self) -> None:
        with open(os.path.join('resources', 'gaussian_100.dat'), 'rb') as f:
            self.data = pickle.load(f)
            x = NumericVariable('x')
            self.jpt = JPT(variables=[x])
            self.jpt.fit(self.data.reshape(-1, 1))

    def test_realset_evidence(self):
        r1 = self.jpt.infer(query={'x': RealSet(['[-1,0.5]', '[1,inf['])})
        r2 = self.jpt.infer(query={'x': ContinuousSet(.5, 1, EXC, INC)})
        self.assertAlmostEqual(r1, 1 - r2, places=10)
