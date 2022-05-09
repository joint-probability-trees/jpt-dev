import os
import pickle
import unittest

import numpy as np

from jpt import SymbolicVariable, JPT
from jpt.learning.distributions import Bool


class JointProbabilityTreesMPE(unittest.TestCase):

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

        cls.jpt = JPT(variables=[cls.E, cls.B, cls.A, cls.M, cls.J], min_impurity_improvement=0).learn(rows=data)
        cls.jpt.learn(rows=cls.data)

    def test_infer_alarm_given_mary(self):
        q = {'Alarm': True}
        e = {'Mary': True}
        res = JointProbabilityTreesMPE.jpt.infer(q, e)
        self.assertAlmostEqual(0.950593, res.result, places=5)

    def test_infer_alarm(self):
        q = {'Alarm': True}
        e = {}
        res = JointProbabilityTreesMPE.jpt.infer(q, e)
        self.assertAlmostEqual(0.210199, res.result, places=5)

    def test_likelihood_discrete(self):
        probs = JointProbabilityTreesMPE.jpt.likelihood(self.data)
        self.assertGreater(sum(np.log(probs)), -np.inf)
