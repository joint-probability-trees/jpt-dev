import os
import pickle
import unittest

import numpy as np

try:
    from jpt.learning.distributions import Bool
    from jpt.trees import JPT
    from jpt.variables import SymbolicVariable
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
    from jpt.learning.distributions import Bool
    from jpt.trees import JPT
    from jpt.variables import SymbolicVariable


class JointProbabilityTreesMPE(unittest.TestCase):

    def setUp(self):
        self.E = SymbolicVariable('Earthquake', Bool)  # .02
        self.B = SymbolicVariable('Burglary', Bool)  # Bool(.01)
        self.A = SymbolicVariable('Alarm', Bool)
        self.M = SymbolicVariable('MaryCalls', Bool)
        self.J = SymbolicVariable('JohnCalls', Bool)

        data = []
        f = os.path.join('../' 'examples', 'data', 'alarm.pkl')
        with open(f, 'rb') as fi:
            data = np.array(pickle.load(fi))

        self.jpt = JPT(variables=[self.E, self.B, self.A, self.M, self.J], min_impurity_improvement=0)
        self.jpt.learn(rows=data)

    def test_infer_alarm_given_mary(self):
        q = {self.A: True}
        e = {self.M: True}
        res = self.jpt.infer(q, e)
        self.assertAlmostEqual(.15009, res.result, places=5)

    def test_infer_alarm(self):
        q = {self.A: True}
        e = {}
        res = self.jpt.infer(q, e)
        self.assertAlmostEqual(0.00252, res.result, places=5)


if __name__ == '__main__':
    unittest.main()
