
import unittest

from jpt.base.utils import Conditional
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

        A_ = Conditional(Bool, [self.E.domain, self.B.domain])
        A_[True, True] = Bool(.95)
        A_[True, False] = Bool(.94)
        A_[False, True] = Bool(.29)
        A_[False, False] = Bool(.001)

        M_ = Conditional(Bool, [self.A])
        M_[True] = Bool(.7)
        M_[False] = Bool(.01)

        J_ = Conditional(Bool, [self.A])
        J_[True] = Bool(.9)
        J_[False] = Bool(.05)

        data = []
        for i in range(10000):
            e = self.E.dist(.2).sample_one()
            b = self.B.dist(.1).sample_one()
            a = A_.sample_one([e, b])
            m = M_.sample_one(a)
            j = J_.sample_one(a)

            data.append([e, b, a, m, j])

        self.jpt = JPT(variables=[self.E, self.B, self.A, self.M, self.J], name='Alarm', min_impurity_improvement=0)
        self.jpt.learn(data)

    def test_infer_alarm_given_mary(self):
        q = {self.A: True}
        e = {self.M: True}
        res = self.jpt.infer(q, e)
        # self.assertAlmostEqual(res.result, 0.95, delta=0.1)
        self.assertAlmostEqual(res.result, 0.95, places=1)

    def test_infer_alarm(self):
        q = {self.A: True}
        e = {}
        res = self.jpt.infer(q, e)
        self.assertAlmostEqual(res.result, 0.2158, places=2)


if __name__ == '__main__':
    unittest.main()
