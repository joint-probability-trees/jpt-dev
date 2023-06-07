import os
import pickle
import unittest
from unittest import TestCase

import numpy as np
from dnutils import out

from jpt import SymbolicVariable, JPT, NumericVariable, SymbolicType
from jpt.base.intervals import ContinuousSet, RealSet, EXC, INC
from jpt.distributions import Bool, Numeric
from jpt.trees import MPESearchState, MPESolver
from jpt.variables import VariableMap, ValueAssignment


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

        cls.jpt = JPT(
            variables=[cls.E, cls.B, cls.A, cls.M, cls.J],
            min_impurity_improvement=0
        ).learn(rows=cls.data)
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


class MPESolverTest(TestCase):

    GAUSSIAN = None

    @classmethod
    def setUp(cls) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            cls.GAUSSIAN = pickle.load(f).reshape(-1, 1)

    def test_search_state(self):
        dom1 = SymbolicType('Dom1', labels=['A', 'B', 'C'])
        var1 = SymbolicVariable('V1', dom1)

        dom2 = SymbolicType('Dom2', labels=['D', 'E', 'F'])
        var2 = SymbolicVariable('V2', dom2)

        state = MPESearchState(
            domains=VariableMap({
                var1: list(dom1.labels.values()),
                var2: list(dom2.labels.values())
            })
        )
        self.assertEqual(
            ['A', 'B', 'C'],
            state.domains['V1']
        )
        self.assertEqual(
            VariableMap(variables=[var1, var2]),
            state.assignment
        )
        state = state.assign('V1', 'B')
        self.assertNotIn(
            'V1',
            state.domains
        )
        self.assertEqual(
            VariableMap({
                'V1': 'B'
            }, variables=[var1, var2]),
            state.assignment
        )
        state = state.assign('V2', 'D')
        self.assertNotIn(
            'V2',
            state.domains
        )
        self.assertEqual(
            VariableMap({
                'V1': 'B',
                'V2': 'D'
            }, variables=[var1, var2]),
            state.assignment
        )
        self.assertFalse(state.domains)
        self.assertRaises(ValueError, state.assign, 'V2', 'NaN')

    def test_mpe_numeric(self):
        dist1 = Numeric().fit(self.GAUSSIAN)
        dist2 = Numeric().fit(self.GAUSSIAN)

        v1 = NumericVariable('X')
        v2 = NumericVariable('Y')

        mpe = MPESolver(
            VariableMap({
                v1: dist1,
                v2: dist2
            })
        )
        for mpe in mpe.solve(10):
            out(mpe)
        dist1.plot(view=True)
