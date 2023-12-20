import itertools
import json
import os
import pickle
import random
import statistics
import tempfile
import unittest
from math import prod
from operator import itemgetter
from unittest import TestCase

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.datasets
from dnutils import project

from jpt.base.utils import pairwise
from jpt.distributions import Gaussian, Numeric, Bool, IntegerType
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal
from pandas import DataFrame
from scipy.stats import norm

import jpt.variables
from jpt import SymbolicType
from jpt.base.errors import Unsatisfiability
from jpt.trees import JPT, Leaf
from jpt.variables import NumericVariable, VariableMap, infer_from_dataframe, SymbolicVariable, LabelAssignment, \
    IntegerVariable

try:
    from jpt.base.functions import __module__
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import ConstantFunction, LinearFunction
    from jpt.base.intervals import ContinuousSet


class JPTTest(TestCase):

    def setUp(self) -> None:
        with open(os.path.join('resources', 'gaussian_100.dat'), 'rb') as f:
            self.data = pickle.load(f)

    def test_hyperparameter_serialization(self):
        '''Serialization with complete hyperparameters without training'''
        x = NumericVariable('X')
        y = NumericVariable('Y')
        dependencies = VariableMap([(x, [x, y]), (y, [x])])
        jpt = JPT(
            variables=[x, y],
            targets=[x, y],
            min_samples_leaf=.1,
            min_impurity_improvement=0.1,
            max_leaves=100,
            max_depth=10,
            dependencies=dependencies
        )
        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

    def test_serialization(self):
        '''(de)serialization of JPTs with training'''
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))

        self.assertIsNone(jpt.root.parent)
        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

        q = jpt.bind(X=[-1, 1])
        q_ = jpt_.bind(X=[-1, 1])
        self.assertEqual(jpt_.infer(q_), jpt.infer(q))

    def test_copy(self):
        # Arrange
        var = NumericVariable('X')
        jpt = JPT([var])
        # Act
        jpt_ = jpt.copy()
        # Assert
        self.assertEqual(
            jpt,
            jpt_,
            msg='Copied JPT is not equal to the original one.'
        )
        for v in jpt_.variables:
            self.assertIs(
                v,
                jpt.varnames[v.name],
                msg='Variables of copied JPT must be identical to the original ones.'
            )

    def test_check_variable_assignment(self):
        # Arrange
        x = SymbolicVariable('X', domain=Bool)
        x_ = SymbolicVariable('X', domain=Bool)
        y = SymbolicVariable('Y', domain=Bool)
        jpt = JPT([x])
        jpt_ = JPT([x_])
        assignx = jpt.bind(X=True)
        assignx_ = jpt_.bind(X=True)
        assigny = LabelAssignment([
            ('Y', False)
        ], variables=[y])

        # Act & Assert
        jpt._check_variable_assignment(assignx)
        jpt_._check_variable_assignment(assignx_)
        # Check None
        jpt._check_variable_assignment(assignment=None)
        # Check containment
        self.assertRaises(ValueError, jpt._check_variable_assignment, assignx_)
        self.assertRaises(ValueError, jpt_._check_variable_assignment, assignx)
        self.assertRaises(ValueError, jpt._check_variable_assignment, assigny)

    def test_serialization_integer(self):
        '''(de)serialization of JPTs with training'''
        # Arrange
        data = pd.DataFrame(
            np.array([list(range(-10, 10))]).T,
            columns=["X"]
        )
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables, min_samples_leaf=.1)
        jpt.fit(data)

        # Act
        jpt_ = JPT.from_json(
            json.loads(
                json.dumps(
                    jpt.to_json()
                )
            )
        )

        # Assert
        self.assertIsNone(jpt.root.parent)
        self.assertEqual(jpt, jpt_)

        q = jpt.bind(X=[-1, 1])
        q_ = jpt_.bind(X=[-1, 1])
        self.assertEqual(
            jpt_.infer(q_),
            jpt.infer(q)
        )

    def test_serialization_symbolic(self):
        '''(de)serialization of JPTs with training'''
        data = pd.DataFrame([["A"], ["B"]], columns=["X"])
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables, min_samples_leaf=.1)
        jpt.fit(data)

        self.assertIsNone(jpt.root.parent)
        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

        q_ = jpt_.bind(X="A")
        q = jpt.bind(X="A")

        self.assertEqual(jpt_.infer(q_), jpt.infer(q))

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
        self.assertTrue(all(probs > 0))

    def test_unsatisfiability(self):
        df = pd.read_csv(
            os.path.join('..', 'examples', 'data', 'restaurant.csv'),
            na_filter=False
        )
        jpt = JPT(
            variables=infer_from_dataframe(df),
            targets=['WillWait'],
            min_samples_leaf=1
        )
        jpt.fit(df)
        self.assertRaises(
            Unsatisfiability,
            jpt.posterior,
            evidence={'WillWait': False, 'Patrons': 'Some'},
            fail_on_unsatisfiability=True
        )
        self.assertIsNone(
            jpt.posterior(
                evidence={'WillWait': False, 'Patrons': 'Some'},
                fail_on_unsatisfiability=False
            )
        )

    def test_unsatisfiability_with_reasons(self):
        df = pd.read_csv(
            os.path.join('..', 'examples', 'data', 'restaurant.csv'),
            na_filter=False
        )
        jpt = JPT(variables=infer_from_dataframe(df), targets=['WillWait'], min_samples_leaf=1)
        jpt.fit(df)
        try:
            jpt.posterior(
                evidence={'WillWait': False, 'Patrons': 'Some'},
                report_inconsistencies=True
            )
        except Unsatisfiability as e:
            self.assertEqual(
                [
                    (4, VariableMap([(jpt.varnames['WillWait'], {False})]))
                ],
                list(e.reasons)
            )
        else:
            raise RuntimeError('jpt.posterior did not raise Unsatisfiability.')

    def test_exact_mpe_discrete(self):
        # Arrange
        df = pd.read_csv(
            os.path.join('..', 'examples', 'data', 'restaurant.csv'),
            na_filter=False
        )
        tree = JPT(variables=infer_from_dataframe(df), min_samples_leaf=0.2)
        tree.fit(df)

        # Act
        mpe, likelihood = tree.mpe()

        # Assert
        self.assertEqual(len(mpe), 1)

    def test_mpe_serialization(self):
        # Arrange
        df = pd.read_csv(
            os.path.join('..', 'examples', 'data', 'restaurant.csv'),
            na_filter=False
        )
        tree = JPT(variables=infer_from_dataframe(df), min_samples_leaf=0.2)
        tree.fit(df)

        # Act
        maxima, likelihood = tree.mpe()

        # Assert
        # Trying to recreate the maxima Variables from json
        maxima_json = [m.to_json() for m in maxima]
        # Creating json maxima infos
        for maxi_json in maxima_json:
            maxi = LabelAssignment.from_json(variables=tree.variables, d=maxi_json)
            self.assertIsInstance(maxi, LabelAssignment)

    def test_exact_mpe_continuous(self):
        var = NumericVariable('X')
        tree = JPT([var], min_samples_leaf=.1)
        tree.learn(self.data.reshape(-1, 1))

        mpe, likelihood = tree.mpe()
        self.assertEqual(len(mpe), 1)

    def test_conditional_jpt(self):
        jpt = JPT.load(os.path.join('resources', 'berlin_crimes.jpt'))
        evidence = jpt.bind(Arson=[20, 30])
        cjpt = jpt.conditional_jpt(evidence)
        marginals = cjpt.posterior(evidence=VariableMap())
        self.assertEqual(marginals["Arson"].p(evidence["Arson"]), 1.)

    def test_reverse_inference(self):
        pass
        jpt = JPT.load(os.path.join('resources', 'berlin_crimes.jpt'))
        q = {
            "District": ["Spandau"],
            "Graffiti": ContinuousSet(20, 40),
            "Drugs": ContinuousSet(30, 40)
        }
        p = jpt.reverse(q)
        expres = [(17.035751235106073, 34), (16.854094288878343, 59), (16.792804125698865, 58), (16.789359254673656, 60), (16.73021346469622, 50), (16.640170414346798, 16), (16.526547303271443, 53), (16.407173212401243, 44)]
        self.assertEqual(expres, [(sum(c.values()), l.idx) for c, l in p])

    @unittest.skip
    def test_parameter_count(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        self.assertEqual(71, jpt.number_of_parameters())

    def test_independence(self):
        x = NumericVariable('X')
        y = NumericVariable('Y')
        jpt = JPT([x, y], min_samples_leaf=1, min_impurity_improvement=.01)
        jpt.learn(np.array([ContinuousSet(0, 1).sample(10), ContinuousSet(0, 1).sample(10)]).T)

    def test_impurity_inversion(self):
        df = pd.DataFrame.from_records([
            ['a', 'c'],
            ['a', 'd'],
            ['a', 'c'],
            ['b', 'd'],
            ['b', 'c'],
            ['b', 'd']
        ], columns=['fst', 'snd'])
        AT = SymbolicType('AType', labels=['a', 'b'])
        BT = SymbolicType('BType', labels=['c', 'd'])
        A = SymbolicVariable('fst', AT, invert_impurity=True)
        B = SymbolicVariable('snd', BT)
        jpt = JPT([A, B])
        jpt.fit(df)
        for leaf in jpt.leaves.values():
            if leaf.applies(jpt.bind(snd='c')):
                self.assertEqual(AT().set(params=[2 / 3, 1 / 3]), leaf.distributions['fst'])

    def test_bind(self):
        # Arrange
        n = NumericVariable('n')
        s = SymbolicVariable('s', domain=Bool)
        i = IntegerVariable('i', IntegerType('Die', 1, 6))
        jpt = JPT(variables=[n, s, i])

        # Act
        bind1 = jpt.bind(n=1, s=True, i=3)
        bind2 = jpt.bind({'n': 1, s: True, 'i': [3, 5]})

        # Assert
        self.assertIsInstance(bind1, LabelAssignment)
        truth1 = {n: ContinuousSet(1, 1), s: {True}, i: {3}}
        truth2 = {n: ContinuousSet(1, 1), s: {True}, i: {3, 4, 5}}
        for var, val in bind1.items():
            self.assertEqual(truth1[var], val)
        for var, val in truth2.items():
            self.assertEqual(val, bind2[var])

    def test_bind_from_json(self):
        """This test only checks if it works with maps coming from json based communication."""
        # Arrange
        n = NumericVariable('n')
        s = SymbolicVariable('s', domain=Bool)
        i = IntegerVariable('i', IntegerType('Die', 1, 6))
        jpt = JPT(variables=[n, s, i])

        # Act
        map1 = {"n": 1,
                "s": True,
                "i": 3}
        jpt.bind(map1)

        map2 = {"n": [1, 2],
                "s": {True, False},
                "i": {3, 4, 5}}
        jpt.bind(map2)

    def test_postprocess_leaves_1dim(self):
        # Arrange
        x = NumericVariable('x')
        jpt = JPT([x])
        data = np.linspace(0, 9, 10).reshape(-1, 1)
        jpt.fit(data, close_convex_gaps=False)
        for _, f in jpt.posterior([x])[x].cdf.iter():
            self.assertIsInstance(f, ConstantFunction)

        # Act
        jpt.postprocess_leaves()

        # Assert
        for i, f in jpt.posterior([x])[x].cdf.iter():
            if i.isinf():
                self.assertIsInstance(f, ConstantFunction)
            else:
                self.assertIsInstance(f, LinearFunction)


class TestCasePosteriorNumeric(TestCase):
    varx = None
    vary = None
    jpt = None
    df = None

    @classmethod
    def f(cls, x):
        """The function to predict."""
        # return x * np.sin(x)
        return x

    @classmethod
    def setUpClass(cls):
        SAMPLES = 200
        gauss1 = Gaussian([-.25, -.25], [[.2, -.07], [-.07, .1]])
        gauss2 = Gaussian([.5, 1], [[.2, .07], [.07, .05]])
        gauss1_data = gauss1.sample(SAMPLES)
        gauss2_data = gauss2.sample(SAMPLES)
        data = np.vstack([gauss1_data, gauss2_data])

        cls.df = DataFrame({'X': data[:, 0], 'Y': data[:, 1], 'Color': ['R'] * SAMPLES + ['B'] * SAMPLES})

        cls.varx = NumericVariable('X', Numeric, precision=.1)
        cls.vary = NumericVariable('Y', Numeric, precision=.1)
        cls.varcolor = SymbolicVariable('Color', SymbolicType('ColorType', ['R', 'B']))

        cls.jpt = JPT(variables=[cls.varx, cls.vary], min_samples_leaf=.01)
        # cls.jpt = JPT(variables=[cls.varx, cls.vary, cls.varcolor], min_samples_leaf=.1)  # TODO use this once symbolic variables are considered in posterior
        cls.jpt.learn(cls.df[['X', 'Y']])
        # cls.jpt.learn(cls.df)  # TODO use this once symbolic variables are considered in posterior

    def test_posterior_numeric_x_given_y_interval(self):
        self.q = [self.varx]
        self.e = {self.vary: ContinuousSet(1, 1.5)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_numeric_y_given_x_interval(self):
        self.q = [self.vary]
        self.e = {self.varx: ContinuousSet(1, 2)}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_posterior_numeric_x_given_y_value(self):
        self.q = [self.varx]
        self.e = {self.vary: 0}
        self.posterior = self.jpt.posterior(self.q, self.e)

    def test_convexity(self):
        self.jpt.postprocess_leaves()

    def plot(self):
        print('Tearing down test method',
              self._testMethodName,
              'with calculated posterior',
              f'Posterior P('
              f'{",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        X = np.linspace(-2, 2, 100)
        mean = statistics.mean(self.df['X'])
        sd = statistics.stdev(self.df['X'])
        meanr = statistics.mean(self.df[self.df['Color'] == 'R']['X'])
        sdr = statistics.stdev(self.df[self.df['Color'] == 'R']['X'])
        meanb = statistics.mean(self.df[self.df['Color'] == 'B']['X'])
        sdb = statistics.stdev(self.df[self.df['Color'] == 'B']['X'])

        xr = self.df[self.df['Color'] == 'R']['X']
        xb = self.df[self.df['Color'] == 'B']['X']
        yr = self.df[self.df['Color'] == 'R']['Y']
        yb = self.df[self.df['Color'] == 'B']['Y']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(xr, yr, color='r', marker='.', label='Training data A')
        plt.scatter(xb, yb, color='b', marker='.', label='Training data B')
        plt.plot(sorted(self.df['X']), norm.pdf(sorted(self.df['X']), mean, sd), label='PDF of combined datasets')
        plt.plot(sorted(xr), norm.pdf(sorted(xr), meanr, sdr), label='PDF of dataset A')
        plt.plot(sorted(xb), norm.pdf(sorted(xb), meanb, sdb), label='PDF of dataset B')

        # plot posterior
        for var in self.q:
            if var not in self.posterior:
                continue
            plt.plot(X, self.posterior[var].cdf.multi_eval(X), label=f'Posterior of combined datasets')

        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        plt.ylim(-2, 5)
        plt.xlim(-2, 2)
        plt.legend(loc='upper left')
        plt.title(f'Posterior P('
                  f'{",".join([v.name for v in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')
        plt.grid()
        plt.show()


# noinspection PyPep8Naming
class TestCasePosteriorSymbolic(TestCase):
    data = None
    variables = None
    jpt = None

    @classmethod
    def setUpClass(cls):
        f_csv = os.path.join('..', 'examples', 'data', 'restaurant.csv')
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data,
                                             scale_numeric_types=True,
                                             precision=.01,
                                             blur=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0--10, 10--30, 30--60, >60
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL (typically target variable)

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)

    def test_posterior_symbolic_single_candidate_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[8]: 'Thai'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_single_candidatet_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[8]: 'Italian'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_evidence_not_in_path_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: True}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_evidence_not_in_path_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior[self.q[-1]].expectation())

    def test_posterior_symbolic_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: '10--30', self.variables[1]: True, self.variables[8]: 'French'}
        self.assertRaises(Unsatisfiability, self.jpt.posterior, self.q, self.e)

    def test_parameter_count(self):
        self.assertEqual(132, self.jpt.number_of_parameters())


# noinspection PyPep8Naming
class TestCasePosteriorSymbolicAndNumeric(TestCase):
    data = None
    variables = None
    jpt = None

    @classmethod
    def setUpClass(cls):
        f_csv = os.path.join('..', 'examples', 'data', 'restaurant-mixed.csv')
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=False, precision=.01, blur=.01)

        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        import logging
        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        JPT.logger.setLevel(logging.DEBUG)
        cls.jpt.learn(columns=cls.data.values.T)

    def test_plot(self):
        # Act
        path = self.jpt.plot(
            plotvars=['Food', 'WaitEstimate'],
            title='Restaurant-Mixed',
            filename='Restaurant-Mixed',
            # directory=tempfile.gettempdir(),
            view=False
        )
        # Assert
        self.assertTrue(
            os.path.exists(path)
        )

    def test_posterior_mixed_single_candidate_T(self):
        self.q = ['WillWait']
        self.e = {'WaitEstimate': [0, 0], 'Food': 'Thai'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True, False}, self.posterior[self.q[-1]].expectation())

    def test_posterior_mixed_single_candidatet_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Italian'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: True}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, self.posterior[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior[self.q[-1]].expectation())

    def test_posterior_mixed_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[1]: True, self.variables[8]: 'French'}
        self.assertRaises(Unsatisfiability, self.jpt.posterior, self.q, self.e)

    @unittest.skip
    def test_parameter_count(self):
        self.assertEqual(330, self.jpt.number_of_parameters())

    def test_posterior_mixed_numeric_query(self):
        self.q = [self.variables[9]]
        self.e = {self.variables[8]: 'Burger', self.variables[0]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        xr = self.data[(self.data['Food'] == 'Burger') & (self.data['Alternatives'] == False)]['WaitEstimate']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(self.data['WaitEstimate'], [0] * len(self.data), color='b', marker='*', label='All training data')
        plt.scatter(xr, [0] * len(xr), color='r', marker='.', label='Filtered training data')

    def test_sampling(self):
        samples = self.jpt.sample(1000)
        self.assertTrue(all(self.jpt.likelihood(samples) > 0))

    def plot(self):
        print('Tearing down test method',
              self._testMethodName,
              'with calculated posterior',
              f'Posterior P('
              f'{",".join([qv.name for qv in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})')
        # plot posterior
        X = np.linspace(-5, 65, 100)
        for var in self.q:
            if var not in self.posterior.distributions: continue
            plt.plot(X,
                     self.posterior.distributions[var].cdf.multi_eval(np.array([var.domain.values[x] for x in X])),
                     label=f'Posterior of dataset')

        plt.xlabel('$WaitEstimate [min]$')
        plt.ylabel('$f(x)$')
        plt.ylim(-2, 2)
        plt.xlim(-5, 65)
        plt.legend(loc='upper left')
        plt.title(f'Posterior P('
                  f'{",".join([v.name for v in self.q])}|{",".join([f"{k.name}={v}" for k, v in self.e.items()])})'
                  .replace('$', r'\$'))
        plt.grid()
        plt.show()


class TestCaseExpectation(TestCase):
    jpt = None
    data = None
    variables = None

    @classmethod
    def setUpClass(cls):
        f_csv = os.path.join('..', 'examples', 'data', 'restaurant-mixed.csv')
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data, scale_numeric_types=True, precision=.01, blur=.01)

        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)

    def test_plot(self):
        # Act
        path = self.jpt.plot(
            plotvars=['WaitEstimate'],
            title='Restaurant-Mixed',
            filename='Restaurant-Mixed',
            view=False
        )
        # Assert
        self.assertTrue(
            os.path.exists(path)
        )

    def test_expectation_mixed_single_candidate_T(self):
        # Arrange
        q = ['WillWait', 'Friday']
        e = {
            'WaitEstimate': [10, 30],
            'Food': 'Thai'
        }

        # Act
        expectation = self.jpt.expectation(q, e)

        # Assert
        self.assertEqual([{True}, {True}], list(expectation.values()))

    def test_expectation_mixed_unsatisfiable(self):
        # Arrange
        q = ['WillWait']
        e = {
            'WaitEstimate': [70, 80],
            'Bar': True,
            'Food': 'French'
        }
        # Assert
        self.assertRaises(Unsatisfiability, self.jpt.expectation, q, e)


class TestCaseInference(TestCase):
    jpt = None
    data = None
    variables = None

    @classmethod
    def setUpClass(cls):
        f_csv = '../examples/data/restaurant-mixed.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(
            cls.data,
            scale_numeric_types=False,
            precision=.01,
            # blur=.01
        )
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], 0, 1, 2
        # 5 Price[PRICE_TYPE(SYM)], 1, 2, 3
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T, close_convex_gaps=False)

    def test_plot(self):
        # Act
        path = self.jpt.plot(
            title='Restaurant-Mixed',
            filename='Restaurant-Mixed',
            # directory=tempfile.gettempdir(),
            view=False
        )
        # Assert
        self.assertTrue(
            os.path.exists(path)
        )

    def test_inference_mixed_single_candidate_T(self):
        # Arrange
        q = self.jpt.bind(WillWait=True)
        e = self.jpt.bind(WaitEstimate=[0, 10], Food='Thai')
        # Act
        result = self.jpt.infer(q, e)
        # Assert
        self.assertAlmostEqual(
            .6,
            result,
            places=10
        )

    def test_inference_mixed_new(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[-1]: True}
        posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, posterior['WillWait'].expectation())

    def test_conditional_jpt(self):
        mpe, likelihood = self.jpt.mpe()
        cjpt = self.jpt.conditional_jpt(mpe[0])
        self.assertEqual(cjpt.infer(mpe[0]), 1)

    # def tearDown(self):
    #     print('Tearing down test method',
    #           self._testMethodName,
    #           'with calculated posterior',
    #           f'Posterior P(' +
    #           f'{",".join([qv.name for qv in self.q])}|{",".join([f"{k}={v}" for k, v in self.e.items()])})')


class TestJPTFeaturesTargets(TestCase):
    jpt = None
    data = None
    variables = None

    @classmethod
    def setUpClass(cls):
        f_csv = '../examples/data/restaurant-mixed.csv'
        cls.data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        cls.variables = infer_from_dataframe(cls.data,
                                             scale_numeric_types=True,
                                             precision=.01,
                                             blur=.01)
        # 0 Alternatives[ALTERNATIVES_TYPE(SYM)], BOOL
        # 1 Bar[BAR_TYPE(SYM)], BOOl
        # 2 Friday[FRIDAY_TYPE(SYM)], BOOL
        # 3 Hungry[HUNGRY_TYPE(SYM)], BOOl
        # 4 Patrons[PATRONS_TYPE(SYM)], None, Some, Full
        # 5 Price[PRICE_TYPE(SYM)], $, $$, $$$
        # 6 Rain[RAIN_TYPE(SYM)], BOOL
        # 7 Reservation[RESERVATION_TYPE(SYM)], BOOL
        # 8 Food[FOOD_TYPE(SYM)], French, Thai, Burger, Italian
        # 9 WaitEstimate[WAITESTIMATE_TYPE(SYM)], 0, 9, 10, 29, 30, 59, 60 NUMERIC!
        # 10 WillWait[WILLWAIT_TYPE(SYM)]  BOOL

    def test_no_features_no_targets(self):
        model = JPT(variables=self.variables, min_samples_leaf=1)
        self.assertEqual(list(model.variables), model.features)
        self.assertEqual(list(model.variables), model.targets)

    def test_no_features_targets(self):
        model = JPT(variables=self.variables, targets=["WillWait"], min_samples_leaf=1)
        self.assertEqual([model.varnames["WillWait"]], model.targets)
        self.assertEqual([v for n, v in model.varnames.items() if v not in model.targets], model.features)

    def test_features_no_targets(self):
        model = JPT(variables=self.variables, features=["Price"], min_samples_leaf=1)
        self.assertEqual(list(model.variables), model.targets)
        self.assertEqual([model.varnames["Price"]], model.features)

    def test_features_targets(self):
        model = JPT(variables=self.variables, features=["Price", "Food"], targets=["Price", "WillWait"],
                    min_samples_leaf=1)
        self.assertEqual([model.varnames["Price"], model.varnames["WillWait"]], model.targets)
        self.assertEqual([model.varnames["Price"], model.varnames["Food"]], model.features)


class TestGaussianConditionalJPT(TestCase):

    def setUp(self) -> None:
        np.random.seed(69)
        self.data = pd.DataFrame(data=np.random.multivariate_normal([100, 100], [[5.96, -2.85], [-2.85, 3.47]], 1000),
                                 columns=["X", "Y"])
        variables = infer_from_dataframe(self.data, scale_numeric_types=True, precision=0.01)
        self.tree = JPT(variables, min_samples_leaf=0.1)
        self.tree.fit(self.data)

    def test_mpe(self):
        mpe, likelihood = self.tree.mpe()
        self.assertEqual(len(mpe), 1)

    def test_mpe_serialization(self):
        # Creating json maxima infos
        maxima, likelihood = self.tree.mpe()
        maxima_json = [m.to_json() for m in maxima]

        # Trying to recreate the maxima Variables from json
        for maxi_json in maxima_json:
            maxi = LabelAssignment.from_json(variables=self.tree.variables, d=maxi_json)
            self.assertIsInstance(maxi, LabelAssignment)

    def test_conditioning_chain(self):
        cjpt = self.tree.conditional_jpt()
        cjpt = cjpt.conditional_jpt()
        self.assertTrue(cjpt is not None)

    def test_conditioning(self):
        mpe, likelihood = self.tree.mpe()
        cjpt: JPT = self.tree.conditional_jpt(mpe[0])
        self.assertEqual(cjpt.infer(mpe[0]), 1)

    def test_moment(self):

        expectation = self.tree.expectation([v for v in self.tree.variables if v.numeric or v.integer])

        for order in range(3):
            moments = self.tree.moment(order, expectation)
            for variable, moment in moments.items():
                scipy_moment = scipy.stats.moment(self.data[variable.name], order)
                self.assertAlmostEqual(scipy_moment, moment, delta=0.05)


class TestConstantColumns(unittest.TestCase):
    """Test JPTs for datasets for constant columns occur."""

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(69)
        z = np.ones((100, 1), dtype=str)
        xy = np.random.uniform(0, 1, (100, 2))
        rotation = np.array([np.cos(np.pi / 4), -np.sin(np.pi / 4), np.sin(np.pi / 4), np.cos(np.pi / 4)]).reshape(2, 2)
        xy = xy @ rotation
        data = np.concatenate([xy, z], axis=-1)
        cls.data = pd.DataFrame(data, columns=["x", "y", "z"]).astype({"x": float, "y": float, "z": str})

    def test_learning_single_constant_symbolic(self):
        model = JPT(variables=infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.1,
                    min_impurity_improvement=0)
        model = model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_single_constant_numeric(self):
        data = self.data.copy()
        data["z"] = data["z"].astype(float)
        model = JPT(variables=infer_from_dataframe(data, scale_numeric_types=False), min_samples_leaf=0.1,
                    min_impurity_improvement=0)
        model = model.fit(data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_multiple_constant_symbolic(self):
        data = self.data.copy()
        data["a"] = data["z"].copy()
        model = JPT(variables=infer_from_dataframe(data, scale_numeric_types=False), min_samples_leaf=0.1,
                    min_impurity_improvement=0)
        model = model.fit(data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_multiple_constant_numeric(self):
        data = self.data.copy()
        data["z"] = data["z"].astype(float)
        data["a"] = data["z"].copy()
        model = JPT(variables=infer_from_dataframe(data, scale_numeric_types=False), min_samples_leaf=0.1,
                    min_impurity_improvement=0)
        model = model.fit(data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_multiple_constant_mixed(self):
        data = self.data.copy()
        data["a"] = data["z"].astype(float).copy()
        data["b"] = np.zeros(100, dtype=int)
        data["c"] = np.zeros(100, dtype=int)
        data["d"] = np.zeros(100, dtype=str)
        data["e"] = np.zeros(100, dtype=float)
        model = JPT(variables=infer_from_dataframe(data, scale_numeric_types=False), min_samples_leaf=0.1,
                    min_impurity_improvement=0)
        model = model.fit(data)
        self.assertTrue(len(model.leaves) > 1)


# ----------------------------------------------------------------------------------------------------------------------

class PreprocessingTest(TestCase):

    # noinspection PyMethodMayBeStatic
    def test_preprocessing_dataframe(self):
        # Arrange
        DOMAIN_A = SymbolicType(
            name='DOM_A',
            labels=list(sorted(['250000', '0']))
        )
        DOMAIN_B = SymbolicType(
            name='DOM_B',
            labels=list(sorted([
                '250000',
                '250005',
                '250001',
                '250006',
                '250002',
                '0'
            ]))
        )
        DOMAIN_C = IntegerType(
            name='DOM_C',
            lmin=1,
            lmax=7
        )

        va = SymbolicVariable('V1', DOMAIN_A)
        vb = SymbolicVariable('V2', DOMAIN_B)
        vc = IntegerVariable('V3', DOMAIN_C)

        jpt = JPT(variables=[va, vb, vc])

        data = DataFrame.from_records([
            ['250000', '250000', 3],
            ['0', '0', 7],
            ['250000', '250001', 7],
            ['250000', '250006', 5],
            ['0', '0', 7]
        ],
            columns=['V1', 'V2', 'V3']
        )

        # Act
        data_ = jpt._preprocess_data(data)

        # Assert
        assert_array_equal(
            np.array(
                [[1., 1., 2.],
                 [0., 0., 6.],
                 [1., 2., 6.],
                 [1., 5., 4.],
                 [0., 0., 6.]]
            ),
            data_
        )


# ----------------------------------------------------------------------------------------------------------------------

class ConditionalJPTTest(TestCase):
    data: pd.DataFrame
    model: JPT

    @classmethod
    def setUpClass(cls) -> None:
        dataset = sklearn.datasets.load_iris()
        df = pd.DataFrame(columns=dataset.feature_names, data=dataset.data)

        target = dataset.target.astype(object)
        for idx, target_name in enumerate(dataset.target_names):
            target[target == idx] = target_name

        df["plant"] = target

        cls.data = df
        cls.model = JPT(infer_from_dataframe(cls.data, scale_numeric_types=False, precision=0.1), min_samples_leaf=0.3)
        cls.model.fit(cls.data)

    def apply_evidence(self, evidence: jpt.variables.LabelAssignment) -> pd.DataFrame:

        result = self.data.copy()

        for variable, assignment in evidence.items():
            if variable.symbolic:
                result = result[result[variable.name].isin(assignment)]

            if variable.numeric:
                if isinstance(assignment, ContinuousSet):
                    result = result[
                        (result[variable.name] < assignment.upper) & (result[variable.name] >= assignment.lower)]

        return result

    def test_identity(self):
        self.assertEqual(
            np.average(
                np.log(
                    self.model.likelihood(self.data)
                )
            ),
            np.average(
                np.log(
                    self.model.conditional_jpt().likelihood(self.data)
                )
            )
        )

    def test_likelihood_symbolic(self):

        # get original likelihood
        likelihood = np.average(np.log(self.model.likelihood(self.data)))

        # create evidence
        evidence = self.model.bind({"plant": {"setosa", "virginica"}})

        # create conditional jpt using the method
        conditional_model = self.model.conditional_jpt(evidence)

        # crop the dataframe to match evidence
        cropped_df = self.apply_evidence(evidence)

        # calculate conditional likelihood using model
        conditional_likelihood = np.average(np.log(conditional_model.likelihood(cropped_df)))

        self.assertTrue(conditional_likelihood > likelihood)

    def test_likelihood_numeric_intervals(self):

        # get original likelihood
        likelihood = np.average(np.log(self.model.likelihood(self.data)))

        # create evidence
        evidence = self.model.bind({"sepal length (cm)": [5, 6]})

        # create conditional jpt using the method
        conditional_model = self.model.conditional_jpt(evidence)

        # crop the dataframe to match evidence
        cropped_df = self.apply_evidence(evidence)

        # calculate conditional likelihood using model
        conditional_likelihood = np.average(np.log(conditional_model.likelihood(cropped_df)))

        self.assertTrue(conditional_likelihood > likelihood)

    def test_posterior(self):

        evidence = {"sepal length (cm)": [5, 6]}

        # create conditional jpt using the method
        conditional_model = self.model.conditional_jpt(self.model.bind(evidence))

        posteriors = conditional_model.posterior(evidence=conditional_model.bind(evidence))

        for variable, value in conditional_model.bind(evidence).items():
            self.assertAlmostEqual(posteriors[variable].p(value), 1.)

    def test_priors(self):
        # create evidence
        evidence = self.model.bind({"sepal length (cm)": [5, 6]})

        # create conditional jpt using the method
        conditional_model = self.model.conditional_jpt(evidence)

        for variable, value in evidence.items():
            self.assertAlmostEqual(conditional_model.priors[variable].p(value), 1.)

    def test_copy_leaf(self):
        evidence = self.model.bind({"sepal length (cm)": 5})
        for leaf in self.model.leaves.values():
            copy = leaf.copy()
            self.assertEqual(leaf.distributions, copy.distributions)
            copy_ = copy.conditional_leaf(evidence)
            self.assertTrue(leaf.distributions != copy_.distributions)

    def test_conditional_leaf(self):
        evidence = self.model.bind({"sepal length (cm)": [5, 6]})
        for leaf in self.model.apply(evidence):
            l_ = leaf.conditional_leaf(evidence)
            self.assertAlmostEqual(1, l_.probability(evidence))


# ----------------------------------------------------------------------------------------------------------------------

class TestCaseTargetLearning(TestCase):
    data: pd.DataFrame

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(69)

        # create random positive semi definite matrix
        covariance = np.random.rand(10, 10)
        covariance = covariance @ covariance.T

        # sample a bunch of points
        numeric_data = np.random.multivariate_normal(mean=np.zeros(10), cov=covariance,
                                                     size=(2000,))

        # sort by first numeric column
        numeric_data = np.sort(numeric_data, 0)
        # add dependent symbolic variable
        symbolic_column = np.concatenate((np.zeros((1000, 1)), np.ones((1000, 1))))

        # sort by second numeric column
        numeric_data = np.sort(numeric_data, 1)
        # add dependent integer variable
        integer_column = np.concatenate((np.zeros((1000, 1)), np.ones((1000, 1))))

        cls.data = pd.DataFrame(columns=["s0"] + ["i0"] + [f"n{i}" for i in range(10)],
                                data=np.concatenate((symbolic_column, integer_column, numeric_data), axis=1))
        cls.data["s0"] = cls.data["s0"].astype("str")
        cls.data["i0"] = cls.data["i0"].astype("int")

    def test_variale_setup(self):
        vars = infer_from_dataframe(self.data)
        self.assertEqual(1, len([v for v in vars if v.symbolic]))
        self.assertEqual(1, len([v for v in vars if v.integer]))
        self.assertEqual(10, len([v for v in vars if v.numeric]))

    def test_learning_discriminative_single_symbolic(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["s0"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_discriminative_single_numeric(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["n0"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_discriminative_mixed_symbolic_numeric(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["s0", "n0"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_discriminative_mixed_symbolic_integer(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["s0", "i0"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_discriminative_mixed_numeric_integer(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["n5", "i0"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)

    def test_learning_discriminative_mixed_all(self):
        model = JPT(infer_from_dataframe(self.data, scale_numeric_types=False), min_samples_leaf=0.3,
                    targets=["s0", "i0", "n9"])
        model.fit(self.data)
        self.assertTrue(len(model.leaves) > 1)


# ----------------------------------------------------------------------------------------------------------------------

class KMPELeafTest(TestCase):
    data: pd.DataFrame
    model: JPT

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(69)
        dataset = sklearn.datasets.load_iris()
        df = pd.DataFrame(columns=dataset.feature_names, data=dataset.data)

        target = dataset.target.astype(object)
        for idx, target_name in enumerate(dataset.target_names):
            target[target == idx] = target_name

        df["plant"] = target

        cls.data = df
        cls.model = JPT(
            variables=infer_from_dataframe(
                cls.data,
                scale_numeric_types=False,
                precision=0.05
            ),
            min_samples_leaf=0.9
        )
        cls.model.fit(cls.data)

    def test_k3_mpe(self):
        # Arrange
        # Act
        k_mpe = list(
            self.model.kmpe(k=3)
        )

        # Assert
        s1 = (
            LabelAssignment({
                'sepal length (cm)': ContinuousSet.parse('[4.300,6.900)'),
                'sepal width (cm)': ContinuousSet.parse('[2.500,3.500)'),
                'petal width (cm)': ContinuousSet.parse('[0.100,0.200)'),
                'petal length (cm)': ContinuousSet.parse('[1.000,1.700)'),
                'plant': {'versicolor', 'virginica', 'setosa'},
            }, variables=self.model.variables),
            0.07714629444766113
        )
        s2 = (
            LabelAssignment({
                'sepal length (cm)': ContinuousSet.parse('[4.300,6.900)'),
                'sepal width (cm)': ContinuousSet.parse('[2.500,3.500)'),
                'petal width (cm)': ContinuousSet.parse('[0.100,0.200)'),
                'petal length (cm)': ContinuousSet.parse('[3.300,6.100)'),
                'plant': {'versicolor', 'virginica', 'setosa'},
            }, variables=self.model.variables),
            0.037342089333708306
        )
        s3 = (
            LabelAssignment({
                'sepal length (cm)': ContinuousSet.parse('[4.300,6.900)'),
                'sepal width (cm)': ContinuousSet.parse('[2.000,2.500)'),
                'petal width (cm)': ContinuousSet.parse('[0.100,0.200)'),
                'petal length (cm)': ContinuousSet.parse('[1.000,1.700)'),
                'plant': {'versicolor', 'virginica', 'setosa'},
            }, variables=self.model.variables),
            0.024797023215319652
        )
        self.assertEqual(
            len(k_mpe),
            3
        )
        self.assertEqual(
            [s1, s2, s3],
            k_mpe
        )

    def test_k_mpe_brute(self):
        # Arrange
        assert len(self.model.leaves) == 1
        leaf = next(
            iter(self.model.leaves.values())
        )

        # calculate likelihood wise unique solutions
        all_states = [list(d.k_mpe()) for d in leaf.distributions.values()]
        all_joint_states = [
            (project(pair, 0), prod(project(pair, 1))) for pair in itertools.product(*all_states)
        ]
        sorted_joint_states = list(
            sorted(
                all_joint_states,
                reverse=True,
                key=itemgetter(1)
            )
        )
        sorted_joint_states = [
            (LabelAssignment((var, val) for var, val in zip(self.model.variables, s)), l) for s, l in sorted_joint_states
        ]

        # Act
        k_mpe = list(
            self.model.kmpe(k=len(all_joint_states) + 1000)
        )

        # Assert
        self.assertTrue(
            all([l1 > l2 for (_, l1), (_, l2) in pairwise(k_mpe)]),
            msg="Not all solutions are ordered by descending likelihood"
        )

        self.assertEqual(
            project(sorted_joint_states, 0),
            project(k_mpe, 0),
            msg='MPE state sequences differ from the brute force solution.'
        )

        self.assertEqual(
            [round(v, 8) for v in project(sorted_joint_states, 1)],
            [round(v, 8) for v in project(k_mpe, 1)],
            msg="These should be equal to the number of solutions"
                "that produce different likelihoods (set-wise),"
                "which is 72 for this experiment. 216 (current)"
                "is the number of unique solutions iff sets are not"
                "regarded."
        )


# ----------------------------------------------------------------------------------------------------------------------

class PruningTest(TestCase):

    def test_pruning(self):
        # Arrange
        n = 1000
        random.seed(42)
        data = np.array([
            [random.random() for _ in range(n)],
            [random.random() for _ in range(n)]
        ])

        jpt = JPT(
            variables=[NumericVariable('x'), NumericVariable('y')],
            min_samples_leaf=100
        ).learn(rows=data.T)
        # jpt.plot(plotvars=jpt.variables, view=True)

        # Act
        pruned_jpt = jpt.prune(.6)
        # pruned_jpt.plot(plotvars=pruned_jpt.variables, view=True)

        # Assert
        self.assertEqual(
            1,
            len(pruned_jpt.leaves)
        )

        self.assertIsInstance(
            pruned_jpt.root,
            Leaf
        )

        self.assertAlmostEqual(
            1,
            pruned_jpt.root.prior,
            places=8
        )


