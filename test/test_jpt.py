import tempfile

import statistics

import json
import numpy as np
import os
import pickle
from unittest import TestCase

import pandas as pd
from dnutils import out
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy.stats import norm

from jpt import SymbolicType
from jpt.base.errors import Unsatisfiability
from jpt.base.intervals import ContinuousSet
from jpt.distributions import Gaussian, Numeric, Bool, IntegerType
from jpt.trees import JPT
from jpt.variables import NumericVariable, VariableMap, infer_from_dataframe, SymbolicVariable, LabelAssignment, \
    IntegerVariable


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
        self.assertEqual(jpt_.infer(q), jpt.infer(q))

    def test_serialization_integer(self):
        '''(de)serialization of JPTs with training'''
        data = pd.DataFrame(np.array([list(range(-10, 10))]).T, columns=["X"])
        variables = infer_from_dataframe(data, scale_numeric_types=False)
        jpt = JPT(variables, min_samples_leaf=.1)
        jpt.fit(data)

        self.assertIsNone(jpt.root.parent)
        jpt_ = JPT.from_json(json.loads(json.dumps(jpt.to_json())))
        self.assertEqual(jpt, jpt_)

        q = jpt.bind(X=[-1, 1])
        self.assertEqual(jpt_.infer(q), jpt.infer(q))

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

    def test_exact_mpe_discrete(self):
        df = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))
        jpt = JPT(variables=infer_from_dataframe(df), min_samples_leaf=0.2)
        jpt.fit(df)

        mpe, likelihood = jpt.mpe()
        self.assertEqual(len(mpe), 1)

    def test_exact_mpe_continuous(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))

        mpe, likelihood = jpt.mpe()
        self.assertEqual(len(mpe), 1)

    def test_conditional_jpt(self):
        jpt = JPT.load(os.path.join('resources', 'berlin_crimes.jpt'))
        evidence = jpt.bind(Arson=[20, 30])
        cjpt = jpt.conditional_jpt(evidence)
        marginals = cjpt.posterior(evidence=VariableMap())
        self.assertEqual(marginals["Arson"].p(evidence["Arson"]), 1.)

    def test_parameter_count(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        self.assertEqual(12, jpt.number_of_parameters())

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
        self.jpt.plot(plotvars=['Food', 'WaitEstimate'], title='Restaurant-Mixed',
                      filename='Restaurant-Mixed',
                      directory=tempfile.gettempdir(),
                      view=False)

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

    def test_parameter_count(self):
        self.assertEqual(200, self.jpt.number_of_parameters())

    def test_posterior_mixed_numeric_query(self):
        self.q = [self.variables[9]]
        self.e = {self.variables[8]: 'Burger', self.variables[0]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        xr = self.data[(self.data['Food'] == 'Burger') & (self.data['Alternatives'] == False)]['WaitEstimate']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(self.data['WaitEstimate'], [0]*len(self.data), color='b', marker='*', label='All training data')
        plt.scatter(xr, [0]*len(xr), color='r', marker='.', label='Filtered training data')

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
        self.jpt.plot(#plotvars=['WaitEstimate'],
                      title='Restaurant-Mixed',
                      filename='Restaurant-Mixed',
                      directory=tempfile.gettempdir(),
                      view=False)

    def test_expectation_mixed_single_candidate_T(self):
        self.q = ['WillWait', 'Friday']
        self.e = {'WaitEstimate': [10, 30],
                  'Food': 'Thai'}
        self.expectation = self.jpt.expectation(self.q, self.e)
        self.assertEqual([{True}, {True}], [e for e in self.expectation.values()])

    def test_expectation_mixed_unsatisfiable(self):
        self.q = ['WillWait']
        self.e = {'WaitEstimate': [70, 80],
                  'Bar': True,
                  'Food': 'French'}
        self.assertRaises(Unsatisfiability, self.jpt.expectation, self.q, self.e)


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
            blur=.01
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
        cls.jpt.learn(columns=cls.data.values.T)

    def test_plot(self):
        self.jpt.plot(
            title='Restaurant-Mixed',
            filename='Restaurant-Mixed',
            directory=tempfile.gettempdir(),
            view=False
        )

    def test_inference_mixed_single_candidate_T(self):
        q = self.jpt.bind(WillWait=True)
        e = self.jpt.bind(WaitEstimate=[0, 10], Food='Thai')
        inference = self.jpt.infer(q, e)
        (self.jpt.conditional_jpt(e))
        self.assertAlmostEqual(.5, inference, places=10)

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
