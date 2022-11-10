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
from jpt.distributions import Gaussian, Numeric
from jpt.trees import JPT
from jpt.variables import NumericVariable, VariableMap, infer_from_dataframe, SymbolicVariable


class JPTTest(TestCase):

    def setUp(self) -> None:
        with open(os.path.join('resources', 'gaussian_100.dat'), 'rb') as f:
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

    def test_exact_mpe_discrete(self):
        df = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))
        jpt = JPT(variables=infer_from_dataframe(df), min_samples_leaf=0.2)
        jpt.fit(df)

        mpe = jpt.mpe()
        self.assertEqual(len(mpe), 1)

    def test_exact_mpe_continuous(self):
        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))

        mpe = jpt.mpe()
        self.assertEqual(len(mpe), 1)

    def test_independent_marginals(self):
        df = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))
        jpt = JPT(variables=infer_from_dataframe(df), min_samples_leaf=0.2)
        jpt.fit(df)

        im = jpt.independent_marginals()
        self.assertEqual(len(im), len(jpt.variables))

        evidence = {jpt.varnames["Hungry"]: {False}}
        im = jpt.independent_marginals(VariableMap(evidence.items()))
        self.assertEqual(len(im), len(jpt.variables))

    def test_conditional_jpt(self):
        jpt = JPT.load(os.path.join('resources', 'berlin_crimes.jpt'))
        evidence = {jpt.varnames["Arson"]: [20, 30]}
        evidence = jpt._preprocess_query(VariableMap(evidence.items()))
        cjpt = jpt.conditional_jpt(evidence)
        marginals = cjpt.independent_marginals()
        self.assertEqual(marginals["Arson"].p(evidence["Arson"]), 1.)


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
        self.assertEqual({True, False}, self.posterior.distributions[self.q[-1]].expectation())

    def test_posterior_mixed_single_candidatet_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[8]: 'Italian'}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior.distributions[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_T(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: True}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, self.posterior.distributions[self.q[-1]].expectation())

    def test_posterior_mixed_evidence_not_in_path_F(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[8]: 'Burger', self.variables[3]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({False}, self.posterior.distributions[self.q[-1]].expectation())

    def test_posterior_mixed_unsatisfiable(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[9]: ContinuousSet(10, 30), self.variables[1]: True, self.variables[8]: 'French'}
        self.assertRaises(Unsatisfiability, self.jpt.posterior, self.q, self.e)

    def test_posterior_mixed_numeric_query(self):
        self.q = [self.variables[9]]
        self.e = {self.variables[8]: 'Burger', self.variables[0]: False}
        self.posterior = self.jpt.posterior(self.q, self.e)

        # Mesh the input space for evaluations of the real function, the prediction and its MSE
        xr = self.data[(self.data['Food'] == 'Burger') & (self.data['Alternatives'] == False)]['WaitEstimate']

        # Plot the data, the pdfs of each dataset and of the datasets combined
        plt.scatter(self.data['WaitEstimate'], [0]*len(self.data), color='b', marker='*', label='All training data')
        plt.scatter(xr, [0]*len(xr), color='r', marker='.', label='Filtered training data')

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
        self.assertEqual([True, True], [e.result for e in self.expectation.values()])

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

        cls.jpt = JPT(variables=cls.variables, min_samples_leaf=1)
        cls.jpt.learn(columns=cls.data.values.T)

    def test_plot(self):
        self.jpt.plot(title='Restaurant-Mixed',
                      filename='Restaurant-Mixed',
                      directory=tempfile.gettempdir(),
                      view=False)

    def test_inference_mixed_single_candidate_T(self):
        self.q = {'WillWait': True}
        self.e = {'WaitEstimate': [0, 10],
                  'Food': 'Thai'}
        inference = self.jpt.infer(self.q, self.e)
        self.assertAlmostEqual(.6, inference.result, places=10)

    def test_inference_mixed_neu(self):
        self.q = [self.variables[-1]]
        self.e = {self.variables[-1]: True}
        posterior = self.jpt.posterior(self.q, self.e)
        self.assertEqual({True}, posterior.distributions['WillWait'].expectation())

    # def tearDown(self):
    #     print('Tearing down test method',
    #           self._testMethodName,
    #           'with calculated posterior',
    #           f'Posterior P(' +
    #           f'{",".join([qv.name for qv in self.q])}|{",".join([f"{k}={v}" for k, v in self.e.items()])})')

