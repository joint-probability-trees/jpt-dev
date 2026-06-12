import json
import pickle
import unittest

import numpy as np
import pandas as pd

from jpt.ensembles import JPTBoost, JPTForest, JPTLikelihoodBoost, MixtureJPT
from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


# ----------------------------------------------------------------------------------------------------------------------

def bimodal_data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    '''Two well-separated Gaussian clusters with a correlated class label.

    Deliberately multi-modal: a single shallow JPT underfits it, so both
    bagging and likelihood-boosting have signal to pick up.
    '''
    rng = np.random.RandomState(seed)
    half = n // 2
    x = np.concatenate([
        rng.normal(-3., .5, half),
        rng.normal(3., .5, n - half)
    ])
    y = np.concatenate([
        2. * x[:half] + rng.normal(0, .3, half),
        -2. * x[half:] + rng.normal(0, .3, n - half)
    ])
    label = np.array(['lo'] * half + ['hi'] * (n - half))
    return pd.DataFrame({'x': x, 'y': y, 'label': label})


def regression_data(n: int = 400, seed: int = 0) -> pd.DataFrame:
    '''Nonlinear regression task y = sin(x1) + .5 x2 + noise.'''
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(-3, 3, n)
    x2 = rng.uniform(-2, 2, n)
    y = np.sin(x1) + .5 * x2 + rng.normal(0, .1, n)
    return pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})


# ----------------------------------------------------------------------------------------------------------------------

class JPTForestTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = bimodal_data()
        cls.forest = JPTForest(
            n_estimators=5,
            min_samples_leaf=.1,
            random_state=42
        ).learn(cls.data)

    def test_members_and_weights(self):
        self.assertEqual(5, len(self.forest))
        self.assertAlmostEqual(1., sum(self.forest.weights))
        self.assertTrue(all(w == .2 for w in self.forest.weights))

    def test_likelihood_is_convex_combination(self):
        mixture = self.forest.likelihood(self.data)
        members = np.array([
            m.likelihood(self.data) for m in self.forest.members
        ])
        expected = np.average(members, axis=0, weights=self.forest.weights)
        self.assertTrue(np.allclose(mixture, expected))
        self.assertTrue(np.all(mixture >= 0))

    def test_infer(self):
        p = self.forest.infer({'label': 'lo'})
        self.assertAlmostEqual(.5, p, places=1)
        p_cond = self.forest.infer({'label': 'lo'}, {'x': [-4., -2.]})
        self.assertGreater(p_cond, .9)

    def test_infer_responsibilities(self):
        # An ensemble of identical members must reproduce the member's
        # conditional exactly -- this fails for the naive (unweighted)
        # average of member conditionals if responsibilities are wrong.
        tree = self.forest.members[0]
        clone = JPTForest(n_estimators=3)
        clone._variables = list(tree.variables)
        clone.members = [tree] * 3
        clone.weights = [1 / 3] * 3
        query, evidence = {'label': 'lo'}, {'x': [-4., -2.]}
        self.assertAlmostEqual(
            tree.infer(query, evidence),
            clone.infer(query, evidence)
        )

    def test_posterior_and_expectation(self):
        posterior = self.forest.posterior(['y'], {'x': [-4., -2.]})
        expectation = posterior['y'].expectation()
        # in the left cluster, y = 2x: E[y | x in [-4, -2]] is around -6
        self.assertLess(expectation, -3)
        e = self.forest.expectation(['y'], {'x': [-4., -2.]})
        self.assertAlmostEqual(expectation, e['y'])

    def test_mpe(self):
        states, score = self.forest.mpe({'label': 'lo'})
        self.assertGreater(score, 0)
        self.assertEqual({'lo'}, states[0]['label'])

    def test_sample(self):
        samples = self.forest.sample(50)
        self.assertEqual(50, len(samples))

    def test_unsatisfiable_evidence(self):
        from jpt.base.errors import Unsatisfiability
        with self.assertRaises(Unsatisfiability):
            self.forest.infer({'label': 'lo'}, {'x': [100., 200.]})
        self.assertIsNone(
            self.forest.infer(
                {'label': 'lo'},
                {'x': [100., 200.]},
                fail_on_unsatisfiability=False
            )
        )

    def test_json_roundtrip(self):
        recovered = MixtureJPT.from_json(
            json.loads(json.dumps(self.forest.to_json()))
        )
        self.assertIsInstance(recovered, JPTForest)
        self.assertEqual(self.forest, recovered)
        self.assertTrue(np.allclose(
            self.forest.likelihood(self.data),
            recovered.likelihood(self.data)
        ))

    def test_pickle_roundtrip(self):
        recovered = pickle.loads(pickle.dumps(self.forest))
        self.assertEqual(self.forest, recovered)

    def test_parallel_training(self):
        forest = JPTForest(
            n_estimators=4,
            min_samples_leaf=.1,
            random_state=42
        ).learn(self.data, multicore=2)
        self.assertEqual(4, len(forest))
        self.assertTrue(np.all(forest.likelihood(self.data) >= 0))

    def test_variance_reduction(self):
        # the forest's held-out log-likelihood should not be (much) worse
        # than a single tree's
        train, test = self.data.iloc[:300], self.data.iloc[300:]
        single = JPT(
            variables=infer_from_dataframe(train, scale_numeric_types=False),
            min_samples_leaf=.1
        ).learn(train)
        forest = JPTForest(
            n_estimators=5,
            min_samples_leaf=.1,
            random_state=0
        ).learn(train)
        ll_single = float(np.mean(np.log(
            np.clip(single.likelihood(test), 1e-12, None)
        )))
        ll_forest = forest.log_likelihood(test)
        self.assertGreater(ll_forest, ll_single - .5)


# ----------------------------------------------------------------------------------------------------------------------

class JPTLikelihoodBoostTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = bimodal_data()
        cls.boost = JPTLikelihoodBoost(
            n_rounds=5,
            min_samples_leaf=.2,
            random_state=42
        ).learn(cls.data)

    def test_weights_normalized(self):
        self.assertAlmostEqual(1., sum(self.boost.weights))
        self.assertTrue(all(w > 0 for w in self.boost.weights))

    def test_history_monotone(self):
        # accepted rounds must strictly improve the training log-likelihood
        for earlier, later in zip(self.boost.history, self.boost.history[1:]):
            self.assertGreater(later, earlier)

    def test_improves_over_single_tree(self):
        single_ll = self.boost.history[0]
        self.assertGreater(self.boost.log_likelihood(self.data), single_ll)

    def test_query_surface(self):
        p = self.boost.infer({'label': 'lo'}, {'x': [-4., -2.]})
        self.assertGreater(p, .5)
        e = self.boost.expectation(['y'], {'x': [-4., -2.]})
        self.assertLess(e['y'], -3)

    def test_json_roundtrip(self):
        recovered = MixtureJPT.from_json(
            json.loads(json.dumps(self.boost.to_json()))
        )
        self.assertIsInstance(recovered, JPTLikelihoodBoost)
        self.assertEqual(self.boost, recovered)
        self.assertEqual(self.boost.history, recovered.history)

    def test_pickle_roundtrip(self):
        recovered = pickle.loads(pickle.dumps(self.boost))
        self.assertEqual(self.boost, recovered)


# ----------------------------------------------------------------------------------------------------------------------

class JPTBoostTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = regression_data()
        cls.train, cls.test = cls.data.iloc[:300], cls.data.iloc[300:]
        cls.boost = JPTBoost(
            target='y',
            n_rounds=20,
            learning_rate=.3,
            min_samples_leaf=.2
        ).learn(cls.train)

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        residual = np.sum((y_true - y_pred) ** 2)
        total = np.sum((y_true - y_true.mean()) ** 2)
        return 1. - residual / total

    def test_beats_mean_predictor(self):
        y_test = self.test['y'].to_numpy()
        score = self.r2(y_test, self.boost.predict(self.test))
        self.assertGreater(score, .5)

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            JPTBoost(target='y').predict(self.test)

    def test_missing_target_raises(self):
        with self.assertRaises(ValueError):
            JPTBoost(target='nonexistent').learn(self.train)

    def test_json_roundtrip(self):
        recovered = JPTBoost.from_json(
            json.loads(json.dumps(self.boost.to_json()))
        )
        self.assertEqual(self.boost, recovered)
        self.assertTrue(np.allclose(
            self.boost.predict(self.test),
            recovered.predict(self.test)
        ))

    def test_pickle_roundtrip(self):
        recovered = pickle.loads(pickle.dumps(self.boost))
        self.assertTrue(np.allclose(
            self.boost.predict(self.test),
            recovered.predict(self.test)
        ))


# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
