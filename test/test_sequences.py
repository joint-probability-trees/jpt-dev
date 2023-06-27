import unittest
import numpy as np
import jpt.variables
import jpt.trees
import jpt.sequential_trees
from jpt.base.errors import Unsatisfiability
import jpt.base.intervals

class UniformSeries:

    def __init__(self, basis_function=np.sin, epsilon=0.05):
        self.epsilon = 0.05
        self.basis_function = basis_function

    def sample(self, samples) -> np.array:
        samples = self.basis_function(samples)
        samples = samples + np.random.uniform(-self.epsilon, self.epsilon, samples.shape)
        return samples


class LearningTest(unittest.TestCase):

    def setUp(self) -> None:
        self.g = UniformSeries()
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1).reshape(-1, 1)
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data])
        self.assertEqual(1, np.sum(sequence_tree.transition_model))


class UniformInferenceTest(unittest.TestCase):

    g: UniformSeries
    data: np.ndarray
    sequence_tree: jpt.sequential_trees.SequentialJPT

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(420)
        cls.g = UniformSeries()
        cls.data = np.expand_dims(cls.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1).reshape(-1, 1)
        variables = [jpt.variables.NumericVariable("X", precision=1)]
        template_tree = jpt.trees.JPT(variables, min_samples_leaf=2500)
        cls.sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        cls.sequence_tree.fit([cls.data, cls.data])
        # cls.sequence_tree.template_tree.plot(plotvars=cls.sequence_tree.template_tree.variables)

    def test_posterior_mid_hard_evidence(self):

        evidence = [{}, {"X": [0.95, 1.05]}, {}]
        evidence = self.sequence_tree.bind(evidence)

        r = self.sequence_tree.posterior(evidence)

        for tree in r:
            # assert valid tree
            self.assertEqual(sum(l.prior for l in tree.leaves.values()), 1.)

        # assert that the state chain of the sequence model is 1 2 1
        self.assertEqual([1, 2, 1], [l.idx for tree in r for l in tree.leaves.values() if l.prior > 0.])

    def test_posterior_begin_hard_evidence(self):

        evidence = [{"X": [0.95, 1.05]}, {}, {}]
        evidence = self.sequence_tree.bind(evidence)

        r = self.sequence_tree.posterior(evidence)

        for tree in r:
            # assert valid tree
            self.assertEqual(sum(l.prior for l in tree.leaves.values()), 1.)

        # assert that the state chain of the sequence model is 1 2 1
        self.assertEqual([2, 1, 2], [l.idx for tree in r for l in tree.leaves.values() if l.prior > 0.])

    def test_posterior_end_hard_evidence(self):
        evidence = [{}, {}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)

        r = self.sequence_tree.posterior(evidence)

        for tree in r:
            # assert valid tree
            self.assertEqual(sum(l.prior for l in tree.leaves.values()), 1.)

        # assert that the state chain of the sequence model is 1 2 1
        self.assertEqual([2, 1, 2], [l.idx for tree in r for l in tree.leaves.values() if l.prior > 0.])

    def test_posterior_multiple_evidence(self):
        evidence = [{"X": [0.95, 1.05]}, {"X": [-1, -0.95]}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result = self.sequence_tree.posterior(evidence)
        self.assertEqual([1, 1, 1], [len(r.leaves) for r in result])

    def test_posterior_overlapping_evidence(self):
        evidence = [{"X": [0.95, 1.05]}, {"X": [-1, 1]}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result = self.sequence_tree.posterior(evidence)
        self.assertEqual([1, 1, 1], [len([l for l in r.leaves.values() if l.prior > 0]) for r in result])

    def test_posterior_overlapping_evidence_multiple_leaves(self):
        evidence = [{"X": [-1, 1]}, {"X": [-1, 1]}, {"X": [-1, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result = self.sequence_tree.posterior(evidence)
        self.assertEqual([2, 2, 2], [len([l for l in r.leaves.values() if l.prior > 0]) for r in result])

    def test_posterior_impossible(self):
        evidence = [{"X": [0, 1]}, {"X": [0, 1]}]
        evidence = self.sequence_tree.bind(evidence)
        result = self.sequence_tree.posterior(evidence, fail_on_unsatisfiability=False)
        self.assertEqual(None, result)
        self.assertRaises(Unsatisfiability, self.sequence_tree.posterior, evidence)

    def test_mpe_mid_hard_evidence(self):
        evidence = [{}, {"X": [0.95, 1.05]}, {}]
        evidence = self.sequence_tree.bind(evidence)

        mpes, likelihood = self.sequence_tree.mpe(evidence)
        self.assertEqual(len(mpes), len(evidence))
        mpe_by_hand = [{"X": [-1.05, -0.95]}, {"X": [0.95, 1.05]}, {"X": [-1.05, -0.95]}]
        mpe_by_hand = self.sequence_tree.bind(mpe_by_hand)

        for mpe, mpe_bh in zip(mpes, mpe_by_hand):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, mpe_bh[variable].lower, delta=0.01)
                self.assertAlmostEqual(state.upper, mpe_bh[variable].upper, delta=0.01)

    def test_mpe_begin_hard_evidence(self):
        evidence = [{"X": [0.95, 1.05]}, {}, {}]
        evidence = self.sequence_tree.bind(evidence)

        mpes, likelihood = self.sequence_tree.mpe(evidence)
        self.assertEqual(len(mpes), len(evidence))
        mpe_by_hand = [{"X": [0.95, 1.05]}, {"X": [-1.05, -0.95]}, {"X": [0.95, 1.05]}]
        mpe_by_hand = self.sequence_tree.bind(mpe_by_hand)

        for mpe, mpe_bh in zip(mpes, mpe_by_hand):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, mpe_bh[variable].lower, delta=0.01)
                self.assertAlmostEqual(state.upper, mpe_bh[variable].upper, delta=0.01)

    def test_mpe_end_hard_evidence(self):
        evidence = [{}, {}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)

        mpes, likelihood = self.sequence_tree.mpe(evidence)
        self.assertEqual(len(mpes), len(evidence))
        mpe_by_hand = [{"X": [0.95, 1.05]}, {"X": [-1.05, -0.95]}, {"X": [0.95, 1.05]}]
        mpe_by_hand = self.sequence_tree.bind(mpe_by_hand)

        for mpe, mpe_bh in zip(mpes, mpe_by_hand):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, mpe_bh[variable].lower, delta=0.01)
                self.assertAlmostEqual(state.upper, mpe_bh[variable].upper, delta=0.01)

    def test_mpe_multiple_evidence(self):
        evidence = [{"X": [0.95, 1.05]}, {"X": [-1, -0.95]}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result, _ = self.sequence_tree.mpe(evidence)
        for evi, mpe in zip(evidence, result):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, evi[variable].lower, delta=0.01)
                self.assertAlmostEqual(state.upper, evi[variable].upper, delta=0.01)

    def test_mpe_overlapping_evidence(self):
        evidence = [{"X": [0.95, 1.05]}, {"X": [-1, 1]}, {"X": [0.95, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result, _ = self.sequence_tree.mpe(evidence)
        for evi, mpe in zip(evidence, result):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, evi[variable].lower, delta=0.01)
                self.assertTrue(state.upper < evi[variable].upper)

    def test_mpe_overlapping_evidence_multiple_leaves(self):
        evidence = [{"X": [-1, 1]}, {"X": [-1, 1]}, {"X": [-1, 1.05]}]
        evidence = self.sequence_tree.bind(evidence)
        result, _ = self.sequence_tree.mpe(evidence)

        mpe_by_hand = [{"X": [0.95, 1.]}, {"X": [-1., -0.95]}, {"X": [0.95, 1.05]}]
        mpe_by_hand = self.sequence_tree.bind(mpe_by_hand)

        for mpe, mpe_bh in zip(result, mpe_by_hand):
            for variable, state in mpe.items():
                state = state.simplify()
                self.assertAlmostEqual(state.lower, mpe_bh[variable].lower, delta=0.01)
                self.assertAlmostEqual(state.upper, mpe_bh[variable].upper, delta=0.01)


    @unittest.skip("Waiting for pgmpy to implement joint maxima.")
    def test_mpe_impossible(self):
        evidence = [{"X": [0, 1]}, {"X": [0, 1]}]
        evidence = self.sequence_tree.bind(evidence)
        result = self.sequence_tree.mpe(evidence, fail_on_unsatisfiability=False)
        self.assertEqual(None, result)
        self.assertRaises(Unsatisfiability, self.sequence_tree.mpe, evidence)

    def test_likelihood(self):
        likelihoods = self.sequence_tree.likelihood([self.data, self.data])
        self.assertTrue(all([np.all(l > 0) for l in likelihoods]))

    @unittest.skip
    def test_probability_trivial(self):
        query = [{"X": [0.95, 1.05]}]
        query = self.sequence_tree.bind(query)
        result = self.sequence_tree.probability(query)
        self.assertEqual(0.5, result)

    @unittest.skip
    def test_probability_trivial_multiple_steps(self):
        query = [{}, {}, {"X": [0.95, 1.05]}]
        query = self.sequence_tree.bind(query)
        result = self.sequence_tree.probability(query)
        self.assertEqual(0.5, result)

    @unittest.skip
    def test_probability_multiple_steps(self):
        query = [{"X": [0.95, 1.05]}, {}, {"X": [0.95, 1.05]}]
        query = self.sequence_tree.bind(query)
        result = self.sequence_tree.probability(query)
        self.assertEqual(0.5, result)

    @unittest.skip
    def test_probability_multiple_steps_half(self):
        query = [{"X": [0.95, 1.05]}, {"X": [-1, -0.95]}, {"X": [0.95, 1.05]}]
        query = self.sequence_tree.bind(query)
        result = self.sequence_tree.probability(query)
        self.assertEqual(0.25, result)


if __name__ == '__main__':
    unittest.main()
