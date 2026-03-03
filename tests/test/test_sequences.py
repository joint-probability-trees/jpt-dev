import unittest
import numpy as np
import jpt.variables
import jpt.trees
from jpt.distributions.univariate import SymbolicType
import jpt.sequential_trees


class UniformSeries:

    def __init__(self, basis_function=np.sin, epsilon=0.05):
        self.epsilon = 0.05
        self.basis_function = basis_function

    def sample(self, samples) -> np.array:
        samples = self.basis_function(samples)
        samples = samples + np.random.uniform(-self.epsilon, self.epsilon, samples.shape)
        return samples


class SequenceTest(unittest.TestCase):

    def setUp(self) -> None:
        self.g = UniformSeries()
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1).reshape(-1, 1)
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=2500)
        sequence_tree = jpt.sequential_trees.SequentialJPT(template_tree)
        sequence_tree.fit([self.data, self.data])

        r = sequence_tree.independent_marginals([
            {},
            template_tree.bind(X=[0.95, 1.05]),
            {}
        ])

        for tree in r:
            self.assertEqual(sum(l.prior for l in tree.leaves.values()), 1.)


if __name__ == '__main__':
    unittest.main()
