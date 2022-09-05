import unittest
import numpy as np
import jpt.variables
import jpt.trees
from jpt.learning.distributions import SymbolicType
import factor_graphs

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
        self.data = np.expand_dims(self.g.sample(np.arange(np.pi / 2, 10000, np.pi)), -1)
        self.variables = [jpt.variables.NumericVariable("X", precision=0.1)]

    def test_learning(self):
        template_tree = jpt.trees.JPT(self.variables, min_samples_leaf=500)
        template_tree.fit(self.data)

        tree_factor_1 = factor_graphs.JPTFactor("t1", template_tree.copy())
        tree_factor_2 = factor_graphs.JPTFactor("t2", template_tree.copy())
        latent_factor = factor_graphs.LatentFactor("t1t2", [tree_factor_1, tree_factor_2])

        graph = factor_graphs.FactorGraph([tree_factor_1, tree_factor_2, latent_factor])

if __name__ == '__main__':
    unittest.main()
