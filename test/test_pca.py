import unittest
import sklearn.datasets
import jpt.pca_trees
import jpt.trees
import jpt.variables
import dnutils
import plotly.express as px
import numpy.random
import numpy.linalg
import numpy as np
import plotly.graph_objects as go
import jpt.distributions
import matplotlib.pyplot as plt

def plot_numeric_pdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1):
    x = []
    y = []
    for interval, function in zip(distribution.pdf.intervals[1:-1], distribution.pdf.functions[1:-1]):
        x += [interval.lower, interval.upper, interval.upper]
        y += [function.value, function.value, None]

    range = x[-1] - x[0]
    x = [x[0] - (range * padding), x[0], x[0]] + x + [x[-1], x[-1], x[-1] + (range * padding)]
    y = [0, 0, None] + y + [None, 0, 0]
    trace = go.Scatter(x=x, y=y, name="PDF")
    return trace


class TestPCAJPTiris(unittest.TestCase):

    def test_pca_tree_iris(self):

        dataset = sklearn.datasets.load_iris(as_frame=True)

        target_column = dataset.target
        target_column[target_column == 0] = "setosa"
        target_column[target_column == 1] = "versicolor"
        target_column[target_column == 2] = "virginica"

        df = dataset.data
        df["leaf_type"] = target_column

        variables = jpt.variables.infer_from_dataframe(df, scale_numeric_types=False)

        model = jpt.pca_trees.PCAJPT(variables, min_samples_leaf=0.2)
        model = model.fit(df)
        # model.plot(plotvars=model.variables)


class TestPCAJPT2D(unittest.TestCase):

    def setUp(self) -> None:
        self.distribution_1 = numpy.random.uniform([1, 1], [2, 2], (100, 2)).astype(np.float32)
        distribution_2 = numpy.random.uniform([6, 7], [8, 9], (120, 2)).astype(np.float32)
        rotation = np.array([[0.239746, 1.],
                             [0., 0.320312]])
        self.distribution_2 = distribution_2.dot(rotation) + [3, 0]

        self.data = np.concatenate((self.distribution_1, self.distribution_2))

        variables = [jpt.variables.NumericVariable("X"), jpt.variables.NumericVariable("Y")]

        self.model = jpt.pca_trees.PCAJPT(variables, min_samples_leaf=0.3)

        self.model.fit(self.data)

    def test_separating_line(self):
        """
        This test case has to be verified visually, by confirming that the drawn line looks distances maximizing.
        """
        def eval_(x):
            return (self.model.root.splits[0].upper - self.model.root.variables["X"] * x) / \
                   self.model.root.variables["Y"]

        # create line coordinates describing the separating plane
        p1 = np.array([[0, eval_(0)]])
        p2 = np.array([[6, eval_(6)]])
        ps = np.concatenate((p1, p2),)

        fig = go.Figure()

        # plot data
        fig.add_trace(go.Scatter(x=self.data[:, 0], y=self.data[:, 1], mode="markers", name="Ground Truth"))

        # plot separating plane
        fig.add_trace(go.Scatter(x=ps[:, 0], y=ps[:, 1], name="Separating Plane"))

        coordinate_frame = np.array([[0, 1], [0, 0], [1, 0]])
        for leaf in self.model.leaves.values():
            transformed_coordinate_frame = leaf.inverse_transform(coordinate_frame)
            fig.add_trace(go.Scatter(x=transformed_coordinate_frame[:, 0], y=transformed_coordinate_frame[:, 1],
                                     mode="lines+markers", name="Principal Components of Leaf %s" % leaf.idx))
        # fig.show()

    def test_likelihood(self):
        fig = go.Figure()
        fig.add_trace(plot_numeric_pdf(self.model.leaves[2].distributions["X"]))
        fig.add_trace(go.Scatter(x=self.model.leaves[2].transform(self.distribution_2)[:, 0],
                                 y=np.zeros(len(self.distribution_2)), mode="markers"))
        fig.show()
        fig = go.Figure()
        fig.add_trace(plot_numeric_pdf(self.model.leaves[2].distributions["Y"]))
        fig.add_trace(go.Scatter(x=self.model.leaves[2].transform(self.distribution_2)[:, 1],
                                 y=np.zeros(len(self.distribution_2)), mode="markers"))
        fig.show()
        likelihood = self.model.likelihood(self.distribution_2)

        # todo ask daniel how to solve that problem on the quantile distribution level
        print(sum(np.log(likelihood)))

    def test_expectation_leaf_no_evidence(self):
        """Test expectation function in each leaf"""

        mean_1 = self.distribution_1.mean(axis=0).reshape(1, -1)
        mean_2 = self.distribution_2.mean(axis=0).reshape(1, -1)
        means = np.concatenate((mean_1, mean_2))

        for leaf in self.model.leaves.values():
            expectation = leaf.expectation()
            for idx, (variable, value) in enumerate(expectation.items()):
                closer_mean = means[np.argmin(np.abs(means[:, idx] - value)), idx]
                self.assertAlmostEqual(closer_mean, value)

    def test_transforms(self):
        leaf = self.model.leaves[1]
        data_ = leaf.inverse_transform(leaf.transform(self.data.copy()))
        self.assertAlmostEqual(0., np.sum(self.data-data_))

    def test_posterior(self):
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = [0, 10]
        posterior = self.model.posterior(evidence=evidence)
        for v, d in posterior.result.items():
            d.plot()
            plt.show()

    def test_expectation_tree_no_evidence(self):
        pass

if __name__ == '__main__':
    unittest.main()
