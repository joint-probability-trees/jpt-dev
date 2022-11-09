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
from jpt.base.utils import list2interval
from jpt.distributions.quantile.quantiles import QuantileDistribution


def plot_numeric_pdf(distribution: jpt.distributions.univariate.Numeric, padding=0.1, name="PDF", transpose=False):
    x = []
    y = []
    for interval, function in zip(distribution.pdf.intervals[1:-1], distribution.pdf.functions[1:-1]):
        x += [interval.lower, interval.upper, interval.upper]
        y += [function.value, function.value, None]

    range = x[-1] - x[0]
    x = [x[0] - (range * padding), x[0], x[0]] + x + [x[-1], x[-1], x[-1] + (range * padding)]
    y = [0, 0, None] + y + [None, 0, 0]

    if not transpose:
        trace = go.Scatter(x=x, y=y, name=name)
    else:
        trace = go.Scatter(x=y, y=x, name=name)
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


class TestPCAJPT2D(unittest.TestCase):

    def setUp(self) -> None:
        numpy.random.seed(42)
        self.distribution_1 = numpy.random.uniform([1, 1], [2, 2], (100, 2)).astype(np.float64)
        distribution_2 = numpy.random.uniform([6, 7], [8, 9], (120, 2)).astype(np.float64)
        rotation = np.array([[0.239746, 1.],
                             [0., 0.320312]])
        self.distribution_2 = distribution_2.dot(rotation) + [3, 0]

        self.data = np.concatenate((self.distribution_1, self.distribution_2))

        variables = [jpt.variables.NumericVariable("X", precision=0.1),
                     jpt.variables.NumericVariable("Y", precision=0.1)]

        self.model = jpt.pca_trees.PCAJPT(variables, min_samples_leaf=0.3)

        self.model.fit(self.data)


    def test_separating_line(self):
        """
        This test case has to be verified visually, by confirming that the drawn line looks distances maximizing.
        """

        show = False

        def eval_(x):
            return (self.model.root.splits[0].upper - self.model.root.variables["X"] * x) / \
                   self.model.root.variables["Y"]

        # create line coordinates describing the separating plane
        p1 = np.array([[0, eval_(0)]])
        p2 = np.array([[6, eval_(6)]])
        ps = np.concatenate((p1, p2),)

        fig = go.Figure()
        fig.update_layout(title="Visualization of PCAJPTs")

        # plot data
        fig.add_trace(go.Scatter(x=self.data[:, 0], y=self.data[:, 1], mode="markers", name="Ground Truth"))

        # plot separating plane
        fig.add_trace(go.Scatter(x=ps[:, 0], y=ps[:, 1], name="Separating Plane"))

        for leaf in self.model.leaves.values():
            x_axis_coordinates_eigen = [[leaf.numeric_domains_eigen["X"].lower, 0],
                                        [leaf.numeric_domains_eigen["X"].upper, 0]]
            x_axis_coordinates_data = leaf.inverse_transform(x_axis_coordinates_eigen)

            fig.add_trace(go.Scatter(x=x_axis_coordinates_data[:, 0], y=x_axis_coordinates_data[:, 1],
                                     mode="lines+markers", name="Principal Component 1 of Leaf %s" % leaf.idx))

            y_axis_coordinates_eigen = [[0, leaf.numeric_domains_eigen["Y"].lower],
                                        [0, leaf.numeric_domains_eigen["Y"].upper]]
            y_axis_coordinates_data = leaf.inverse_transform(y_axis_coordinates_eigen)

            fig.add_trace(go.Scatter(x=y_axis_coordinates_data[:, 0], y=y_axis_coordinates_data[:, 1],
                                     mode="lines+markers", name="Principal Component 2 of Leaf %s" % leaf.idx))

            likelihoods = leaf.parallel_likelihood(self.data)
            corresponding_data = self.data[likelihoods > 0]
            corresponding_data_eigen = leaf.transform(corresponding_data)
            leaf_figure = go.Figure()
            leaf_figure.update_layout(title="Leaf %s" % leaf.idx)
            leaf_figure.add_trace(go.Scatter(x=corresponding_data_eigen[:, 0], y=corresponding_data_eigen[:, 1],
                                             mode="markers", name="Ground Truth"))
            leaf_figure.add_trace(plot_numeric_pdf(leaf.distributions["X"], name="PDF of X"))
            leaf_figure.add_trace(plot_numeric_pdf(leaf.distributions["Y"], name="PDF of Y", transpose=True))

            if show:
                leaf_figure.show()

        if show:
            fig.show()

        self.model.plot(plotvars=self.model.variables)

    def test_likelihood(self):
        """Check that the likelihood of each datapoint is > 0"""
        fig = go.Figure()
        fig.add_trace(plot_numeric_pdf(self.model.leaves[2].distributions["X"]))
        fig.add_trace(go.Scatter(x=self.model.leaves[2].transform(self.distribution_1)[:, 0],
                                 y=np.zeros(len(self.distribution_1)), mode="markers"))
        # fig.show()

        fig = go.Figure()
        fig.add_trace(plot_numeric_pdf(self.model.leaves[2].distributions["Y"]))
        fig.add_trace(go.Scatter(x=self.model.leaves[2].transform(self.distribution_1)[:, 1],
                                 y=np.zeros(len(self.distribution_1)), mode="markers"))
        # fig.show()

        likelihood = self.model.likelihood(self.data)

        self.assertTrue(all(likelihood > 0))

    def test_expectation_variance_leaf_no_evidence(self):
        """Test expectation function and variance in each leaf"""

        mean_1 = self.distribution_1.mean(axis=0).reshape(1, -1)
        mean_2 = self.distribution_2.mean(axis=0).reshape(1, -1)
        means = np.concatenate((mean_1, mean_2))

        var_1 = self.distribution_1.var(axis=0).reshape(1, -1)
        var_2 = self.distribution_2.var(axis=0).reshape(1, -1)
        vars = np.concatenate((var_1, var_2))

        for leaf in self.model.leaves.values():
            expectation = leaf.expectation()
            for idx, (variable, value) in enumerate(expectation.items()):
                closer_mean = means[np.argmin(np.abs(means[:, idx] - value)), idx]
                self.assertAlmostEqual(closer_mean, value, places=5)
                self.assertAlmostEqual(vars[np.argmin(np.abs(means[:, idx] - value)), idx], leaf.scaler.var_[idx],
                                       places=5)

    def test_transforms(self):
        """Check that transform and inverse transform are indeed inverse of each other"""
        for leaf in self.model.leaves.values():
            data_ = leaf.inverse_transform(leaf.transform(self.data.copy()))
            self.assertAlmostEqual(0., np.sum(np.abs(self.data-data_)))

    def test_transform_variable_map(self):
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = list2interval([0, 3])
        evidence[self.model.varnames["Y"]] = list2interval([0, 1])

        for leaf in self.model.leaves.values():
            e_ = leaf.transform_variable_map(evidence)
            print(e_)

    def test_posterior(self):
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = list2interval([1, 2])
        evidence[self.model.varnames["Y"]] = list2interval([1, 2])
        posterior = self.model.posterior(evidence=evidence)
        for v, d in posterior.distributions.items():
            self.assertEqual(1., d.p(evidence[v]))
            d.plot(title=v.name)
            plt.show()

        # px.scatter(x=self.data[:, 0], y=self.data[:, 1]).show()

    def test_expectation(self):
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = list2interval([1, 2])
        evidence[self.model.varnames["Y"]] = list2interval([1, 2])

        expectation = self.model.expectation(evidence=evidence)
        print(expectation["X"])


    def test_conditional_tree(self):
        # px.scatter(x=self.data[:, 0], y=self.data[:, 1]).show()
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = list2interval([0, 3])
        conditional_model = self.model.conditional_jpt(evidence)

    def test_infer(self):
        evidence = jpt.variables.VariableMap()
        evidence[self.model.varnames["X"]] = list2interval([1, 2])
        evidence[self.model.varnames["Y"]] = list2interval([1, 2])

        # check base case
        # self.assertEqual(self.model.infer().result, 1.)

        # check with evidence
        # self.assertEqual(self.model.infer(evidence=evidence).result, 1.)
        print("mins", self.distribution_1.min(axis=0))
        print("maxs", self.distribution_1.max(axis=0))
        print(self.model.infer(evidence).result)

if __name__ == '__main__':
    unittest.main()
