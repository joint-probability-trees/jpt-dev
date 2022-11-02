import unittest
import sklearn.datasets
import jpt.pca_trees
import jpt.trees
import jpt.variables
import dnutils
import plotly.express as px
import numpy.random
import numpy as np
import plotly.graph_objects as go

class TestPCAJPT(unittest.TestCase):

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

    def test_pca_tree_2_distributions(self):
        distribution_1 = numpy.random.uniform([1, 1], [2, 2], (100, 2))
        distribution_2 = numpy.random.uniform([6, 7], [8, 9], (120, 2))
        rotation = np.array([[0.239746, 1.],
                             [0., 0.320312]])
        distribution_2 = distribution_2.dot(rotation) + [3, 0]

        data = np.concatenate((distribution_1, distribution_2))

        variables = [jpt.variables.NumericVariable("X"), jpt.variables.NumericVariable("Y")]

        model = jpt.pca_trees.PCAJPT(variables, min_samples_leaf=0.3)

        model.fit(data)

        def eval_(x):
            return (model.root.splits[0].upper - model.root.variables["X"] * x) / model.root.variables["Y"]

        p1 = np.array([[0, eval_(0)]])
        p2 = np.array([[6, eval_(6)]])
        ps = np.concatenate((p1, p2),)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers"))
        fig.add_trace(go.Scatter(x=ps[:, 0], y=ps[:, 1]))
        # fig.show()
        # px.scatter(x=data[:, 0], y=data[:, 1]).show()

    def test_likelihood(self):
        distribution_1 = numpy.random.uniform([1, 1], [2, 2], (100, 2))
        distribution_2 = numpy.random.uniform([6, 7], [8, 9], (120, 2))
        rotation = np.array([[0.239746, 1.],
                             [0., 0.320312]])
        distribution_2 = distribution_2.dot(rotation) + [3, 0]

        data = np.concatenate((distribution_1, distribution_2))

        variables = [jpt.variables.NumericVariable("X"), jpt.variables.NumericVariable("Y")]

        model = jpt.pca_trees.PCAJPT(variables, min_samples_leaf=0.3)

        model.fit(data)
        model.plot(plotvars=model.variables)
        likelihood = model.likelihood(data)
        print(likelihood)


if __name__ == '__main__':
    unittest.main()
