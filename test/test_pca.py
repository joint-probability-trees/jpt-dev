import unittest
import sklearn.datasets
import jpt.pca_trees
import jpt.variables
import dnutils


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
        jpt.pca_trees.PCAJPT.logger.level = dnutils.DEBUG
        model.fit(df)


if __name__ == '__main__':
    unittest.main()
