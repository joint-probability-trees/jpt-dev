from unittest import TestCase

import numpy as np
import pandas as pd

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


# ----------------------------------------------------------------------

class LikelihoodTest(TestCase):

    # noinspection PyMethodMayBeStatic
    def test_likelihood(self):
        # Arrange
        df = pd.DataFrame([
                [1.2, 2, 'A'],
                [1.5, 3, 'B'],
                [1.6, 2, 'B']
            ],
            columns=['a', 'b', 'c']
        )
        jpt = JPT(
            infer_from_dataframe(
                df, scale_numeric_types=False
            )
        )
        jpt.learn(df)

        # Act
        likelihoods = jpt.likelihood(
            df,
            dirac_scaling=.1,
            variables=jpt.variables,
        )

        # Assert
        np.testing.assert_array_almost_equal(
            np.ones(3),
            likelihoods,
            decimal=8
        )

    def test_single_likelihoods(self):
        # Arrange
        df = pd.DataFrame([
                [1.2, 2, 'A'],
                [1.5, 3, 'B'],
                [1.6, 2, 'B']
            ],
            columns=['a', 'b', 'c']
        )
        jpt = JPT(
            infer_from_dataframe(
                df, scale_numeric_types=False
            )
        )
        jpt.learn(df)

        # Act
        likelihoods_single = jpt.likelihood(
            df,
            dirac_scaling=1,
            single_likelihoods=True,
            variables=jpt.variables
        )

        likelihoods = jpt.likelihood(
            df,
            dirac_scaling=1,
            single_likelihoods=False,
            variables=jpt.variables
        )

        # Assert
        np.testing.assert_array_equal(
            likelihoods,
            likelihoods_single.prod(axis=1)
        )

    def test_single_likelihoods(self):
        # Arrange
        df = pd.DataFrame([
                [1.2, 2, 'A'],
                [1.5, 3, 'B'],
                [1.6, 2, 'B']
            ],
            columns=['a', 'b', 'c']
        )
        jpt = JPT(
            infer_from_dataframe(
                df, scale_numeric_types=False
            )
        )
        jpt.learn(df)

        # Act
        likelihoods_single = jpt.likelihood(
            df,
            dirac_scaling=1,
            single_likelihoods=True
        )

        likelihoods = jpt.likelihood(
            df,
            dirac_scaling=1,
            single_likelihoods=False
        )

        # Assert
        np.testing.assert_array_equal(
            likelihoods,
            likelihoods_single.prod(axis=1)
        )
