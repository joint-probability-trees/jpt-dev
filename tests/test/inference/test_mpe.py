import itertools
from math import prod
from operator import itemgetter
from unittest import TestCase

import numpy as np
import pandas as pd
import sklearn.datasets
from dnutils import project

from jpt.base.utils import pairwise
from jpt.distributions import Numeric
from jpt.trees import MPESolver, JPT
from jpt.variables import (
    VariableMap,
    NumericVariable,
    infer_from_dataframe,
    LabelAssignment
)
from jpt.base.intervals import ContinuousSet


# ----------------------------------------------------------------------

class MPESolverTest(TestCase):

    def test_mpe_numeric(self):
        """Verify MPE solver returns solutions in descending likelihood order."""
        # Arrange
        data = np.array(
            [[1], [2], [8], [9]], dtype=np.float64
        )
        dist1 = Numeric().fit(data)
        dist2 = Numeric().fit(data)

        v1 = NumericVariable('X')
        v2 = NumericVariable('Y')
        mpe = MPESolver(
            VariableMap({
                v1: dist1,
                v2: dist2
            })
        )

        # Act
        solutions = list(mpe.solve())

        # Assert
        self.assertEqual(
            9,
            len(solutions)
        )
        self.assertTrue(
            all(
                l1 >= l2 for l1, l2 in
                pairwise(project(solutions, 1))
            )
        )


# ----------------------------------------------------------------------

class KMPELeafTest(TestCase):
    data: pd.DataFrame
    model: JPT

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(69)
        dataset = sklearn.datasets.load_iris()
        df = pd.DataFrame(
            columns=dataset.feature_names,
            data=dataset.data
        )

        target = dataset.target.astype(object)
        for idx, target_name in enumerate(
            dataset.target_names
        ):
            target[target == idx] = target_name

        df["plant"] = target

        cls.data = df
        cls.model = JPT(
            variables=infer_from_dataframe(
                cls.data,
                scale_numeric_types=False,
                precision=0.05
            ),
            min_samples_leaf=0.9
        )
        cls.model.fit(cls.data)

    def test_k3_mpe(self):
        """Verify top-3 MPE solutions match expected assignments."""
        # Arrange
        # Act
        k_mpe = list(
            self.model.kmpe(k=3)
        )

        # Assert
        s1 = (
            LabelAssignment({
                'sepal length (cm)':
                    ContinuousSet.parse(
                        '[4.800,5.100)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.900,3.000)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[1.200,1.600)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.4249362406671988
        )
        s2 = (
            LabelAssignment({
                'sepal length (cm)':
                    ContinuousSet.parse(
                        '[5.100,6.900)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.900,3.000)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[1.200,1.600)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.27195919402700697
        )
        s3 = (
            LabelAssignment({
                'sepal length (cm)':
                    ContinuousSet.parse(
                        '[4.800,5.100)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.900,3.000)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[3.900,5.200)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.1863181978310025
        )
        self.assertEqual(
            len(k_mpe),
            3
        )
        self.assertEqual(
            [s1, s2, s3],
            k_mpe
        )

    def test_k_mpe_brute(self):
        """Verify k-MPE results match brute-force enumeration."""
        # Arrange
        assert len(self.model.leaves) == 1
        leaf = next(
            iter(self.model.leaves.values())
        )

        # calculate likelihood wise unique solutions
        all_states = [
            list(d.k_mpe())
            for d in leaf.distributions.values()
        ]
        all_joint_states = [
            (project(pair, 0), prod(project(pair, 1)))
            for pair in itertools.product(*all_states)
        ]
        sorted_joint_states = list(
            sorted(
                all_joint_states,
                reverse=True,
                key=itemgetter(1)
            )
        )
        sorted_joint_states = [
            (
                LabelAssignment(
                    (var, val)
                    for var, val in zip(
                        self.model.variables, s
                    )
                ),
                l
            )
            for s, l in sorted_joint_states
        ]

        # Act
        k_mpe = list(
            self.model.kmpe(
                k=len(all_joint_states) + 1000
            )
        )

        # Assert — allow tiny float noise in ties
        self.assertTrue(
            all([
                l1 >= l2 - 1e-12
                for (_, l1), (_, l2) in pairwise(k_mpe)
            ]),
            msg="Not all solutions are ordered by "
                "descending likelihood"
        )

        # Compare the multiset of likelihoods: ties may
        # reorder between kmpe and brute force, but the
        # rounded set of likelihood values must match.
        self.assertEqual(
            sorted(
                round(v, 8)
                for v in project(
                    sorted_joint_states, 1
                )
            ),
            sorted(
                round(v, 8)
                for v in project(k_mpe, 1)
            ),
            msg='Likelihood multisets differ between '
                'kmpe and brute-force enumeration.'
        )

        # The count of distinct solutions must match.
        self.assertEqual(
            len(sorted_joint_states),
            len(k_mpe),
            msg='Solution count differs from brute force'
        )
