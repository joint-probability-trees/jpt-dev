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
            4,
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
                        '[4.300,6.900)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.500,3.500)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[1.000,1.700)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.07714629444766113
        )
        s2 = (
            LabelAssignment({
                'sepal length (cm)':
                    ContinuousSet.parse(
                        '[4.300,6.900)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.500,3.500)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[3.300,6.100)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.037342089333708306
        )
        s3 = (
            LabelAssignment({
                'sepal length (cm)':
                    ContinuousSet.parse(
                        '[4.300,6.900)'
                    ),
                'sepal width (cm)':
                    ContinuousSet.parse(
                        '[2.000,2.500)'
                    ),
                'petal width (cm)':
                    ContinuousSet.parse(
                        '[0.100,0.200)'
                    ),
                'petal length (cm)':
                    ContinuousSet.parse(
                        '[1.000,1.700)'
                    ),
                'plant': {
                    'versicolor',
                    'virginica',
                    'setosa'
                },
            }, variables=self.model.variables),
            0.024797023215319652
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

        # Assert
        self.assertTrue(
            all([
                l1 > l2
                for (_, l1), (_, l2) in pairwise(k_mpe)
            ]),
            msg="Not all solutions are ordered by "
                "descending likelihood"
        )

        self.assertEqual(
            project(sorted_joint_states, 0),
            project(k_mpe, 0),
            msg='MPE state sequences differ from the '
                'brute force solution.'
        )

        self.assertEqual(
            [
                round(v, 8)
                for v in project(
                    sorted_joint_states, 1
                )
            ],
            [round(v, 8) for v in project(k_mpe, 1)],
            msg="These should be equal to the number of "
                "solutions that produce different "
                "likelihoods (set-wise), which is 72 "
                "for this experiment. 216 (current) is "
                "the number of unique solutions iff "
                "sets are not regarded."
        )
