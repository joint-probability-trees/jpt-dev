import itertools
from typing import Tuple, Union

import numpy as np

from jpt import NumericVariable, JPT


def pdf_grid_3d(
        jpt: JPT,
        variable1: Union[NumericVariable, str],
        variable2: Union[NumericVariable, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Create a mesh grid consistent of ``x``, ``y``, ``z`` coordinate values
    representing all distinct interval boundaries forming the joint probability
    density function a projection of ``variable1`` and ``variable2``
    represented the JPT passed the first argument.

    :param jpt:
    :param variable1:
    :param variable2:
    :return:
    '''

    var1_boundaries = set()
    var2_boundaries = set()

    for leaf in jpt.leaves.values():

        var1_boundaries.update(
            itertools.chain(
                *[(i.min, i.max) for i in leaf.distributions[variable1].pdf.intervals]
            )
        )
        var2_boundaries.update(
            itertools.chain(
                *[(i.min, i.max) for i in leaf.distributions[variable2].pdf.intervals]
            )
        )

    v1_values = list(sorted(filter(np.isfinite, var1_boundaries)))
    v2_values = list(sorted(filter(np.isfinite, var2_boundaries)))

    X, Y = np.meshgrid(v1_values, v2_values)

    Z = np.zeros(X.shape)
    for i, (row_x, row_y) in enumerate(zip(X, Y)):
        for j, (x, y) in enumerate(zip(row_x, row_y)):
            Z[i, j] = jpt.pdf(
                jpt.bind(
                    {
                        variable1: x,
                        variable2: y
                    }
                )
            )
    return X, Y, Z
