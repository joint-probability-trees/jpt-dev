"""Spatial classification with evidence: breakfast
object placement.

Predicts breakfast object placement success using
spatial coordinates and object type. A JPT is learned
over X/Y positions, object class, and success labels,
then queried with interval-based spatial evidence.

Demonstrates:
    - ContinuousSet queries with interval-based evidence
    - 3D conditional probability surfaces
    - Mixed symbolic and numeric variable types
    - ``expectation()`` with symbolic evidence
"""
import logging
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import (
    NumericVariable,
    SymbolicVariable,
)
from jpt.base.intervals import ContinuousSet


logger = logging.getLogger(__name__)


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def plot_conditional(
        jpt,
        qvarx,
        qvary,
        evidence=None,
        fuzziness=0.01,
        visualize=True,
):
    """Plot a 3D surface of conditional probability.

    Evaluates P(X, Y | evidence) over a grid and renders
    the result as a matplotlib 3D surface plot.

    :param jpt:        the learned JPT
    :param qvarx:      the X variable
    :param qvary:      the Y variable
    :param evidence:   optional evidence dictionary
    :param fuzziness:  half-width of query intervals
    :param visualize:  whether to show the plot
    """
    x = np.linspace(.7, 1.05, 50)
    y = np.linspace(.15, .55, 50)
    X, Y = np.meshgrid(x, y)

    Z = np.array([
        jpt.infer(
            {
                qvarx: ContinuousSet(
                    xi - fuzziness, xi + fuzziness
                ),
                qvary: ContinuousSet(
                    yi - fuzziness, yi + fuzziness
                ),
            },
            evidence=evidence
        ).result
        for xi, yi in zip(X.ravel(), Y.ravel())
    ]).reshape(X.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(
        X, Y, Z, cmap='viridis', edgecolor='none'
    )
    evidence_str = (
        ', '.join(
            f'{k.name}={v}' for k, v in evidence.items()
        )
        if evidence else r'$\emptyset$'
    )
    ax.set_title(
        f'P({qvarx.name}, {qvary.name}'
        f' | {evidence_str})'
    )

    if visualize:
        plt.show()


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT from breakfast placement data and
    run spatial queries.

    :param visualize: whether to show interactive plots
    """
    # Load the muesli breakfast dataset
    data = pd.read_csv(
        os.path.join(_DATA_DIR, 'muesli.csv')
    )
    data["Success"] = data["Success"].astype(str)

    # Define variable types
    ObjectType = SymbolicType(
        'ObjectType', data['Class'].unique()
    )
    SuccessType = SymbolicType(
        'Success', data['Success'].unique()
    )

    # Create variables
    x = NumericVariable('X', Numeric, blur=.01)
    y = NumericVariable('Y', Numeric, blur=.01)
    o = SymbolicVariable('Class', ObjectType)
    s = SymbolicVariable('Success', SuccessType)

    # Learn the JPT
    jpt = JPT(
        [x, y, o, s], min_samples_leaf=.2
    )
    jpt.learn(data)

    # Query expected positions per object class
    for clazz in data['Class'].unique():
        for exp in jpt.expectation(
                [x.name, y.name],
                evidence={o.name: clazz}
        ):
            logging.info(exp)

    logging.info(jpt.to_string())

    # Visualize conditional density for one object class
    if visualize:
        plot_conditional(
            jpt, x, y,
            evidence={o: 'BaerenMarkeFrischeAlpenmilch'},
            visualize=visualize
        )

    # Query object class probabilities at a specific
    # spatial location
    fuzziness = 0.01
    for clazz in data['Class'].unique():
        res = jpt.infer(
            query={o.name: clazz},
            evidence={
                x.name: [
                    .95 - fuzziness,
                    .95 + fuzziness,
                ],
                y.name: [
                    .45 - fuzziness,
                    .45 + fuzziness,
                ],
            }
        )
        logger.info(res)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(visualize=True)
