"""Xi-correlation pruning for JPTs.

Demonstrates using Chatterjee's xi correlation
coefficient as a pre-pruning criterion via the
``prune_or_split`` callback. Splits are pruned when
no feature-target pair shows significant functional
dependence in the current partition.

Demonstrates:
    - ``XiPruningCriterion`` as pruning callback
    - ``XiDependencyDiscovery`` for automatic
      dependency structure learning
    - Comparison of pruned vs. unpruned tree sizes
"""
import logging

import numpy as np
from pandas import DataFrame

from jpt.base.correlation import xi_correlation
from jpt.distributions import Numeric
from jpt.learning.dependency import (
    XiDependencyDiscovery,
)
from jpt.learning.pruning import XiPruningCriterion
from jpt.trees import JPT
from jpt.variables import NumericVariable


logger = logging.getLogger(__name__)


# ------------------------------------------------------


def main():
    """Compare tree sizes with and without xi
    pruning.
    """
    np.random.seed(42)
    n = 2000

    # Generate data: Y depends on X1 (quadratic),
    # X2 is pure noise. Noise is high so that
    # dependence fades in smaller partitions.
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    noise = np.random.normal(0, 1.5, n)
    y = x1 ** 2 + noise

    df = DataFrame({'X1': x1, 'X2': x2, 'Y': y})

    varx1 = NumericVariable(
        'X1', Numeric, precision=.1
    )
    varx2 = NumericVariable(
        'X2', Numeric, precision=.1
    )
    vary = NumericVariable(
        'Y', Numeric, precision=.1
    )

    # --- Unpruned tree ---
    tree = JPT(
        variables=[varx1, varx2, vary],
        targets=[vary],
        min_samples_leaf=.01
    )
    tree.learn(df)
    tree.plot(
        plotvars=tree.variables,
        filename='unpruned'
    )
    n_leaves = len(tree.leaves)

    # --- Xi-pruned tree with dependency discovery ---
    tree_pruned = JPT(
        variables=[varx1, varx2, vary],
        targets=[vary],
        dependencies=XiDependencyDiscovery(
            alpha=0.05
        ),
        min_samples_leaf=.01
    )
    tree_pruned.learn(
        df,
        prune_or_split=XiPruningCriterion(
            alpha=0.05
        )
    )
    tree_pruned.plot(
        plotvars=tree_pruned.variables,
        filename='xi-pruned'
    )
    n_leaves_pruned = len(tree_pruned.leaves)

    logger.info(
        'Unpruned: %d leaves, Xi-pruned: %d leaves',
        n_leaves,
        n_leaves_pruned
    )

    # Verify: xi detects X1->Y but not X2->Y
    xi_x1y = xi_correlation(x1, y)
    xi_x2y = xi_correlation(x2, y)
    logger.info(
        'xi(X1, Y) = %.3f, xi(X2, Y) = %.3f',
        xi_x1y,
        xi_x2y
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
