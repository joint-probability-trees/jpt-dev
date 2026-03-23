"""Xi-correlation-based pruning criterion.

Uses Chatterjee's xi coefficient to decide
whether a split is justified by statistically
significant functional dependence between
features and targets in the current partition.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from jpt.base.correlation.xi import xi_correlation


# --------------------------------------------------


class XiPruningCriterion:
    """Prune-or-split callback based on xi.

    Tests whether any feature-target pair shows
    significant functional dependence in the
    current partition. Under H0 (independence),
    sqrt(n) * xi ~ N(0, 2/5).

    :param alpha: significance level
    :param min_n: minimum samples to apply test
    """

    def __init__(
            self,
            alpha: float = 0.05,
            min_n: int = 30
    ) -> None:
        self.z_crit: float = norm.ppf(1. - alpha)
        self.min_n: int = min_n

    def __call__(
            self,
            jpt: 'JPT',
            partition: 'JPTPartition',
            indices: np.ndarray,
            data: np.ndarray
    ) -> bool:
        """Return True to prune (stop splitting).

        :param jpt:       the JPT being learned
        :param partition: current data partition
        :param indices:   index buffer mapping
                          positions to data rows
        :param data:      the full training data
                          array (n_samples x n_vars)
        :returns:         True if the node should
                          become a leaf
        """
        n: int = partition.n_samples
        if n < self.min_n:
            return False

        start: int = partition.start
        end: int = partition.end
        idx: np.ndarray = indices[start:end]

        sqrt_n: float = np.sqrt(n)
        denom: float = np.sqrt(2. / 5.)

        for feat in jpt.features:
            fi: int = jpt.variables.index(feat)
            x: np.ndarray = data[idx, fi]

            deps = jpt.dependencies.get(
                feat, []
            )
            for tgt in deps:
                ti: int = (
                    jpt.variables.index(tgt)
                )
                y: np.ndarray = data[idx, ti]

                xi: float = xi_correlation(x, y)
                z: float = sqrt_n * xi / denom

                if z > self.z_crit:
                    return False

        return True
