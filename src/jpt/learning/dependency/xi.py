"""Xi-correlation-based dependency discovery.

Uses Chatterjee's xi coefficient to determine
which feature-target pairs exhibit statistically
significant functional dependence.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from jpt.base.correlation.xi import (
    xi_correlation_matrix,
)
from jpt.learning.dependency.base import (
    DependencyDiscovery,
)
from jpt.variables import Variable


# ------------------------------------------------------


class XiDependencyDiscovery(DependencyDiscovery):
    """Dependency discovery via Chatterjee's xi.

    Computes the xi correlation between all
    feature-target pairs and retains only those
    where the correlation is statistically
    significant under the asymptotic null
    distribution of xi.

    Under H0 (independence with continuous Y),
    sqrt(n) * xi ~ N(0, 2/5).

    :param alpha: significance level for the
                  independence test
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha: float = alpha

    def __call__(
            self,
            data: np.ndarray,
            features: list[Variable],
            targets: list[Variable],
            variables: list[Variable]
    ) -> dict[Variable, list[Variable]]:
        """Discover dependencies via xi correlation.

        :param data:      data array (n x d)
        :param features:  list of feature Variables
        :param targets:   list of target Variables
        :param variables: list of all Variables
        :returns:         dict mapping features to
                          their dependent targets
        """
        n: int = data.shape[0]
        z_crit: float = norm.ppf(1. - self.alpha)

        feat_idx: np.ndarray = np.array(
            [variables.index(f) for f in features],
            dtype=np.int64
        )
        tgt_idx: np.ndarray = np.array(
            [variables.index(t) for t in targets],
            dtype=np.int64
        )

        xi_mat: np.ndarray = xi_correlation_matrix(
            np.ascontiguousarray(
                data, dtype=np.float64
            ),
            np.ascontiguousarray(feat_idx),
            np.ascontiguousarray(tgt_idx)
        )

        denom: float = np.sqrt(2. / 5.)
        dependencies: dict[
            Variable, list[Variable]
        ] = {}

        for i, feat in enumerate(features):
            deps: list[Variable] = []
            for j, tgt in enumerate(targets):
                z: float = (
                    np.sqrt(n)
                    * xi_mat[i, j]
                    / denom
                )
                if z > z_crit:
                    deps.append(tgt)
            dependencies[feat] = deps

        return dependencies

    def to_json(self) -> dict[str, Any]:
        """Serialize configuration.

        :returns: JSON-serializable dict
        """
        return {
            'type': self.__class__.__name__,
            'alpha': self.alpha,
        }

    @classmethod
    def from_json(
            cls,
            data: dict[str, Any]
    ) -> XiDependencyDiscovery:
        """Restore from JSON.

        :param data: dict from ``to_json()``
        :returns:    XiDependencyDiscovery instance
        """
        return cls(alpha=data['alpha'])
