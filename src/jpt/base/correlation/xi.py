"""Chatterjee's xi correlation coefficient.

A rank-based measure of functional dependence
introduced by Chatterjee (JASA, 2021). The
coefficient equals 0 iff X and Y are independent
and 1 iff Y is a measurable function of X.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------


def xi_correlation(
        x: np.ndarray,
        y: np.ndarray
) -> float:
    """Compute Chatterjee's xi correlation.

    Measures the degree to which ``y`` is a
    measurable function of ``x``. Requires only
    two sorts and a linear pass, giving
    O(n log n) complexity.

    :param x: feature values, shape (n,)
    :param y: target values, shape (n,)
    :returns: xi coefficient in [-0.5, 1]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n: int = len(x)

    if n < 3:
        return 0.

    order: np.ndarray = np.argsort(
        x, kind='mergesort'
    )
    y_sorted: np.ndarray = y[order]

    y_ranks: np.ndarray = np.empty(
        n, dtype=np.float64
    )
    y_order: np.ndarray = np.argsort(
        y_sorted, kind='mergesort'
    )
    y_ranks[y_order] = np.arange(1, n + 1)

    diff_sum: float = np.sum(
        np.abs(np.diff(y_ranks))
    )

    return 1. - 3. * diff_sum / (n * n - 1.)


# --------------------------------------------------


def xi_correlation_matrix(
        data: np.ndarray,
        feature_indices: np.ndarray,
        target_indices: np.ndarray,
        row_indices: np.ndarray | None = None
) -> np.ndarray:
    """Compute xi for all feature-target pairs.

    Returns a matrix M where
    M[i, j] = xi(data[:, feature_indices[i]],
                  data[:, target_indices[j]]).

    :param data:            array (n x d)
    :param feature_indices: feature column indices
    :param target_indices:  target column indices
    :param row_indices:     optional row subset
    :returns:               xi matrix (nf x nt)
    """
    data = np.asarray(data, dtype=np.float64)

    if row_indices is not None:
        data = data[row_indices]

    nf: int = len(feature_indices)
    nt: int = len(target_indices)
    result: np.ndarray = np.zeros(
        (nf, nt), dtype=np.float64
    )

    for i, fi in enumerate(feature_indices):
        x: np.ndarray = data[:, fi]
        for j, ti in enumerate(target_indices):
            y: np.ndarray = data[:, ti]
            result[i, j] = xi_correlation(x, y)

    return result
