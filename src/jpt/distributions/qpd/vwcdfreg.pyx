# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

"""Visvalingam-Whyatt CDF regressor.

A greedy L∞-optimal piecewise-linear approximator for empirical
CDFs, based on the Visvalingam-Whyatt line simplification
algorithm, adapted from 2D geometry to the 1D CDF setting by
replacing the original triangular-area cost with the max
absolute residual (sup-norm) of the chord fit at every spanned
input point.
"""

import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport INFINITY
from libcpp.queue cimport priority_queue
from libcpp.pair cimport pair

from jpt.base.cutils.cutils cimport DTYPE_t, SIZE_t


# ------------------------------------------------------------------
# Heap entries: (negated_cost, idx). Negation makes the C++ max-heap
# behave as a min-heap on cost. Stale entries are detected by
# comparing the popped cost against the up-to-date cost stored in
# ``costs[idx]``; if they differ the entry is ignored (lazy delete).
# ------------------------------------------------------------------

ctypedef pair[DTYPE_t, SIZE_t] HeapEntry


cdef inline DTYPE_t _removal_cost(
        DTYPE_t[::1] xs,
        DTYPE_t[::1] ys,
        SIZE_t[::1] left,
        SIZE_t[::1] right,
        SIZE_t n,
        SIZE_t i,
) noexcept nogil:
    """Max absolute residual of the chord through
    (xs[left[i]], ys[left[i]]) -> (xs[right[i]], ys[right[i]])
    evaluated at every original point between them in x.
    Returns +inf for endpoints (left < 0 or right >= n), which
    are never candidates for removal.
    """
    cdef:
        SIZE_t l = left[i]
        SIZE_t r = right[i]
        DTYPE_t xl, xr, yl, slope, dx, chord_y, err, max_err
        SIZE_t k
    if l < 0 or r >= n:
        return INFINITY
    xl = xs[l]
    xr = xs[r]
    dx = xr - xl
    if dx == 0.0:
        return INFINITY
    yl = ys[l]
    slope = (ys[r] - yl) / dx
    max_err = 0.0
    for k in range(l + 1, r):
        chord_y = yl + slope * (xs[k] - xl)
        err = chord_y - ys[k]
        if err < 0.0:
            err = -err
        if err > max_err:
            max_err = err
    return max_err


cdef class VWCDFRegressor:
    """Visvalingam-Whyatt piecewise-linear CDF regressor.

    Greedy bottom-up L∞-optimal piecewise-linear approximator for
    empirical CDFs.  The algorithm is the Visvalingam-Whyatt line
    simplification procedure [1]_, adapted from 2D cartographic
    line generalisation to the 1D CDF setting: the original
    triangular-area cost function is replaced by the maximum
    absolute residual (sup-norm) of the chord fit over every
    original sample the chord now spans.  Starting from the full
    empirical support, interior knots are removed one by one --
    cheapest-removal-cost first -- until no further removal can
    be done without exceeding the user-supplied tolerance
    ``eps``.  A lazy-deleted C++ ``priority_queue`` keyed on
    (negated) removal cost drives the loop.

    The sup-norm cost gives a strong guarantee: every original
    input point lies within ``eps`` of the fitted piecewise-linear
    CDF.  Because the algorithm starts from the full empirical
    support, probability mass is preserved by construction; it
    cannot be silently absorbed by early termination the way a
    top-down recursive fitter can.

    Example
    -------
    >>> reg = VWCDFRegressor(eps=0.01)
    >>> reg.fit(xs, ys)
    >>> pts = reg.support_points   # (M, 2) array of surviving (x, y) knots

    References
    ----------
    .. [1] Visvalingam, M. and Whyatt, J. D. (1993).
       "Line generalisation by repeated elimination of points."
       *The Cartographic Journal*, 30(1):46-51.
       https://doi.org/10.1179/000870493786962263
    """

    cdef readonly DTYPE_t eps
    cdef readonly object _fit_xs
    cdef readonly object _fit_ys

    def __init__(self, eps: float = 0.0):
        """
        :param eps: maximum absolute deviation tolerated by the
                    fit; every original input point is guaranteed
                    to lie within ``eps`` of the resulting
                    piecewise-linear CDF.
        """
        self.eps = eps
        self._fit_xs = None
        self._fit_ys = None

    cpdef VWCDFRegressor fit(
            self,
            DTYPE_t[::1] xs,
            DTYPE_t[::1] ys,
    ):
        """Run the simplification on the empirical CDF support
        ``(xs, ys)`` and store the surviving knots.

        :param xs: sorted, unique x-coordinates.
        :param ys: monotone non-decreasing y-values in [0, 1],
                   same length as ``xs``.
        :returns: ``self`` -- for chainable calls.
        """
        fit_xs, fit_ys = _simplify_cdf(xs, ys, self.eps)
        self._fit_xs = fit_xs
        self._fit_ys = fit_ys
        return self

    @property
    def fit_xs(self):
        """X-coordinates of the surviving knots (float64 array)
        or ``None`` if ``fit`` has not been called yet."""
        return self._fit_xs

    @property
    def fit_ys(self):
        """Y-coordinates of the surviving knots (float64 array)
        or ``None`` if ``fit`` has not been called yet."""
        return self._fit_ys

    @property
    def support_points(self):
        """Return the surviving knots as an ``(M, 2)`` float64
        array. Empty if ``fit`` has not run.
        """
        if self._fit_xs is None:
            return np.empty((0, 2), dtype=np.float64)
        return np.column_stack([self._fit_xs, self._fit_ys])


cdef tuple _simplify_cdf(
        DTYPE_t[::1] xs,
        DTYPE_t[::1] ys,
        DTYPE_t eps,
):
    """Internal: return the L∞-minimal subset of the given
    ``(xs, ys)`` knots whose piecewise-linear interpolant matches
    every input point to within ``eps`` in sup-norm.
    """
    cdef:
        SIZE_t n = xs.shape[0]
        SIZE_t i, neighbour
        SIZE_t l, r, idx
        DTYPE_t cost, new_cost, neg_cost
        HeapEntry entry

    if ys.shape[0] != n:
        raise ValueError(
            'xs and ys length mismatch: %d vs %d'
            % (n, ys.shape[0])
        )
    if n <= 2:
        return (
            np.ascontiguousarray(xs, dtype=np.float64).copy(),
            np.ascontiguousarray(ys, dtype=np.float64).copy()
        )

    # Linked-list over active indices. ``left[i] = -1`` marks the
    # left endpoint (no left neighbour); ``right[i] = n`` marks the
    # right endpoint.
    cdef np.ndarray[np.int64_t, ndim=1] left_arr = (
        np.arange(n, dtype=np.int64) - 1
    )
    cdef np.ndarray[np.int64_t, ndim=1] right_arr = (
        np.arange(n, dtype=np.int64) + 1
    )
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] alive_arr = (
        np.ones(n, dtype=bool)
    )
    cdef np.ndarray[np.float64_t, ndim=1] costs_arr = (
        np.zeros(n, dtype=np.float64)
    )
    cdef SIZE_t[::1] left = left_arr
    cdef SIZE_t[::1] right = right_arr
    cdef np.uint8_t[::1] alive = alive_arr
    cdef DTYPE_t[::1] costs = costs_arr

    cdef priority_queue[HeapEntry] heap

    # Seed the heap with initial removal costs for every interior
    # support point.
    for i in range(1, n - 1):
        cost = _removal_cost(xs, ys, left, right, n, i)
        costs[i] = cost
        heap.push(HeapEntry(-cost, i))

    # Greedy loop: pop the cheapest live candidate. If its cost is
    # still current (matches ``costs[idx]``) and below eps, remove
    # the point and push fresh costs for its two neighbours.
    while not heap.empty():
        entry = heap.top()
        heap.pop()
        neg_cost = entry.first
        idx = entry.second
        cost = -neg_cost
        if not alive[idx] or costs[idx] != cost:
            continue
        if cost >= eps:
            break
        # Remove point ``idx`` from the active list.
        alive[idx] = 0
        l = left[idx]
        r = right[idx]
        if l >= 0:
            right[l] = r
        if r < n:
            left[r] = l
        # Recompute cost for its two surviving neighbours.
        for neighbour in (l, r):
            if 0 < neighbour < n - 1 and alive[neighbour]:
                new_cost = _removal_cost(
                    xs, ys, left, right, n, neighbour
                )
                costs[neighbour] = new_cost
                heap.push(HeapEntry(-new_cost, neighbour))

    # Gather the survivors in order.
    cdef list kept_idx = []
    for i in range(n):
        if alive[i]:
            kept_idx.append(i)
    cdef np.ndarray[np.int64_t, ndim=1] idx_arr = np.asarray(
        kept_idx, dtype=np.int64
    )
    return (
        np.asarray(xs)[idx_arr].copy(),
        np.asarray(ys)[idx_arr].copy(),
    )
