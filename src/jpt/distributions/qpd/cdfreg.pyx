# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np

from libc.stdio cimport printf

from dnutils import ifnone

from jpt.base.utils import pairwise

cdef DTYPE_t pinf = np.inf

from libc.math cimport fabs as abs


# A quantile step is classified as a jump when it exceeds this multiple
# of the median quantile step across the dataset.
JUMP_THR_FACTOR = 10


# ------------------------------------------------------------------------------

cdef class CDFRegressor:
    '''
    Piecewise-linear approximator for empirical CDFs.

    .. deprecated::
        ``CDFRegressor`` is retained for backward compatibility but
        is no longer used by ``QuantileDistribution.fit()``, which
        now delegates to :class:`jpt.distributions.qpd.vwcdfreg
        .VWCDFRegressor` — a Visvalingam-Whyatt based greedy
        bottom-up L∞ regressor with a stronger fit guarantee
        (sup-norm ≤ eps against every input point, not just against
        subsampled breakpoints) and cleaner jump semantics (real
        probability-mass jumps from duplicate samples are
        represented as explicit step discontinuities rather than
        steep linear ramps). New code should use
        :class:`VWCDFRegressor` directly.

    Fits a minimal set of linear segments to a sorted 2×N array of
    (x, quantile) pairs. Historically detected probability-mass jumps
    in a dedicated pre-scan; that heuristic has been removed in
    favour of the deduplication-driven representation used by the
    simplifier (see the ``CHANGELOG`` / commit history).
    '''

    def __init__(
            self,
            eps: float = None,
            max_splits: int = None,
            delta_min: float = np.nan,
            jump_factor: float = 10.0
    ):
        '''
        :param eps:         Maximum absolute deviation tolerance. The fit
                            guarantees that every data point lies within
                            ``eps`` of the piecewise-linear approximation
                            (sup-norm / L-infinity bound). Default: 0.
        :param max_splits:  Maximum recursion depth for the forward pass
                            (-1 = unlimited).
        :param delta_min:   Deprecated; ignored.
        :param jump_factor: Sensitivity of the jump detector. An x-gap
                            between consecutive samples is classified as
                            a jump when ``Δx > jump_factor × median(Δx)``.
                            Lower values => more sensitive (more jumps);
                            higher values => stricter. Default: 10.0.
        '''
        import warnings
        warnings.warn(
            'CDFRegressor is deprecated; '
            'QuantileDistribution.fit() now uses '
            'jpt.distributions.qpd.vwcdfreg.VWCDFRegressor. '
            'Use VWCDFRegressor directly for new code.',
            DeprecationWarning,
            stacklevel=2,
        )
        self.eps = ifnone(eps, .0)
        self.max_splits = ifnone(max_splits, -1)
        self.jump_factor = jump_factor
        self.data = None

    @property
    def support_points(self):
        '''
        Return the selected breakpoints as an (M, 2) array of (x, quantile)
        pairs.

        Jump discontinuities are represented as two consecutive rows sharing
        the same x: the first carries the quantile value just *before* the
        jump, the second carries the value just *after*.
        '''
        cdef SIZE_t idx
        jump_set = set(self._jump_indices)
        pts = []
        for idx in self.points:
            if idx in jump_set:
                # Emit the left edge of the jump: same x, previous y
                prev_y = pts[len(pts) - 1][1] if pts else 0.
                pts.append(np.array([
                    np.nextafter(self.data[0, idx], self.data[0, idx] - 1.),
                    prev_y
                ]))
            pts.append(np.asarray(self.data[:, idx]))
        return np.array(pts) if pts else np.empty((0, 2))

    cpdef void fit(self, DTYPE_t[:, ::1] data):
        '''
        Fit a piecewise-linear CDF to ``data``.

        :param data: 2×N C-contiguous float64 array.  Row 0 is the sorted
                     x-coordinate; row 1 is the corresponding quantile in
                     [0, 1].
        '''
        self.data = data
        cdef SIZE_t n_samples = data.shape[1]

        self._jump_indices.clear()
        self.points.clear()
        while self._points.size():
            self._points.pop()

        if n_samples <= 1:
            self._points.push(0)
            self._backward()
            return

        # Automatic jump detection is disabled. On real empirical
        # data, neither Δy-based nor Δx-based detectors cleanly
        # distinguish genuine probability-mass discontinuities from
        # random order-statistic variation: smooth samples produce
        # many false positives while true clusters (duplicate
        # samples) leave no x-space trace after dedup. The L∞
        # forward-backward pass approximates steep regions — including
        # cluster-induced steep segments — to within ``eps`` without
        # needing special treatment. ``_jump_indices`` is left empty;
        # ``support_points`` accordingly returns a pure-linear knot
        # sequence.
        pass

        # ------------------------------------------------------------------
        # 2. Build contiguous segments separated by detected jumps.
        #    Jump at index k means segment k-1 ends at k-1 and segment k
        #    starts at k.
        # ------------------------------------------------------------------
        seg_starts = [0]
        seg_ends = []
        for j in self._jump_indices:
            seg_ends.append(j - 1)
            seg_starts.append(j)
        seg_ends.append(n_samples - 1)

        # ------------------------------------------------------------------
        # 3. Fit each segment independently and collect breakpoints in order.
        # ------------------------------------------------------------------
        all_bp = []
        cdef (SIZE_t, SIZE_t, DTYPE_t, SIZE_t) args

        for seg_start, seg_end in zip(seg_starts, seg_ends):
            # Reset internal state
            while self._points.size():
                self._points.pop()
            self.points.clear()
            while self._queue.size():
                self._queue.pop_front()

            self._points.push(seg_start)

            if seg_end > seg_start:
                self._points.push(seg_end)
                self._queue.push_back((seg_start, seg_end, -1, 0))

                while self._queue.size():
                    args = self._queue.front()
                    self._queue.pop_front()
                    self._forward(args[0], args[1], args[2], args[3])

            self._backward()
            for bp in self.points:
                all_bp.append(bp)

        # Populate self.points with the merged, ordered breakpoints.
        self.points.clear()
        for bp in all_bp:
            self.points.push_back(bp)

    def verify(self, data):
        '''
        Assert that every ``(x, quantile)`` pair in ``data`` lies within
        ``self.eps`` (absolute deviation) of the fitted piecewise-linear
        CDF.

        :param data: iterable of (x, quantile) pairs to check.
        :raises AssertionError: if any point deviates by more than eps.
        '''
        pts = np.asarray(self.support_points)
        xs = pts[:, 0]
        ys = pts[:, 1]
        for dx, dy in data:
            cdfy = np.interp(dx, xs, ys)
            assert abs(cdfy - dy) < self.eps

    cdef inline void _forward(
            self,
            SIZE_t start,
            SIZE_t end,
            DTYPE_t parent_err,
            SIZE_t depth
    ):
        '''
        Recursive forward pass: find the split of the interval
        ``[start, end]`` that minimises the worst-case absolute residual
        of a two-segment chord fit, and enqueue the two sub-intervals.

        :param start:      index of the leftmost point of the interval.
        :param end:        index of the rightmost point of the interval.
        :param parent_err: parent max absolute residual (-1 on the
                           initial call, meaning "accept any split").
        :param depth:      current recursion depth.
        '''
        # Base case: no inner points, or depth limit reached
        if end - start == 1 or 0 < self.max_splits <= depth:
            return

        cdef:
            DTYPE_t err_min = pinf
            SIZE_t best_split = -1
            DTYPE_t best_err_left = 0., best_err_right = 0.
            DTYPE_t[:, ::1] data = self.data

            DTYPE_t xl_ = data[0, start], yl_ = data[1, start]
            DTYPE_t xr_ = data[0, end],   yr_ = data[1, end]

            DTYPE_t m1, m2, c1, c2, r, err_left, err_right, overall
            SIZE_t split, i

        for split in range(start + 1, end):
            # Chord on the left segment passes through
            # (xl_, yl_) and (data[0, split], data[1, split])
            m1 = (
                (data[1, split] - yl_)
                / (data[0, split] - xl_)
            )
            c1 = yl_ - m1 * xl_
            # Chord on the right segment passes through
            # (data[0, split], data[1, split]) and (xr_, yr_)
            m2 = (
                (yr_ - data[1, split])
                / (xr_ - data[0, split])
            )
            c2 = yr_ - m2 * xr_

            err_left = 0.
            for i in range(start + 1, split):
                r = m1 * data[0, i] + c1 - data[1, i]
                if r < 0:
                    r = -r
                if r > err_left:
                    err_left = r

            err_right = 0.
            for i in range(split + 1, end):
                r = m2 * data[0, i] + c2 - data[1, i]
                if r < 0:
                    r = -r
                if r > err_right:
                    err_right = r

            overall = (
                err_left if err_left > err_right
                else err_right
            )

            if overall < err_min:
                best_split = split
                err_min = overall
                best_err_left = err_left
                best_err_right = err_right

        if best_split == -1:
            return

        self._points.push(best_split)

        if best_err_left >= self.eps:
            self._queue.push_back(
                (start, best_split, best_err_left, depth + 1)
            )
        if best_err_right >= self.eps:
            self._queue.push_back(
                (best_split, end, best_err_right, depth + 1)
            )

    cdef void _backward(self) nogil:
        '''
        Backward pruning pass: starting from the rightmost selected
        breakpoint and working left, drop any breakpoint whose removal
        keeps the max absolute residual below ``self.eps``.
        '''
        cdef DTYPE_t[:, ::1] data = self.data

        self.points.push_front(self._points.top())
        self._points.pop()

        cdef:
            SIZE_t pos = self._points.top() if self._points.size() else 0
            SIZE_t start, p
            DTYPE_t lx, ly, rx, ry, m, c, err_max, r

        while self._points.size() > 1:
            rx = data[0, self.points.front()]
            ry = data[1, self.points.front()]
            p = self._points.top()
            self._points.pop()
            lx = data[0, self._points.top()]
            ly = data[1, self._points.top()]

            m = (ry - ly) / (rx - lx)
            c = ly - m * lx
            start = pos
            err_max = 0.

            while pos > self._points.top():
                r = m * data[0, pos] + c - data[1, pos]
                if r < 0:
                    r = -r
                if r > err_max:
                    err_max = r
                pos -= 1

            if err_max >= self.eps:
                self.points.push_front(p)
                pos = p - 1
            else:
                pos = start

        if self._points.size():
            self.points.push_front(self._points.top())
