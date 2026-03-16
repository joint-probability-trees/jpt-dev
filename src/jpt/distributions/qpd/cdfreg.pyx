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

    Fits a minimal set of linear segments to a sorted 2×N array of
    (x, quantile) pairs.  Probability-mass jumps (large discontinuous
    steps in the quantile sequence) are detected in a dedicated pre-scan
    and stored explicitly, keeping the regression loop free of
    jump-related branching.
    '''

    def __init__(
            self,
            eps: float = None,
            max_splits: int = None,
            delta_min: float = np.nan
    ):
        '''
        :param eps:        RMSE tolerance for segment acceptance.  The value
                           is stored internally as its square (the MSE
                           threshold) so that the inner-loop comparison
                           avoids a square-root.  Passing ``eps=0.01``
                           means "tolerate up to 1 % root-mean-square
                           residual per segment", stored as ``self.eps =
                           0.0001``.  Default: 0 (exact fit).
        :param max_splits: Maximum recursion depth for the forward pass
                           (-1 = unlimited).
        :param delta_min:  Deprecated; ignored.  Jump detection now uses
                           the median quantile step derived from the data.
        '''
        self.eps = ifnone(eps, .0, lambda e: e * e)
        self.max_splits = ifnone(max_splits, -1)
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

        # ------------------------------------------------------------------
        # 1. Pre-identify jumps using a median-based threshold.
        #    A quantile step is a "jump" when it is more than
        #    JUMP_THR_FACTOR times the median step.
        # ------------------------------------------------------------------
        deltas = np.diff(np.asarray(data[1, :n_samples]))
        cdef DTYPE_t threshold = JUMP_THR_FACTOR * np.median(deltas)

        cdef SIZE_t i
        for i in range(n_samples - 1):
            if deltas[i] - threshold > 1e-8:
                self._jump_indices.push_back(i + 1)

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
        Assert that the fitted piecewise-linear CDF approximates each
        (x, quantile) pair in ``data`` within ``self.eps``.

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
            DTYPE_t mse,
            SIZE_t depth
    ):
        '''
        Recursive forward pass: find the best MSE-minimising split of the
        interval [start, end] and enqueue the two sub-intervals.

        :param start: index of the leftmost point of the interval.
        :param end:   index of the rightmost point of the interval.
        :param mse:   parent MSE (negative on the initial call).
        :param depth: current recursion depth.
        '''
        # Base case: no inner points, or depth limit reached
        if end - start == 1 or 0 < self.max_splits <= depth:
            return

        cdef:
            DTYPE_t n_samples = <DTYPE_t> end - start + 1
            DTYPE_t mse_min = mse
            SIZE_t best_split = -1
            DTYPE_t best_mse1 = pinf, best_mse2 = pinf
            DTYPE_t[:, ::1] data = self.data

            DTYPE_t sum_xy_left = 0, sum_y_sq_left = 0, sum_x_sq_left = 0

            DTYPE_t xl_ = data[0, start], yl_ = data[1, start]
            DTYPE_t xr_ = data[0, end],   yr_ = data[1, end]

            DTYPE_t sum_x_sq_right = 0, sum_y_sq_right = 0, sum_xy_right = 0
            SIZE_t i
            DTYPE_t x, y

        for i in range(start + 1, end):
            x = data[0, i] - xr_
            y = data[1, i] - yr_
            sum_x_sq_right += x * x
            sum_y_sq_right += y * y
            sum_xy_right += x * y

        cdef:
            DTYPE_t x_start = data[0, start] - xl_
            DTYPE_t y_start = data[1, start] - yl_
            DTYPE_t x_end = data[0, end] - xr_
            DTYPE_t y_end = data[1, end] - yr_
            DTYPE_t xl, yl, xr, yr, m1, m2
            SIZE_t samples_left, samples_right
            DTYPE_t err_left, err_right, mse_

        cdef SIZE_t split

        for split in range(start + 1, end):
            xl = data[0, split] - xl_
            yl = data[1, split] - yl_
            xr = data[0, split] - xr_
            yr = data[1, split] - yr_
            m1 = (yl - y_start) / (xl - x_start)
            m2 = (y_end - yr) / (x_end - xr)

            samples_left = split - start - 1
            samples_right = end - split - 1

            if samples_left > 0:
                sum_y_sq_left += yl * yl
                sum_x_sq_left += xl * xl
                sum_xy_left += xl * yl

                err_left = (
                    sum_y_sq_left
                    - 2 * sum_xy_left * m1
                    + sum_x_sq_left * m1 * m1
                )
                err_left /= samples_left
            else:
                err_left = 0

            if samples_right > 0:
                sum_y_sq_right -= yr * yr
                sum_x_sq_right -= xr * xr
                sum_xy_right -= xr * yr

                err_right = (
                    sum_y_sq_right
                    - 2 * sum_xy_right * m2
                    + sum_x_sq_right * m2 * m2
                )
                err_right /= samples_right
            else:
                err_right = 0

            mse_ = (
                samples_left / n_samples * err_left
                + samples_right / n_samples * err_right
            )

            if mse_min < 0 or mse_ < mse_min:
                best_split = split
                mse_min = mse_
                best_mse1 = err_left
                best_mse2 = err_right

        if best_split == -1:
            return

        self._points.push(best_split)

        if best_mse1 >= self.eps:
            self._queue.push_back((start, best_split, best_mse1, depth + 1))
        if best_mse2 >= self.eps:
            self._queue.push_back((best_split, end, best_mse2, depth + 1))

    cdef void _backward(self) nogil:
        '''
        Backward pruning pass: starting from the rightmost selected
        breakpoint and working left, drop any breakpoint whose removal
        keeps the linear approximation error below ``self.eps``.
        '''
        cdef DTYPE_t[:, ::1] data = self.data

        self.points.push_front(self._points.top())
        self._points.pop()

        cdef:
            SIZE_t pos = self._points.top() if self._points.size() else 0
            SIZE_t start, p
            DTYPE_t lx, ly, rx, ry, m, c, err

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
            err = 0

            while pos > self._points.top():
                err += (m * data[0, pos] + c - data[1, pos]) ** 2
                pos -= 1

            if err / <DTYPE_t> (start - pos) >= self.eps:
                self.points.push_front(p)
                pos = p - 1
            else:
                pos = start

        if self._points.size():
            self.points.push_front(self._points.top())
