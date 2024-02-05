# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np

from libc.stdio cimport printf

from dnutils import ifnone

from jpt.base.utils import pairwise

cdef DTYPE_t pinf = np.PINF


cdef DTYPE_t DELTA_MIN_THR = 2


cdef class CDFRegressor:
    '''Experimental quantile regression.'''

    def __init__(self, eps: float = None, max_splits: int = None, delta_min: float = None):
        self.eps = ifnone(eps, .000, lambda _: _ * _)
        self.max_splits = ifnone(max_splits, -1)
        self.data = None
        self.delta_min = delta_min

    @property
    def support_points(self):
        points = []
        if self.points.size() < 2:
            return points.append(self.data[:, self.points[0]])
        else:
            for i, j in pairwise([_ for _ in self.points]):
                if i == j:
                    points.append(
                        np.array([
                            np.nextafter(self.data[0, i], self.data[0, i] - 1.),
                            0 if not i else points[len(points) - 1][1]
                        ])
                    )
                    # print(points[len(points) - 1])
                else:
                    points.append(self.data[:, i])

            points.append(self.data[:, j])
        return np.array(points)

    cpdef void fit(self, DTYPE_t[:, ::1] data):
        '''
        
        :param data: 2xN array, where the first row is the x ccordinate of the data, and the 2nd row is the quantile.
         
        :return: 
        '''
        self.data = data
        cdef SIZE_t n_samples = self.data.shape[1]

        self._points.push(0)
        cdef (SIZE_t, SIZE_t, DTYPE_t, SIZE_t) args

        if n_samples > 1:
            self._points.push(n_samples - 1)
            if n_samples > 2 and (data[1, n_samples - 1] - data[1, n_samples - 2]) - DELTA_MIN_THR * self.delta_min > 1e-8:
                self._points.push(n_samples - 1)

            self._queue.push_back((
                0,
                n_samples,
                -1,
                0
            ))

            while self._queue.size():
                args = self._queue.front()
                self._queue.pop_front()
                self._forward(
                    args[0],
                    args[1],
                    args[2],
                    args[3]
                )

        self._backward()

    def verify(self, data):
        cdf = self.cdf
        for dx, dy in data:
            cdfy = cdf.eval(dx)
            assert abs(cdfy - dy) > self.eps

    cdef inline void _forward(
            self,
            SIZE_t start,
            SIZE_t end,
            DTYPE_t mse,
            SIZE_t depth
    ) nogil:
        if end - start < 2 or 0 < self.max_splits <= depth:
            return

        cdef:
            DTYPE_t n_samples = <DTYPE_t> end - start
            DTYPE_t mse_min = mse
            SIZE_t best_split = -1
            DTYPE_t best_mse1 = pinf, best_mse2 = pinf
            DTYPE_t[:, ::1] data = self.data
            # SIZE_t[::1] indices = self.indices

            DTYPE_t sum_xy_left = 0, sum_y_sq_left = 0, sum_x_sq_left = 0

            DTYPE_t xl_ = data[0, start], yl_ = data[1, start]
            DTYPE_t xr_ = data[0, end - 1], yr_ = data[1, end - 1]

            DTYPE_t sum_x_sq_right = 0, sum_y_sq_right = 0, sum_xy_right = 0
            SIZE_t i,
            DTYPE_t x, y

        for i in range(start + 1, end - 1):
            x = data[0, i] - xr_
            y = data[1, i] - yr_
            sum_x_sq_right += x * x
            sum_y_sq_right += y * y
            sum_xy_right += x * y

        cdef:
            DTYPE_t x_start = data[0, start] - xl_
            DTYPE_t y_start = data[1, start] - yl_
            DTYPE_t x_end = data[0, end - 1] - xr_
            DTYPE_t y_end = data[1, end - 1] - yr_
            DTYPE_t xl, yl, xr, yr, m1, m2
            SIZE_t samples_left, samples_right
            DTYPE_t err_left, err_right, mse_

        cdef SIZE_t split
        cdef DTYPE_t delta

        for split in range(start + 1, end - 1):
            delta = data[1, split] - data[1, split - 1]

            xl = data[0, split] - xl_
            yl = data[1, split] - yl_
            xr = data[0, split] - xr_
            yr = data[1, split] - yr_
            m1 = (yl - y_start) / (xl - x_start)
            m2 = (y_end - yr) / (x_end - xr)

            samples_left = split - start
            samples_right = end - split

            sum_y_sq_left += yl * yl
            sum_x_sq_left += xl * xl
            sum_xy_left += xl * yl

            err_left = sum_y_sq_left - 2 * sum_xy_left * m1 + sum_x_sq_left * m1 * m1
            err_left /= samples_left

            sum_y_sq_right -= yr * yr
            sum_x_sq_right -= xr * xr
            sum_xy_right -= xr * yr

            err_right = sum_y_sq_right - 2 * sum_xy_right * m2 + sum_x_sq_right * m2 * m2
            err_right /= samples_right

            mse_ = samples_left / n_samples * err_left + samples_right / n_samples * err_right

            if mse_min < 0 or mse_ < mse_min or delta - DELTA_MIN_THR * self.delta_min > 1e-8:
                best_split = split
                mse_min = mse_
                best_mse1 = err_left
                best_mse2 = err_right
                if delta - DELTA_MIN_THR * self.delta_min > 1e-8:
                    self._points.push(best_split)
                    if best_split > 1:
                        self._points.push(best_split - 1)
                    break

        if best_split == -1:
            return

        self._points.push(best_split)

        if best_mse1 >= self.eps:
            self._queue.push_back((start, best_split, best_mse1, depth + 1))
        if best_mse2 >= self.eps:
            self._queue.push_back((best_split, end, best_mse2, depth + 1))

    cdef void _backward(self) nogil:
        cdef DTYPE_t[:, ::1] data = self.data

        self.points.push_front(
            self._points.top()
        )
        self._points.pop()

        cdef:
            SIZE_t pos = self._points.top(), start, p
            DTYPE_t lx, ly, rx, ry, m, c, err

        while self._points.size() > 1:
            # printf('front: %d\n', self.points.front())
            rx = data[0, self.points.front()]
            ry = data[1, self.points.front()]
            p = self._points.top()
            self._points.pop()
            lx = data[0, self._points.top()]
            ly = data[1, self._points.top()]

            if p == self._points.top():  # Jump
                self.points.push_front(p)
                continue
            elif self.points.front() == p:
                self.points.push_front(p)
                self.points.push_front(self._points.top())
                self._points.pop()
                pos = self.points.front() - 1
                continue

            m = (ry - ly) / (rx - lx)
            c = ly - m * lx
            start = pos
            err = 0

            while pos > self._points.top():
                # printf('%d, top: %d, lx=%.4f\n', pos, self._points.top(), lx)
                err += (m * data[0, pos] + c - data[1, pos]) ** 2
                pos -= 1
            if err / <DTYPE_t> (start - pos) >= self.eps:
                self.points.push_front(p)
                pos = p - 1
            else:
                pos = start

        if self._points.size():
            self.points.push_front(
                self._points.top()
            )
