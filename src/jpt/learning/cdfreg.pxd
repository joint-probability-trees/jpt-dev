# cython: language_level=3

from ..base.cutils cimport DTYPE_t, SIZE_t, nan

from libcpp.queue cimport priority_queue
from libcpp.deque cimport  deque


cdef class CDFRegressor:
    '''Experimental quantile regression.'''

    cdef DTYPE_t eps
    cdef SIZE_t max_splits
    cdef DTYPE_t[:, ::1] data
    cdef SIZE_t[::1] indices
    cdef priority_queue[SIZE_t] _points
    cdef deque[SIZE_t] points
    cdef deque[(SIZE_t, SIZE_t, DTYPE_t, SIZE_t)] _queue

    cpdef void fit(self, DTYPE_t[:, ::1] data)

    cdef inline void _forward(CDFRegressor self,
                              SIZE_t start,
                              SIZE_t end,
                              DTYPE_t mse,
                              SIZE_t depth) nogil

    cdef void _backward(self) nogil
