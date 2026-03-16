# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: nonehcheck=False
# ----------------------------------------------------------------------------------------------------------------------

__module__ = 'cutils.pyx'
import numpy as np
cimport numpy as np
nan = np.nan
ninf = -np.inf
pinf = np.inf


cdef class ConfInterval:
    '''Represents a prediction interval with a mean, lower und upper bound'''

    def __init__(self, mean, lower, upper):
        self.mean = mean
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return '|<-- %s -- %s -- %s -->|' % (self.lower, self.mean, self.upper)

    def __repr__(self):
        return str(self)

    cpdef tuple totuple(ConfInterval self):
        return self.lower, self.mean, self.upper

    cpdef DTYPE_t[::1] tomemview(ConfInterval self, DTYPE_t[::1] result=None):
        if result is None:
            result = np.ndarray(shape=3, dtype=np.float64)
        result[0] = self.lower
        result[1] = self.mean
        result[2] = self.upper