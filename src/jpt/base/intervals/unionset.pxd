# distutils: language = c++
# cython: auto_cpdef=False
# cython: infer_types=True
# cython: language_level=3



cimport cython
from .base cimport NumberSet

from .base cimport SIZE_t, DTYPE_t


# ----------------------------------------------------------------------------------------------------------------------

cdef class UnionSet(NumberSet):

    # Class attributes
    cdef public list intervals

    # Class Methods

    cpdef NumberSet simplify(self, SIZE_t keep_type=*)