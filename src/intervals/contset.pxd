# distutils: language = c++
# cython: auto_cpdef=False
# cython: infer_types=True
# cython: language_level=3

cimport numpy as np

from .base cimport (
    SIZE_t,
    DTYPE_t,
    Interval,
    NumberSet
)

# ----------------------------------------------------------------------------------------------------------------------
# Constant definitions

cdef int _INC
cdef int _EXC
cdef int CLOSED
cdef int HALFOPEN
cdef int OPEN


# ----------------------------------------------------------------------------------------------------------------------

cdef class ContinuousSet(Interval):

    # Class attributes
    cdef public:
        SIZE_t left, right

    cpdef SIZE_t itype(self)

    cpdef SIZE_t isclosed(self)

    @staticmethod
    cdef ContinuousSet c_allnumbers()

    cpdef SIZE_t contains_interval(self, NumberSet other, int proper_containment=*)

    cpdef ContinuousSet intersection_with_ends(
            self,
            ContinuousSet other,
            int left=*,
            int right=*
    )

    cpdef DTYPE_t[::1] linspace(self, SIZE_t num, DTYPE_t default_step=*, DTYPE_t[::1] result=*)

    cpdef DTYPE_t uppermost(self)

    cpdef DTYPE_t lowermost(self)

    cpdef ContinuousSet boundaries(self, int left=*, int right=*)

    cpdef ContinuousSet ends(self, int left=*, int right=*)
