# distutils: language = c++
# cython: auto_cpdef=True
# cython: infer_types=True
# cython: language_level=3
# cython: cdividion=True


cimport numpy as np
cimport cython

from .cutils cimport DTYPE_t


# ----------------------------------------------------------------------------------------------------------------------
# Constant definitions

cdef int _INC
cdef int _EXC
cdef int CLOSED = 2
cdef int HALFOPEN = 3
cdef int OPEN = 4


# ----------------------------------------------------------------------------------------------------------------------

cdef class NumberSet:
    pass


# ----------------------------------------------------------------------------------------------------------------------

@cython.final
cdef class RealSet(NumberSet):

    # Class attributes
    cdef public list intervals

    # Class Methods
    cpdef DTYPE_t size(RealSet self)

    cpdef DTYPE_t[::1] sample(RealSet self, np.int32_t n=*, DTYPE_t[::1] result=*)

    cpdef inline np.int32_t contains_value(RealSet self, DTYPE_t value)

    cpdef inline np.int32_t contains_interval(RealSet self, NumberSet other)

    cpdef inline np.int32_t isempty(RealSet self)

    cpdef inline np.int32_t intersects(RealSet self, NumberSet other)

    cpdef inline np.int32_t isdisjoint(RealSet self, NumberSet other)

    cpdef inline NumberSet intersection(RealSet self, NumberSet other)

    cpdef inline NumberSet union(RealSet self, NumberSet other)

    cpdef inline NumberSet difference(RealSet self, NumberSet other)

    cpdef inline NumberSet complement(RealSet self)

    cpdef inline DTYPE_t fst(RealSet self)

    cpdef inline NumberSet simplify(RealSet self)

    cpdef inline RealSet copy(RealSet self)

    cpdef inline RealSet xmirror(self)

    cpdef inline np.int32_t isninf(self)

    cpdef inline np.int32_t ispinf(self)

    cpdef inline np.int32_t isinf(self)

# ----------------------------------------------------------------------------------------------------------------------

@cython.final
cdef class ContinuousSet(NumberSet):

    # Class attributes
    cdef public:
        np.int32_t left, right
        DTYPE_t lower, upper

    cpdef inline np.int32_t itype(ContinuousSet self)

    cpdef inline np.int32_t isempty(ContinuousSet self)

    cpdef inline np.int32_t isclosed(ContinuousSet self)

    @staticmethod
    cdef inline ContinuousSet c_emptyset()

    @staticmethod
    cdef inline ContinuousSet c_allnumbers()

    cpdef np.ndarray[np.float64_t] sample(ContinuousSet self, np.int32_t k=*, DTYPE_t[::1] result=*)

    cpdef DTYPE_t[::1] _sample(ContinuousSet self, np.int32_t k=*, DTYPE_t[::1] result=*)

    cpdef inline ContinuousSet copy(ContinuousSet self)

    cpdef inline np.int32_t contains_value(ContinuousSet self, DTYPE_t value)

    cpdef inline np.int32_t contains_interval(ContinuousSet self, NumberSet other, int proper_containment=*)

    cpdef inline np.int32_t contiguous(ContinuousSet self, ContinuousSet other)

    cpdef inline np.int32_t intersects(ContinuousSet self, NumberSet other)

    cpdef inline np.int32_t isdisjoint(ContinuousSet self, NumberSet other)

    cpdef inline ContinuousSet intersection(ContinuousSet self, ContinuousSet other, int left=*, int right=*)

    cpdef inline NumberSet union(ContinuousSet self, ContinuousSet other)

    cpdef inline NumberSet difference(ContinuousSet self, NumberSet other)

    cpdef inline NumberSet complement(ContinuousSet self)

    cpdef inline DTYPE_t size(ContinuousSet self)

    cpdef DTYPE_t[::1] linspace(ContinuousSet self, np.int32_t num, DTYPE_t default_step=*, DTYPE_t[::1] result=*)

    cpdef inline DTYPE_t fst(ContinuousSet self)

    cpdef inline DTYPE_t uppermost(ContinuousSet self)

    cpdef inline DTYPE_t lowermost(ContinuousSet self)

    cpdef inline ContinuousSet boundaries(ContinuousSet self, int left=*, int right=*)

    cpdef inline NumberSet simplify(self)

    cpdef inline ContinuousSet ends(self, int left=*, int right=*)

    cpdef inline ContinuousSet xmirror(self)

    cpdef inline np.int32_t isninf(ContinuousSet self)

    cpdef inline np.int32_t ispinf(ContinuousSet self)

    cpdef inline np.int32_t isinf(ContinuousSet self)