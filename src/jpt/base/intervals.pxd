# cython: auto_cpdef=True, infer_types=True, language_level=3

cimport numpy as np
cimport cython


cdef class NumberSet:
    pass


@cython.final
cdef class RealSet(NumberSet):

    # Class attributes
    cdef public list intervals

    # Class Methods
    cpdef np.float64_t size(RealSet self)

    cpdef np.float64_t[::1] sample(RealSet self, np.int32_t n=*, np.float64_t[::1] result=*)

    cpdef inline np.int32_t contains_value(RealSet self, np.float64_t value)

    cpdef inline np.int32_t contains_interval(RealSet self, ContinuousSet other)

    cpdef inline np.int32_t isempty(RealSet self)

    cpdef inline np.int32_t intersects(RealSet self, RealSet other)

    cpdef inline RealSet intersection(RealSet self, RealSet other)

    cpdef inline RealSet union(RealSet self, RealSet other) except +

    cpdef inline RealSet difference(RealSet self, RealSet other)

    cpdef inline RealSet complement(RealSet self)

    cpdef inline np.float64_t fst(RealSet self)


@cython.final
cdef class ContinuousSet(NumberSet):

    # Class attributes
    cdef public:
        np.int32_t left, right
        np.float64_t lower, upper

    cpdef inline np.int32_t itype(ContinuousSet self)

    cpdef inline np.int32_t isempty(ContinuousSet self)

    cpdef inline np.int32_t isclosed(ContinuousSet self)

    cpdef inline ContinuousSet emptyset(ContinuousSet self)

    cpdef inline ContinuousSet allnumbers(ContinuousSet self)

    cpdef np.float64_t[::1] sample(ContinuousSet self, np.int32_t k=*, np.float64_t[::1] result=*)

    cpdef inline ContinuousSet copy(ContinuousSet self)

    cpdef inline np.int32_t contains_value(ContinuousSet self, np.float64_t value)

    cpdef inline np.int32_t contains_interval(ContinuousSet self, ContinuousSet other)

    cpdef inline np.int32_t contiguous(ContinuousSet self, ContinuousSet other)

    cpdef inline np.int32_t intersects(ContinuousSet self, ContinuousSet other)

    cpdef inline ContinuousSet intersection(ContinuousSet self, ContinuousSet other)

    cpdef inline NumberSet union(ContinuousSet self, ContinuousSet other)

    cpdef inline NumberSet difference(ContinuousSet self, ContinuousSet other)

    cpdef inline NumberSet complement(ContinuousSet self)

    cpdef inline np.float64_t size(ContinuousSet self)

    cpdef np.float64_t[::1] linspace(ContinuousSet self, np.int32_t num, np.float64_t default_step=*, np.float64_t[::1] result=*)

    cpdef inline np.float64_t fst(ContinuousSet self)