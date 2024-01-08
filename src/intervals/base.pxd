# distutils: language = c++
# cython: auto_cpdef=False
# cython: infer_types=True
# cython: language_level=3
cimport numpy as np

ctypedef double DTYPE_t                  # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.int64_t SIZE_t               # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef NumberSet _Z
cdef NumberSet _R

# ----------------------------------------------------------------------------------------------------------------------

cdef class NumberSet:

    cpdef SIZE_t contains_value(self, DTYPE_t x)

    @staticmethod
    cdef NumberSet _emptyset()

    cpdef SIZE_t issuperseteq(self, NumberSet other)

    cpdef SIZE_t issuperset(self, NumberSet other)

    cpdef NumberSet union(self, NumberSet other)

    cpdef NumberSet difference(self, NumberSet other)

    cpdef SIZE_t isdisjoint(self, NumberSet other)

    cpdef SIZE_t intersects(self, NumberSet other)

    cpdef NumberSet intersection(self, NumberSet other)

    cpdef SIZE_t isempty(self)

    cpdef DTYPE_t size(self)

    cpdef NumberSet copy(self)

    cpdef DTYPE_t fst(self)

    cpdef DTYPE_t lst(self)

    cpdef NumberSet xmirror(self)

    cpdef SIZE_t isninf(self)

    cpdef SIZE_t ispinf(self)

    cpdef SIZE_t isinf(self)

    cpdef np.ndarray[DTYPE_t] sample(self, SIZE_t k=*, DTYPE_t[::1] result=*)

    cpdef DTYPE_t[::1] _sample(self, SIZE_t k=*, DTYPE_t[::1] result=*)


# ----------------------------------------------------------------------------------------------------------------------

cdef class Interval(NumberSet):

    cdef public DTYPE_t _lower, _upper

    cpdef SIZE_t contiguous(self, Interval other)

    cpdef NumberSet complement(self)
