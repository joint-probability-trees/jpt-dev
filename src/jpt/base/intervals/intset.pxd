# distutils: language = c++
# cython: auto_cpdef=False
# cython: infer_types=True
# cython: language_level=3


from .base cimport Interval


cdef class IntSet(Interval):
    pass
