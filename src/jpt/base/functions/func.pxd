# cython: language_level=3

from jpt.base.intervals.contset cimport ContinuousSet
from jpt.base.intervals.unionset cimport UnionSet

import numpy as np
cimport numpy as np


from .cutils cimport DTYPE_t, SIZE_t


# ----------------------------------------------------------------------------------------------------------------------

cdef class NotInvertibleError(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------------

cdef class Function:
    '''
    Abstract base type of functions.
    '''

    cpdef DTYPE_t eval(self, DTYPE_t x)

    cpdef DTYPE_t[::1] multi_eval(self, DTYPE_t[::1] x, DTYPE_t[::1] result=*)

    cpdef Function set(self, Function f)

    cpdef Function mul(self, Function f)

    cpdef Function add(self, Function f)

    cpdef Function simplify(self)

    cpdef Function copy(self)

    cpdef Function xmirror(self)


# ----------------------------------------------------------------------------------------------------------------------

cdef class Undefined(Function):
    '''
    This class represents an undefined function.
    '''
    cpdef Function xshift(self, DTYPE_t delta)


# ----------------------------------------------------------------------------------------------------------------------

cdef class ConstantFunction(Function):
    '''
    Represents a constant function.
    '''

    # Class attributes

    cdef public DTYPE_t value

    # Class methods

    cpdef Function differentiate(self)

    cpdef SIZE_t is_invertible(self)

    cpdef SIZE_t intersects(self, Function f)

    cpdef ContinuousSet intersection(self, Function f)

    cpdef Function copy(self)

    cpdef DTYPE_t integrate(self, DTYPE_t x1, DTYPE_t x2)

    cpdef ConstantFunction xshift(self, DTYPE_t delta)


# ----------------------------------------------------------------------------------------------------------------------

cdef class LinearFunction(Function):
    '''
    Implementation of univariate linear functions.
    '''

    # Class attributes

    cdef public DTYPE_t m, c

    # Class methods

    cpdef DTYPE_t root(self)

    cpdef Function invert(self)

    cpdef Function hmirror(self)

    cpdef SIZE_t intersects(self, Function f)

    cpdef ContinuousSet intersection(self, Function f)

    cpdef Function differentiate(self)

    cpdef SIZE_t is_invertible(self)

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y)

    cpdef DTYPE_t integrate(self, DTYPE_t x1, DTYPE_t x2)

    cpdef LinearFunction xshift(self, DTYPE_t delta)


# ----------------------------------------------------------------------------------------------------------------------

cdef class QuadraticFunction(Function):
    '''
    Implementation of a univariate quadratic function
    '''

    # Class attributes

    cdef public DTYPE_t a, b, c

    # Class methods

    cpdef DTYPE_t[::1] roots(self)

    cpdef Function invert(self)

    cpdef SIZE_t intersects(self, Function f)

    cpdef ContinuousSet intersection(self, Function f)

    cpdef Function differentiate(self)

    cpdef SIZE_t is_invertible(self)

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y, DTYPE_t[::1] z)

    cpdef DTYPE_t argvertex(self)


# ----------------------------------------------------------------------------------------------------------------------

cdef class PiecewiseFunction(Function):
    '''
    Represents a function that is piece-wise defined by constant values.
    '''
    # Class attributes

    cdef public list intervals
    cdef public list functions

    # Class methods

    cpdef Function at(self, DTYPE_t x)

    cpdef ContinuousSet interval_at(self, DTYPE_t x)

    cpdef Function copy(self)

    cpdef Function differentiate(self)

    cpdef DTYPE_t integrate(self, ContinuousSet interval=*)

    cpdef ensure_left(self, Function left, DTYPE_t x)

    cpdef ensure_right(self, Function right, DTYPE_t x)

    cpdef DTYPE_t[::1] xsamples(self, np.int32_t sort=*)

    cpdef UnionSet eq(self, DTYPE_t y)

    cpdef UnionSet lt(self, DTYPE_t y)

    cpdef UnionSet gt(self, DTYPE_t y)

    cpdef tuple split(self, DTYPE_t splitpoint)

    cpdef str pfmt(self)

    cpdef list knots(self, DTYPE_t lastx=*)

    cpdef PiecewiseFunction add_knot(self, DTYPE_t x, DTYPE_t y)

    cpdef PiecewiseFunction crop(self, ContinuousSet interval)

    cpdef int idx_at(self, DTYPE_t x)

    cpdef PiecewiseFunction xshift(self, DTYPE_t delta)