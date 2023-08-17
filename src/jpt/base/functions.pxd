# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False

from .intervals cimport ContinuousSet, RealSet

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

cdef class KnotFunction(Function):
    '''
    Abstract superclass of all knot functions.
    '''
    # Class attributes

    cdef public:
        DTYPE_t knot, weight


# ----------------------------------------------------------------------------------------------------------------------

cdef class Hinge(KnotFunction):
    '''
    Implementation of hinge functions as used in MARS regression.

    alpha = 1:  hinge is zero to the right of knot
    alpha = -1: hinge is zero to the left of knot
    '''

    # Class attributes

    cdef public:
        np.int32_t alpha

    # Class methods

    cpdef Function differentiate(self)

    cpdef np.int32_t is_invertible(self)


# ----------------------------------------------------------------------------------------------------------------------

cdef class Jump(KnotFunction):
    '''
    Implementation of jump functions.
    '''

    # Class attributes

    cdef public:
        np.int32_t alpha

    # Class methods

    cpdef Function differentiate(self)


# ----------------------------------------------------------------------------------------------------------------------

cdef class Impulse(KnotFunction):
    '''
    Represents a function that is non-zero at exactly one x-position and zero at all other positions.
    '''

    # Class methods

    cpdef Function differentiate(self)

    cpdef np.int32_t is_invertible(self)


# ----------------------------------------------------------------------------------------------------------------------

cdef class ConstantFunction(Function):
    '''
    Represents a constant function.
    '''

    # Class attributes

    cdef public DTYPE_t value

    # Class methods

    cpdef Function differentiate(self)

    cpdef np.int32_t is_invertible(self)

    cpdef np.int32_t intersects(self, Function f) except +

    cpdef ContinuousSet intersection(self, Function f) except +

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

    cpdef DTYPE_t root(self) except +

    cpdef Function invert(self) except +

    cpdef Function hmirror(self)

    cpdef np.int32_t intersects(self, Function f) except +

    cpdef ContinuousSet intersection(self, Function f) except +

    cpdef Function differentiate(self)

    cpdef np.int32_t is_invertible(self)

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y) except +

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

    cpdef Function invert(self) except +

    cpdef np.int32_t intersects(self, Function f) except +

    cpdef ContinuousSet intersection(self, Function f) except +

    cpdef Function differentiate(self)

    cpdef np.int32_t is_invertible(self)

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y, DTYPE_t[::1] z) except +

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

    cpdef RealSet eq(self, DTYPE_t y)

    cpdef RealSet lt(self, DTYPE_t y)

    cpdef RealSet gt(self, DTYPE_t y)

    cpdef tuple split(self, DTYPE_t splitpoint)

    cpdef str pfmt(self)

    cpdef list knots(self, DTYPE_t lastx=*)

    cpdef PiecewiseFunction add_knot(self, DTYPE_t x, DTYPE_t y)

    cpdef PiecewiseFunction crop(self, ContinuousSet interval)

    cpdef int idx_at(self, DTYPE_t x)

    cpdef PiecewiseFunction xshift(self, DTYPE_t delta)