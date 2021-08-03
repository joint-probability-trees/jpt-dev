# cython: auto_cpdef=True, infer_types=True, language_level=3

from .intervals cimport ContinuousSet, RealSet

import numpy as np
cimport numpy as np
cimport cython

from ..base.cutils cimport DTYPE_t, SIZE_t


cdef class NotInvertibleError(Exception):
    pass


cdef class ConfInterval:
    '''Represents a prediction interval with a predicted value, lower und upper bound'''

    cdef readonly:
        DTYPE_t mean, lower, upper

    cpdef tuple totuple(ConfInterval self)

    cpdef DTYPE_t[::1] tomemview(ConfInterval self, DTYPE_t[::1] result=*)



cdef class Function:
    '''
    Abstract base type of functions.
    '''


@cython.final
cdef class Undefined(Function):
    '''
    This class represents an undefined function.
    '''

    cpdef inline DTYPE_t eval(Undefined self, DTYPE_t x)


cdef class KnotFunction(Function):
    '''
    Abstract superclass of all knot functions.
    '''
    # Class attributes

    cdef public:
        DTYPE_t knot, weight


@cython.final
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

    cpdef inline DTYPE_t eval(Hinge self, DTYPE_t x)

    cpdef Function differentiate(Hinge self)

    cpdef np.int32_t is_invertible(Hinge self)


@cython.final
cdef class Jump(KnotFunction):
    '''
    Implementation of jump functions.
    '''

    # Class attributes

    cdef public:
        np.int32_t alpha

    # Class methods

    cpdef inline DTYPE_t eval(Jump self, DTYPE_t x)

    cpdef inline Function differentiate(Jump self)


@cython.final
cdef class Impulse(KnotFunction):
    '''
    Represents a function that is non-zero at exactly one x-position and zero at all other positions.
    '''

    # Class methods

    cpdef DTYPE_t eval(self, DTYPE_t x)

    cpdef inline Function differentiate(Impulse self)

    cpdef inline np.int32_t is_invertible(Impulse self)


@cython.final
cdef class ConstantFunction(Function):
    '''
    Represents a constant function.
    '''

    # Class attributes

    cdef public DTYPE_t value

    # Class methods

    cpdef inline DTYPE_t eval(ConstantFunction self, DTYPE_t x)

    cpdef inline ConstantFunction differentiate(ConstantFunction self)

    cpdef inline np.int32_t is_invertible(ConstantFunction self)

    cpdef np.int32_t crosses(ConstantFunction self, Function f) except +

    cpdef ContinuousSet xing_point(ConstantFunction self, Function f) except +

    cpdef inline ConstantFunction copy(ConstantFunction self)


@cython.final
cdef class LinearFunction(Function):
    '''
    Implementation of univariate linear functions.
    '''

    # Class attributes

    cdef public DTYPE_t m, c

    # Class methods

    cpdef inline DTYPE_t eval(LinearFunction self, DTYPE_t x)

    cpdef DTYPE_t root(LinearFunction self) except +

    cpdef LinearFunction invert(LinearFunction self) except +

    cpdef LinearFunction hmirror(LinearFunction self)

    cpdef LinearFunction copy(LinearFunction self)

    cpdef np.int32_t crosses(LinearFunction self, Function f) except +

    cpdef ContinuousSet xing_point(LinearFunction self, Function f) except +

    cpdef inline Function differentiate(LinearFunction self)

    cpdef inline Function simplify(LinearFunction self)

    cpdef inline np.int32_t is_invertible(LinearFunction self)

    cpdef inline LinearFunction fit(LinearFunction self, DTYPE_t[::1] x, DTYPE_t[::1] y) except +


# cdef class Quantiles:
#     '''
#     This class implements basic representation and handling of quantiles
#     in a data distribution.
#     '''
#
#     # Class attributes
#
#     cdef readonly DTYPE_t[::1] data
#     cdef readonly DTYPE_t[::1] weights
#     cdef readonly DTYPE_t epsilon
#     cdef readonly DTYPE_t penalty
#     cdef readonly DTYPE_t _lower
#     cdef readonly DTYPE_t _upper
#     cdef readonly DTYPE_t _mean
#     # cdef readonly DTYPE_t[::1] _lower_half, _upper_half
#     cdef readonly Function _cdf, _pdf, _invcdf
#     cdef readonly np.int32_t dtype
#     cdef public np.int32_t verbose
#     cdef object _stuff
#
#     # Class methods
#
#     cpdef Function cdf(Quantiles self, DTYPE_t epsilon=*, DTYPE_t penalty=*)
#
#     cpdef Function invcdf(Quantiles self, DTYPE_t epsilon=*, DTYPE_t penalty=*)
#
#     cpdef Function pdf(Quantiles self, np.int32_t simplify=*, np.int32_t samples=*,
#                                 DTYPE_t epsilon=*, DTYPE_t penalty=*)
#
#     cpdef DTYPE_t[::1] sample(Quantiles self, np.int32_t n=*, DTYPE_t[::1] result=*)
#
#     cpdef DTYPE_t[::1] gt(Quantiles self, DTYPE_t q)
#
#     cpdef DTYPE_t[::1] lt(Quantiles self, DTYPE_t q)
#
#     cpdef ConfInterval interval(Quantiles self, DTYPE_t alpha)


@cython.final
cdef class PiecewiseFunction(Function):
    '''
    Represents a function that is piece-wise defined by constant values.
    '''
    # Class attributes

    cdef readonly list intervals
    cdef readonly list functions

    # Class methods

    cpdef inline DTYPE_t eval(PiecewiseFunction self, DTYPE_t x)

    cpdef inline DTYPE_t[::1] multi_eval(PiecewiseFunction self, DTYPE_t[::1] x, DTYPE_t[::1] result=*)

    cpdef inline Function at(PiecewiseFunction self, DTYPE_t x)

    cpdef inline ContinuousSet interval_at(PiecewiseFunction self, DTYPE_t x)

    cpdef PiecewiseFunction copy(PiecewiseFunction self)

    cpdef PiecewiseFunction differentiate(PiecewiseFunction self)

    cpdef ensure_left(PiecewiseFunction self, Function left, DTYPE_t x)

    cpdef ensure_right(PiecewiseFunction self, Function right, DTYPE_t x)

    cpdef DTYPE_t[::1] xsamples(PiecewiseFunction self, np.int32_t sort=*)

    # cpdef PiecewiseFunction simplify(PiecewiseFunction self, np.int32_t n_samples=*, DTYPE_t epsilon=*,
    #                                  DTYPE_t penalty=*)

    cpdef RealSet eq(PiecewiseFunction self, DTYPE_t y)

    cpdef RealSet lt(PiecewiseFunction self, DTYPE_t y)

    cpdef RealSet gt(PiecewiseFunction self, DTYPE_t y)

    cpdef tuple split(PiecewiseFunction self, DTYPE_t splitpoint)

    cpdef inline str pfmt(PiecewiseFunction self)

    cpdef inline list knots(PiecewiseFunction self, DTYPE_t lastx=*)

    cpdef PiecewiseFunction add_knot(PiecewiseFunction self, DTYPE_t x, DTYPE_t y)

    cpdef PiecewiseFunction crop(PiecewiseFunction self, ContinuousSet interval)



