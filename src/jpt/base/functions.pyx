# cython: infer_types=False
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False
__module__ = 'functions.pyx'

import heapq
from functools import cmp_to_key

from itertools import chain

import itertools
import numbers
from collections import deque
from operator import attrgetter
from typing import Iterator, List, Iterable, Tuple, Union, Dict, Any, Optional

from dnutils import ifnot, ifnone, pairwise, fst, last
from dnutils.tools import ifstr, first
from scipy import stats
from scipy.stats import norm

from .constants import eps
from .intervals cimport ContinuousSet, RealSet
from .intervals import R, EMPTY, EXC, INC, NumberSet, ContinuousSet

import numpy as np
cimport numpy as np
cimport cython

import warnings

from .cutils cimport DTYPE_t
from .utils import Heap

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------

@cython.freelist(1000)
cdef class Function:
    """
    Abstract base type of functions.
    """

    cpdef DTYPE_t eval(self, DTYPE_t x):
        """
        Evaluate this function at position ``x``
        :param x: the value where to eval this function 
        :return: f(x)
        """
        return np.nan

    cpdef DTYPE_t[::1] multi_eval(self, DTYPE_t[::1] x, DTYPE_t[::1] result = None):
        if result is None:
            result = np.ndarray(shape=x.shape[0], dtype=np.float64)
        cdef int i
        for i in range(x.shape[0]):
            result[i] = self.eval(x[i])
        return result

    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return self.multi_eval(x)
        else:
            return self.eval(x)

    cpdef Function set(self, Function f):
        """
        Overwrite this functions parameters with the parameters of ``f``
        :param f: the function to overwrite from
        :return: The altered function
        """
        raise NotImplementedError()

    cpdef Function mul(self, Function f):
        """
        Calculate the product of this and the given function.
        :param f: the other function
        :return: The product of this function and ``f``
        """
        raise NotImplementedError()

    cpdef Function add(self, Function f):
        """
        Calculate the sum of this and the given function.
        :param f: the other function
        :return: The sum of this function and ``f``
        """
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            other = ConstantFunction(other)
        if isinstance(other, Function):
            return self.add(other).simplify()
        else:
            raise TypeError(
                'Unsupported operand type(s) for +: %s and %s' % (
                    type(self).__name__,
                    type(other).__name__
                )
            )

    def __mul__(self, other: Union[float, Function]) -> Function:
        if isinstance(other, numbers.Real):
            return self.mul(
                ConstantFunction(other)
            ).simplify()
        elif isinstance(other, Function):
            return self.mul(other).simplify()
        else:
            raise TypeError(
                'Unsupported operand type(s) for *: %s and %s' % (
                    type(self).__name__,
                    type(other).__name__
                )
            )

    def __iadd__(self, other):
        return self.set(self + other)

    def __imul__(self, other):
        return self.set(self * other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        raise NotImplementedError(f'{type(self).__qualname__}')

    def __neg__(self):
        raise NotImplementedError(f'{type(self).__qualname__}')

    cpdef Function simplify(self):
        """
        Get a simpler version of this function. Simpler refers to a lower amount of parameters.
        :return: The simplified function.
        """
        return self.copy()

    cpdef Function copy(self):
        """
        Get a fresh copy of this function.
        :return: the fresh copy
        """
        raise NotImplementedError()

    cpdef Function xmirror(self):
        raise NotImplementedError(f'{type(self)}')


# ----------------------------------------------------------------------------------------------------------------------

cdef class Undefined(Function):
    """
    This class represents an undefined function.
    An undefined function returns nan for every value.
    """

    def __str__(self):
        return 'undef.'

    def __repr__(self):
        return '<Undefined 0x%X>' % id(self)

    def __hash__(self):
        return hash(Undefined)

    def __eq__(self, other):
        if isinstance(other, Undefined):
            return True
        elif isinstance(other, ConstantFunction) and np.isnan(other.value):
            return True
        elif isinstance(other, LinearFunction) and any([np.isnan(other.m), np.isnan(other.c)]):
            return True
        return False

    cpdef Function set(self, Function f):
        return self

    cpdef Function mul(self, Function f):
        return Undefined()

    cpdef Function add(self, Function f):
        return Undefined()

    cpdef Function copy(self):
        return Undefined()

    cpdef Function xshift(self, DTYPE_t delta):
        return Undefined()


# ----------------------------------------------------------------------------------------------------------------------

cdef class KnotFunction(Function):
    """
    Abstract superclass of all knot functions.
    """

    def __init__(self, DTYPE_t knot, DTYPE_t weight):
        self.knot = knot
        self.weight = weight


# ----------------------------------------------------------------------------------------------------------------------

cdef class Hinge(KnotFunction):
    """
    Implementation of hinge functions as used in MARS regression.

    alpha = 1:  hinge is zero to the right of knot
    alpha = -1: hinge is zero to the left of knot
    """

    def __init__(self, DTYPE_t knot, np.int32_t alpha, DTYPE_t weight):
        super(Hinge, self).__init__(knot, weight)
        assert alpha in (1, -1), 'alpha must be in {1,-1}'
        self.alpha = alpha

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return max(0, (self.knot - x) if self.alpha == 1 else (x - self.knot)) * self.weight

    def __str__(self):
        return '%.3f * max(0, %s)' % (self.weight,
                                      ('x - %s' % self.knot) if self.alpha == 1 else ('%s - x' % self.knot))

    def __repr__(self):
        return '<Hinge 0x%X: k=%.3f a=%d w=%.3f>' % (id(self), self.knot, self.alpha, self.weight)

    cpdef Function differentiate(self):
        """
        Calculate the derivative of this function.
        :return: the derivative
        """
        return Jump(self.knot, self.alpha, -self.weight if self.alpha == 1 else self.weight)

    cpdef np.int32_t is_invertible(self):
        return False


# ----------------------------------------------------------------------------------------------------------------------

cdef class Jump(KnotFunction):
    """
    Implementation of jump functions.
    """

    def __init__(self, DTYPE_t knot, np.int32_t alpha, DTYPE_t weight):
        super(Jump, self).__init__(knot, weight)
        assert alpha in (1, -1), 'alpha must be in {1,-1}'
        self.alpha = alpha

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return max(0, (-1 if ((self.knot - x) if self.alpha == 1 else (x - self.knot)) < 0 else 1)) * self.weight

    def __str__(self):
        return '%.3f * max(0, sgn(%s))' % (self.weight,
                                           ('x - %s' % self.knot) if self.alpha == 1 else ('%s - x' % self.knot))

    def __repr__(self):
        return '<Jump 0x%X: k=%.3f a=%d w=%.3f>' % (id(self), self.knot, self.alpha, self.weight)

    @staticmethod
    def from_point(p1, alpha):
        x, y = p1
        return Jump(x, alpha, y)

    cpdef Function differentiate(self):
        return Impulse(self.knot, self.weight)


# ----------------------------------------------------------------------------------------------------------------------

cdef class Impulse(KnotFunction):
    """
    Represents a function that is non-zero at exactly one x-position and zero at all other positions.
    """

    def __init__(self, DTYPE_t knot, DTYPE_t weight):
        super(Impulse, self).__init__(knot, weight)

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.weight if x == self.knot else 0

    cpdef Function differentiate(self):
        return Impulse(self.knot, np.nan)

    cpdef np.int32_t is_invertible(self):
        return False

    def __repr__(self):
        return '<Impulse )x%X: k=%.3f w=%.3f>' % (id(self), self.knot, self.weight)

    def __str__(self):
        return '%.3f if x=%.3f else 0' % (self.weight, self.knot)


# ----------------------------------------------------------------------------------------------------------------------

cdef class ConstantFunction(Function):
    """
    Represents a constant function.
    """

    def __init__(self, DTYPE_t value):
        self.value = value

    def __hash__(self):
        return hash((LinearFunction, 0, self.value))

    def __str__(self):
        return '%s = const.' % self.value

    def __repr__(self):
        return '<ConstantFunction 0x%X: %s>' % (id(self), str(self.value))

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.value

    cpdef Function differentiate(self):
        return ConstantFunction(0)

    cpdef SIZE_t is_invertible(self):
        return False

    cpdef Function copy(self):
        return ConstantFunction(self.value)

    def __eq__(self, other):
        if not isinstance(other, Function):
            raise TypeError(
                'Cannot compare object of type %s to %s.' % (
                    type(self).__name__,
                    type(other).__name__
                )
            )
        if isinstance(other, ConstantFunction):
            return np.isnan(self.value) and np.isnan(other.value) or self.value == other.value
        elif isinstance(other, LinearFunction):
            return (
                (other.m == 0 and self.value == other.c)
                or (np.isnan(self.value) and (np.isnan(other.c) or np.isnan(other.m)))
            )
        elif isinstance(other, Undefined) and np.isnan(self.value):
            return True
        return False

    cpdef Function set(self, Function f):
        if isinstance(f, ConstantFunction):
            self.value = f.value
            return self
        else:
            raise TypeError(
                'Object of type %s can only be set to parameters '
                'of objects of the same type' % type(self).__name__
            )

    cpdef Function mul(self, Function f):
        cdef DTYPE_t v = np.nan
        if isinstance(f, ConstantFunction):
            if not self.value or not f.value:
                v = 0
            else:
                v = self.value * f.value
            return ConstantFunction(v)
        elif isinstance(f, (LinearFunction, QuadraticFunction, Undefined)):
            return f.mul(self)
        else:
            raise TypeError(
                'Unsupported operand type(s) for mul(): %s and %s.' % (
                    type(self).__name__, type(f).__name__
                )
            )

    cpdef Function add(self, Function f):
        if isinstance(f, ConstantFunction):
            return ConstantFunction(self.value + f.value)
        elif isinstance(f, (LinearFunction, QuadraticFunction, Undefined)):
            return f.add(self)
        else:
            raise TypeError(
                'Unsupported operand type(s) for add(): %s and %s.' % (
                    type(self).__name__, type(f).__name__
                )
            )

    @property
    def m(self):
        return 0

    @property
    def c(self):
        return self.value

    cpdef SIZE_t intersects(self, Function f):
        """
        Determine if the function crosses another function ``f``.
        :param f: the other ``Function``
        :return: True if they intersect, False else
        """
        if isinstance(f, LinearFunction):
            return f.intersects(self)
        elif isinstance(f, ConstantFunction):
            return f.value == self.value
        else:
            raise TypeError('Argument must be of type LinearFunction '
                            'or ConstantFunction, not %s' % type(f).__name__)

    cpdef ContinuousSet intersection(self, Function f):
        """
        Determine where the function crosses another function ``f``.
        :param f: the other ``Function``
        :return: the ``ContinuousSet`` where they intersect. ContinuousSet.empty() if they dont intersect.
        """
        if isinstance(f, LinearFunction):
            return f.intersection(self)
        elif isinstance(f, ConstantFunction):
            return R.copy() if self.value == f.value else EMPTY.copy()
        else:
            raise TypeError('Argument must be of type LinearFunction '
                            'or ConstantFunction, not %s' % type(f).__name__)

    def to_json(self):
        return {'type': 'constant', 'value': self.value}

    cpdef DTYPE_t integrate(self, DTYPE_t x1, DTYPE_t x2):
        if x2 < x1:
            raise ValueError(
                'The x2 argument must be greater than x1. '
                'Got x1=%s, x2=%s' % (
                    x1, x2
                )
            )
        return 0 if not self.c else (self.c * (x2 - x1))

    cpdef ConstantFunction xshift(self, DTYPE_t delta):
        return self.copy()

    cpdef Function xmirror(self):
        '''
        Returns a modification of the functino that has been mirrored
        at position x=0.
        :return:
        '''
        return self.copy()

    def __neg__(self):
        return ConstantFunction(-self.value)


# ----------------------------------------------------------------------------------------------------------------------

cdef class LinearFunction(Function):
    """
    Implementation of univariate linear functions.
    """

    def __init__(self, DTYPE_t m, DTYPE_t c):
        self.m = m
        self.c = c

    def __hash__(self):
        return hash((LinearFunction, self.m, self.c))

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.m * x + self.c

    def __str__(self):
        l = (str(self.m) + 'x') if self.m else ''
        op = '' if (not l and self.c > 0 or not self.c) else ('+' if self.c > 0 else '-')
        c = '0' if (not self.c and not self.m) else ('' if not self.c else str(abs(self.c)))
        return ('%s %s %s' % (l, op, c)).strip()

    def __repr__(self):
        return '<LinearFunction 0x%X: %s>' % (id(self), str(self))

    @staticmethod
    def parse(s):
        if s == 'undef.' or s is None:
            return Undefined()
        if isinstance(s, numbers.Number):
            return ConstantFunction(s)
        match = s.split('x')
        if not match:
            raise ValueError('Illegal format for linear function: "%s"' % s)
        elif len(match) > 1:
            linear = float(ifnot(match[0].replace(' ', ''), 1))
            const = float(match[1].replace(' ', '')) if match[1] else 0
            if not linear:
                return ConstantFunction(const)
            return LinearFunction(linear, const)
        else:
            return ConstantFunction(float(match[0].replace(' ', '')))

    cpdef DTYPE_t root(self):
        """
        Find the root of the function, i.e. the ``x`` positions subject to ``self.eval(x) = 0``.
        :return: root of this function as float
        """
        return -self.c / self.m

    cpdef Function invert(self):
        """
        Return the inverted linear function of this LF.
        :return: the inverted function as ``LinearFunction``
        """
        return LinearFunction(1 / self.m, -self.c / self.m)

    cpdef Function hmirror(self):
        """
        Return the linear function that is obtained by horizontally mirroring this LF.
        :return: the mirrored function as ``LinearFunction``
        """
        return LinearFunction(-self.m, -self.c)

    cpdef Function copy(self):
        return LinearFunction(self.m, self.c)

    cpdef SIZE_t intersects(self, Function f):
        """
        Determine if the function crosses another function ``f``.
        :param f: the other ``Function``
        :return: True if they intersect, False if not
        """
        if isinstance(f, LinearFunction):
            if self.m == f.m:
                return False
            else:
                return True
        elif isinstance(f, ConstantFunction):
            if self.m == 0:
                return False
            else:
                return True
        else:
            raise TypeError('Argument must be of type '
                            'LinearFunction or ConstantFunction, not %s' % type(f).__name__)

    cpdef ContinuousSet intersection(self, Function f):
        """
        Determine the interval where this function intersects the other function ``f``.
        :param f: the other ``Function``
        :return: The intersection of the function as ''ContinuousSet''
        """
        cdef DTYPE_t x
        if isinstance(f, LinearFunction):
            if self.m == f.m:
                if self.c == f.c:
                    return R.copy()
                else:
                    return EMPTY.copy()
            else:
                x = (self.c - f.c) / (f.m - self.m)
                return ContinuousSet(x, x)
        elif isinstance(f, ConstantFunction):
            if self.m == 0:
                if self.c == f.value:
                    return R.copy()
                else:
                    return EMPTY.copy()
            x = (self.c - f.value) / -self.m
            return ContinuousSet(x, x)
        else:
            raise TypeError('Argument must be of type LinearFunction '
                            'or ConstantFunction, not %s' % type(f).__name__)

    cpdef Function mul(self, Function f):
        if isinstance(f, ConstantFunction):
            return LinearFunction(self.m * f.value, self.c * f.value)
        elif isinstance(f, LinearFunction):
            return QuadraticFunction(
                self.m * f.m,
                self.m * f.c + f.m * self.c,
                self.c * f.c
            ).simplify()
        else:
            raise TypeError('No operator "*" defined for objects of '
                            'types %s and %s' % (type(self).__name__, type(f).__name__))

    cpdef Function add(self, Function f):
        if isinstance(f, LinearFunction):
            return LinearFunction(self.m + f.m, self.c + f.c)
        elif isinstance(f, numbers.Number):
            return LinearFunction(self.m, self.c + f)
        elif isinstance(f, ConstantFunction):
            return LinearFunction(self.m, self.c + f.value)
        else:
            raise TypeError(
                'Operator "+" undefined for types %s '
                'and %s' % (type(f).__name__, type(self).__name__)
            )

    def __sub__(self, x):
        return -x + self

    def __iadd__(self, other):
        return self.set(self + other)

    def __imul__(self, other):
        return self.set(self * other)

    def __eq__(self, other):
        if isinstance(other, LinearFunction):
            return (
                ((np.isnan(self.c) or np.isnan(self.m)) and (np.isnan(other.c) or np.isnan(other.m)))
                or (self.m == other.m and self.c == other.c)
            )
        elif isinstance(other, ConstantFunction):
            return (
                (self.m == 0 and other.value == self.c)
                or (np.isnan(self.c) or np.isnan(self.m)) and np.isnan(other.value)
            )
        elif isinstance(other, Undefined) and (np.isnan(self.c) or np.isnan(self.m)):
            return True
        elif isinstance(other, Function):
            return False
        else:
            raise TypeError('Can only compare objects of type "Function", but got type "%s".' % type(other).__name__)

    def __neg__(self):
        return self * -1

    cpdef Function set(self, Function f):
        if not isinstance(f, LinearFunction):
            raise TypeError('Unable to assign object parameters of '
                            'type %s to object of type %s' % (type(self).__name__, type(f).__name__))
        self.m = f.m
        self.c = f.c
        return self

    cpdef Function differentiate(self):
        return ConstantFunction(self.m)

    cpdef Function simplify(self):
        if self.m == 0:
            return ConstantFunction(self.c)
        else:
            return self.copy()

    @classmethod
    def from_points(cls, (DTYPE_t, DTYPE_t) p1, (DTYPE_t, DTYPE_t) p2) -> Function:
        """
        Construct a linear function for two points.
        :param p1: the first point
        :param p2: the second point
        :return: the ``LinearFunction`` that connects the points.
        """
        cdef DTYPE_t x1 = p1[0], y1 = p1[1]
        cdef DTYPE_t x2 = p2[0], y2 = p2[1]
        if x1 == x2:
            raise ValueError('Points must have different coordinates '
                             'to fit a line: p1=%s, p2=%s' % (p1, p2))
        if any(np.isnan(p) for p in itertools.chain(p1, p2)):
            raise ValueError('Arguments %s, %s are invalid.' % (p1, p2))
        if y2 == y1:
            return ConstantFunction(y2)
        cdef DTYPE_t m = (y2 - y1) / (x2 - x1)
        cdef DTYPE_t c = y1 - m * x1
        assert np.isfinite(m) and np.isfinite(c), \
            'Fitting linear function from %s to %s resulted in m=%s, c=%s' % (p1, p2, m, c)
        return cls(m, c)

    cpdef SIZE_t is_invertible(self):
        """
        Checks if this function can be inverted.
        :return: True if it is possible, False if not
        """
        return abs(self.m) >= 1e-4

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y):
        """
        Perform a linear regression of the form x -> y.
        :param x: The x values
        :param y: The y values
        :return: The best solution as ``LinearFunction``
        """
        self.m, self.c, _, _, _ = stats.linregress(x, y)
        return self

    def to_json(self):
        return {'type': 'linear',
                'slope': self.m,
                'intercept': self.c}

    @staticmethod
    def from_json(data):
        if data['type'] == 'linear':
            return LinearFunction(data['slope'], data['intercept'])
        elif data['type'] == 'constant':
            return ConstantFunction(data['value'])
        else:
            raise TypeError('Unknown function type or type not given (%s)' % data.get('type'))

    cpdef DTYPE_t integrate(self, DTYPE_t x1, DTYPE_t x2):
        '''
        Compute the integral over this function within the bounds ``x1`` and ``x2``.
        '''
        if x2 < x1:
            raise ValueError('The x2 argument must be greater than x1.')
        elif x2 == x1:
            return 0
        elif np.isinf(x1) and np.isinf(x2):
            if self.m != 0:
                return np.nan
            return np.inf if self.c > 0 else -np.inf
        elif np.isinf(x1):
            if self.m <= 0:
                return np.inf
            return -np.inf
        elif np.isinf(x2):
            if self.m >= 0:
                return np.inf
            return -np.inf
        return (.5 * self.m * x2 ** 2 + self.c * x2) - (.5 * self.m * x1 ** 2 + self.c * x1)

    cpdef LinearFunction xshift(self, DTYPE_t delta):
        '''
        Shift the function along the x-axis by ``delta``, i.e. f(x - delta).
        '''
        cdef LinearFunction f = self.copy()
        f.c = f(delta)
        return f

    cpdef Function xmirror(self):
        '''
        Returns a modification of the functino that has been mirrored
        at position x=0.
        :return:
        '''
        return LinearFunction(
            -self.m,
            self.c
        )


# ----------------------------------------------------------------------------------------------------------------------

cdef class QuadraticFunction(Function):
    """
    Implementation of a univariate quadratic function
    """

    def __init__(self, DTYPE_t a, DTYPE_t b, DTYPE_t c):
        self.a = a
        self.b = b
        self.c = c

    def __eq__(self, other):
        return isinstance(other, QuadraticFunction) and (self.a, self.b, self.c) == (other.a, other.b, other.c)

    cpdef Function set(self, Function f):
        self.a = f.a
        self.b = f.b
        self.c = f.c
        return self

    cpdef Function mul(self, Function f):
        if isinstance(f, ConstantFunction):
            return QuadraticFunction(self.a * f.value, self.b * f.value, self.c * f.value).simplify()
        else:
            TypeError('Unsupported operand type(s) for mul(): %s and %s.' % (type(self).__name__,
                                                                             type(f).__name__))

    cpdef Function add(self, Function f):
        if isinstance(f, ConstantFunction):
            return QuadraticFunction(self.a, self.b, self.c + f.value)
        elif isinstance(f, LinearFunction):
            return QuadraticFunction(self.a, self.b + f.m, self.c + f.c)
        elif isinstance(f, QuadraticFunction):
            return QuadraticFunction(self.a + f.a, self.b + f.b, self.c + f.c)

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.a * x * x + self.b * x + self.c

    cpdef DTYPE_t[::1] roots(self):
        cdef DTYPE_t det = self.b * self.b - 4. * self.a * self.c
        cdef DTYPE_t[::1] result
        if det > 0:
            result = np.ndarray(shape=(2,), dtype=np.float64)
            result[0] = (-self.b - np.sqrt(det)) / (2. * self.a)
            result[1] = (-self.b + np.sqrt(det)) / (2. * self.a)
        elif det == 0:
            result = np.ndarray(shape=(1,), dtype=np.float64)
            result[0] = -self.b / (2. * self.a)
        else:
            result = np.ndarray(shape=(0,), dtype=np.float64)
        return result

    cpdef Function invert(self):
        raise NotImplementedError()

    cpdef Function copy(self):
        return QuadraticFunction(self.a, self.b, self.c)

    cpdef SIZE_t intersects(self, Function f):
        raise NotImplementedError()

    cpdef ContinuousSet intersection(self, Function f):
        raise NotImplementedError()

    cpdef Function differentiate(self):
        return LinearFunction(2 * self.a, self.b)

    cpdef Function simplify(self):
        if not self.a:
            if not self.b:
                return ConstantFunction(self.c)
            return LinearFunction(self.b, self.c)
        return self.copy()

    cpdef SIZE_t is_invertible(self):
        return False

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y, DTYPE_t[::1] z):
        cdef DTYPE_t denom = (x[0] - y[0]) * (x[0] - z[0]) * (y[0] - z[0])
        self.a = (z[0] * (y[1] - x[1]) + y[0] *
                          (x[1] - z[1]) + x[0] * (z[1] - y[1])) / denom
        self.b = (z[0] * z[0] * (x[1] - y[1]) + y[0] * y[0]
                          * (z[1] - x[1]) + x[0] * x[0] * (y[1] - z[1])) / denom
        self.c = (y[0] * z[0] * (y[0] - z[0]) * x[1] + z[0] * x[0] *
                          (z[0] - x[0]) * y[1] + x[0] * y[0] * (x[0] - y[0]) * z[1]) / denom
        return self

    def to_json(self):
        return {'a': self.a, 'b': self.b, 'c': self.c}

    @staticmethod
    def from_json(data):
        return QuadraticFunction(data.get('a', 0), data.get('b', 0), data.get('c', 0))

    @classmethod
    def from_points(cls, (DTYPE_t, DTYPE_t) p1, (DTYPE_t, DTYPE_t) p2, (DTYPE_t, DTYPE_t) p3):
        if any(np.isnan(p) for p in itertools.chain(p1, p2, p3)):
            raise ValueError('Arguments %s, %s are invalid.' % (p1, p2))
        if p1 == p2 or p2 == p3 or p1 == p3:
            raise ValueError('Points must have different coordinates p1=%s, p2=%s, p3=%s' % (p1, p2, p3))
        x = np.array(list(p1), dtype=np.float64)
        y = np.array(list(p2), dtype=np.float64)
        z = np.array(list(p3), dtype=np.float64)
        return cls(np.nan, np.nan, np.nan).fit(x, y, z)

    cpdef DTYPE_t argvertex(self):
        return self.differentiate().simplify().root()

    def __str__(self):
        return ('%.3fx² %s%.3fx %s%.3f' % (self.a,
                                           {1: '+ ', 0: ''}[self.b >= 0],
                                           self.b,
                                           {1: '+ ', 0: ''}[self.c >= 0],
                                           self.c)).strip()

    def __repr__(self):
        return '<QuadraticFunction 0x%X: %s>' % (id(self), str(self))

# ----------------------------------------------------------------------------------------------------------------------

cdef class GaussianCDF(Function):
    """
    Integral of a univariate Gaussian distribution.
    """

    cdef readonly DTYPE_t mu, sigma

    def __init__(self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return norm.cdf(x, loc=self.mu, scale=self.sigma)


# ----------------------------------------------------------------------------------------------------------------------

cdef class GaussianPDF(Function):
    """
    Density of a univariate Gaussian distribution.
    """

    cdef readonly DTYPE_t mu, sigma

    def __init__(self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)

    def __str__(self):
        return '<GaussianPDF mu=%s sigma=%s>' % (self.mu, self.sigma)


# ----------------------------------------------------------------------------------------------------------------------

cdef class GaussianPPF(Function):
    """
    PPF of Gaussian distribution
    """

    cdef readonly DTYPE_t mu, sigma

    def __init__(self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return norm.ppf(x, loc=self.mu, scale=self.sigma)


# ----------------------------------------------------------------------------------------------------------------------

cdef class PiecewiseFunction(Function):
    """
    Represents a function that is piece-wise defined by constant values.
    """
    def __init__(self):
        self.functions: List[Function] = []
        self.intervals: List[ContinuousSet] = []

    def __hash__(self):
        return hash((PiecewiseFunction, ((i, f) for i, f in self.iter())))

    def append(self, interval: ContinuousSet, f: Function) -> None:
        self.intervals.append(interval)
        self.functions.append(f)

    def iter(self) -> Iterator[(ContinuousSet, Function)]:
        """ Iterate over intervals and functions at the same time. """
        return zip(self.intervals, self.functions)

    def __eq__(self, other):
        if not isinstance(other, PiecewiseFunction):
            raise TypeError(
                'An object of type "PiecewiseFunction" '
                'is required, but got "%s".' % type(other).__name__
            )
        for i1, f1, i2, f2 in zip(self.intervals, self.functions, other.intervals, other.functions):
            if i1 != i2 or f1 != f2:
                return False
        else:
            return True

    def __sub__(self, other: Function) -> PiecewiseFunction:
        return self.add(
            other.mul(
                ConstantFunction(-1)
            )
        )

    def __truediv__(self, other: numbers.Real) -> PiecewiseFunction:
        return self.mul(
            ConstantFunction(1 / float(other))
        )

    @classmethod
    def zero(cls, interval: ContinuousSet = None) -> PiecewiseFunction:
        '''
        Return a constant 'zero'-function on the specified interval, or on |R, if
        no interval is passed.
        '''
        return cls.from_dict({
            ifnone(interval, R.copy()): 0
        })

    def drop_undef(self) -> PiecewiseFunction:
        '''
        Return a copy of this ``PiecewiseFunction``, in which all segments
        of ``Undefined`` function instances have been removed.
        '''
        result = PiecewiseFunction()
        for i, f in self.iter():
            if isinstance(f, Undefined):
                continue
            result.append(i.copy(), f.copy())
        return result

    @classmethod
    def from_dict(cls, d: Dict[Union[ContinuousSet, str], Union[Function, str, float]]) -> PiecewiseFunction:
        '''
        Construct a ``PiecewiseFunction`` object from a set of key-value pairs mapping
        ``ContinuousSet``s to ``Function``s.

        Objects of ``ContinuousSet`` and ``Function`` are allowed or their respective string
        representation. ``ConstantFunction``s may be passed by means of their constant float value.

        :rtype: PiecewiseFunction
        '''
        intervals = []
        functions = []
        for interval, function in d.items():
            if type(interval) is str:
                interval = ContinuousSet.fromstring(interval)
            intervals.append(interval)
            if type(function) is str:
                function = LinearFunction.parse(function)
            elif function is None:
                function = Undefined()
            elif isinstance(function, numbers.Number):
                function = ConstantFunction(function)
            functions.append(function)
        fcts = sorted([(i, f) for (i, f) in zip(intervals, functions)], key=lambda a: a[0].lower)
        plf = PiecewiseFunction()
        plf.intervals.extend([i for i, _ in fcts])
        plf.functions.extend([f for _, f in fcts])
        return plf

    @classmethod
    def from_points(cls, points: Iterable[Tuple[float, float]]) -> PiecewiseFunction:
        '''
        Construct a contiguous piecewise-linear function from a sequence of (x, y) coordinates, where
        each point represents an interval border.
        '''
        plf = PiecewiseFunction()
        for p1, p2 in pairwise(points):
            x1, _ = p1
            x2, _ = p2
            i = ContinuousSet(x1, x2, INC, EXC)
            if not i.isempty():
                plf.functions.append(
                    LinearFunction.from_points(p1, p2)
                )
                plf.intervals.append(i)
        plf.intervals[-1].right = INC if np.isfinite(plf.intervals[-1].upper) else EXC
        plf.intervals[-1] = plf.intervals[-1].ends(right=EXC)
        return plf

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.at(x).eval(x)

    def __call__(self, x: float or np.ndarray):
        if isinstance(x, numbers.Real):
            return self.eval(x)
        elif isinstance(x, np.ndarray):
            return self.multi_eval(x)

    def __getitem__(self, key: int):
        return self.intervals[key], self.functions[key]

    cpdef DTYPE_t[::1] multi_eval(self, DTYPE_t[::1] x, DTYPE_t[::1] result=None):
        """
        Eval multiple points.
        :param x: The points
        :param result: Array of function values at x.
        :return: 
        """
        if result is None:
            result = np.ndarray(shape=len(x), dtype=np.float64)
        cdef int i
        for i in range(len(x)):
            result[i] = self.eval(x[i])
        return result

    cpdef Function at(self, DTYPE_t x):
        """
        Return the function segment at position ``x``.
        :param x: the point
        :return: The Function the contains x
        """
        cdef int idx = self.idx_at(x)
        if idx != -1:
            return self.functions[idx]
        return None

    cpdef int idx_at(self, DTYPE_t x):
        """
        Return the index of the function segment at position ``x``.
        :param x: The point
        :return: The idx of interval and function that describe x.
        """
        cdef int i
        cdef ContinuousSet interval
        for i, interval in enumerate(self.intervals):
            if x in interval or ((x == -np.inf and interval.lower == -np.inf) or
                                 (x == np.inf and interval.upper == np.inf)):
                return i
        return -1

    def __len__(self):
        return len(self.intervals)

    cpdef ContinuousSet interval_at(self, DTYPE_t x):
        """
        Get the interval that contains ``x``
        :param x: the point
        :return: The interval that contains ``x``
        """
        cdef int i
        cdef ContinuousSet interval
        for i, interval in enumerate(self.intervals):
            if x in interval:
                return interval
        else:
            return EMPTY

    cpdef Function set(self, Function f):
        if not isinstance(f, PiecewiseFunction):
            raise TypeError('Object of type %s can only be set '
                            'to attributes of the same type.' % type(self).__name__)
        self.intervals = [i.copy() for i in f.intervals]
        self.functions = [g.copy() for g in f.functions]
        return self

    # noinspection DuplicatedCode
    cpdef Function add(self, Function f):
        if isinstance(f, (ConstantFunction, LinearFunction)):
            result = self.copy()
            result.functions = [g + f for g in result.functions]
            return result
        elif isinstance(f, PiecewiseFunction):
            domain = RealSet(self.intervals).intersections(RealSet(f.intervals))
            undefined = self.domain().union(f.domain()).difference(domain)
            if not isinstance(undefined, RealSet):
                undefined = RealSet([undefined])
            result = PiecewiseFunction.from_dict({
                i: Undefined() for i in undefined.intervals
            })
            for interval in domain.intervals:
                if np.isfinite(interval.lower) and np.isfinite(interval.upper):
                    middle = (interval.lower + interval.upper) * .5
                elif np.isinf(interval.lower):
                    middle = interval.lower
                elif np.isinf(interval.upper):
                    middle = interval.upper
                f1 = ifnone(self.at(middle), Undefined())
                f2 = ifnone(f.at(middle), Undefined())
                result = result.overwrite_at(interval, f1 + f2)
            return result.simplify()
        else:
            raise TypeError(
                'Addition of type %s and %s currently unsupported.' % (
                    type(self).__name__, type(f).__name__
                )
            )

    # noinspection DuplicatedCode
    cpdef Function mul(self, Function f):
        if isinstance(f, (ConstantFunction, LinearFunction)):
            result = self.copy()
            result.functions = [g * f for g in result.functions]
            return result

        elif isinstance(f, PiecewiseFunction):
            domain = RealSet(self.intervals).intersections(
                RealSet(f.intervals)
            )

            undefined = self.domain().union(
                f.domain()
            ).difference(domain)

            if not isinstance(undefined, RealSet):
                undefined = RealSet([undefined])

            result = PiecewiseFunction.from_dict({
                i: Undefined() for i in undefined.intervals
            } if undefined else {})

            for interval in domain.intervals:
                pivot = interval.min
                if interval.isninf():
                    pivot = interval.max
                f1 = ifnone(self.at(pivot), Undefined())
                f2 = ifnone(f.at(pivot), Undefined())
                result = result.overwrite_at(interval, f1 * f2)
            return result.simplify()

        else:
            raise TypeError(
                'Multiplication of type %s and %s currently unsupported.' % (
                    type(self).__name__, type(f).__name__
                )
            )

    cpdef Function copy(self):
        cdef PiecewiseFunction result = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        result.functions = [f.copy() for f in self.functions]
        result.intervals = [i.copy() for i in self.intervals]
        return result

    def overwrite_at(self, interval: ContinuousSet or str, func: Function) -> 'PiecewiseFunction':
        """
        Overwrite this function in the specified interval range with the passed function ``func``.

        :param interval: The interval to update
        :param func: The function to replace the old function at ``interval``
        :return: The update function
        """
        interval = ifstr(interval, ContinuousSet.parse)
        result = self.copy()
        if not result.intervals:
            result.intervals.append(interval)
            result.functions.append(func)
            return result
        intervals = []
        functions = []
        added_ = False
        segments = deque(zip(result.intervals, result.functions))
        insert_pos = 0
        while segments:
            i, f = segments.popleft()
            intersection = i.intersection(interval, left=INC, right=EXC)
            i_ = i.difference(intersection).simplify()
            if i_.isempty():  # The original interval is subsumed by the new one and thus disappears
                continue
            if isinstance(i_, RealSet) and len(i_.intervals) > 1:
                segments.appendleft((i_.intervals[1], f.copy()))
                i_ = i_.intervals[0]
            if isinstance(i_, ContinuousSet):
                intervals.append(i_)
                functions.append(f.copy())
            if interval.max > i_.min:
                insert_pos += 1
        intervals.insert(insert_pos, interval.boundaries(left=0, right=EXC))
        functions.insert(insert_pos, func)
        result.intervals = list(intervals)
        result.functions = list(functions)
        return result

    def overwrite(self, segments: Dict[ContinuousSet or str, Function or float]) -> PiecewiseFunction:
        result = self.copy()
        for i, f in PiecewiseFunction.from_dict(segments).iter():
            result = result.overwrite_at(i, f)
        return result

    cpdef tuple split(self, DTYPE_t splitpoint):
        """
        Get two functions originating if this function is splitted at ``splitpoint``
        :param splitpoint: The position to split the function
        :return: the two splitted function
        """
        cdef PiecewiseFunction f1 = PiecewiseFunction(), f2 = PiecewiseFunction()
        cdef ContinuousSet interval, i_
        cdef Function f
        cdef int i
        for i, interval in enumerate(self.intervals):
            if splitpoint in interval:
                f1.intervals = [i_.copy() for i_ in self.intervals[:i]]
                f1.functions = [f.copy() for f in self.functions[:i]]
                f2.intervals = [i_.copy() for i_ in self.intervals[i:]]
                f2.functions = [f.copy() for f in self.functions[i:]]
                if f2.functions:
                    f1.intervals.append(f2.intervals[0].copy())
                    f1.functions.append(f2.functions[0].copy())
                    f2.intervals[0].lower = splitpoint
                    f2.intervals[0].left = 1
                f1.intervals[-1].upper = splitpoint
                f1.intervals[-1].right = 2
                break
        else:
            raise ValueError('This function is not defined at point %s' % splitpoint)
        return f1, f2

    def stretch(self, alpha: float) -> PiecewiseFunction:
        """
        Multiply every function by ``alpha``.
        :param alpha: The factor
        """
        plf = self.copy()
        for f in plf.functions:
            if isinstance(f, ConstantFunction):
                f.value *= alpha
            elif isinstance(f, LinearFunction):
                f.c *= alpha
                f.m *= alpha
        return plf

    cpdef str pfmt(self):
        """
        Pretty format of this function
        :return: pretty string
        """
        assert len(self.intervals) == len(self.functions), \
            ('Intervals: %s, Functions: %s' % (self.intervals, self.functions))
        return str('\n'.join([f'{str(i): <50} ↦ {str(f)}' for i, f in zip(self.intervals, self.functions)]))

    cpdef Function differentiate(self):
        cdef PiecewiseFunction diff = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        for i, f in zip(self.intervals, self.functions):
            diff.intervals.append(i.copy())
            diff.functions.append(f.differentiate())
        return diff

    cpdef DTYPE_t integrate(self, ContinuousSet interval = None):
        '''
        Compute the area under this ``PiecewiseFunction`` in the ``interval``.
        '''
        interval = ifnone(interval, R)
        cdef DTYPE_t area = 0
        cdef ContinuousSet intersect = None
        for i, f in self.iter():
            intersect = interval & i
            if intersect:
                area += f.integrate(intersect.lower, intersect.upper)
        return area

    cpdef ensure_left(self, Function left, DTYPE_t x):
        cdef ContinuousSet xing = self.functions[0].intersection(left)
        if xing == R:  # With all points are crossing points, we are done
            return
        elif xing and xing.lower < self.intervals[0].upper and xing.upper >= x:
            self.intervals.insert(0, ContinuousSet(-np.inf, xing.lower, EXC, EXC))
            self.intervals[1].lower = xing.lower
            self.intervals[1].left = INC
            self.functions.insert(0, left)
        else:
            self.intervals.insert(0, ContinuousSet(-np.inf, x, EXC, EXC))
            self.intervals[1].lower = x
            self.intervals[1].left = INC
            self.functions.insert(0, left)

            if not self.intervals[1]:
                del self.intervals[1]
                del self.functions[1]

    cpdef ensure_right(self, Function right, DTYPE_t x):
        cdef ContinuousSet xing = self.functions[-1].intersection(right)
        if xing == R:  # With all points are crossing points, we are done
            return
        elif xing and xing.upper > self.intervals[-1].lower and xing.lower <= x:
            self.intervals.append(ContinuousSet(xing.lower, np.inf, INC, EXC))
            self.intervals[-2].upper = xing.upper
            self.intervals[-2].left = INC
            self.functions.append(right)
        else:
            self.intervals.append(ContinuousSet(x, np.inf, INC, EXC))
            self.intervals[-2].upper = x
            self.intervals[-2].left = INC
            self.functions.append(right)
            if not self.intervals[-2]:
                del self.intervals[-2]
                del self.functions[-2]

    cpdef DTYPE_t[::1] xsamples(self, np.int32_t sort=True):
        cdef np.int32_t n = 2
        cdef np.int32_t samples_total = len(self.intervals) * (n + 1)
        cdef DTYPE_t[::1] samples_x = np.ndarray(shape=samples_total, dtype=np.float64)
        samples_x[:] = 0
        cdef object f
        cdef np.int32_t i = 0
        cdef DTYPE_t stepsize = 1
        cdef ContinuousSet interval
        cdef np.int32_t nopen = 2
        cdef DTYPE_t tmp_
        for interval, f in zip(self.intervals, self.functions):
            if interval.lower != -np.inf and interval.upper != np.inf:
                interval.linspace(n, default_step=1, result=samples_x[i:i + n])
                i += n
        if len(self.intervals) >= 2:
            if self.intervals[0].lower != -np.inf:
                nopen -= 1
            if self.intervals[-1].upper != np.inf:
                nopen -= 1
            if nopen:
                stepsize = ifnot(np.mean(np.abs(np.diff(samples_x[:i]))), 1)
        if self.intervals[0].lower == -np.inf:
            self.intervals[0].linspace(n, default_step=stepsize, result=samples_x[i:i + n])
            i += n
        if self.intervals[-1].upper == np.inf:
            self.intervals[-1].linspace(n, default_step=stepsize, result=samples_x[i:i + n])
            i += n
        if sort:
            return np.sort(samples_x[:i])
        else:
            return samples_x[:i]

    cpdef RealSet eq(self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        prev_f = None
        for i, f in zip(self.intervals, self.functions):
            x = f.intersection(y_)
            if x and x in i:
                result_set.intervals.append(i.intersection(x))
            elif prev_f and (prev_f(i.lower) < y < f(i.lower) or prev_f(i.lower) > y > f(i.lower)):
                result_set.intervals.append(ContinuousSet(i.lower, i.lower))
            prev_f = f
        return result_set

    cpdef RealSet lt(self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        prev_f = None
        current = None
        for i, f in zip(self.intervals, self.functions):
            x = f.intersection(y_)
            if x.size() == 1 and x.lower in i:
                if f.m > 0:
                    if current is None:
                        result_set.intervals.append(ContinuousSet(-np.inf, x.upper, 2, 2))
                    else:
                        current.upper = x.upper
                        result_set.intervals.append(current)
                        current = None
                elif f.m < 0:
                    current = ContinuousSet(x.lower, np.inf, 2, 2)
            elif not x and f(i.lower) < y and current is None:
                current = ContinuousSet(i.lower, np.inf, 2, 2)
            elif not x and f(i.lower) >= y and current is not None:
                current.upper = i.lower
                result_set.intervals.append(current)
                current = None
            prev_f = f
        if current is not None:
            result_set.intervals.append(current)
        return result_set

    cpdef RealSet gt(self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        current = None
        for i, f in zip(self.intervals, self.functions):
            x = f.intersection(y_)
            if x.size() == 1 and x.lower in i:
                if f.m < 0:
                    if current is None:
                        result_set.intervals.append(ContinuousSet(-np.inf, x.upper, 2, 2))
                    else:
                        current.upper = x.upper
                        result_set.intervals.append(current)
                        current = None
                elif f.m > 0:
                    current = ContinuousSet(x.lower, np.inf, 2, 2)
            elif not x and f(i.lower) > y and current is None:
                current = ContinuousSet(i.lower, np.inf, 1, 2)
            elif (not x and f(i.lower) < y or x == R and f(i.lower) == y) and current is not None:
                current.upper = i.lower
                result_set.intervals.append(current)
                current = None
        if current is not None:
            result_set.intervals.append(current)
        return result_set

    @staticmethod
    def merge(f1: PiecewiseFunction, f2: PiecewiseFunction, mergepoint: float) -> PiecewiseFunction:
        """
        Merge two piecewise linear functions ``f1`` and ``f2`` at the merge point ``mergepoint``.

        The result will be a CDF-conform function
        :param f1: the first function
        :param f2: the second function
        :param mergepoint: the mergepoint
        :return: merged CDF
        """
        lower, _ = f1.split(mergepoint)
        _, upper = f2.split(mergepoint)
        v1 = lower.eval(lower.intervals[-1].lower)
        v2 = upper.eval(upper.intervals[0].upper)
        upper.add_const(v1 - v2)  # Ensure the function is continuous at the merge point
        result = PiecewiseFunction()
        result.intervals = lower.intervals + upper.intervals
        result.functions = lower.functions + upper.functions
        result.stretch(1. / result.eval(result.intervals[-1].lower))  # Ensure that the function ends with value 1 on the right
        return result

    cpdef list knots(PiecewiseFunction self, DTYPE_t lastx=np.nan):
        result = []
        for i, f in zip(self.intervals, self.functions):
            if i.lower != -np.inf:
                result.append((i.lower, f.eval(i.lower)))
        if self.intervals and self.intervals[-1].upper != np.inf:
            result.append((self.intervals[-1].upper, self.functions[-1].eval(self.intervals[-1].upper)))
        elif not np.isnan(lastx):
            result.append((lastx, self.functions[-1].eval(lastx)))
        return result

    cpdef PiecewiseFunction add_knot(PiecewiseFunction self, DTYPE_t x, DTYPE_t y):
        ...

    cpdef PiecewiseFunction crop(PiecewiseFunction self, ContinuousSet interval):
        '''
        Return a copy of this ``PiecewiseFunction``, whose domain is cropped to the passed interval.
        '''
        cdef PiecewiseFunction result = PiecewiseFunction()
        for i, f in self.iter():
            if i.intersects(interval):
                intersection = i.intersection(interval, left=INC, right=EXC)
                result.intervals.append(intersection)
                result.functions.append(f.copy())
        return result

    def to_json(self) -> Dict[str, Any]:
        return {
            'intervals': [i.to_json() for i in self.intervals],
            'functions': [f.to_json() for f in self.functions]
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]):
        function = cls()
        function.intervals = [ContinuousSet.from_json(d) for d in data['intervals']]
        function.functions = [LinearFunction.from_json(d) for d in data['functions']]
        return function

    def __repr__(self):
        return self.pfmt()

    def round(self, digits: int = None, include_intervals: bool = True) -> PiecewiseFunction:
        """
        Return a copy of this PLF, in which all parameters of sub-functions have been rounded by
        the specified number of digits.

        If ``include_intervals`` is ``False``, the parameter values of the intervals will not be affected by
        this operation.
        
        :param digits: the amount of digits to round to. 
        :param include_intervals: Rather to round interval borders or not.
        :return: A new, rounded function.
        """
        digits = ifnone(digits, 3)
        round_ = lambda x: np.round(x, decimals=digits)

        plf = PiecewiseFunction()
        for interval, function in zip(self.intervals, self.functions):
            if include_intervals:
                interval = interval.copy()
                interval.lower = round_(interval.lower) if not np.isinf(interval.lower) else interval.lower
                interval.upper = round_(interval.upper) if not np.isinf(interval.upper) else interval.upper
            plf.intervals.append(interval)
            if isinstance(function, LinearFunction):
                function = LinearFunction(round_(function.m), round_(function.c))
            elif isinstance(function, ConstantFunction):
                function = ConstantFunction(round_(function.value))
            else:
                raise TypeError('Unknown function type in PiecewiseFunction: "%s"' % type(function).__name__)
            plf.functions.append(function)
        return plf

    def domain(self) -> NumberSet:
        '''
        Return the domain of this PLF, i.e. the range of input values the PLF is defined on.
        '''
        return RealSet(self.intervals).simplify()

    cpdef PiecewiseFunction simplify(self):
        segments = sorted(
            list(self.iter()),
            key=cmp_to_key(self.cmp_segments)
        )
        queue = deque(segments)
        plf = PiecewiseFunction()
        while queue:
            i, f = queue.popleft()
            if i.isempty():
                continue
            if not plf.functions:
                plf.append(i, f)
                continue
            i_, f_ = plf[-1]
            if i.contiguous(i_) and f == f_:
                i_.upper = i.upper
                i_.right = i.right
            else:
                plf.append(i, f)
        return plf

    @staticmethod
    def combine(f1: PiecewiseFunction, f2: PiecewiseFunction, operator: str) -> PiecewiseFunction:
        # The combination of two functions is only defined on their intersecting domains
        domain = (f1.domain() & f2.domain()).simplify()
        intervals = [
            (i.min, i.max + eps) for i in f1.intervals
        ] + [
            (i.min, i.max + eps) for i in f2.intervals
        ]
        intervals = domain.chop({i for i in chain(*intervals) if np.isfinite(i)})
        queue = deque(intervals)

        plf = PiecewiseFunction()
        while queue:
            i = queue.popleft()
            i_center = (i.min + i.max) * .5
            l1 = f1.at(i_center)
            l2 = f2.at(i_center)
            if (not isinstance(l1, (LinearFunction, ConstantFunction)) or
                    not isinstance(l2, (LinearFunction, ConstantFunction))):
                raise TypeError(
                    'combine() currently only support on linear functions, got %s and %s.' % (type(l1), type(l2))
                )
            intersect = l1.intersection(l2)
            if intersect.size() == 1 and intersect.min in i and intersect.min not in (i.min, i.max):
                queue.extendleft(reversed(list(i.chop([intersect.min]))))
                continue
            delta = l1.eval(i_center) - l2.eval(i_center)
            if operator == 'max':
                delta *= -1
            plf.append(
                i.copy(),
                (l1 if delta <= 0 else l2).copy()
            )
        return plf.simplify()

    @staticmethod
    def min(f1: PiecewiseFunction, f2: PiecewiseFunction) -> PiecewiseFunction:
        return PiecewiseFunction.combine(f1, f2, operator='min')

    @staticmethod
    def max(f1: PiecewiseFunction, f2: PiecewiseFunction) -> PiecewiseFunction:
        return PiecewiseFunction.combine(f1, f2, operator='max')

    @staticmethod
    def abs(f: PiecewiseFunction) -> PiecewiseFunction:
        return PiecewiseFunction.max(f, f * -1)

    def is_impulse(self) -> float:
        '''
        Determine wether or not this ``PiecewiseFunction`` represents a Dirac impulse.
        If yes, the return value is the non-zero function value at that impulse.
        Otherwise, it returns 0.
        :return:
        '''
        if (
            len(self) == 3 and
            self.intervals[0].isninf() and
            self.functions[0] == ConstantFunction(0) and
            self.intervals[-1].ispinf() and
            self.functions[-1] == ConstantFunction(0) and
            not self.intervals[1].isempty() and
            self.intervals[1].min + eps >= self.intervals[1].upper - eps
        ):
            return self.functions[1].eval(self.intervals[1].lower)
        elif (
            len(self) == 1 and
            self.domain() == R and
            isinstance(self.functions[0], Impulse)
        ):
            return self.functions[0].weight
        return False

    @staticmethod
    def cmp_segments(s1: Tuple[ContinuousSet, Function], s2: Tuple[ContinuousSet, Function]) -> int:
        '''
        A comparator for <interval, function> pais. Uses the ``ContinuousSet.comparator()`` function.
        '''
        return ContinuousSet.comparator(s1[0], s2[0])

    @staticmethod
    def jaccard_similarity(
            f1: PiecewiseFunction,
            f2: PiecewiseFunction,
            interval: ContinuousSet = None
    ) -> float:
        '''
        Compute the Jaccard index given by the quotient of the intersection over the union of the two
        '''
        interval = ifnone(interval, R)
        f_max = PiecewiseFunction.max(f1, f2)
        f_min = PiecewiseFunction.min(f1, f2)
        return f_min.integrate(interval) / f_max.integrate(interval)

    cpdef PiecewiseFunction xshift(self, DTYPE_t delta):
        '''
        Returns a copy of this function, which is shifted the on the x-axis by ``delta``.
        
        Corresponds to a translation of $f(x + \Delta)$, i.e. positive values of
        $\Delta$ will cause the function to "move to the left", negative value will
        move it to the right. 
        '''
        cdef PiecewiseFunction f = self.copy()
        for j, i in enumerate(f.intervals):
            if np.isfinite(i.lower):
                i.lower -= delta
            if np.isfinite(i.upper):
                i.upper -= delta
            f.functions[j] = f.functions[j].xshift(delta)
        return f

    def boundaries(self) -> np.ndarray:
        points = [fst(self.intervals, attrgetter('lower'))]
        for i1, i2 in pairwise(self.intervals):
            if i1.contiguous(i2):
                points.append(i2.min)
            else:
                points.extend([i1.max, i2.min])
        points.append(last(self.intervals, attrgetter('upper')))
        return np.sort(
            np.array(
                list(
                    set(
                        filter(
                            np.isfinite,
                            points
                        )
                    )
                )
            )
        )

    cpdef Function xmirror(self):
        cdef PiecewiseFunction result = PiecewiseFunction()
        result.intervals = [
            i.ends(
                left=INC if np.isfinite(i.lower) else EXC,
                right=EXC
            ) for i in RealSet(self.intervals).xmirror().intervals
        ]
        result.functions = list(reversed([f.xmirror() for f in self.functions]))
        return result

    def convolution(self, g: PiecewiseFunction) -> PiecewiseFunction:
        '''
        Compute the convolution of this function $f$ with another function $g$.

        .. math::
            f*g(z)=\int_{-\infty}^{+\infty} f(y)g(z-y) dy

        At the moment, convolution only supports piecewise constant functions.

        :param g:
        :return:
        '''
        f = self
        if f.is_impulse():
            return g.copy()
        elif g.is_impulse():
            return f.copy()
        for func in itertools.chain(f.functions, g.functions):
            if (not isinstance(func, (ConstantFunction, Undefined)) and
                    isinstance(func, LinearFunction) and func.m != 0):
                raise TypeError(
                    'Only constant functions are supported for convolution, got %s.' %
                    type(func).__name__
                )
        # Compute all interval transitions
        boundaries_f = f.boundaries()
        boundaries_g = np.sort(-g.boundaries())

        # z is the position to which g needs to be shifted, so that
        # g's and f's domains are just contiguous
        z = boundaries_g.max() - boundaries_f.min()  # positive z means g's right border is right hand of f's left border
        g_ = g.xmirror().xshift(z)  # positive z shifts g to the left
        boundaries_g -= z  # boundaries move to the left
        z *= -1
        support_points = [(z, (f * g_).integrate())]
        iteration = 1
        while 1:
            distances = np.array(  # compute the pairwise distances of boundaries
                [b_g - b_f for b_g in boundaries_g for b_f in boundaries_f],
                dtype=np.float64
            )
            if (distances >= 0).all():
                break
            delta_min = np.abs(distances[distances < 0].max())
            # shift g ahead by delta_min
            b = g_
            g_ = g_.xshift(-delta_min)
            boundaries_g += delta_min
            f_times_g = f * g_
            z += delta_min
            # print(f, '\n---', g_, '\n===', f_times_g)
            integral = f_times_g.integrate()
            if (z, integral) not in support_points:
                support_points.append((z, integral))
            iteration += 1
        domain = f.domain().union(g.domain())
        if isinstance(domain, ContinuousSet):
            domain = RealSet([domain])
        result = PiecewiseFunction.from_dict({
            i: v for i, v in [
                (first(domain.intervals), first(support_points)[1]),
                (last(domain.intervals), last(support_points)[1])
            ]
        })

        for i, h in PiecewiseFunction.from_points(support_points).iter():
            result = result.overwrite_at(i, h)
        return result.simplify()

    def rectify(self) -> PiecewiseFunction:
        '''
        Returns a modification of this ``PiecewiseFunction``, in which all linear non-constant
        function components have been replaced by constants given by the mean of the
        original linear function, such that the result is a function of "rectangles".

        :return:
        '''
        result = PiecewiseFunction()
        for i, f in self.iter():
            if isinstance(f, LinearFunction):
                if f.m != 0 and np.isinf([i.lower, i.upper]).any():
                    raise ValueError(
                        'Expected finite interval, got %s on function %s' % (i, f)
                    )
                f_ = ConstantFunction((f(i.min) + f(i.max)) * .5)
            elif isinstance(f, (ConstantFunction, Undefined)):
                f_ = f.copy()
            else:
                raise TypeError(
                    'Rectification of functions of type %s are currently unsupported.' %
                    type(f).__name__
                )
            result.append(i.copy(), f_)
        return result

    def maximize(self) -> Tuple[RealSet, float]:
        '''
        Determine the global maxima of this ``PiecewiseFunction``
        :return:
        '''
        f_max = np.nan
        f_argmax = EMPTY.copy()
        d = self.domain()

        for b in itertools.chain(
                [i.min for i in self.intervals],
                [i.max for i in self.intervals]
        ):
            idx = self.idx_at(b)
            f = self.functions[idx]
            i = self.intervals[idx]
            f_ = f(b)
            if np.isnan(f_) or not np.isnan(f_max) and f_max > f_:
                continue
            if isinstance(f, LinearFunction):
                argmax = ContinuousSet(b, b)
            elif isinstance(f, ConstantFunction):
                argmax = i.copy()
            else:
                raise TypeError(
                    'Maximizing functions of type %s '
                    'is currently not supported.' % type(self).__name__
                )
            if f_max == f_:
                f_argmax = f_argmax.union(argmax)
            else:
                f_argmax = argmax
            f_max = f_
        return f_argmax, f_max

    def approximate(
            self,
            error_max: float = None,
            n_segments = None,
            replace_by: type = LinearFunction
    ) -> PiecewiseFunction:
        '''
        Compute an approximation of this `PiecewiseFunction`, which comprises fewer
        function segments than the original PLF.

        This is done by iteratively replacing subsequent function
        segments by an approximation thereof.

        :param error_max:       the maximal error allowed for constructing an approximation
        :param n_segments:      the desired number of function segments of the approximation result
        :param replace_by:      (ConstantFunction, LinearFunction) the type of function to be used for
                                the approximations
        :return:
        '''
        return PLFApproximator(
            self,
            replace_by=replace_by
        ).run(
            error_max=error_max,
            k=n_segments
        )


# ----------------------------------------------------------------------------------------------------------------------

class PLFApproximator:
    """
    This class implements an algorithm for simplifying a `PiecewiseFunction`
    by iteratively "merging" function segments of contiguous intervals by some
    approximation. The algorithm maintains a heap of all pairwise consecutive
    itervals ("FunctionSegments") and always chooses the replacement that
    causes the minimal increase in the mean squared error.
    """

    class FunctionSegment:
        def __init__(self, i: ContinuousSet, f: Function):
            self.i: ContinuousSet = i
            self.f: Function = f

        def __eq__(self, other):
            return self.i == other.i and self.f == other.f

    class FunctionSegmentReplacement:

        def __init__(
                self,
                error: float,
                left: 'PLFApproximator.FunctionSegment',
                right: 'PLFApproximator.FunctionSegment',
                new: 'PLFApproximator.FunctionSegment',
                next_repl: Optional['PLFApproximator.FunctionSegmentReplacement'] = None,
                prev_repl: Optional['PLFApproximator.FunctionSegmentReplacement'] = None
        ):
            self.left = left
            self.right = right
            self.next = next_repl
            self.prev = prev_repl
            self.error = error
            self.new = new

        def __eq__(self, other):
            if other is None:
                return False
            return all((
                self.left == other.left,
                self.right == other.right,
                self.error == other.error,
                self.new == other.new
            ))

    def __init__(
            self,
            plf: PiecewiseFunction,
            replace_by: type = LinearFunction
        ):
        self.plf: PiecewiseFunction = plf
        self.replace_by: type = replace_by

    def _construct_replacement(
            self,
            left: FunctionSegment,
            right: FunctionSegment
    ) -> FunctionSegmentReplacement:
        '''
        Compute a function of type `replace_by` spanning the union
        interval of i1 and i2 and estimate the mean sqaured error
        induced by this new function segment.
        '''
        if not left.i.contiguous(right.i) or left.i.isninf() or right.i.ispinf():
            raise ValueError(
                'left and right must be finite, contiguous segments, got left.i=%s, right.i=%s.' % (
                    left.i, right.i
                )
            )

        # Create a replacement function segment spanning both intervals
        i = ContinuousSet(left.i.lower, right.i.upper, left.i.left, right.i.right)
        if self.replace_by is LinearFunction:
            f = LinearFunction.from_points(
                (i.lower, left.f(i.lower)),
                (i.upper, right.f(i.upper))
            )
        elif self.replace_by is ConstantFunction:
            f = ConstantFunction(
                left.f(i.lower) * left.i.width / i.width + right.f(i.upper) * right.i.width / i.width
            )
        else:
            raise TypeError(
                'Unsupported type of replacement function: %s.' % self.replace_by.__name__
            )

        # Estimate the error induced by the replacement
        error = self.plf.crop(i) - f
        # error_sq = error.mul(error)
        # mse = error_sq.integrate() / i.width
        mae = PiecewiseFunction.abs(error).integrate() / i.width  # .maximize()[1]
        return PLFApproximator.FunctionSegmentReplacement(
            mae,
            left,
            right,
            PLFApproximator.FunctionSegment(i, f)
        )

    def run(
            self,
            error_max: float = None,
            k = None,
    ) -> PiecewiseFunction:
        '''Compute an approximation of the function under consideration.'''
        if k is not None and k < 3:
            raise ValueError(
                'Minimum value for k is 3, got %s.' % k
            )
        error_max = ifnone(error_max, np.inf)
        result = self.plf.copy()
        replacements = []
        # Loop through all pairs of consecutive function segments
        for (i1, f1), (i2, f2) in pairwise(result.iter()):
            # We only consider finite, contiguous segments
            if not i1.contiguous(i2) or i1.isninf() or i2.ispinf():
                continue
            # Store admissible replacement candidates on the heap
            replacement = self._construct_replacement(
                PLFApproximator.FunctionSegment(i1, f1),
                PLFApproximator.FunctionSegment(i2, f2)
            )
            if replacements:  # Update the double-linked list
                replacement.prev = replacements[-1]
                replacements[-1].next = replacement
            replacements.append(replacement)

        queue = Heap(replacements, key=attrgetter('error'))
        # Keep replacing the contiguous segments until we
        # hit one of the abortion criteria
        while queue:
            replacement = queue.pop()
            # Abortion: either the maximal number of segments is hit
            # or we exceed the mse_max parameter
            if (
                    len(result) <= ifnone(k, 3) or
                    error_max < replacement.error
            ):
                break

            # Remove the left and right segments of the replacement from the original function
            del result.intervals[result.intervals.index(replacement.left.i)]
            del result.functions[result.functions.index(replacement.left.f)]
            del result.intervals[result.intervals.index(replacement.right.i)]
            del result.functions[result.functions.index(replacement.right.f)]

            # Insert the new segment at the interval union of the former left and right segments
            result = result.overwrite_at(
                replacement.new.i,
                replacement.new.f
            )

            # Remove the "prev" and "next" replacements from the heap as one counterpart of
            # both their "left" and "right" segments has become obsolete.
            # Also, insert new replacement candidates given by the new segment and the "left" segment
            # of the "prev" replacement, and by the "new" segment and the "right" segment of
            # the "next" replacement
            new_replacement_left = None
            if replacement.prev is not None:
                del queue[queue.index(replacement.prev)]
                new_replacement_left = self._construct_replacement(
                    replacement.prev.left,
                    replacement.new
                )
                if replacement.prev.prev is not None:
                    replacement.prev.prev.next = new_replacement_left
                    new_replacement_left.prev = replacement.prev.prev
                queue.push(
                    new_replacement_left
                )
            new_replacement_right = None
            if replacement.next is not None:
                del queue[queue.index(replacement.next)]
                new_replacement_right = self._construct_replacement(
                    replacement.new,
                    replacement.next.right
                )
                if replacement.next.next is not None:
                    replacement.next.next.prev = new_replacement_right
                    new_replacement_right.next = replacement.next.next
                queue.push(
                    new_replacement_right
                )
            if None not in (new_replacement_left, new_replacement_right):
                new_replacement_left.next = new_replacement_right
                new_replacement_right.prev = new_replacement_left

        return result
