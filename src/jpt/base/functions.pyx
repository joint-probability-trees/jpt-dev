# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False

__module__ = 'functions.pyx'

import itertools
import numbers
from collections import deque
from typing import Iterator, List

from dnutils import ifnot, ifnone, pairwise
from scipy import stats
from scipy.stats import norm

from .intervals cimport ContinuousSet, RealSet
from .intervals import R, EMPTY, EXC, INC

import numpy as np
cimport numpy as np
cimport cython

import warnings

from .cutils cimport DTYPE_t

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------------------------------------------------------

@cython.freelist(1000)
cdef class Function:
    """
    Abstract base type of functions.
    """

    def __call__(self, x):
        return self.eval(x)

    cpdef DTYPE_t eval(self, DTYPE_t x):
        """
        Evaluate this function at position ``x``
        :param x: the value where to eval this function 
        :return: f(x)
        """
        return np.nan

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
        if isinstance(other, numbers.Real):
            return self.add(ConstantFunction(other)).simplify()
        elif isinstance(other, Function):
            return self.add(other).simplify()
        else:
            raise TypeError('Unsupported operand type(s) for +: %s and %s' % (type(self).__name__,
                                                                              type(other).__name__))

    def __mul__(self, other):
        if isinstance(other, numbers.Real):
            return self.mul(ConstantFunction(other)).simplify()
        elif isinstance(other, Function):
            return self.mul(other).simplify()
        else:
            raise TypeError('Unsupported operand type(s) for *: %s and %s' % (type(self).__name__,
                                                                              type(other).__name__))

    def __iadd__(self, other):
        return self.set(self + other)

    def __imul__(self, other):
        return self.set(self * other)

    def __radd__(self, other):
        return other + self

    def __rmul__(self, other):
        return other * self

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
        return False

    cpdef Function set(self, Function f):
        return self

    cpdef Function mul(self, Function f):
        return Undefined()

    cpdef Function add(self, Function f):
        return Undefined()

    cpdef Function copy(self):
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

    cpdef np.int32_t is_invertible(self):
        return False

    cpdef Function copy(self):
        return ConstantFunction(self.value)

    def __eq__(self, other):
        if not isinstance(other, Function):
            raise TypeError('Cannot compare object of type %s to %s.' % (type(self).__name__,
                                                                         type(other).__name__))
        if isinstance(other, (LinearFunction, ConstantFunction)):
            return self.m == other.m and self.c == other.c
        return False

    cpdef Function set(self, Function f):
        if isinstance(f, ConstantFunction):
            self.value = f.value
            return self
        else:
            raise TypeError('Object of type %s can only be set to '
                            'parameters of objects of the same type' % type(self).__name__)

    cpdef Function mul(self, Function f):
        if isinstance(f, ConstantFunction):
            return ConstantFunction(self.value * f.value)
        elif isinstance(f, (LinearFunction, QuadraticFunction)):
            return f.mul(self)
        else:
            raise TypeError('Unsupported operand type(s) for '
                            'mul(): %s and %s.' % (type(self).__name__, type(f).__name__))

    cpdef Function add(self, Function f):
        if isinstance(f, ConstantFunction):
            return ConstantFunction(self.value + f.value)
        elif isinstance(f, (LinearFunction, QuadraticFunction)):
            return f.add(self)
        else:
            raise TypeError('Unsupported operand type(s) for '
                            'add(): %s and %s.' % (type(self).__name__, type(f).__name__))

    @property
    def m(self):
        return 0

    @property
    def c(self):
        return self.value

    cpdef np.int32_t intersects(self, Function f) except +:
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

    cpdef ContinuousSet intersection(self, Function f) except +:
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

    cpdef DTYPE_t root(self) except +:
        """
        Find the root of the function, i.e. the ``x`` positions subject to ``self.eval(x) = 0``. 
        :return: root of this function as float
        """
        return -self.c / self.m

    cpdef Function invert(self) except +:
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

    cpdef np.int32_t intersects(self, Function f) except +:
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

    cpdef ContinuousSet intersection(self, Function f) except +:
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
            return QuadraticFunction(self.m * f.m, self.m * f.c + f.m * self.c, self.c * f.c).simplify()
        else:
            raise TypeError('No operator "*" defined for objects of '
                            'types %s and %s' % (type(self).__name__, type(f).__name__))

    cpdef Function add(self, Function f):
        if isinstance(f, LinearFunction):
            return LinearFunction(self.m + f.m, self.c + f.c)
        elif isinstance(f, (int, float)):
            return LinearFunction(self.m, self.c + f)
        elif isinstance(f, ConstantFunction):
            return LinearFunction(self.m, self.c + f.value)
        else:
            raise TypeError('Operator "+" undefined for types %s '
                            'and %s' % (type(f).__name__, type(self).__name__))

    def __sub__(self, x):
        return -x + self

    def __radd__(self, x):
        return self + x

    def __rsub__(self, x):
        return self - x

    def __rmul__(self, o):
        return self * o

    def __iadd__(self, other):
        return self.set(self + other)

    def __imul__(self, other):
        return self.set(self * other)

    def __eq__(self, other):
        if isinstance(other, (ConstantFunction, LinearFunction)):
            return self.m == other.m and self.c == other.c
        elif isinstance(other, Function):
            return False
        else:
            raise TypeError('Can only compare objects of type "Function", but got type "%s".' % type(other).__name__)

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

    @staticmethod
    def from_points((DTYPE_t, DTYPE_t) p1, (DTYPE_t, DTYPE_t) p2):
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
        assert not np.isnan(m) and not np.isnan(c), \
            'Fitting linear function from %s to %s resulted in m=%s, c=%s' % (p1, p2, m, c)
        return LinearFunction(m, c)

    cpdef np.int32_t is_invertible(self):
        """
        Checks if this function can be inverted.
        :return: True if it is possible, False if not
        """
        return abs(self.m) >= 1e-4

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y) except +:
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

    cpdef DTYPE_t root(self) except +:
        raise NotImplementedError()

    cpdef Function invert(self) except +:
        raise NotImplementedError()

    cpdef Function copy(self):
        return QuadraticFunction(self.a, self.b, self.c)

    cpdef np.int32_t intersects(self, Function f) except +:
        raise NotImplementedError()

    cpdef ContinuousSet intersection(self, Function f) except +:
        raise NotImplementedError()

    cpdef Function differentiate(self):
        return LinearFunction(2 * self.a, self.b)

    cpdef Function simplify(self):
        if not self.a:
            if not self.b:
                return ConstantFunction(self.c)
            return LinearFunction(self.b, self.c)
        return self.copy()

    cpdef np.int32_t is_invertible(self):
        return False

    cpdef Function fit(self, DTYPE_t[::1] x, DTYPE_t[::1] y, DTYPE_t[::1] z) except +:
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

    @staticmethod
    def from_points((DTYPE_t, DTYPE_t) p1, (DTYPE_t, DTYPE_t) p2, (DTYPE_t, DTYPE_t) p3):
        if any(np.isnan(p) for p in itertools.chain(p1, p2, p3)):
            raise ValueError('Arguments %s, %s are invalid.' % (p1, p2))
        if p1 == p2 or p2 == p3 or p1 == p3:
            raise ValueError('Points must have different coordinates p1=%s, p2=%s, p3=%s' % (p1, p2, p3))
        x = np.array(list(p1), dtype=np.float64)
        y = np.array(list(p2), dtype=np.float64)
        z = np.array(list(p3), dtype=np.float64)
        return QuadraticFunction(np.nan, np.nan, np.nan).fit(x, y, z)

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

    def iter(self) -> Iterator[(ContinuousSet, Function)]:
        """ Iterate over intervals and functions at the same time. """
        return zip(self.intervals, self.functions)

    def __eq__(self, other):
        if not isinstance(other, PiecewiseFunction):
            raise TypeError('An object of type "PiecewiseFunction" '
                            'is required, but got "%s".' % type(other).__name__)
        for i1, f1, i2, f2 in zip(self.intervals, self.functions, other.intervals, other.functions):
            if i1 != i2 or f1 != f2:
                return False
        else:
            return True

    @staticmethod
    def from_dict(d):
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

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.at(x).eval(x)

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
            if x in interval or ((x == np.NINF and interval.lower == np.NINF) or
                                 (x == np.PINF and interval.upper == np.PINF)):
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
        if isinstance(f, (ConstantFunction, LinearFunction, QuadraticFunction)):
            result = self.copy()
            result.functions = [g + f for g in result.functions]
            return result
        elif isinstance(f, PiecewiseFunction):
            result = PiecewiseFunction()
            knots = sorted(set((itertools.chain(*[(i.lower, i.upper) for i in f.intervals] +
                                                 [(i.lower, i.upper) for i in self.intervals]))))
            for lower, upper in pairwise(knots):
                result.intervals.append(ContinuousSet(lower, upper, INC, EXC))
                if not np.isinf(lower) and not np.isinf(upper):
                    middle = (lower + upper) * .5
                elif np.isinf(lower):
                    middle = lower
                elif np.isinf(upper):
                    middle = upper
                result.functions.append(ifnone(self.at(middle),
                                               Undefined()) +
                                        ifnone(f.at(middle),
                                               Undefined()))
            if result.intervals:
                if np.isinf(result.intervals[0].lower):
                    result.intervals[0].left = EXC
            return result.simplify()

    # noinspection DuplicatedCode
    cpdef Function mul(self, Function f):
        if isinstance(f, ConstantFunction):
            result = self.copy()
            for i, g in result.iter():
                g *= f
            return result
        elif isinstance(f, PiecewiseFunction):
            result = PiecewiseFunction()
            knots = sorted(set((itertools.chain(*[(i.lower, i.upper) for i in f.intervals] +
                                                 [(i.lower, i.upper) for i in self.intervals]))))
            for lower, upper in pairwise(knots):
                result.intervals.append(ContinuousSet(lower, upper, INC, EXC))
                if not np.isinf(lower) and not np.isinf(upper):
                    middle = (lower + upper) * .5
                elif np.isinf(lower):
                    middle = lower
                elif np.isinf(upper):
                    middle = upper
                result.functions.append(ifnone(self.at(middle),
                                               Undefined()) *
                                        ifnone(f.at(middle),
                                               Undefined()))
            if result.intervals:
                if np.isinf(result.intervals[0].lower):
                    result.intervals[0].left = EXC
            return result.simplify()

    cpdef Function copy(self):
        cdef PiecewiseFunction result = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        result.functions = [f.copy() for f in self.functions]
        result.intervals = [i.copy() for i in self.intervals]
        return result

    def add_function(self, interval, func):
        """
        Add a function and interval into this function.
        :param interval: The interval to update
        :param func: The function to replace the old function at ``interval``
        :return: The update function
        """
        if not self.intervals:
            self.intervals.append(interval)
            self.functions.append(func)
            return
        intervals = deque()
        functions = deque()
        added_ = False
        for j, i in enumerate(self.intervals):
            f = self.functions[j]
            i_ = i.difference(interval)
            if not i_.isempty():
                if isinstance(i_, ContinuousSet):
                    i_ = RealSet(i_)
                intervals.append(i_.intervals[0].copy())
                functions.append(f)
            if i_ != i and not added_:
                intervals.append(interval)
                functions.append(func)
                added_ = True
            if isinstance(i_, RealSet) and len(i_.intervals) > 1:
                intervals.append(i_.intervals[1].copy())
                functions.append(f)
        self.intervals = list(intervals)
        self.functions = list(functions)

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

    def stretch(self, alpha):
        """
        Multiply every function by ``alpha`` inplace.
        :param alpha: The factor
        """
        for f in self.functions:
            if isinstance(f, ConstantFunction):
                f.value *= alpha
            elif isinstance(f, LinearFunction):
                f.c *= alpha
                f.m *= alpha

    cpdef str pfmt(self):
        """
        Pretty format of this function
        :return: pretty string
        """
        assert len(self.intervals) == len(self.functions), \
            ('Intervals: %s, Functions: %s' % (self.intervals, self.functions))
        return str('\n'.join([f'{str(i): <50} |--> {str(f)}' for i, f in zip(self.intervals, self.functions)]))

    cpdef Function differentiate(self):
        cdef PiecewiseFunction diff = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        for i, f in zip(self.intervals, self.functions):
            diff.intervals.append(i.copy())
            diff.functions.append(f.differentiate())
        return diff

    cpdef ensure_left(self, Function left, DTYPE_t x):
        cdef ContinuousSet xing = self.functions[0].intersection(left)
        if xing == R:  # With all points are crossing points, we are done
            return
        elif xing and xing.lower < self.intervals[0].upper and xing.upper >= x:
            self.intervals.insert(0, ContinuousSet(np.NINF, xing.lower, EXC, EXC))
            self.intervals[1].lower = xing.lower
            self.intervals[1].left = INC
            self.functions.insert(0, left)
        else:
            self.intervals.insert(0, ContinuousSet(np.NINF, x, EXC, EXC))
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
            self.intervals.append(ContinuousSet(xing.lower, np.PINF, INC, EXC))
            self.intervals[-2].upper = xing.upper
            self.intervals[-2].left = INC
            self.functions.append(right)
        else:
            self.intervals.append(ContinuousSet(x, np.PINF, INC, EXC))
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
            if interval.lower != np.NINF and interval.upper != np.PINF:
                interval.linspace(n, default_step=1, result=samples_x[i:i + n])
                i += n
        if len(self.intervals) >= 2:
            if self.intervals[0].lower != np.NINF:
                nopen -= 1
            if self.intervals[-1].upper != np.PINF:
                nopen -= 1
            if nopen:
                stepsize = ifnot(np.mean(np.abs(np.diff(samples_x[:i]))), 1)
        if self.intervals[0].lower == np.NINF:
            self.intervals[0].linspace(n, default_step=stepsize, result=samples_x[i:i + n])
            i += n
        if self.intervals[-1].upper == np.PINF:
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
                        result_set.intervals.append(ContinuousSet(np.NINF, x.upper, 2, 2))
                    else:
                        current.upper = x.upper
                        result_set.intervals.append(current)
                        current = None
                elif f.m < 0:
                    current = ContinuousSet(x.lower, np.PINF, 2, 2)
            elif not x and f(i.lower) < y and current is None:
                current = ContinuousSet(i.lower, np.PINF, 2, 2)
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
                        result_set.intervals.append(ContinuousSet(np.NINF, x.upper, 2, 2))
                    else:
                        current.upper = x.upper
                        result_set.intervals.append(current)
                        current = None
                elif f.m > 0:
                    current = ContinuousSet(x.lower, np.PINF, 2, 2)
            elif not x and f(i.lower) > y and current is None:
                current = ContinuousSet(i.lower, np.PINF, 1, 2)
            elif (not x and f(i.lower) < y or x == R and f(i.lower) == y) and current is not None:
                current.upper = i.lower
                result_set.intervals.append(current)
                current = None
        if current is not None:
            result_set.intervals.append(current)
        return result_set

    @staticmethod
    def merge(f1, f2, mergepoint):
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
            if i.lower != np.NINF:
                result.append((i.lower, f.eval(i.lower)))
        if self.intervals and self.intervals[-1].upper != np.PINF:
            result.append((self.intervals[-1].upper, self.functions[-1].eval(self.intervals[-1].upper)))
        elif not np.isnan(lastx):
            result.append((lastx, self.functions[-1].eval(lastx)))
        return result

    cpdef PiecewiseFunction add_knot(PiecewiseFunction self, DTYPE_t x, DTYPE_t y):
        ...

    cpdef PiecewiseFunction crop(PiecewiseFunction self, ContinuousSet interval):
        cdef PiecewiseFunction result = PiecewiseFunction()
        for i, f in zip(self.intervals, self.functions):
            if i.intersects(interval):
                intersection = i.intersection(interval, left=INC, right=EXC)
                result.intervals.append(intersection)
                result.functions.append(f.copy())
        return result

    def to_json(self):
        return {'intervals': [i.to_json() for i in self.intervals],
                'functions': [f.to_json() for f in self.functions]}

    @staticmethod
    def from_json(data):
        function = PiecewiseFunction()
        function.intervals = [ContinuousSet.from_json(d) for d in data['intervals']]
        function.functions = [LinearFunction.from_json(d) for d in data['functions']]
        return function

    def __repr__(self):
        return self.pfmt()

    def round(self, digits=None, include_intervals=True):
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
