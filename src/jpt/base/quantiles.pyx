# cython: auto_cpdef=True,
# cython: infer_types=True,
# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
import itertools
import numbers
import random
import re
from collections import deque
from operator import itemgetter, attrgetter

from dnutils import ifnot, out, first, stop, ifnone
# from pyearth import Earth
# from pyearth._basis import ConstantBasisFunction, HingeBasisFunctionBase, LinearBasisFunction, HingeBasisFunction
from scipy import stats
from scipy.stats import norm

from .intervals cimport ContinuousSet, RealSet, _INC, _EXC
from .intervals import R, EMPTY, EXC, INC

import numpy as np
cimport numpy as np
cimport cython

from numpy cimport float64_t

import warnings

from .utils import pairwise, normalized
from .cutils cimport SIZE_t, DTYPE_t, sort

from ..learning.cdfreg import CDFRegressor

warnings.filterwarnings("ignore")


cpdef DTYPE_t ifnan(DTYPE_t if_, DTYPE_t else_, transform=None):
    '''
    Returns the condition ``if_`` iff it is not ``Nan``, or if a transformation is
    specified, ``transform(if_)``. Returns ``else_`` if the condition is ``NaN``.
    ``transform`` can be any callable, which will be passed ``if_`` in case ``if_`` is not ``NaN``.
    '''
    if np.isnan(if_):
        return else_
    else:
        if transform is not None:
            return transform(if_)
        else:
            return if_


cdef inline np.int32_t equal(DTYPE_t x1, DTYPE_t x2, DTYPE_t tol=1e-7):
    return abs(x1 - x2) < tol


cpdef DTYPE_t[::1] linspace(DTYPE_t start, DTYPE_t stop, np.int64_t num):
    '''
    Modification of the ``numpy.linspace`` function to return an array of ``num``
    equally spaced samples in the range of ``start`` and ``stop`` (both inclusive).

    In contrast to the original numpy function, this variant return the centroid of
    ``start`` and ``stop`` in the case where ``num`` is ``1``.

    :param start:
    :param stop:
    :param num:
    :return:
    '''
    cdef DTYPE_t[::1] samples = np.ndarray(shape=num, dtype=np.float64)
    cdef DTYPE_t n
    cdef DTYPE_t space, val = start
    cdef np.int64_t i
    if num == 1:
        samples[0] = (stop - start) / 2
    else:
        n = <DTYPE_t> num - 1
        space = (stop - start) / n
        for i in range(num):
            samples[i] = val
            val += space
    return samples


cdef class ConfInterval:
    '''Represents a prediction interval with a mean, lower und upper bound'''

    def __init__(self, mean, lower, upper):
        self.mean = mean
        self.lower = lower
        self.upper = upper

    def __str__(self):
        return '|<-- %s -- %s -- %s -->|' % (self.lower, self.mean, self.upper)

    def __repr__(self):
        return str(self)

    cpdef tuple totuple(ConfInterval self):
        return self.lower, self.mean, self.upper

    cpdef DTYPE_t[::1] tomemview(ConfInterval self, DTYPE_t[::1] result=None):
        if result is None:
            result = np.ndarray(shape=3, dtype=np.float64)
        result[0] = self.lower
        result[1] = self.mean
        result[2] = self.upper


@cython.freelist(1000)
cdef class Function:
    '''
    Abstract base type of functions.
    '''

    def __init__(self):
        pass

    def __call__(self, x):
        return self.eval(x)


@cython.final
cdef class Undefined(Function):
    '''
    This class represents an undefined function.
    '''

    cpdef inline DTYPE_t eval(Undefined self, DTYPE_t x):
        return np.nan

    def __str__(self):
        return 'undef.'

    def __repr__(self):
        return '<Undefined>'

    def __eq__(self, other):
        if isinstance(other, Undefined):
            return True
        return False


cdef class KnotFunction(Function):
    '''
    Abstract superclass of all knot functions.
    '''

    def __init__(KnotFunction self, DTYPE_t knot, DTYPE_t weight):
        self.knot = knot
        self.weight = weight


@cython.final
cdef class Hinge(KnotFunction):
    '''
    Implementation of hinge functions as used in MARS regression.

    alpha = 1:  hinge is zero to the right of knot
    alpha = -1: hinge is zero to the left of knot
    '''

    def __init__(Hinge self, DTYPE_t knot, np.int32_t alpha, DTYPE_t weight):
        super(Hinge, self).__init__(knot, weight)
        assert alpha in (1, -1), 'alpha must be in {1,-1}'
        self.alpha = alpha

    cpdef inline DTYPE_t eval(Hinge self, DTYPE_t x):
        return max(0, (self.knot - x) if self.alpha == 1 else (x - self.knot)) * self.weight

    def __str__(self):
        return '%.3f * max(0, %s)' % (self.weight, ('x - %s' % self.knot) if self.alpha == 1 else ('%s - x' % self.knot))

    def __repr__(self):
        return '<Hinge k=%.3f a=%d w=%.3f>' % (self.knot, self.alpha, self.weight)

    cpdef Function differentiate(Hinge self):
        return Jump(self.knot, self.alpha, -self.weight if self.alpha == 1 else self.weight)

    cpdef np.int32_t is_invertible(Hinge self):
        return False


@cython.final
cdef class Jump(KnotFunction):
    '''
    Implementation of jump functions.
    '''

    def __init__(Jump self, DTYPE_t knot, np.int32_t alpha, DTYPE_t weight):
        super(Jump, self).__init__(knot, weight)
        assert alpha in (1, -1), 'alpha must be in {1,-1}'
        self.alpha = alpha

    cpdef inline DTYPE_t eval(Jump self, DTYPE_t x):
        return max(0, (-1 if ((self.knot - x) if self.alpha == 1 else (x - self.knot)) < 0 else 1)) * self.weight

    def __str__(self):
        return '%.3f * max(0, sgn(%s))' % (self.weight, ('x - %s' % self.knot) if self.alpha == 1 else ('%s - x' % self.knot))

    def __repr__(self):
        return '<Jump k=%.3f a=%d w=%.3f>' % (self.knot, self.alpha, self.weight)

    @staticmethod
    def from_point(p1, alpha):
        x, y = p1
        return Jump(x, alpha, y)

    cpdef inline Function differentiate(Jump self):
        return Impulse(self.knot, self.weight)


@cython.final
cdef class Impulse(KnotFunction):
    '''
    Represents a function that is non-zero at exactly one x-position and zero at all other positions.
    '''

    def __init__(Impulse self, DTYPE_t knot, DTYPE_t weight):
        super(Impulse, self).__init__(knot, weight)

    cpdef DTYPE_t eval(self, DTYPE_t x):
        return self.weight if x == self.knot else 0

    cpdef inline Function differentiate(Impulse self):
        return Impulse(self.knot, np.nan)

    cpdef inline np.int32_t is_invertible(Impulse self):
        return False

    def __repr__(self):
        return '<Impulse k=%.3f w=%.3f>' % (self.knot, self.weight)

    def __str__(self):
        return '%.3f if x=%.3f else 0' % (self.weight, self.knot)


@cython.final
cdef class ConstantFunction(Function):
    '''
    Represents a constant function.
    '''

    def __init__(ConstantFunction self, DTYPE_t value):
        self.value = value

    def __call__(self, x=None):
        return self.eval(x)

    def __str__(self):
        return '%s = const.' % self.value

    def __repr__(self):
        return 'const=%s' % self.value

    cpdef inline DTYPE_t eval(ConstantFunction self, DTYPE_t x):
        return self.value

    cpdef inline ConstantFunction differentiate(ConstantFunction self):
        return ConstantFunction(0)

    cpdef inline np.int32_t is_invertible(ConstantFunction self):
        return False

    cpdef inline ConstantFunction copy(ConstantFunction self):
        return ConstantFunction(self.value)

    def __eq__(self, other):
        if not isinstance(other, Function):
            raise TypeError('Cannot compare object of type %s to %s.' % (type(self).__name__, type(other).__name__))
        if isinstance(other, (LinearFunction, ConstantFunction)):
            return self.m == other.m and self.c == other.c
        return False

    @property
    def m(self):
        return 0

    @property
    def c(self):
        return self.value

    cpdef np.int32_t crosses(ConstantFunction self, Function f) except +:
        '''
        Determine if the function crosses another linear function ``f``.
        :param f: 
        :return: 
        '''
        if isinstance(f, LinearFunction):
            return f.crosses(self)
        elif isinstance(f, ConstantFunction):
            return f.value == self.value
        else:
            raise TypeError('Argument must be of type LinearFunction or ConstantFunction, not %s' % type(f).__name__)

    cpdef ContinuousSet xing_point(ConstantFunction self, Function f) except +:
        '''
        Determine if the function crosses another linear function ``f``.
        :param f: 
        :return: 
        '''
        if isinstance(f, LinearFunction):
            return f.crosses_at(self)
        elif isinstance(f, ConstantFunction):
            return R.copy() if self.value == f.value else EMPTY.copy()
        else:
            raise TypeError('Argument must be of type LinearFunction or ConstantFunction, not %s' % type(f).__name__)

    def to_json(self):
        return {'type': 'constant', 'value': self.value}


@cython.final
cdef class LinearFunction(Function):
    '''
    Implementation of univariate linear functions.
    '''

    def __init__(LinearFunction self, DTYPE_t m, DTYPE_t c):
        self.m = m
        self.c = c

    def __call__(self, DTYPE_t x):
        return self.eval(x)

    cpdef inline DTYPE_t eval(LinearFunction self, DTYPE_t x):
        return self.m * x + self.c

    def __str__(self):
        l = (str(self.m) + 'x') if self.m else ''
        op = '' if (not l and self.c > 0 or not self.c) else ('+' if self.c > 0 else '-')
        c = '0' if (not self.c and not self.m) else ('' if not self.c else str(abs(self.c)))
        return ('%s %s %s' % (l, op, c)).strip()

    def __repr__(self):
        return '<%s>' % str(self)

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
            linear = float(match[0].replace(' ', ''))
            const = float(match[1].replace(' ', '')) if match[1] else 0
            if not linear:
                return ConstantFunction(const)
            return LinearFunction(linear, const)
        else:
            return ConstantFunction(float(match[0].replace(' ', '')))


    cpdef DTYPE_t root(LinearFunction self) except +:
        '''
        Find the root of the function, i.e. the ``x`` positions subject to ``self.eval(x) = 0``.
        :return: 
        '''
        return -self.c / self.m

    cpdef LinearFunction invert(LinearFunction self) except +:
        '''
        Return the inverted linear function of this LF.
        :return: 
        '''
        return LinearFunction(1 / self.m, -self.c / self.m)

    cpdef LinearFunction hmirror(LinearFunction self):
        '''
        Return the linear function that is obtained by horizonally mirroring this LF.
        :return: 
        '''
        return LinearFunction(-self.m, -self.c)

    cpdef LinearFunction copy(LinearFunction self):
        return LinearFunction(self.m, self.c)

    cpdef np.int32_t crosses(LinearFunction self, Function f) except +:
        '''
        Determine if the function crosses another linear function ``f``.
        :param f: 
        :return: 
        '''
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
            raise TypeError('Argument must be of type LinearFunction or ConstantFunction, not %s' % type(f).__name__)

    cpdef ContinuousSet xing_point(LinearFunction self, Function f) except +:
        '''
        Determine if the function crosses another linear function ``f``.
        :param f: 
        :return: 
        '''
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
            raise TypeError('Argument must be of type LinearFunction or ConstantFunction, not %s' % type(f).__name__)

    def __add__(self, x):
        if isinstance(x, LinearFunction):
            return LinearFunction(self.m + x.m, self.c + x.c)
        elif isinstance(x, (int, float)):
            return LinearFunction(self.m, self.c + x)
        else:
            raise TypeError('Operator "+" undefined for types %s and %s' % (type(x).__name__, type(self).__name__))

    def __sub__(self, x):
        return -x + self

    def __radd__(self, x):
        return self + x

    def __rsub__(self, x):
        return self - x

    def __eq__(self, other):
        if isinstance(other, (ConstantFunction, LinearFunction)):
            return self.m == other.m and self.c == other.c
        elif isinstance(other, Function):
            return False
        else:
            raise TypeError('Can only compare objects of type "Function", but got type "%s".' % type(other).__name__)

    cpdef inline Function differentiate(LinearFunction self):
        return ConstantFunction(self.m)

    cpdef inline Function simplify(LinearFunction self):
        if self.m == 0:
            return ConstantFunction(self.c)
        else:
            return self.copy()

    @staticmethod
    def from_points((DTYPE_t, DTYPE_t) p1, (DTYPE_t, DTYPE_t) p2):
        cdef DTYPE_t x1 = p1[0], y1 = p1[1]
        cdef DTYPE_t x2 = p2[0], y2 = p2[1]
        if x1 == x2:
            raise ValueError('Points must have different coordinates to fit a line: p1=%s, p2=%s' % (p1, p2))
        if any(np.isnan(p) for p in itertools.chain(p1, p2)):
            raise ValueError('Arguments %s, %s are invalid.' % (p1, p2))
        if y2 == y1:
            return ConstantFunction(y2)
        cdef DTYPE_t m = (y2 - y1) / (x2 - x1)
        cdef DTYPE_t c = y1 - m * x1
        assert not np.isnan(m) and not np.isnan(c), \
            'Fitting linear function from %s to %s resulted in m=%s, c=%s' % (p1, p2, m, c)
        return LinearFunction(m, c)

    cpdef inline np.int32_t is_invertible(LinearFunction self):
        return abs(self.m) >= 1e-4

    cpdef inline LinearFunction fit(LinearFunction self, DTYPE_t[::1] x, DTYPE_t[::1] y) except +:
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


@cython.final
cdef class GaussianCDF(Function):

    cdef readonly DTYPE_t mu, sigma

    def __init__(GaussianCDF self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef inline double eval(GaussianCDF self, DTYPE_t x):
        return norm.cdf(x, loc=self.mu, scale=self.sigma)


@cython.final
cdef class GaussianPDF(Function):

    cdef readonly DTYPE_t mu, sigma

    def __init__(GaussianPDF self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef inline double eval(GaussianPDF self, DTYPE_t x):
        return norm.pdf(x, loc=self.mu, scale=self.sigma)

    def __str__(self):
        return '<GaussianPDF mu=%s sigma=%s>' % (self.mu, self.sigma)


@cython.final
cdef class GaussianPPF(Function):

    cdef readonly DTYPE_t mu, sigma

    def __init__(GaussianPPF self, DTYPE_t mu, DTYPE_t sigma):
        self.mu = mu
        self.sigma = sigma

    cpdef inline double eval(GaussianPPF self, DTYPE_t x):
        return norm.ppf(x, loc=self.mu, scale=self.sigma)


cdef np.int32_t _FLOAT = 1
cdef np.int32_t _INT   = 2
cdef np.int32_t _GAUSSIAN = 3
cdef np.int32_t _UNIFORM = 4

FLOAT = np.int32(_FLOAT)
INT = np.int32(_INT)
GAUSSIAN = np.int32(_GAUSSIAN)
UNIFORM = np.int32(_UNIFORM)


cdef class QuantileDistribution:
    '''
    Abstract base class for any quantile-parameterized cumulative data distribution.
    '''

    cdef DTYPE_t epsilon
    cdef DTYPE_t penalty
    cdef np.int32_t verbose
    cdef np.int32_t min_samples_mars
    cdef PiecewiseFunction _cdf, _pdf, _ppf

    def __init__(self, epsilon=.01, penalty=3., min_samples_mars=5, verbose=False):
        self.epsilon = epsilon
        self.penalty = penalty
        self.verbose = verbose
        self.min_samples_mars = min_samples_mars
        self._cdf = None
        self._pdf = None
        self._ppf = None

    @staticmethod
    def from_cdf(cdf):
        d = QuantileDistribution()
        d._cdf = cdf
        return d

    cpdef QuantileDistribution fit(self, DTYPE_t[:, ::1] data, SIZE_t[::1] rows, SIZE_t col):
        if rows is None:
            rows = np.arange(data.shape[0], dtype=np.int64)

        cdef SIZE_t i, n_samples = rows.shape[0]
        cdef DTYPE_t[:, ::1] data_buffer = np.ndarray(shape=(2, n_samples), dtype=np.float64, order='C')
        for i in range(n_samples):
            data_buffer[0, i] = data[rows[i], col]
        np.asarray(data_buffer[0, :]).sort()

        cdef SIZE_t count = 0,
        i = 0
        for i in range(n_samples):
            if i > 0 and data_buffer[0, i] == data_buffer[0, i - 1]:
                data_buffer[1, count - 1] += 1
            else:
                data_buffer[0, count] = data_buffer[0, i]
                data_buffer[1, count] = <DTYPE_t> i + 1
                count += 1
        for i in range(count):
            data_buffer[1, i] -= 1
            data_buffer[1, i] /= <DTYPE_t> (n_samples - 1)
        data_buffer = np.ascontiguousarray(data_buffer[:, :count])
        cdef DTYPE_t[::1] x, y
        n_samples = count

        self._ppf = self._pdf = None
        # Use simple linear regression when fewer than min_samples_mars points are available
        # if 1 < n_samples < self.min_samples_mars:
        #     x = data_buffer[0, :]
        #     y = data_buffer[1, :]
        #     self._cdf = PiecewiseFunction()
        #     self._cdf.intervals.append(R.copy())
        #     self._cdf.functions.append(LinearFunction(0, 0).fit(x, y))
        #     self._cdf.ensure_left(ConstantFunction(0), x[0])
        #     self._cdf.ensure_right(ConstantFunction(1), x[-1])
        #
        # elif self.min_samples_mars <= n_samples:
        alert = False
        if n_samples > 1:
            regressor = CDFRegressor(eps=self.epsilon)
            regressor.fit(data_buffer)
            self._cdf = PiecewiseFunction()
            self._cdf.functions.append(ConstantFunction(0))
            self._cdf.intervals.append(ContinuousSet(np.NINF, np.PINF, EXC, EXC))
            for left, right in pairwise(regressor.support_points):
                self._cdf.functions.append(LinearFunction.from_points(tuple(left), tuple(right)))
                if self._cdf.functions[-1].m < 1e-3:
                    alert = True
                self._cdf.intervals[-1].upper = left[0]
                self._cdf.intervals.append(ContinuousSet(left[0], right[0], 1, 2))
            self._cdf.functions.append(ConstantFunction(1))
            self._cdf.intervals.append(ContinuousSet(self._cdf.intervals[-1].upper, np.PINF, INC, EXC))
            if alert and len(self._cdf.intervals) == 3:
                raise ValueError(self._cdf.pfmt() + '\n' + str(np.asarray(data_buffer)))
        else:
            x = data_buffer[0, :]
            y = data_buffer[1, :]
            self._cdf = PiecewiseFunction()
            self._cdf.intervals.append(ContinuousSet(np.NINF, x[0], EXC, EXC))
            self._cdf.functions.append(ConstantFunction(0))
            self._cdf.intervals.append(ContinuousSet(x[0], np.PINF, INC, EXC))
            self._cdf.functions.append(ConstantFunction(1))

        return self

    cpdef crop(self, ContinuousSet interval):
        '''
        Return a copy this quantile distribution that is cropped to the ``interval``.
        '''
        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        cdf_ = self.cdf.crop(interval)
        cdf_.add_const(-cdf_.eval(cdf_.intervals[0].lowermost()))

        cdf = PiecewiseFunction()
        cdf.intervals.append(ContinuousSet(np.NINF, cdf_.intervals[0].lower, EXC, EXC))
        cdf.functions.append(ConstantFunction(0.))

        alpha = cdf_.functions[-1].eval(interval.uppermost())

        f_ = cdf_.functions[0]
        for i, f in zip(cdf_.intervals, cdf_.functions):
            if f == ConstantFunction(0.):
                cdf.intervals[-1].upper = i.lower
                continue
            y = cdf.functions[-1].eval(i.lower)
            c = (f.eval(i.lower) - f_.eval(i.lower)) * alpha  # If the function is continuous (no jump), c = 0

            cdf.intervals.append(ContinuousSet(i.lower, i.upper, INC, EXC))
            upper_ = np.nextafter(i.upper, i.upper - 1)

            if isinstance(f, ConstantFunction) or i.size() == 1:
                cdf.functions.append(ConstantFunction(c))
            else:
                cdf.intervals[-1].lower = cdf.intervals[-1].lower = cdf.intervals[-2].upper
                cdf.functions.append(LinearFunction.from_points((i.lower, y + c),
                                                                (upper_, (f.m / alpha) * (upper_ - i.lower) + y + c)))
        if cdf.functions[-1] == ConstantFunction(1.):
            cdf.intervals[-1].upper = np.PINF
            cdf.intervals[-1].right = EXC
            if len(cdf.intervals) > 1:
                cdf.intervals[-1].lower = cdf.intervals[-2].upper
        else:
            if interval.uppermost() in cdf.intervals[-1]:
                cdf.intervals[-1].upper = np.nextafter(cdf.intervals[-1].upper, cdf.intervals[-1].upper - 1)

            cdf.functions.append(ConstantFunction(1.))
            cdf.intervals.append(ContinuousSet(cdf.intervals[-1].upper, np.PINF, INC, EXC))

        # Clean the function segments that might have become empty
        for idx, i in enumerate(cdf.intervals):
            if i.isempty():
                del cdf.intervals[idx]
                del cdf.functions[idx]

        result = QuantileDistribution(self.epsilon, self.penalty, min_samples_mars=self.min_samples_mars)
        result._cdf = cdf
        return result

    @property
    def cdf(self):
        return self._cdf

    @property
    def pdf(self):
        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        elif self._pdf is None:
            pdf = self._cdf.differentiate()
            if len(self._cdf.intervals) == 2:
                pdf.intervals.insert(1, ContinuousSet(pdf.intervals[0].upper,
                                                      np.nextafter(pdf.intervals[0].upper,
                                                                   pdf.intervals[0].upper + 1), INC, EXC))
                pdf.intervals[-1].lower = np.nextafter(pdf.intervals[-1].lower,
                                                       pdf.intervals[-1].lower + 1)
                pdf.functions.insert(1, ConstantFunction(np.PINF))
            # if simplify:
            #     pdf = pdf.simplify(samples, epsilon=ifnan(epsilon, self.epsilon), penalty=ifnan(penalty, self.penalty))
            self._pdf = pdf
        return self._pdf

    @property
    def ppf(self):
        cdef PiecewiseFunction ppf
        cdef ContinuousSet interval
        cdef Function f
        cdef DTYPE_t one_plus_eps = np.nextafter(1, 2)

        if self._cdf is None:
            raise RuntimeError('No quantile distribution fitted. Call fit() first.')

        elif self._ppf is None:
            ppf = PiecewiseFunction()

            ppf.intervals.append(ContinuousSet(np.NINF, np.PINF, EXC, EXC))

            assert len(self._cdf.functions) > 1, self._cdf.pfmt()

            if len(self._cdf.functions) == 2:
                ppf.intervals[-1].upper = 1
                ppf.functions.append(Undefined())
                ppf.intervals.append(ContinuousSet(1, one_plus_eps, INC, EXC))
                ppf.functions.append(ConstantFunction(self._cdf.intervals[-1].lower))
                ppf.intervals.append(ContinuousSet(one_plus_eps, np.PINF, INC, EXC))
                ppf.functions.append(Undefined())
                self._ppf = ppf
                return ppf

            y_cdf = 0
            for interval, f in zip(self._cdf.intervals, self._cdf.functions):

                if f.is_invertible():
                    ppf.functions.append(f.invert())
                    const = 0
                elif not f.c:
                    ppf.functions.append(Undefined())
                elif f(interval.lower) > y_cdf:
                    ppf.functions.append(ConstantFunction(interval.lower))
                else:
                    continue

                if not np.isinf(interval.upper):
                    ppf.intervals[-1].upper = min(one_plus_eps, f(interval.upper))
                else:
                    ppf.intervals[-1].upper = one_plus_eps

                ppf.intervals.append(ContinuousSet(ppf.intervals[-1].upper, np.PINF, INC, EXC))
                y_cdf = min(one_plus_eps, f(interval.upper))

            ppf.intervals[-2].upper = ppf.intervals[-1].lower = one_plus_eps
            ppf.functions.append(Undefined())

            if not np.isnan(ppf.eval(one_plus_eps)):
                print(ppf.functions[ppf.idx_at(one_plus_eps)],
                      ppf.functions[ppf.idx_at(one_plus_eps)].eval(one_plus_eps),
                      ppf.eval(one_plus_eps))
                raise ValueError('ppf(%.16f) = %.20f [fct: %d]:\n %s\n===cdf\n%s' % (one_plus_eps,
                                                                                     ppf.eval(one_plus_eps),
                                                                                     ppf.idx_at(one_plus_eps),
                                                                                     ppf.pfmt(),
                                                                                     self._cdf.pfmt()))

            if np.isnan(ppf.eval(1.)):
                raise ValueError(str(one_plus_eps) +
                                 'ppf:\n %s\n===cdf\n%s\nval' % (ppf.pfmt(),
                                                                 self._cdf.pfmt() + str(ppf.intervals[-1].lower) + ' ' + str(ppf.intervals[-2].upper) + ' ' + str(ppf.eval(1.)) + ' ' + str(ppf.eval(one_plus_eps))))

            self._ppf = ppf
        return self._ppf

    @staticmethod
    def merge(distributions, weights=None):
        '''
        Construct a merged quantile-distribution from the passed distributions using the ``weights``.
        '''
        intervals = [ContinuousSet(np.NINF, np.PINF, EXC, EXC)]
        functions = [ConstantFunction(0)]
        if weights is None:
            weights = [1. / len(distributions)] * len(distributions)

        # --------------------------------------------------------------------------------------------------------------
        # We preprocess the CDFs that are in the form of "jump" functions
        jumps = {}
        for w, cdf in [(w, d.cdf) for w, d in zip(weights, distributions) if len(d.cdf) == 2]:
            jumps[cdf.intervals[0].upper] = jumps.get(cdf.intervals[0].upper, Jump(cdf.intervals[0].upper, 1, 0))
            jumps.get(cdf.intervals[0].upper).weight += w

        # --------------------------------------------------------------------------------------------------------------
        lower = sorted([(i.lower, f, w)
                        for d, w in zip(distributions, weights)
                        for i, f in zip(d.cdf.intervals, d.cdf.functions) if not isinstance(f, ConstantFunction)]
                       + [(j.knot, j, j.weight)
                          for j in jumps.values()],
                       key=itemgetter(0))
        upper = sorted([(i.upper, f, w)
                        for d, w in zip(distributions, weights)
                        for i, f in zip(d.cdf.intervals, d.cdf.functions) if not isinstance(f, ConstantFunction)],
                       key=itemgetter(0))

        # --------------------------------------------------------------------------------------------------------------
        m = 0
        c = 0

        while lower or upper:
            pivot = None
            m_ = m
            offset = 0

            # Process all function intervals whose lower bound is minimal and
            # smaller than the smallest upper interval bound
            while lower and (pivot is None and first(lower, first) <= first(upper, first, np.PINF) or
                   pivot == first(lower, first, np.PINF)):
                l, f, w = lower.pop(0)
                if isinstance(f, ConstantFunction) or l == np.NINF or isinstance(f, LinearFunction) and f.m == 0:
                    continue
                if isinstance(f, Jump):  # and isinstance(functions[-1], LinearFunction):
                    offset += w
                if isinstance(f, LinearFunction):
                    m_ += f.m * w
                pivot = l

            # Do the same for the upper bounds...
            while upper and (pivot is None and first(upper, first) <= first(lower, first, np.PINF) or
                   pivot == first(upper, first, np.PINF)):
                u, f, w = upper.pop(0)
                if isinstance(f, (ConstantFunction, Jump)) or u == np.PINF or isinstance(f, LinearFunction) and f.m == 0:
                    continue
                m_ -= f.m * w
                pivot = u

            if pivot is None:
                continue

            y = m * pivot + c
            m = m_ if abs(m_) > 1e-8 else 0
            c = y - m * pivot + offset

            intervals[-1].upper = pivot
            if (c or m) and (m != functions[-1].m or c != functions[-1].c):
                # Split the last interval at the pivot point
                intervals.append(ContinuousSet(pivot, np.PINF, INC, EXC))
                # Evaluate the old function at the new pivot point to get the intercept
                functions.append(LinearFunction(m, c) if abs(m) > 1e-8 else ConstantFunction(c))

        # If the merging ends with an "approximate" constant function
        # remove it. This may happen for numerical imprecision.
        while len(functions) > 1 and abs(functions[-1].m) <= 1e-08 and functions[-1].m:
            del intervals[-1]
            del functions[-1]

        cdf = PiecewiseFunction()
        cdf.functions = functions
        cdf.intervals = intervals

        cdf.ensure_right(ConstantFunction(1), cdf.intervals[-1].lower)

        distribution = QuantileDistribution()
        distribution._cdf = cdf

        # if len(cdf.functions) == 3 and cdf.functions[1].m < 1e-4:
        #     raise ValueError(cdf.pfmt() + '\n\n' + '\n---\n'.join([d.cdf.pfmt()
        #                                                            for d in distributions]) + '\n' + str(jumps))

        return distribution

    def to_json(self):
        return {'epsilon': self.epsilon,
                'penalty': self.penalty,
                'min_samples_mars': self.min_samples_mars,
                'cdf': self._cdf.to_json()}

    @staticmethod
    def from_json(data):
        q = QuantileDistribution(epsilon=data['epsilon'],
                                 penalty=data['penalty'],
                                 min_samples_mars=data['min_samples_mars'])
        q._cdf = PiecewiseFunction.from_json(data['cdf'])
        return q


@cython.final
cdef class PiecewiseFunction(Function):
    '''
    Represents a function that is piece-wise defined by constant values.
    '''

    def __init__(PiecewiseFunction self):
        self.functions = []
        self.intervals = []

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

    cpdef inline DTYPE_t eval(PiecewiseFunction self, DTYPE_t x):
        return self.at(x).eval(x)

    cpdef inline DTYPE_t[::1] multi_eval(PiecewiseFunction self, DTYPE_t[::1] x, DTYPE_t[::1] result=None):
        if result is None:
            result = np.ndarray(shape=len(x), dtype=np.float64)
        cdef int i
        for i in range(len(x)):
            result[i] = self.eval(x[i])
        return result

    cpdef inline Function at(PiecewiseFunction self, DTYPE_t x):
        '''
        Return the linear function segment at position ``x``.
        '''
        cdef idx = self.idx_at(x)
        if idx != -1:
            return self.functions[idx]
        return None

    cpdef inline int idx_at(PiecewiseFunction self, DTYPE_t x):
        '''
        Return the index of the function segment at position ``x``.
        '''
        cdef int i
        cdef ContinuousSet interval
        for i, interval in enumerate(self.intervals):
            if x in interval:
                return i
        return -1

    def __len__(self):
        return len(self.intervals)

    cpdef inline ContinuousSet interval_at(PiecewiseFunction self, DTYPE_t x):
        cdef int i
        cdef ContinuousSet interval
        for i, interval in enumerate(self.intervals):
            if x in interval:
                return interval
        else:
            return EMPTY

    def __call__(self, x):
        return self.eval(x)

    cpdef PiecewiseFunction copy(PiecewiseFunction self):
        cdef PiecewiseFunction result = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        result.functions = [f.copy() for f in self.functions]
        result.intervals = [i.copy() for i in self.intervals]
        return result

    def add_function(self, interval, func):
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
                intervals.append(i_.intervals[0])
                functions.append(f)
            if i_ != i and not added_:
                intervals.append(interval)
                functions.append(func)
                added_ = True
            if isinstance(i_, RealSet) and len(i_.intervals) > 1:
                intervals.append(i_.intervals[1])
                functions.append(f)
        self.intervals = list(intervals)
        self.functions = list(functions)

    cpdef tuple split(PiecewiseFunction self, DTYPE_t splitpoint):
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

    def add_const(self, c):
        for f in self.functions:
            if isinstance(f, ConstantFunction):
                f.value += c
            elif isinstance(f, LinearFunction):
                f.c += c

    def stretch(self, alpha):
        for f in self.functions:
            if isinstance(f, ConstantFunction):
                f.value *= alpha
            elif isinstance(f, LinearFunction):
                f.c *= alpha
                f.m *= alpha

    cpdef inline str pfmt(PiecewiseFunction self):
        assert len(self.intervals) == len(self.functions), \
            ('Intervals: %s, Functions: %s' % (self.intervals, self.functions))
        return str('\n'.join([f'{str(i): <50} |--> {str(f)}' for i, f in zip(self.intervals, self.functions)]))

    cpdef PiecewiseFunction differentiate(PiecewiseFunction self):
        cdef PiecewiseFunction diff = PiecewiseFunction()
        cdef ContinuousSet i
        cdef Function f
        for i, f in zip(self.intervals, self.functions):
            diff.intervals.append(i)
            diff.functions.append(f.differentiate())
        return diff

    cpdef ensure_left(PiecewiseFunction self, Function left, DTYPE_t x):
        cdef ContinuousSet xing = self.functions[0].xing_point(left)
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

    cpdef ensure_right(PiecewiseFunction self, Function right, DTYPE_t x):
        cdef ContinuousSet xing = self.functions[-1].xing_point(right)
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

    cpdef DTYPE_t[::1] xsamples(PiecewiseFunction self, np.int32_t sort=True):
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

    # cpdef PiecewiseFunction simplify(PiecewiseFunction self, np.int32_t n_samples=1, DTYPE_t epsilon=.001,
    #                                  DTYPE_t penalty=3.):
    #     cdef np.int32_t samples_total = len(self.intervals) * n_samples
    #     cdef DTYPE_t[::1] samples_x = np.ndarray(shape=samples_total, dtype=np.float64)
    #     samples_x[:] = 0
    #     cdef object f
    #     cdef np.int32_t i = 0
    #     cdef DTYPE_t stepsize = 1
    #     cdef DTYPE_t max_int = max([i_.upper - i_.lower for i_ in self.intervals if i_.lower != np.NINF and i_.upper != np.PINF])
    #     cdef ContinuousSet interval
    #     cdef np.int32_t nopen = 2
    #     for interval, f in zip(self.intervals, self.functions):
    #         if interval.lower != np.NINF and interval.upper != np.PINF:
    #             samples = int(max(1, np.ceil(n_samples * (interval.upper - interval.lower) / max_int)))
    #             interval.linspace(samples, default_step=1, result=samples_x[i:i + samples])
    #             i += samples
    #     if len(self.intervals) > 2:
    #         if self.intervals[0].lower != np.NINF:
    #             nopen -= 1
    #         if self.intervals[-1].upper != np.PINF:
    #             nopen -= 1
    #         if nopen:
    #             stepsize = np.mean(np.abs(np.diff(samples_x[:i])))
    #     if self.intervals[0].lower == np.NINF:
    #         self.intervals[0].linspace(n_samples, default_step=stepsize, result=samples_x[i:i + n_samples])
    #         i += n_samples
    #     if self.intervals[-1].upper == np.PINF:
    #         self.intervals[-1].linspace(n_samples, default_step=stepsize, result=samples_x[i:i + n_samples])
    #         i += n_samples
    #     cdef DTYPE_t[::1] learn_data = samples_x[:i]
    #     cdef DTYPE_t[::1] samples_y = self.multi_eval(learn_data)
    #     return fit_piecewise(learn_data, samples_y, penalty=penalty, epsilon=epsilon)

    cpdef RealSet eq(PiecewiseFunction self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        prev_f = None
        for i, f in zip(self.intervals, self.functions):
            x = f.xing_point(y_)
            if x and x in i:
                result_set.intervals.append(i.intersection(x))
            elif prev_f and (prev_f(i.lower) < y < f(i.lower) or prev_f(i.lower) > y > f(i.lower)):
                result_set.intervals.append(ContinuousSet(i.lower, i.lower))
            prev_f = f
        return result_set

    cpdef RealSet lt(PiecewiseFunction self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        prev_f = None
        current = None
        for i, f in zip(self.intervals, self.functions):
            x = f.xing_point(y_)
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

    cpdef RealSet gt(PiecewiseFunction self, DTYPE_t y):
        result_set = RealSet()
        y_ = ConstantFunction(y)
        current = None
        for i, f in zip(self.intervals, self.functions):
            x = f.xing_point(y_)
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
        '''
        Merge two piecewise linear functions ``f1`` and ``f2`` at the merge point ``mergepoint``.

        The result will be a CDF-conform function
        :param f1:
        :param f2:
        :param mergepoint:
        :return:
        '''
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
                result.functions.append(f)
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
        '''
        Return a copy of this PLF, in which all parameters of sub-functions have been rounded by
        the specified number of digits.

        If ``include_intervals`` is ``False``, the parameter values of the intervals will not be affected by
        this operation.
        '''
        digits = ifnone(digits, 3)
        round_ = lambda x: round(x, ndigits=digits)

        plf = PiecewiseFunction()
        for interval, function in zip(self.intervals, self.functions):
            if include_intervals:
                interval = interval.copy()
                interval.lower = round_(interval.lower)
                interval.upper = round_(interval.upper)
            plf.intervals.append(interval)
            if isinstance(function, LinearFunction):
                function = LinearFunction(round_(function.m), round_(function.c))
            elif isinstance(function, ConstantFunction):
                function = ConstantFunction(round_(function.value))
            else:
                raise TypeError('Unknown function type in PiecewiseFunction: "%s"' % type(function).__name__)
            plf.functions.append(function)
        return plf

# cpdef object fit_piecewise(DTYPE_t[::1] x, DTYPE_t[::1] y, DTYPE_t epsilon=np.nan,
#                            DTYPE_t penalty=np.nan, DTYPE_t[::1] weights=None,
#                            np.int32_t verbose=False):
#     cdef np.int32_t max_terms = 2 * x.shape[0]
#     epsilon = ifnan(epsilon, .001)
#     penalty = ifnan(penalty, 3)
#     cdef object mars = Earth(thresh=epsilon, penalty=penalty)  # thresh=epsilon, penalty=penalty, minspan=1, endspan=1, max_terms=max_terms)  # , check_every=1, minspan=0, endspan=0,
#     mars.fit(np.asarray(x), np.asarray(y)) #, sample_weight=weights)
#     if verbose:
#         print(mars.summary())
#     cdef object f = PiecewiseFunction()
#     cdef list hinges = [h for h in mars.basis_ if not h.is_pruned()]
#     cdef DTYPE_t[::1] coeff = mars.coef_[0,:]
#     # Sort the functions according to their knot positions
#     cdef np.ndarray functions = np.vstack([[h.get_knot() if isinstance(h, HingeBasisFunctionBase) else np.nan for h in hinges], hinges, coeff])
#     functions = functions[:, np.argsort(functions[0,:])]
#     cdef LinearFunction linear = LinearFunction(0, 0)
#     cdef DTYPE_t[:,::1] x_ = np.array([[0]], dtype=np.float64)
#     cdef DTYPE_t k, w
#     for k, f_, w in functions.T:
#         if isinstance(f_, LinearBasisFunction):
#             linear.m += w
#         elif isinstance(f_, HingeBasisFunction):
#             if f_.get_reverse() == 1:
#                 linear.m -= w
#                 linear.c += w * f_.get_knot()
#         elif isinstance(f_, ConstantBasisFunction):
#             linear.c += w
#     cdef DTYPE_t m = linear.m
#     cdef DTYPE_t c = linear.c
#     f.intervals.append(R.copy())
#     f.functions.append(linear)
#     for k, f_, w in functions.T:
#         if not type(f_) is HingeBasisFunction:
#             continue
#         p = m * f_.get_knot() + c
#         m += w
#         c = p - m * f_.get_knot()
#         if f_.get_knot() == f.intervals[-1].lower and m > 0:
#             f.functions[-1].m = m
#             f.functions[-1].c = c
#         elif f_.get_knot() in f.intervals[-1] and m > 0:
#             f.intervals[-1].upper = f_.get_knot()
#             f.intervals.append(ContinuousSet(f_.get_knot(), np.PINF, 1, 2))
#             f.functions.append(LinearFunction(m, c))  # if m != 0 else ConstantFunction(c))
#     return f
#
#
