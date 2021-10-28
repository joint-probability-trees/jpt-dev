# cython: auto_cpdef=True,
# cython: infer_types=True,
# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
import itertools
import random
from collections import deque
from operator import itemgetter

from dnutils import ifnot, out, first, stop
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

from .utils import pairwise
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
        return str(self.value)

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
        l = ('%.3fx' % self.m) if self.m else ''
        op = '' if (not l and self.c > 0 or not self.c) else ('+' if self.c > 0 else '-')
        c = '' if not self.c else '%.3f' % abs(self.c)
        return ('%s %s %s' % (l, op, c)).strip()

    def __repr__(self):
        return '<%s>' % str(self)

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
        result = QuantileDistribution(self.epsilon, self.penalty, min_samples_mars=self.min_samples_mars)
        result._cdf = cdf_
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

            ppf.intervals.append(ContinuousSet(np.NINF, 0, EXC, EXC)) # np.nextafter(f(i.lower), f(i.lower) - 1),
            ppf.functions.append(Undefined())

            assert len(self._cdf.functions) > 1, self._cdf.pfmt()

            if len(self._cdf.functions) == 2:
                ppf.intervals[-1].upper = 1
                ppf.intervals.append(ContinuousSet(1, one_plus_eps, INC, EXC))
                ppf.functions.append(ConstantFunction(self._cdf.intervals[-1].lower))
                ppf.intervals.append(ContinuousSet(one_plus_eps, np.PINF, INC, EXC)) # np.nextafter(f(i.lower), f(i.lower) - 1),
                ppf.functions.append(Undefined())
                return ppf
            lower = None
            for interval, f in zip(self._cdf.intervals[1:-1], self._cdf.functions[1:-1]):
                if f.is_invertible():
                    ppf.intervals.append(ContinuousSet(ppf.intervals[-1].upper if lower is None else lower,
                                                       f(interval.upper),
                                                       INC, EXC))
                    ppf.functions.append(f.invert())
                    lower = None
                else:
                    lower = f.c if isinstance(f, LinearFunction) else f.value
                    continue
            if ppf.intervals[-1].upper >= 1:
                ppf.intervals[-1].upper = one_plus_eps
            elif ppf.intervals[-1].upper < 1:
                try:
                    ppf.functions.append(LinearFunction.from_points((ppf.intervals[-1].upper,
                                                                     ppf.functions[-1].eval(ppf.intervals[-1].upper)),
                                                                    (1,
                                                                     self.cdf.intervals[-1].lower)))
                except ValueError:
                    s = str(ppf.functions[-1].eval(ppf.intervals[-1].upper)) + '\n'
                    s += str(self._cdf.pfmt()) + '\n'
                    s += str(ppf.intervals[-1].upper) + '\n'
                    s += str(ppf.functions[-1]) + '\n'
                    s += str(ppf.pfmt())
                    raise ValueError(s)
                ppf.intervals.append(ContinuousSet(ppf.intervals[-1].upper, one_plus_eps, INC, EXC))

            ppf.intervals.append(ContinuousSet(one_plus_eps, np.PINF, INC, EXC))
            ppf.functions.append(Undefined())
            assert np.isnan(ppf.eval(one_plus_eps)), \
                (ppf.intervals[-1].lower, ppf.intervals[-1],
                 'value:', ppf.intervals[-2].upper, ppf.eval(one_plus_eps), ppf.pfmt(), self._cdf.pfmt())
            assert not np.isnan(ppf.eval(1.)), ppf.pfmt()
            self._ppf = ppf
        return self._ppf

    @staticmethod
    def merge(distributions, weights):
        '''
        Construct a merged quantile-distribution from the passed distributions using the ``weights``.
        '''
        intervals = [ContinuousSet(np.NINF, np.PINF, EXC, EXC)]
        functions = [ConstantFunction(0)]
        lower = sorted([(i.lower, f, w)
                        for d, w in zip(distributions, weights)
                        for i, f in zip(d.cdf.intervals, d.cdf.functions)],
                       key=itemgetter(0))
        upper = sorted([(i.upper, f, w)
                        for d, w in zip(distributions, weights)
                        for i, f in zip(d.cdf.intervals, d.cdf.functions)],
                       key=itemgetter(0))

        # --------------------------------------------------------------------------------------------------------------
        # We preprocess the CDFs that are in the form of "jump" functions
        jumps = {}
        for w, cdf in [(w, d.cdf) for w, d in zip(weights, distributions) if len(d.cdf) == 2]:
            jumps[cdf.intervals[0].upper] = jumps.get(cdf.intervals[0].upper, 0) + w

        # --------------------------------------------------------------------------------------------------------------
        m = 0

        while lower or upper:
            pivot = None

            # Process all function intervals whose lower bound is minimal and
            # smaller than the smallest upper interval bound
            while lower and (pivot is None and first(lower, first) <= first(upper, first, np.PINF) or
                   pivot == first(lower, first, np.PINF)):
                l, f, w = lower.pop(0)
                if isinstance(f, ConstantFunction) or l == np.NINF:
                    continue
                m += f.m * w
                pivot = l

            # Do the same for the upper bounds...
            while upper and (pivot is None and first(upper, first) <= first(lower, first, np.PINF) or
                   pivot == first(upper, first, np.PINF)):
                u, f, w = upper.pop(0)
                if isinstance(f, ConstantFunction) or u == np.PINF:
                    continue
                m -= f.m * w
                pivot = u

            if pivot is None:
                continue

            if functions and m == functions[-1].m and functions[-1].eval(pivot) - m * pivot == functions[-1].c:
                continue
            # Split the last interval at the pivot point
            intervals[-1].upper = pivot
            intervals.append(ContinuousSet(pivot, np.PINF, INC, EXC))
            # Evaluate the old function at the new pivot point to get the intercept
            functions.append(LinearFunction(m, functions[-1].eval(pivot) - m * pivot))

        # If the merging ends with an "approximate" constant function
        # remove it. This may happen for numerical imprecision.
        while len(functions) > 1 and abs(functions[-1].m) <= 1e-08:
            del intervals[-1]
            del functions[-1]

        cdf = PiecewiseFunction()
        cdf.functions = functions
        cdf.intervals = intervals
        cdf.ensure_right(ConstantFunction(1), l or u)
        distribution = QuantileDistribution()
        distribution._cdf = cdf

        if len(cdf.functions) == 3 and cdf.functions[1].m < 1e-4:
            raise ValueError(cdf.pfmt() + '\n\n' + '\n---\n'.join([d.cdf.pfmt()
                                                                   for d in distributions]) + '\n' + str(jumps))

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


# @cython.freelist(500)
# cdef class Quantiles:
#     '''
#     This class implements basic representation and handling of quantiles
#     in a data distribution.
#     '''
#
#     def __init__(Quantiles self, DTYPE_t[:] data, DTYPE_t lower=np.nan, np.int32_t verbose=False,
#                  DTYPE_t upper=np.nan, DTYPE_t epsilon=.001, DTYPE_t penalty=3., np.int32_t dtype=_FLOAT):
#         if data.shape[0] == 0:
#             raise IndexError('Data must contain at least 1 data point, got only %s' % len(data))
#         self.dtype = dtype
#         self.verbose = verbose
#         cdef DTYPE_t[::1] bins
#         cdef DTYPE_t z
#         if self.dtype == _INT:
#             self.data = np.sort(np.unique(data))
#             bins = np.ndarray(shape=self.data.shape[0]+1, dtype=np.float64)
#             bins[0:-1] = self.data
#             bins[-1] = self.data[-1] + 1.
#             self.weights = np.histogram(data, bins=bins, density=True)[0]
#         elif self.dtype == _FLOAT:
#             self.data = np.sort(np.unique(data))
#             self.weights = np.ones(shape=self.data.shape[0], dtype=np.float64) * 1. / self.data.shape[0]
#         elif self.dtype == _GAUSSIAN:
#             self.weights = None
#             self.data = np.unique(data)
#         elif self.dtype == _UNIFORM:
#             self.weights = None
#             self.data = np.sort(np.unique(data))
#         else:
#             raise TypeError('Invalid dtype argument: must be INT or FLOAT, got %s' % dtype)
#         self._mean = np.nan
#         self.epsilon = ifnan(epsilon, .001)
#         self.penalty = ifnan(penalty, 3)
#         self._cdf = self._pdf = self._invcdf = None
#         self._upper = upper
#         self._lower = lower
#         self._stuff = None
#
#     @property
#     def mean(self):
#         if np.isnan(self._mean):
#             if self.weights is None:
#                 self.weights = np.ones(shape=self.data.shape[0], dtype=np.float64) * 1. / self.data.shape[0]
#             self._mean = sum([w * d for w, d in zip(self.weights, self.data)])
#         return self._mean
#
#     def __len__(self):
#         return len(self.data)
#
#     @property
#     def array(self):
#         return np.asarray(self.data, dtype=np.float64)
#
#     cpdef Function cdf(Quantiles self, DTYPE_t epsilon=np.nan, DTYPE_t penalty=np.nan):
#         cdef DTYPE_t[::1] y
#         if self._cdf is None and self.dtype == _UNIFORM:
#             self._cdf = PiecewiseFunction()
#             self._cdf.intervals.append(ContinuousSet(np.NINF, self.data[0], 2, 2))
#             self._cdf.functions.append(ConstantFunction(0))
#             if self.data[0] != self.data[-1]:
#                 self._cdf.intervals.append(ContinuousSet(self.data[0], self.data[-1], 1, 2))
#                 self._cdf.functions.append(LinearFunction.from_points((self.data[0], 0), (self.data[-1], 1)))
#             self._cdf.intervals.append(ContinuousSet(self.data[-1], np.PINF,  1, 2))
#             self._cdf.functions.append(ConstantFunction(1))
#         elif self._cdf is None and self.dtype == _GAUSSIAN:
#             if self._stuff is None:
#                 self._stuff = norm.fit(self.data)
#             self._cdf = GaussianCDF(*self._stuff)
#         elif self._cdf is None:
#             y = np.cumsum(self.weights, dtype=np.float64)
#             if 1 < self.data.shape[0] < 20:  # Use simple linear regression when fewer than 20 points are available
#                 self._cdf = PiecewiseFunction()
#                 self._cdf.intervals.append(R.copy())
#                 self._cdf.functions.append(LinearFunction(0, 0).fit(self.data, y))
#             elif 20 <= self.data.shape[0]:
#                 self._cdf = fit_piecewise(self.data, y, epsilon=ifnan(epsilon, self.epsilon),
#                                           penalty=ifnan(penalty, self.penalty), verbose=self.verbose)
#             else:
#                 self._cdf = PiecewiseFunction()
#                 self._cdf.intervals.append(R.copy())
#                 self._cdf.functions.append(ConstantFunction(1))
#             self._cdf.ensure_left(ConstantFunction(0), self.data[0])
#             self._cdf.ensure_right(ConstantFunction(1), self.data[-1])
#         return self._cdf
#
#     cpdef Function invcdf(Quantiles self, DTYPE_t epsilon=np.nan, DTYPE_t penalty=np.nan):
#         cdef PiecewiseFunction cdf
#         cdef PiecewiseFunction inv
#         cdef ContinuousSet i
#         cdef Function f
#         cdef DTYPE_t mean, stddev
#         if self._invcdf is None and self.dtype == _GAUSSIAN:
#             if self._stuff is None:
#                 self._stuff = norm.fit(self.data)
#             self._invcdf = GaussianPPF(*self._stuff)
#         elif self._invcdf is None:
#             cdf = self.cdf(epsilon=ifnan(epsilon, self.epsilon), penalty=ifnan(penalty, self.penalty))
#             inv = PiecewiseFunction()
#             for i, f in zip(cdf.intervals, cdf.functions):
#                 if f.is_invertible():
#                     inv_ = f.invert()
#                     inv.intervals.append(ContinuousSet(max(0, f(i.lower)),
#                                                        min(np.nextafter(1, 2), f(i.upper)),
#                                                        INC, EXC))
#                     inv.functions.append(inv_)
#                 else:
#                     if i.lower == np.NINF:
#                         inv.intervals.append(ContinuousSet(np.NINF, np.nextafter(f(i.lower), f(i.lower)-1), EXC, EXC))
#                         inv.functions.append(Undefined())
#                     if i.upper == np.PINF:
#                         inv.intervals[-1].upper = 1
#                         inv.intervals.append(ContinuousSet(1, np.nextafter(1, 2), 1, 2))
#                         inv.functions.append(ConstantFunction(i.lower))
#                         inv.intervals.append(ContinuousSet(np.nextafter(1, 2), np.PINF, 1, 2))
#                         inv.functions.append(Undefined())
#             self._invcdf = inv
#         return self._invcdf
#
#     cpdef Function pdf(Quantiles self, np.int32_t simplify=False, np.int32_t samples=False,
#                                 DTYPE_t epsilon=np.nan, DTYPE_t penalty=np.nan):
#         cdef PiecewiseFunction pdf
#         cdef DTYPE_t mean, stddev
#         if self._pdf is None and self.dtype == _GAUSSIAN:
#             if self._stuff is None:
#                 self._stuff = norm.fit(self.data)
#             self._pdf = GaussianPDF(*self._stuff)
#         elif self._pdf is None:
#             pdf = self.cdf().differentiate()
#             if len(pdf.intervals) == 2 and self.data.shape[0] == 1:
#                 pdf.intervals.insert(1, ContinuousSet(self.data[0], np.nextafter(self.data[0], self.data[0]+1), 1, 2))
#                 pdf.intervals[-1].lower = np.nextafter(pdf.intervals[-1].lower, pdf.intervals[-1].lower+1)
#                 pdf.functions.insert(1, ConstantFunction(1))
#             if simplify:
#                 pdf = pdf.simplify(samples, epsilon=ifnan(epsilon, self.epsilon), penalty=ifnan(penalty, self.penalty))
#             self._pdf = pdf
#         return self._pdf
#
#     cpdef DTYPE_t[::1] sample(Quantiles self, np.int32_t n=1, DTYPE_t[::1] result=None):
#         '''
#         Generate from this quantile distribution ``k`` samples.
#
#         This method implements a slice sampling procedure.
#
#         :param n:       (int) the number of samples to draw.
#         :param result:  (optional) buffer the results shall be written to.
#         :return:
#         '''
#         if result is None:
#             result = np.ndarray(shape=n, dtype=np.float64)
#         cdef size_t i
#         if self.dtype == _GAUSSIAN:
#             if self._stuff is None:
#                 self._stuff = norm(*norm.fit(self.data))
#             for i in range(result.shape[0]):
#                 result[i] = self._stuff.rvs(size=1)
#             return result
#         if self.data.shape[0] == 1:
#             result[...] = self.data[0]
#             return result
#         cdef PiecewiseFunction pdf = self.pdf()
#         cdef DTYPE_t[::1] z = np.ndarray(1, dtype=np.float64)
#         z[...] = np.random.uniform(self.data[0], self.data[-1])
#         cdef DTYPE_t slice_
#         for i in range(n):
#             slice_ = random.uniform(0, pdf(z[0]))
#             z[...] = pdf.gt(slice_).sample(1, result=result[i:i+1])
#             if result[i] > 1e6:
#                 # print('sampling from', pdf.pfmt(), 'in', pdf.gt(slice), 'returned', result[i])
#                 raise ValueError('sampling from', pdf.pfmt(), 'in', pdf.gt(slice_), 'returned', result[i])
#         if self.dtype == _INT:
#             np.round(result, out=np.asarray(result))
#         return result
#
#     cpdef DTYPE_t[::1] gt(Quantiles self, DTYPE_t q):
#         cdef np.int32_t points = self.data.shape[0]
#         return self.data[int(np.floor(q * points)):] if points > 1 else self.data
#
#     cpdef DTYPE_t[::1] lt(Quantiles self, DTYPE_t q):
#         cdef np.int32_t points = self.data.shape[0]
#         return self.data[:int(np.floor(q * points))] if points > 1 else self.data
#
#     @property
#     def median(self):
#         return self.invcdf().eval(.5)
#
#     @property
#     def lower_half(self):
#         a = np.asarray(self.data)
#         return a[a <= self.median] if not np.isnan(self.median) else self.data[:]
#
#     @property
#     def upper_half(self):
#         a = np.asarray(self.data)
#         return a[a > self.median] if not np.isnan(self.median) else self.data[:]
#
#     @property
#     def avgdist(self):
#         return np.mean(np.diff(self.data))
#
#     @property
#     def lower(self):
#         return self.invcdf().eval(self._lower)
#
#     @property
#     def upper(self):
#         return self.invcdf().eval(self._upper)
#
#     cpdef ConfInterval interval(Quantiles self, DTYPE_t conf_level):
#         if not 0 <= conf_level <= 1:
#             raise ValueError('Confidence level must be in [0, 1], got %s' % conf_level)
#         return ConfInterval(self.mean, self.invcdf().eval(conf_level), self.invcdf().eval(1 - conf_level))


@cython.final
cdef class PiecewiseFunction(Function):
    '''
    Represents a function that is piece-wise defined by constant values.
    '''

    def __init__(PiecewiseFunction self):
        self.functions = []
        self.intervals = []

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
        cdef int i
        cdef ContinuousSet interval
        for i, interval in enumerate(self.intervals):
            if x in interval:
                break
        else:
            return Undefined()
        return self.functions[i]

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
        intstr = list(map(str, self.intervals))
        funstr = list(map(str, self.functions))
        space = max([len(i) for i in intstr])
        return str('\n'.join(['%s |--> %s' % (i.ljust(space + 2, ' '), f) for i, f in zip(intstr, funstr)]))

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
                intersection = i.intersection(interval)
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
