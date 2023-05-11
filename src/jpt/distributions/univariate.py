'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
from collections import deque, Counter
from itertools import tee
from types import FunctionType
from typing import Any, Iterable, List, Union, Set, Type, Tuple

from jpt.base.utils import classproperty, save_plot, normalized, mapstr, setstr, none2nan
from jpt.base.errors import Unsatisfiability

import copy
import math
import numbers
import os
import re
from operator import itemgetter

from dnutils import first, out, ifnone, stop, ifnot, project, pairwise, edict
from dnutils.stats import Gaussian as Gaussian_, _matshape

from scipy.stats import multivariate_normal, norm

import numpy as np
from numpy import iterable

import matplotlib.pyplot as plt

from jpt.base.constants import sepcomma
from jpt.base.sampling import wsample, wchoice
from .utils import Identity, OrderedDictProxy, DataScalerProxy, DataScaler

try:
    from ..base.intervals import __module__
    from .quantile.quantiles import __module__
    from ..base.functions import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from ..base.intervals import R, ContinuousSet, RealSet, NumberSet
    from ..base.functions import LinearFunction, ConstantFunction, PiecewiseFunction
    from .quantile.quantiles import QuantileDistribution


# ----------------------------------------------------------------------------------------------------------------------
# Constant symbols

SYMBOLIC = 'symbolic'
NUMERIC = 'numeric'
CONTINUOUS = 'continuous'
DISCRETE = 'discrete'


# ----------------------------------------------------------------------------------------------------------------------
# Gaussian distribution. This is somewhat deprecated as we use model-free
# quantile distributions, but this code is used in testing to sample
# from Gaussian distributions.
# TODO: In order to keep the code consistent, this class should inherit from 'Distribution'

class Gaussian(Gaussian_):
    """Extension of :class:`dnutils.stats.Gaussian`"""

    PRECISION = 1e-15

    def __init__(self, mean=None, cov=None, data=None, weights=None):
        """Creates a new Gaussian distribution.

        :param mean:    the mean of the Gaussian
        :type mean:     float if multivariate else [float] if multivariate
        :param cov:     the covariance of the Gaussian
        :type cov:      float if multivariate else [[float]] if multivariate
        :param data:    if ``mean`` and ``cov`` are not provided, ``data`` may be a data set (matrix) from which the
                        parameters of the distribution are estimated.
        :type data:     [[float]]
        :param weights:  **[optional]** weights for the data points. The weight do not need to be normalized.
        :type weights:  [float]
        """
        self._sum_w = 0  # ifnot(weights, 0, sum)
        self._sum_w_sq = 0  # 1 / self._sum_w ** 2 * sum([w ** 2 for w in weights]) if weights else 0
        super().__init__(mean=mean, cov=cov, keepsamples=False)
        self._mean = np.array(self._mean, dtype=np.float64)
        self._cov = np.array(self._cov, dtype=np.float64)
        while self._cov.shape and len(self._cov.shape) < 2:
            self._cov = np.array([self._cov])
        if None not in (weights, data) and weights.shape[0] != data.shape[0]:
            raise ValueError('Weight vector must have same length as data vector.')
        if data:
            self.estimate(data, weights)
        self.data = []

    @Gaussian_.mean.getter
    def mean(self):
        return self._mean

    @Gaussian_.cov.getter
    def cov(self):
        return self._cov

    @Gaussian_.var.getter
    def var(self):
        return np.array([self._cov[i, i] for i in range(self.dim)])

    @property
    def std(self):
        return np.sqrt(self.var)

    def deviation(self, x):
        """
        Computes the deviation of ``x`` in multiples of the standard deviation.

        :param x:
        :type x:
        :returns:
        """
        if isinstance(x, numbers.Number):
            raise TypeError('Argument must be a vector, got a scalar: %s' % x)
        return (np.array(x) - np.array(self._mean)) / self.std

    def __add__(self, alpha):
        if isinstance(alpha, Gaussian):
            if alpha.dim != self.dim:
                raise TypeError('Addition of two Gaussian only with same dimensionality.')
            return Gaussian(mean=[m1 + m2 for m1, m2, in zip(self._mean, alpha._mean)],
                            cov=[[c1 + c2 for c1, c2 in zip(row1, row2)] for row1, row2 in zip(self._cov, alpha._cov)])
        elif isinstance(alpha, numbers.Number):
            return Gaussian(mean=[alpha + m for m in self._mean], cov=[[c for c in row] for row in self._cov])
        else:
            raise TypeError('Undefined operation "+" on types %s and %s' % (type(self).__name__, type(alpha).__name__))

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        result = self + other
        self._mean = result._mean
        self._cov = result._cov
        return self

    def __mul__(self, alpha):
        if isinstance(alpha, Gaussian):
            raise TypeError('Multiplication of two Gaussians not supported yet.')
        elif isinstance(alpha, numbers.Number):
            return Gaussian(mean=[alpha * m for m in self._mean], cov=[[alpha ** 2 * c for c in row] for row in self._cov])
        else:
            raise TypeError('Undefined operation "*" on types %s and %s' % (type(self).__name__, type(alpha).__name__))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        result = self * other
        self._mean = result._mean
        self._cov = result._cov
        return self

    @Gaussian_.dim.getter
    def dim(self):
        if self._mean is None:
            raise ValueError('no dimensionality specified yet.')
        return self._mean.shape[0]

    def sample(self, n):
        return multivariate_normal(self._mean, self._cov, n).rvs(n)

    @property
    def pdf(self):
        try:
            return multivariate_normal(self._mean, self._cov, allow_singular=True).pdf
        except ValueError:
            out(self._mean)
            out(self._cov)
            raise

    def cdf(self, *x):
        return np.array(norm.cdf(x, loc=self._mean, scale=self._cov))[0, :]

    def eval(self, lower, upper):
        return abs(multivariate_normal(self._mean, self._cov).cdf(upper) - multivariate_normal(self._mean, self._cov).cdf(lower))

    def copy(self):
        g = Gaussian()
        g._mean = self._mean
        g._cov = self._cov
        g.samples = copy.copy(self.samples)
        # g._weights = list(self.weights)
        return g

    def __eq__(self, other):
        return (other.mean == self.mean and other.cov == self.cov) if other is not None else False

    def linreg(self):
        """
        Compute a 4-tuple ``<m, b, rss, noise>`` of a linear regression represented by this Gaussian.

        :return:    ``m`` - the slope of the line
                    ``b`` - the intercept of the line
                    ``rss`` - the residual sum-of-squares error
                    ``noise`` - the square of the sample correlation coefficient ``r^2``

        References:
            - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#In_least_squares_regression_analysis
            - https://milnepublishing.geneseo.edu/natural-resources-biometrics/chapter/chapter-7-correlation-and-simple-linear-regression/
            - https://en.wikipedia.org/wiki/Residual_sum_of_squares
            - https://en.wikipedia.org/wiki/Explained_sum_of_squares
        """
        if self.dim != 2:
            raise ValueError('This operation is only supported for 2-dimensional Gaussians.')
        if self.numsamples < 2:
            raise ValueError('Need at least 2 data points. %d are too few.' % self.numsamples)
        rho = self._cov[0][1] / math.sqrt(self.var[0] * self.var[1]) if all(abs(v) > 1e-8 for v in self.var) else 1
        m = self._cov[0][1] / self.var[0]
        b = self._mean[1] - m * self._mean[0]
        rss = self.var[1] * (1 - rho ** 2)
        return m, b, rss, ((self.var[1] - rss) / self.var[1] if self.var[1] else 0)

    def update_all(self, data, weights=None):
        '''Update the distribution with new data points given in ``data``.'''
        weights = ifnone(weights, [1] * len(data))
        if len(data) != len(weights):
            raise ValueError('Weight vector must have the same length as data vector.')
        for x, w in zip(data, weights):
            self.update(x, w)
        return self

    def estimate(self, data, weights=None):
        '''Estimate the distribution parameters with subject to the given data points.'''
        self.mean = self.cov = None
        return self.update_all(data=data, weights=weights)

    def update(self, x, w=1):
        '''update the Gaussian distribution with a new data point ``x`` and weight ``w``.'''
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if self._mean is None or not self._mean.shape or not self._cov.shape:
            self._mean = np.zeros(x.shape)
            self._cov = np.zeros(shape=(x.shape[0], x.shape[0]))
        else:
            assert self._mean.shape == x.shape and self._cov.shape == (x.shape[0], x.shape[0])
        # self.data.append(np.array(x))
        sum_w = self._sum_w  # S_w
        sum_w_ = self._sum_w + w  # S'_w
        sum_w_sq = self._sum_w_sq  # S_{w^2}
        sum_w_sq_ = (sum_w ** 2 * sum_w_sq + w ** 2) / sum_w_ ** 2  # S'_{w^2}
        oldmean = np.array(self._mean)
        oldcov = np.array(self._cov)
        dims = np.where(np.abs(x - self._mean) > Gaussian.PRECISION)[0]
        for dim in dims:  # Check this for numerical stability. Otherwise the cov matrix may not be positive semi-definite
            self._mean[dim] = (sum_w * oldmean[dim] + w * x[dim]) / sum_w_
        self.samples += 1
        if sum_w_ and sum_w_sq_ != 1:
            for j in range(self.dim):
                for k in range(j+1):
                    if k in dims or j in dims:
                        self._cov[j][k] = (oldcov[j][k] * sum_w * (1 - sum_w_sq)
                                           + sum_w * oldmean[j] * oldmean[k]
                                           - sum_w_ * self._mean[j] * self._mean[k]
                                           + w * (x[j] if j in dims else oldmean[j]) * (x[k] if k in dims else oldmean[k])) / (sum_w_ * (1 - sum_w_sq_))
                    else:  # No change in either of the dimensions,
                        self._cov[j][k] = oldcov[j][k] * sum_w * (1 - sum_w_sq) / (sum_w_ * (1 - sum_w_sq_))
                    self._cov[k][j] = self._cov[j][k]
            for j in range(self.dim):
                for k in range(self.dim):
                    if self._cov[k, k] < Gaussian.PRECISION:
                        self._cov[k, :] = 0
                        self._cov[:, k] = 0
                    if self._cov[j, j] < Gaussian.PRECISION:
                        self._cov[j, :] = 0
                        self._cov[:, j] = 0
        self._sum_w = sum_w_
        self._sum_w_sq = sum_w_sq_
        # self.pdf
        if not self.sym():
            out(f'oldmean {oldmean} oldcov\n{np.array(oldcov)}, introducing {x} as +1th example')
            stop(f'newmean {self._mean} newcov\n{np.array(self._cov)}')

    def retract(self, x, w=1):
        """
        Retract the data point `x` with weight `w` from the Gaussian distribution.

        In case the data points are being kept in the distribution, it must actually exist and have the right
        weight associated. Otherwise, a ValueError will be raised.
        """
        if not hasattr(x, '__len__'):
            x = [x]
        if self._mean is None or self._cov is None or not self.numsamples:
            raise ValueError('Cannot retract a value from an empty distribution.')
        else:
            assert len(x) == len(self._mean) and _matshape(self._cov) == (len(x), len(x))
        if type(self.samples) is list:
            idx = [i for i, (v, w_) in enumerate(zip(self.samples, self._weights)) if v == x and w == w_]
            if idx:
                i = first(idx)
                del self.samples[i]
                del self._weights[i]
            else:
                raise ValueError('Only elements from the distribution can be retracted. %s is not an element.' % x)
        sum_w = self._sum_w  # S_w
        sum_w_ = self._sum_w - w  # S'_w
        sum_w_sq = self._sum_w_sq  # S_{w^2}
        sum_w_sq_ = (sum_w ** 2 * self._sum_w_sq - w ** 2) / sum_w_ ** 2  # S'_{w^2}
        oldmean = list(self._mean)
        oldcov = list([list(row) for row in self._cov])
        if sum_w_:
            for i, (m, x_i) in enumerate(zip(self._mean, x)):
                self._mean[i] = ((sum_w * m) - w * x_i) / sum_w_
            if sum_w_sq_ < 1:
                for j in range(self.dim):
                    for k in range(self.dim):
                        self._cov[j][k] = (oldcov[j][k] * sum_w * (1 - sum_w_sq)
                                           + sum_w * oldmean[j] * oldmean[k]
                                           - w * x[j] * x[k]
                                           - sum_w_ * self._mean[j] * self._mean[k]) / (sum_w_ * (1 - sum_w_sq_))
            else:
                self._cov = [[0] * self.dim for _ in range(self.dim)]
        else:
            self._mean = None
            self._cov = None
        self._sum_w = sum_w_
        self._sum_w_sq = sum_w_sq_

    def sym(self):
        for i in range(self.dim):
            for j in range(self.dim):
                if self._cov[i][j] != self._cov[j][i]:
                    out(i, j, self._cov[i][j], self._cov[j][i], self._cov[i][j] == self._cov[j][i])
                    return False
        return True


# ----------------------------------------------------------------------------------------------------------------------

class Distribution:
    '''
    Abstract supertype of all domains and distributions
    '''
    values = None
    labels = None

    SETTINGS = {
    }

    def __init__(self, **settings):
        # used for str and repr methods to be able to print actual type
        # of Distribution when created with jpt.variables.Variable
        self._cl = f'{self.__class__.__name__}' \
                   + (f' ({self.__class__.__mro__[1].__name__})'
                      if self.__module__ != __name__
                      else '')
        self.settings = type(self).SETTINGS.copy()
        for attr in type(self).SETTINGS:
            try:
                super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                raise AttributeError('Attribute ambiguity: Object of type "%s" '
                                     'already has an attribute with name "%s"' % (type(self).__name__,
                                                                                  attr))
        for attr, value in settings.items():
            if attr not in self.settings:
                raise AttributeError('Unknown settings "%s": '
                                     'expected one of {%s}' % (attr, setstr(type(self).SETTINGS)))
            self.settings[attr] = value

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in type(self).SETTINGS:
                return self.settings[name]
            else:
                raise

    def __hash__(self):
        return hash((type(self), self.values, self.labels))

    def __getitem__(self, value):
        return self.p(value)

    @classmethod
    def value2label(cls, value):
        raise NotImplementedError()

    @classmethod
    def label2value(cls, label):
        raise NotImplementedError()

    def _sample(self, n: int) -> Iterable:
        raise NotImplementedError()

    def _sample_one(self):
        raise NotImplementedError()

    def sample(self, n: int) -> Iterable:
        yield from (self.value2label(v) for v in self._sample(n))

    def sample_one(self) -> Any:
        return self.value2label(self._sample_one())

    def p(self, value):
        raise NotImplementedError()

    def _p(self, value):
        raise NotImplementedError()

    def expectation(self) -> numbers.Real:
        raise NotImplementedError()

    def mpe(self):
        raise NotImplementedError()

    def crop(self):
        raise NotImplementedError()

    def _crop(self):
        raise NotImplementedError()

    def merge(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()

    def moment(self, order=1, c=0):
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param c: The constant to subtract in the basis of the exponent
        """
        raise NotImplementedError()

    def fit(self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None) -> 'Distribution':
        raise NotImplementedError()

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Distribution':
        raise NotImplementedError()

    def set(self, params: Any) -> 'Distribution':
        raise NotImplementedError()

    def kl_divergence(self, other: 'Distribution'):
        raise NotImplementedError()

    def number_of_parameters(self) -> int:
        raise NotImplementedError()

    def plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, **kwargs):
        '''Generates a plot of the distribution.

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file
        :type fname:        str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :return:            None
        '''
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = Distribution.from_json(state).__dict__

    @staticmethod
    def from_json(data) -> Type:
        cls = _DISTRIBUTION_TYPES.get(data['type'])
        if cls is None:
            raise TypeError('Unknown distribution type: %s' % data['type'])
        return cls.type_from_json(data)

    type_from_json = from_json


# ----------------------------------------------------------------------------------------------------------------------

class Numeric(Distribution):
    '''
    Wrapper class for numeric domains and distributions.
    '''

    PRECISION = 'precision'

    values = Identity()
    labels = Identity()

    SETTINGS = {
        PRECISION: .01
    }

    def __init__(self, **settings):
        super().__init__(**settings)
        self._quantile: QuantileDistribution = None
        self.to_json = self.inst_to_json

    def __str__(self):
        return self.cdf.pfmt()

    def __getitem__(self, value):
        return self.p(value)

    def __eq__(self, o: 'Numeric'):
        if not issubclass(type(o), Numeric):
            raise TypeError('Cannot compare object of type %s with other object of type %s' % (type(self),
                                                                                               type(o)))
        return type(o).equiv(type(self)) and self._quantile == o._quantile

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[numbers.Real, NumberSet]) -> Union[numbers.Real, NumberSet]:
        if isinstance(value, ContinuousSet):
            return ContinuousSet(cls.labels[value.lower], cls.labels[value.upper], value.left, value.right)
        elif isinstance(value, RealSet):
            return RealSet([cls.value2label(i) for i in value.intervals])
        elif isinstance(value, numbers.Real):
            return cls.labels[value]
        else:
            raise TypeError('Expected float or NumberSet type, got %s.' % type(value).__name__)

    # noinspection DuplicatedCode
    @classmethod
    def label2value(cls, label: Union[numbers.Real, NumberSet]) -> Union[numbers.Real, NumberSet]:
        if isinstance(label, ContinuousSet):
            return ContinuousSet(cls.values[label.lower], cls.values[label.upper], label.left, label.right)
        elif isinstance(label, RealSet):
            return RealSet([cls.label2value(i) for i in label.intervals])
        elif isinstance(label, numbers.Real):
            return cls.values[label]
        else:
            raise TypeError('Expected float or NumberSet type, got %s.' % type(label).__name__)

    @classmethod
    def equiv(cls, other):
        return (issubclass(other, Numeric) and
                cls.__name__ == other.__name__ and
                cls.values == other.values and
                cls.labels == other.labels)

    @property
    def cdf(self):
        return self._quantile.cdf

    @property
    def pdf(self):
        return self._quantile.pdf

    @property
    def ppf(self):
        return self._quantile.ppf

    def _sample(self, n):
        return self._quantile.sample(n)

    def _sample_one(self):
        return self._quantile.sample(1)[0]

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 1 if this is a dirac impulse, number of intervals times two else
        """
        if self.is_dirac_impulse():
            return 1
        else:
            return len(self.cdf.intervals)

    def _expectation(self) -> numbers.Real:
        e = 0
        singular = True  # In case the CDF is jump fct the expectation is where the jump happens
        for i, f in zip(self.cdf.intervals, self.cdf.functions):
            if i.lower == np.NINF or i.upper == np.PINF:
                continue
            e += (self.cdf.eval(i.upper) - self.cdf.eval(i.lower)) * (i.upper + i.lower) / 2
            singular = False
        return e if not singular else i.lower

    def expectation(self) -> numbers.Real:
        return self.moment(1)

    def variance(self) -> numbers.Real:
        return self.moment(2)

    def quantile(self, gamma: numbers.Real) -> numbers.Real:
        return self.ppf.eval(gamma)

    def create_dirac_impulse(self, value):
        """Create a dirac impulse at the given value aus quantile distribution."""
        self._quantile = QuantileDistribution()
        self._quantile.fit(
            np.asarray([[value]], dtype=np.float64),
            rows=np.asarray([0], dtype=np.int64),
            col=0
        )
        return self

    def is_dirac_impulse(self) -> bool:
        """Checks if this distribution is a dirac impulse."""
        return len(self._quantile.cdf.intervals) == 2

    def mpe(self) -> (float, RealSet):
        """
        Calculate the most probable configuration of this quantile distribution.
        :return: The likelihood of the mpe as float and the mpe itself as RealSet
        """
        _max = max(f.value for f in self.pdf.functions)
        return _max, self.value2label(RealSet([interval for interval, function in zip(self.pdf.intervals, self.pdf.functions)
                              if function.value == _max]))

    def _fit(
            self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None
    ) -> 'Numeric':
        self._quantile = QuantileDistribution(epsilon=self.precision)
        self._quantile.fit(
            data,
            rows=ifnone(
                rows,
                np.array(list(range(data.shape[0])), dtype=np.int64)
            ),
            col=ifnone(col, 0)
        )
        return self

    fit = _fit

    def set(self, params: QuantileDistribution) -> 'Numeric':
        self._quantile = params
        return self

    def _p(self, value: Union[numbers.Number, NumberSet]) -> numbers.Real:
        if isinstance(value, numbers.Number):
            value = ContinuousSet(value, value)
        elif isinstance(value, RealSet):
            return sum(self._p(i) for i in value.intervals)
        probspace = self.pdf.gt(0)
        if probspace.isdisjoint(value):
            return 0
        probmass = ((self.cdf.eval(value.upper) if value.upper != np.PINF else 1.) -
                    (self.cdf.eval(value.lower) if value.lower != np.NINF else 0.))
        if not probmass:
            return probspace in value
        return probmass

    def p(self, labels: Union[numbers.Number, NumberSet]) -> numbers.Real:
        if not isinstance(labels, (NumberSet, numbers.Number)):
            raise TypeError('Argument must be numbers.Number or '
                            'jpt.base.intervals.NumberSet (got %s).' % type(labels))
        if isinstance(labels, ContinuousSet):
            return self._p(self.label2value(labels))
        elif isinstance(labels, RealSet):
            self._p(RealSet([ContinuousSet(self.values[i.lower],
                                           self.values[i.upper],
                                           i.left,
                                           i.right) for i in labels.intervals]))
        else:
            return self._p(self.values[labels])

    def kl_divergence(self, other: 'Numeric') -> numbers.Real:
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        self_ = [(i.lower, f.value, None) for i, f in self.pdf.iter()]
        other_ = [(i.lower, None, f.value) for i, f in other.pdf.iter()]
        all_ = deque(sorted(self_ + other_, key=itemgetter(0)))
        queue = deque()
        while all_:
            v, p, q = all_.popleft()
            if queue and v == queue[-1][0]:
                if p is not None:
                    queue[-1][1] = p
                if q is not None:
                    queue[-1][2] = q
            else:
                queue.append([v, p, q])
        result = 0
        p, q = 0, 0
        for (x0, p_, q_), (x1, _, _) in pairwise(queue):
            p = ifnone(p_, p)
            q = ifnone(q_, q)
            i = ContinuousSet(x0, x1)
            result += self._p(i) * abs(self._p(i) - other._p(i))
        return result

    def copy(self):
        dist = type(self)(**self.settings).set(params=self._quantile.copy())
        dist.values = copy.copy(self.values)
        dist.labels = copy.copy(self.labels)
        return dist

    @staticmethod
    def merge(distributions: List['Numeric'], weights: Iterable[numbers.Real]) -> 'Numeric':
        if not all(distributions[0].__class__ == d.__class__ for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        return type(distributions[0])().set(QuantileDistribution.merge(distributions, weights))

    def update(self, dist: 'Numeric', weight: numbers.Real) -> 'Numeric':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if type(dist) is not type(self):
            raise TypeError('Can only update with distribution of the same type, got %s' % type(dist))
        tmp = Numeric.merge([self, dist], normalized([1, weight]))
        self.values = tmp.values
        self.labels = tmp.labels
        self._quantile = tmp._quantile
        return self

    def _crop(self, interval):
        dist = self.copy()
        dist._quantile = self._quantile.crop(interval)
        return dist

    def crop(self, restriction: RealSet or ContinuousSet or numbers.Number):
        """Apply a restriction to this distribution. The restricted distrubtion will only assign mass
        to the given range and will preserve the relativity of the pdf.

        :param restriction: The range to limit this distribution (or singular value)
        :type restriction: float or int or ContinuousSet
        """

        # for real sets the result is a merge of the single ContinuousSet crops
        if isinstance(restriction, RealSet):

            distributions = []

            for idx, continuous_set in enumerate(restriction.intervals):

                distributions.append(self.crop(continuous_set))

            weights = np.full((len(distributions)), 1/len(distributions))

            return self.merge(distributions, weights)

        elif isinstance(restriction, ContinuousSet):
            if restriction.size() == 1:
                return self.crop(restriction.lower)
            else:
                return self._crop(restriction)

        elif isinstance(restriction, numbers.Number):
            return self.create_dirac_impulse(restriction)

        else:
            raise ValueError("Unknown Datatype for cropping a numeric distribution, type is %s" % type(restriction))

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'numeric',
            'class': cls.__name__
        }

    def inst_to_json(self):
        return {
            'class': type(self).__name__,
            'settings': self.settings,
            'quantile': self._quantile.to_json() if self._quantile is not None else None
        }

    to_json = type_to_json

    @staticmethod
    def from_json(data):
        return Numeric(**data['settings']).set(QuantileDistribution.from_json(data['quantile']))

    @classmethod
    def type_from_json(cls, data):
        return cls

    def insert_convex_fragments(self, left: ContinuousSet or None, right: ContinuousSet or None,
                                number_of_samples: int):
        """Insert fragments of distributions on the right and left part of this distribution. This should only be used
        to create a convex hull around the JPTs domain which density is never 0.

        :param right: The right (lower) interval to add on if needed and None else
        :param left: The left (upper) interval to add on if needed and None else
        :param number_of_samples: The number of samples to use as basis for the weight
        """

        # create intervals used in the new distribution
        points = [-float("inf")]

        if left:
            points.extend([left.lower, left.upper])

        if right:
            points.extend([right.lower, right.upper])

        points.append(float("inf"))

        intervals = [ContinuousSet(a, b) for a, b in zip(points[:-1], points[1:])]

        valid_arguments = [e for e in [left, right] if e is not None]
        number_of_intervals = len(valid_arguments)
        functions = [ConstantFunction(0.)]

        for idx, interval in enumerate(intervals[1:]):
            prev_value = functions[idx].eval(interval.lower)

            if interval in valid_arguments:
                functions.append(
                    LinearFunction.from_points(
                        (interval.lower, prev_value),
                        (interval.upper, prev_value + 1 / number_of_intervals)
                    )
                )
            else:
                functions.append(ConstantFunction(prev_value))

        cdf = PiecewiseFunction()
        cdf.intervals = intervals
        cdf.functions = functions
        quantile = QuantileDistribution.from_cdf(cdf)
        self._quantile = QuantileDistribution.merge(
            [self._quantile, quantile],
            [
                1 - (1 / (2 * number_of_samples)),
                1 / (2 * number_of_samples)
            ]
        )

    def moment(self, order=1, center=0):
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param center: The constant (c) to subtract in the basis of the exponent
        """
        # We have to catch the special case in which the
        # PDF is an impulse function
        if self.pdf.is_impulse():
            if order == 1:
                return self.pdf.gt(0).min
            elif order >= 2:
                return 0
        result = 0
        for interval, function in zip(self.pdf.intervals[1:-1], self.pdf.functions[1:-1]):
            interval_ = self.value2label(interval)

            function_value = function.value * interval.range() / interval_.range()
            result += (
                (pow(interval_.upper - center, order+1) - pow(interval_.lower - center, order+1))
            ) * function_value / (order + 1)
        return result

    def plot(self, title=None, fname=None, xlabel='value', directory='/tmp', pdf=False, view=False, **kwargs):
        '''
        Generates a plot of the piecewise linear function representing
        the variable's cumulative distribution function

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file to be stored
        :type fname:        str
        :param xlabel:      the label of the x-axis
        :type xlabel:       str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :return:            None
        '''
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not view:
            plt.ioff()

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"CDF of {self._cl}"}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('%')
        ax.set_ylim(-.1, 1.1)

        if len(self.cdf.intervals) == 2:
            std = abs(self.cdf.intervals[0].upper) * .1
        else:
            std = ifnot(np.std([i.upper - i.lower for i in self.cdf.intervals[1:-1]]),
                        self.cdf.intervals[1].upper - self.cdf.intervals[1].lower) * 2

        # add horizontal line before first interval of distribution
        X = np.array([self.cdf.intervals[0].upper - std])

        for i, f in zip(self.cdf.intervals[:-1], self.cdf.functions[:-1]):
            if isinstance(f, ConstantFunction):
                X = np.append(X, [np.nextafter(i.upper, i.upper - 1), i.upper])
            else:
                X = np.append(X, i.upper)

        # add horizontal line after last interval of distribution
        X = np.append(X, self.cdf.intervals[-1].lower + std)
        X_ = np.array([self.labels[x] for x in X])
        Y = np.array(self.cdf.multi_eval(X))
        ax.plot(X_,
                Y,
                color='cornflowerblue',
                linestyle='dashed',
                label='Piecewise linear CDF from bounds',
                linewidth=2,
                markersize=12)

        bounds = np.array([i.upper for i in self.cdf.intervals[:-1]])
        bounds_ = np.array([self.labels[b] for b in bounds])
        ax.scatter(bounds_,
                   np.asarray(self.cdf.multi_eval(bounds)),
                   color='orange',
                   marker='o',
                   label='Piecewise Function limits')

        ax.legend(loc='upper left', prop={'size': 8})  # do we need a legend with only one plotted line?
        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


class ScaledNumeric(Numeric):
    '''
    Scaled numeric distribution represented by mean and variance.
    '''

    scaler = None

    def __init__(self, **settings):
        super().__init__(**settings)

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'scaled-numeric',
            'class': cls.__name__,
            'scaler': cls.scaler.to_json()
        }

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        clazz = NumericType(data['class'], None)
        clazz.scaler = DataScaler.from_json(data['scaler'])
        clazz.values = DataScalerProxy(clazz.scaler)
        clazz.labels = DataScalerProxy(clazz.scaler, True)
        return clazz

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(QuantileDistribution.from_json(data['quantile']))


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode
class Multinomial(Distribution):
    '''
    Abstract supertype of all symbolic domains and distributions.
    '''

    values: OrderedDictProxy = None
    labels: OrderedDictProxy = None

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Multinomial) or type(self) is Multinomial:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')
        self._params: np.ndarray = None
        self.to_json: FunctionType = self.inst_to_json

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[Any, Set]) -> Union[Any, Set]:
        if type(value) is set:
            return {cls.labels[v] for v in value}
        else:
            return cls.labels[value]

    # noinspection DuplicatedCode
    @classmethod
    def label2value(cls, label: Union[Any, Set]) -> Union[Any, Set]:
        if type(label) is set:
            return {cls.values[l_] for l_ in label}
        else:
            return cls.values[label]

    @classmethod
    def pfmt(cls, max_values=10, labels_or_values='labels') -> str:
        '''
        Returns a pretty-formatted string representation of this class.

        By default, a set notation with value labels is used. By setting
        ``labels_or_values`` to ``"values"``, the internal value representation
        is used. If the domain comprises more than ``max_values`` values,
        the middle part of the list of values is abbreviated by "...".
        '''
        if labels_or_values not in ('labels', 'values'):
            raise ValueError('Illegal Value for "labels_or_values": Expected one out of '
                             '{"labels", "values"}, got "%s"' % labels_or_values)
        return '%s = {%s}' % (cls.__name__, ', '.join(mapstr(cls.values.values()
                                                             if labels_or_values == 'values'
                                                             else cls.labels.values(), limit=max_values)))

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls):
        return len(cls.values)

    def __contains__(self, item):
        return item in self.values

    @classmethod
    def equiv(cls, other):
        if not issubclass(other, Multinomial):
            return False
        return cls.__name__ == other.__name__ and cls.labels == other.labels and cls.values == other.values

    def __getitem__(self, value):
        return self.p([value])

    def __setitem__(self, label, p):
        self._params[self.values[label]] = p

    def __eq__(self, other):
        return type(self).equiv(type(other)) and (self.probabilities == other.probabilities).all()

    def __str__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{";".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self.probabilities)])}]>'

    def __repr__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'\n{self._cl}<p=[\n{sepcomma.join([f" {v}={p:.3}" for v, p in zip(self.labels, self.probabilities)])}]>;'

    def sorted(self):
        return sorted([
            (p, l) for p, l in zip(self._params, self.labels.values())],
            key=itemgetter(0),
            reverse=True
        )

    def items(self):
        '''Return a list of (probability, label) pairs representing this distribution.'''
        return [(p, l) for p, l in zip(self._params, self.labels.values())]

    def copy(self):
        return type(self)(**self.settings).set(params=self._params)

    def p(self, labels):
        if not isinstance(labels, (set, list, tuple, np.ndarray)):
            raise TypeError('Argument must be iterable (got %s).' % type(labels))
        return self._p(self.values[label] for label in labels)

    def _p(self, values):
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) for v in i1):
            raise TypeError('All arguments must be integers.')
        return sum(self._params[v] for v in i2)

    def create_dirac_impulse(self, value):
        result = self.copy()
        result._params = np.zeros(shape=result.n_values, dtype=np.float64)
        result._params[result.values[result.labels[value]]] = 1
        return result

    def _sample(self, n: int) -> Iterable[Any]:
        '''Returns ``n`` sample `values` according to their respective probability'''
        return wsample(list(self.values.values()), self._params, n)

    def _sample_one(self) -> Any:
        '''Returns one sample `value` according to its probability'''
        return wchoice(list(self.values.values()), self._params)

    def _expectation(self):
        '''Returns the value with the highest probability for this variable'''
        return max([(v, p) for v, p in zip(self.values.values(), self._params)], key=itemgetter(1))[0]

    def expectation(self) -> set:
        """
        For symbolic variables the expectation is equal to the mpe.
        :return: The set of all most likely values
        """
        return self.mpe()[1]

    def mpe(self) -> (float, set):
        """
        Calculate the most probable configuration of this distribution.
        :return: The likelihood of the mpe as float and the mpe itself as Set
        """
        _max = max(self.probabilities)
        return _max, set([label for label, p in zip(self.labels.values(), self.probabilities) if p == _max])

    def kl_divergence(self, other):
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        result = 0
        for v in range(self.n_values):
            result += self._params[v] * abs(self._params[v] - other._params[v])
        return result

    def _crop(self, incl_values=None, excl_values=None):
        if incl_values and excl_values:
            raise Unsatisfiability("Admissible and inadmissible values must be disjoint.")
        posterior = self.copy()
        if incl_values:
            posterior._params[...] = 0
            for i in incl_values:
                posterior._params[int(i)] = self._params[int(i)]
        if excl_values:
            for i in excl_values:
                posterior._params[int(i)] = 0
        try:
            params = normalized(posterior._params)
        except ValueError:
            raise Unsatisfiability('All values have zero probability [%s].' % type(self).__name__)
        else:
            posterior._params = np.array(params)
        return posterior

    def crop(self, restriction: set or List or str):
        """
        Apply a restriction to this distribution such that all values are in the given set.

        :param restriction: The values to remain
        :return: Copy of self that is consistent with the restriction
        """
        if not isinstance(restriction, set) and not isinstance(restriction, list):
            return self.create_dirac_impulse(restriction)

        result = self.copy()
        for idx, value in enumerate(result.labels.keys()):
            if value not in restriction:
                result._params[idx] = 0

        if sum(result._params) == 0:
            raise Unsatisfiability('All values have zero probability [%s].' % type(result).__name__)
        else:
            result._params = result._params / sum(result._params)
        return result

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Multinomial':
        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(len(data))):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def set(self, params: Iterable[numbers.Real]) -> 'Multinomial':
        if len(self.values) != len(params):
            raise ValueError('Number of values and probabilities must coincide.')
        self._params = np.array(params, dtype=np.float64)
        return self

    def update(self, dist: 'Multinomial', weight: numbers.Real) -> 'Multinomial':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight
        return self

    @staticmethod
    def merge(distributions: Iterable['Multivariate'], weights: Iterable[numbers.Real]) -> 'Multinomial':
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError('Weights must sum to 1 (but is %s).' % sum(weights))
        params = np.zeros(distributions[0].n_values)
        for d, w in zip(distributions, weights):
            params += d.probabilities * w
        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')
        return type(distributions[0])().set(params)

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'symbolic',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values())
        }

    def inst_to_json(self):
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        return SymbolicType(data['class'], data['labels'])

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(data['params'])

    def is_dirac_impulse(self):
        for p in self._params:
            if p == 1:
                return True
        return False

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 1 if this is a dirac impulse, number of parameters else
        """
        if self.is_dirac_impulse():
            return 1
        return len(self._params)

    @classmethod
    def list2set(cls, values: List[str]) -> Set[str]:
        """
        Convert a list to a set.
        """
        return set(values)

    def plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, horizontal=False, max_values=None):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file to be stored
        :type fname:        str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :type horizontal:   bool
        :param max_values:  maximum number of values to plot
        :type max_values:   int
        :return:            None
        '''
        # Only save figures, do not show
        if not view:
            plt.ioff()

        max_values = min(ifnone(max_values, len(self.labels)), len(self.labels))

        labels = list(sorted(list(enumerate(self.labels.values())),
                             key=lambda x: self._params[x[0]],
                             reverse=True))[:max_values]
        labels = project(labels, 1)
        probs = list(sorted(self._params, reverse=True))[:max_values]

        vals = [re.escape(str(x)) for x in labels]

        x = np.arange(max_values)  # the label locations
        # width = .35  # the width of the bars
        err = [.015] * max_values

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"Distribution of {self._cl}"}')
        if horizontal:
            ax.barh(x, probs, xerr=err, color='cornflowerblue', label='P', align='center')
            ax.set_xlabel('%')
            ax.set_yticks(x)
            ax.set_yticklabels(vals)
            ax.invert_yaxis()
            ax.set_xlim(left=0., right=1.)

            for p in ax.patches:
                h = p.get_width() - .09 if p.get_width() >= .9 else p.get_width() + .03
                plt.text(h, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}',
                         fontsize=10, color='black', verticalalignment='center')
        else:
            ax.bar(x, probs, yerr=err, color='cornflowerblue', label='P')
            ax.set_ylabel('%')
            ax.set_xticks(x)
            ax.set_xticklabels(vals)
            ax.set_ylim(bottom=0., top=1.)

            # print precise value labels on bars
            for p in ax.patches:
                h = p.get_height() - .09 if p.get_height() >= .9 else p.get_height() + .03
                plt.text(p.get_x() + p.get_width() / 2, h,
                         f'{p.get_height():.2f}',
                         rotation=90, fontsize=10, color='black', horizontalalignment='center')

        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------

class Bool(Multinomial):
    '''
    Wrapper class for Boolean domains and distributions.
    '''

    values = OrderedDictProxy([(False, 0), (True, 1)])
    labels = OrderedDictProxy([(0, False), (1, True)])

    def __init__(self, **settings):
        super().__init__(**settings)

    def set(self, params: Union[np.ndarray, numbers.Real]) -> 'Bool':
        if params is not None and not iterable(params):
            params = [1 - params, params]
        super().set(params)
        return self

    def __str__(self):
        if self.p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{",".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self._params)])}]>'

    def __setitem__(self, v, p):
        if not iterable(p):
            p = np.array([p, 1 - p])
        super().__setitem__(v, p)


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode
class Integer(Distribution):

    lmin = None
    lmax = None
    vmin = None
    vmax = None
    values = None
    labels = None

    OPEN_DOMAIN = 'open_domain'
    AUTO_DOMAIN = 'auto_domain'

    SETTINGS = edict(Distribution.SETTINGS) + {
        OPEN_DOMAIN: False,
        AUTO_DOMAIN: False
    }

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Integer) or type(self) is Integer:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')
        self._params: np.ndarray = None
        self.to_json: FunctionType = self.inst_to_json

    @classmethod
    def equiv(cls, other: Type):
        if not issubclass(other, Integer):
            return False
        return all((
            cls.__name__ == other.__name__,
            cls.labels == other.labels,
            cls.values == other.values,
            cls.lmin == other.lmin,
            cls.lmax == other.lmax,
            cls.vmin == other.vmin,
            cls.vmax == other.vmax
        ))

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'integer',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values()),
            'vmin': int(cls.vmin),
            'vmax': int(cls.vmax),
            'lmin': int(cls.lmin),
            'lmax': int(cls.lmax)
        }

    to_json = type_to_json

    def inst_to_json(self):
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    @classmethod
    def list2set(cls, bounds: List[int]) -> Set[int]:
        '''
        Convert a 2-element list specifying a lower and an upper bound into a
        integer set containing the admissible values of the corresponding interval
        '''
        if not len(bounds) == 2:
            raise ValueError('Argument list must have length 2, got length %d.' % len(bounds))
        if bounds[0] < cls.lmin or bounds[1] > cls.lmax:
            raise ValueError(f'Argument must be in [%d, %d].' % (cls.lmin, cls.lmax))
        return set(range(bounds[0], bounds[1] + 1))

    @staticmethod
    def type_from_json(data):
        return IntegerType(data['class'], data['lmin'], data['lmax'])

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(data['params'])

    def copy(self):
        result = type(self)(**self.settings)
        result._params = np.array(self._params)
        return result

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls):
        return cls.lmax - cls.lmin + 1

    # noinspection DuplicatedCode
    @classmethod
    def value2label(cls, value: Union[Any, Set]) -> Union[Any, Set]:
        if type(value) is set:
            # if cls.open_domain:
            #     return {cls.labels[v] for v in value if v in cls.labels}
            # else:
            return {cls.labels[v] for v in value}
        else:
            return cls.labels[value]

    # noinspection DuplicatedCode
    @classmethod
    def label2value(cls, label: Union[Any, Set]) -> Union[Any, Set]:
        if type(label) is set:
            return {cls.values[l_] for l_ in label}
        else:
            return cls.values[label]

    def _sample(self, n) -> Iterable:
        return wsample(list(self.values.values()), weights=self.probabilities, k=n)

    def _sample_one(self) -> int:
        return wchoice(list(self.values.values()), weights=self.probabilities)

    def p(self, labels: Union[int, Iterable[int]]):
        if not isinstance(labels, Iterable):
            labels = {labels}
        i1, i2 = tee(labels, 2)
        if not all(isinstance(v, numbers.Integral) and self.lmin <= v <= self.lmax for v in i1):
            if self.open_domain:
                return 0
            raise ValueError('Arguments must be in %s' % setstr(self.labels.values(), limit=5))
        return self._p(self.values[l] for l in labels)

    def _p(self, values: Union[int, Iterable[int]]):
        if not isinstance(values, Iterable):
            values = {values}
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) and self.vmin <= v <= self.vmax for v in i1):
            raise ValueError('Arguments must be in %s' % setstr(self.values.values(), limit=5))
        return sum(self._params[v] for v in i2)

    def expectation(self) -> numbers.Real:
        return sum(p * v for p, v in zip(self.probabilities, self.values))

    def _expectation(self) -> numbers.Real:
        return sum(p * v for p, v in zip(self.probabilities, self.labels))

    def variance(self) -> numbers.Real:
        e = self.expectation()
        return sum((l - e) ** 2 * p for l, p in zip(self.labels.values(), self.probabilities))

    def _variance(self) -> numbers.Real:
        e = self._expectation()
        return sum((v - e) ** 2 * p for v, p in zip(self.values.values(), self.probabilities))

    def mpe(self) -> Tuple[float, Set[int]]:
        p_max = max(self.probabilities)
        return p_max, {l for l, p in zip(self.labels.values(), self.probabilities) if p == p_max}

    def _mpe(self) -> Tuple[float, Set[int]]:
        p_max = max(self.probabilities)
        return p_max, {l for l, p in zip(self.values.values(), self.probabilities) if p == p_max}

    def crop(self, restriction: Union[Iterable, int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        return self._crop([self.label2value(l) for l in restriction])

    def _crop(self, restriction: Union[Iterable, int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        result = self.copy()
        try:
            params = normalized([
                p if v in restriction else 0
                for p, v in zip(self.probabilities, self.values.values())
            ])
        except ValueError:
            raise Unsatisfiability('Restriction unsatisfiable: probabilities must sum to 1.')
        else:
            return result.set(params=params)

    @staticmethod
    def merge(distributions: Iterable['Integer'], weights: Iterable[numbers.Real]) -> 'Integer':
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError('Weights must sum to 1 (but is %s).' % sum(weights))
        params = np.zeros(distributions[0].n_values)
        for d, w in zip(distributions, weights):
            params += d.probabilities * w
        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')
        return type(distributions[0])().set(params)

    def update(self, dist: 'Integer', weight: numbers.Real) -> 'Integer':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight
        return self

    def fit(self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None) -> 'Integer':
        if rows is None:
            rows = range(data.shape[0])
        data_ = np.array([self.label2value(int(data[row][col])) for row in rows], dtype=data.dtype)
        return self._fit(data_.reshape(-1, 1), None, 0)

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Integer':
        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(data.shape[0])):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def set(self, params: Iterable[numbers.Real]) -> 'Integer':
        if len(self.values) != len(params):
            raise ValueError('Number of values and probabilities must coincide.')
        self._params = np.array(params, dtype=np.float64)
        return self

    def __eq__(self, other):
        return type(self).equiv(type(other)) and (self.probabilities == other.probabilities).all()

    def __str__(self):
        if self._p is None:
            return f'<{type(self).__qualname__} p=n/a>'
        return f'<{self._cl} p=[{"; ".join([f"{v}: {p:.3f}" for v, p in zip(self.values, self.probabilities)])}]>'

    def __repr__(self):
        return str(self)

    def sorted(self):
        return sorted([(p, l) for p, l in zip(self._params, self.labels.values())],
                      key=itemgetter(0), reverse=True)

    def _items(self):
        '''Return a list of (probability, value) pairs representing this distribution.'''
        return [(p, v) for p, v in zip(self._params, self.values.values())]

    def items(self):
        '''Return a list of (probability, value) pairs representing this distribution.'''
        return [(p, self.label2value(l)) for p, l in self._items()]

    def kl_divergence(self, other: 'Distribution'):
        if type(other) is not type(self):
            raise TypeError('Can only compute KL divergence between '
                            'distributions of the same type, got %s' % type(other))
        result = 0
        for v in range(self.n_values):
            result += self._params[v] * abs(self._params[v] - other._params[v])
        return result

    def number_of_parameters(self) -> int:
        return self._params.shape[0]

    def moment(self, order=1, c=0):
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param c: The constant to subtract in the basis of the exponent
        """
        result = 0
        for value, probability in zip(self.labels.values(), self._params):
            result += pow(value - c, order) * probability
        return result

    def plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, horizontal=False, max_values=None):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file to be stored
        :type fname:        str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :type horizontal:   bool
        :param max_values:  maximum number of values to plot
        :type max_values:   int
        :return:            None
        '''
        # Only save figures, do not show
        if not view:
            plt.ioff()

        max_values = min(ifnone(max_values, len(self.labels)), len(self.labels))

        labels = list(sorted(list(enumerate(self.labels.values())),
                             key=lambda x: self._params[x[0]],
                             reverse=True))[:max_values]
        labels = project(labels, 1)
        probs = list(sorted(self._params, reverse=True))[:max_values]

        vals = [re.escape(str(x)) for x in labels]

        x = np.arange(max_values)  # the label locations
        # width = .35  # the width of the bars
        err = [.015] * max_values

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"Distribution of {self._cl}"}')
        if horizontal:
            ax.barh(x, probs, xerr=err, color='cornflowerblue', label='P', align='center')
            ax.set_xlabel('P')
            ax.set_yticks(x)
            ax.set_yticklabels(vals)
            ax.invert_yaxis()
            ax.set_xlim(left=0., right=1.)

            for p in ax.patches:
                h = p.get_width() - .09 if p.get_width() >= .9 else p.get_width() + .03
                plt.text(h, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}',
                         fontsize=10, color='black', verticalalignment='center')
        else:
            ax.bar(x, probs, yerr=err, color='cornflowerblue', label='P')
            ax.set_ylabel('P')
            ax.set_xticks(x)
            ax.set_xticklabels(vals)
            ax.set_ylim(bottom=0., top=1.)

            # print precise value labels on bars
            for p in ax.patches:
                h = p.get_height() - .09 if p.get_height() >= .9 else p.get_height() + .03
                plt.text(p.get_x() + p.get_width() / 2, h,
                         f'{p.get_height():.2f}',
                         rotation=90, fontsize=10, color='black', horizontalalignment='center')

        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming
def SymbolicType(name: str, labels: List[Any]) -> Type:
    if len(labels) < 1:
        raise ValueError('At least one value is needed for a symbolic type.')
    if len(set(labels)) != len(labels):
        duplicates = [item for item, count in Counter(labels).items() if count > 1]
        raise ValueError('List of labels  contains duplicates: %s' % duplicates)
    t = type(name, (Multinomial,), {})
    t.values = OrderedDictProxy([(lbl, int(val)) for val, lbl in zip(range(len(labels)), labels)])
    t.labels = OrderedDictProxy([(int(val), lbl) for val, lbl in zip(range(len(labels)), labels)])
    return t


# noinspection PyPep8Naming
def NumericType(name: str, values: Iterable[float]) -> Type:
    t = type(name, (ScaledNumeric,), {})
    if values is not None:
        values = np.array(list(none2nan(values)))
        if (~np.isfinite(values)).any():
            raise ValueError('Values contain nan or inf.')
        t.scaler = DataScaler(values)
        t.values = DataScalerProxy(t.scaler, inverse=False)
        t.labels = DataScalerProxy(t.scaler, inverse=True)
    return t


# noinspection PyPep8Naming
def IntegerType(name: str, lmin: int, lmax: int) -> Type:
    if lmin > lmax:
        raise ValueError('Min label is greater tham max value: %s > %s' % (lmin, lmax))
    t = type(name, (Integer,), {})
    t.values = OrderedDictProxy([(l, v) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.labels = OrderedDictProxy([(v, l) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.lmin = lmin
    t.lmax = lmax
    t.vmin = 0
    t.vmax = lmax - lmin
    return t


# ----------------------------------------------------------------------------------------------------------------------

_DISTRIBUTION_TYPES = {
    'numeric': Numeric,
    'scaled-numeric': ScaledNumeric,
    'symbolic': Multinomial,
    'integer': Integer
}

_DISTRIBUTIONS = {
    'Numeric': Numeric,
    'ScaledNumeric': ScaledNumeric,
    'Multinomial': Multinomial,
    'Integer': Integer
}
