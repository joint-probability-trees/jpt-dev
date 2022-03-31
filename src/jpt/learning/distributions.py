'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
from collections import OrderedDict, deque
from itertools import tee

from sklearn.preprocessing import StandardScaler

from jpt.base.utils import classproperty, save_plot, Unsatisfiability, normalized

import copy
import math
import numbers
import os
import re
from operator import itemgetter

import dnutils
from dnutils import first, out, ifnone, stop, ifnot, project, mapstr, pairwise
from dnutils.stats import Gaussian as Gaussian_, _matshape

from scipy.stats import multivariate_normal, mvn, norm

import numpy as np
from numpy import iterable, isnan

import matplotlib.pyplot as plt

from jpt.base.constants import sepcomma
from jpt.base.sampling import wsample, wchoice

try:
    from ..base.intervals import R, ContinuousSet
    from ..base.quantiles import QuantileDistribution, LinearFunction
except ImportError:
    import pyximport
    pyximport.install()
    from ..base.intervals import R
    from ..base.quantiles import QuantileDistribution, LinearFunction


logger = dnutils.getlogger(name='GaussianLogger', level=dnutils.ERROR)


class Gaussian(Gaussian_):
    '''Extension of :class:`dnutils.stats.Gaussian`'''

    PRECISION = 1e-15

    def __init__(self, mean=None, cov=None, data=None, weights=None):
        '''Creates a new Gaussian distribution.

        :param mean:    the mean of the Gaussian
        :type mean:     float if multivariate else [float] if multivariate
        :param cov:     the covariance of the Gaussian
        :type cov:      float if multivariate else [[float]] if multivariate
        :param data:    if ``mean`` and ``cov`` are not provided, ``data`` may be a data set (matrix) from which the
                        parameters of the distribution are estimated.
        :type data:     [[float]]
        :param weights:  **[optional]** weights for the data points. The weight do not need to be normalized.
        :type weights:  [float]
        '''
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
        '''
        Computes the deviation of ``x`` in multiples of the standard deviation.

        :param x:
        :type x:
        :returns:
        '''
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
        '''
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
        '''
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
        '''Retract the a data point `x` with eight `w` from the Gaussian distribution.

        In case the data points are being kept in the distribution, it must actually exist and have the right
        weight associated. Otherwise, a ValueError will be raised.
        '''
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


class MultiVariateGaussian(Gaussian):

    def __init__(self, mean=None, cov=None, data=None, ignore=-6000000):
        '''A Multivariate Gaussian distribution that can be incrementally updated with new samples
        '''
        self.ignore = ignore
        super(MultiVariateGaussian, self).__init__(mean=mean, cov=cov, data=data)

    def cdf(self, intervals):
        '''Computes the CDF for a multivariate normal distribution.

        :param intervals: the boundaries of the integral
        :type intervals: list of matcalo.utils.utils.Interval
        '''
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], self.mean, self.cov))

    def pdf(self):
        var = multivariate_normal(mean=self.mean, cov=self.cov)
        return var.pdf

    @property
    def mvg(self):
        '''Computes the multivariate Gaussian distribution.
        '''
        return multivariate_normal(self.mean, self.cov, allow_singular=True)

    @property
    def dim(self):
        '''Returns the dimension of the distribution.
        '''
        if self._mean is None:
            raise ValueError('no dimensionality specified yet.')
        return len(self._mean) if hasattr(self.mean, '__len__') else 1

    @property
    def cov_(self):
        '''Returns the covariance matrix for prettyprinting (precision .2).
        '''
        return list([round(c, 2) for c in r] for r in self.cov) if hasattr(self.cov, '__len__') else round(self.cov, 2)

    @property
    def mean_(self):
        '''Returns the mean vector for prettyprinting (precision .2).
        '''
        return list([round(c, 2) for c in self.mean]) if hasattr(self.mean, '__len__') else round(self.mean, 2)

    def conditional(self, evidence):
        r'''Returns a distribution conditioning on the variables in ``evidence`` following the calculations described
        in `Conditional distributions <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions>`_,
        i.e., after determining the partitions of :math:`\mu`, i.e. :math:`\mu_{1}` and :math:`\mu_{2}` as well as
        the partitions of :math:`\Sigma`, i.e. :math:`\Sigma_{11}, \Sigma_{12}, \Sigma_{21} \text{ and } \Sigma_{22}`, we
        calculate the multivariate normal :math:`N(\overline\mu,\overline\Sigma)` using

        .. math::
            \overline\mu = \mu_{1} + \Sigma_{12}\Sigma_{22}^{-1}(a-\mu_{2})
            :label: mu

        .. math::
            \overline\Sigma = \Sigma_{11} + \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
            :label: sigma

        :param evidence: the variables the returned distribution conditions on (mapping indices to values or Intervals of values)
        :type evidence: dict
        '''
        indices = sorted(list(evidence.keys()))
        k = self.dim - len(indices)
        a = np.array([evidence[i] for i in indices])

        # sort conditioning variables to the bottom right corner of the covariance matrix
        order = [i for i in range(self.dim) if i not in indices] + indices
        sigma = self.cov[:, order][order]

        # determining the partitions of µ, i.e. µ_{1} and µ_{2}
        mu1 = self.mean[order][:k]
        mu2 = self.mean[order][k:]

        # determining the partitions of Σ, i.e. Σ_{11}, Σ_{12}, Σ_{21} and Σ_{22}
        sigma11 = sigma[:k, :k]
        sigma12 = sigma[k:, :k]
        sigma21 = sigma[:k, k:]
        sigma22 = sigma[k:, k:]

        # determine the inverse for matrix Σ_{22}
        sigma22inv = np.linalg.inv(sigma22)

        # µ' = µ_{1} + Σ_{12}Σ_{22}^{-1}(a-µ_{2})
        mu_ = mu1 + sigma12.dot(sigma22inv).dot((a-mu2).T).T

        # Σ' = Σ_{11} - Σ_{12}Σ{22}^{-1}Σ_{21}
        sigma_ = sigma11 - sigma12.dot(sigma22inv).dot(sigma21)
        return MultiVariateGaussian(mean=mu_, cov=sigma_)

    def plot(self):
        '''
        .. highlight:: python
        .. code-block:: python

            import sys
            self.dim==1
        '''
        if self.dim == 1:
            x = np.linspace(self.mean - 2 * self.cov, self.mean + 2 * self.cov, 500)
            y = multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

            fig1 = plt.figure(f'Distribution Leaf N{self.mean, self.cov}')
            ax = fig1.add_subplot(111)
            ax.plot(x, y)

        elif self.dim == 2:
            x = np.linspace(self.mean[0]-2*self.cov[0][0], self.mean[0]+2*self.cov[0][0], 500)
            y = np.linspace(self.mean[1]-2*self.cov[1][1], self.mean[1]+2*self.cov[1][1], 500)
            rv = multivariate_normal(self.mean, self.cov)
            pos = np.dstack((x, y))

            # plot
            fig2 = plt.figure(f'Distribution Leaf N{self.mean, self.cov}')
            ax2 = fig2.add_subplot(111)
            ax2.contourf(x, y, rv.pdf(pos))

        elif self.dim == 3:
            # grid and mvn
            x = np.linspace(self.mean[0]-2*self.cov[0][0], self.mean[0]+2*self.cov[0][0], 500)
            y = np.linspace(self.mean[1]-2*self.cov[1][1], self.mean[1]+2*self.cov[1][1], 500)
            rv = multivariate_normal(self.mean, self.cov)
            X, Y = np.meshgrid(x, y)
            pos = np.empty(X.shape + (2,))
            pos[:, :, 0] = X
            pos[:, :, 1] = Y

            # plot
            fig = plt.figure(f'Distribution Leaf N{self.mean, self.cov}')
            ax = fig.gca(projection='3d')
            ax.plot_surface(X, Y, rv.pdf(pos), cmap='viridis', linewidth=0)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.show()


# ----------------------------------------------------------------------------------------------------------------------


class Distribution:
    '''
    Abstract supertype of all domains and distributions
    '''
    values = None
    labels = None

    def __init__(self):
        # used for str and repr methods to be able to print actual type
        # of Distribution when created with jpt.variables.Variable
        self._cl = f'{self.__class__.__name__}' \
                   + (f' ({self.__class__.__mro__[1].__name__})'
                      if self.__module__ != __name__
                      else '')

    def __hash__(self):
        return hash((type(self), self.values, self.labels))

    def __getitem__(self, value):
        return self.p(value)

    def sample(self, n):
        raise NotImplementedError()

    def sample_one(self):
        raise NotImplementedError()

    def p(self, value):
        raise NotImplementedError()

    def _p(self, value):
        raise NotImplementedError()

    def expectation(self):
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

    def fit(self, data, rows=None, col=None):
        raise NotImplementedError()

    def kl_divergence(self, other):
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

    @staticmethod
    def from_json(data):
        clazz = _DISTRIBUTIONS.get(data['class'])
        if clazz is None:
            raise TypeError('Unknown distribution class: %s' % data['class'])
        return clazz.from_json(data)

    @staticmethod
    def type_from_json(data):
        typ = _DISTRIBUTION_TYPES.get(data['type'])
        if typ is None:
            raise TypeError('Unknown distribution type: %s' % data['type'])
        clazz = typ.type_from_json(data)
        if clazz.__name__ in _DISTRIBUTIONS:
            if not clazz.equiv(_DISTRIBUTIONS[clazz.__name__]):
                raise TypeError('Distribution class named "%s" is ambiguous.' % clazz.__name__)
        else:
            _DISTRIBUTIONS[clazz.__name__] = clazz
        return clazz


class DataScaler(StandardScaler):
    '''
    A numeric data transformation that represents data points in form of a translation
    by their mean and a scaling by their variance. After the transformation, the transformed
    input data have zero mean and unit variance.
    '''

    def __init__(self, data):
        super().__init__()
        if data is not None:
            self.fit(data.reshape(-1, 1))

    @property
    def mean(self):
        return self.mean_[0]

    @property
    def variance(self):
        return self.scale_[0]

    def __getitem__(self, item):
        if item in (np.NINF, np.PINF):
            return item
        return self.inverse_transform(np.array([item]))[0, 0]

    def to_json(self):
        return {'mean_': list(self.mean_),
                'scale_': list(self.scale_)}

    def __eq__(self, other):
        return self.mean == other.mean and self.variance == other.variance

    @staticmethod
    def from_json(data):
        scaler = DataScaler(None)
        scaler.mean_ = np.array(data['mean_'])
        scaler.scale_ = np.array(data['scale_'])
        scaler.n_features_in_ = 1
        return scaler


class Identity:
    '''
    Simple identity mapping that mimics the __getitem__ protocol of dicts.
    '''

    def __getitem__(self, item):
        return item

    __call__ = __getitem__

    def transformer(self):
        return lambda a: self[a]

    def __eq__(self, o):
        return type(o) is Identity

    def __hash__(self):
        return hash(Identity)


class DataScalerProxy:

    def __init__(self, datascaler, inverse=False):
        self.datascaler = datascaler
        self.inverse = inverse
        self.mean = self.datascaler.mean
        self.variance = self.datascaler.variance

    def __hash__(self):
        return hash((self.inverse, self.mean, self.variance))

    def __call__(self, arg):
        return self[arg]

    def __eq__(self, o):
        return (isinstance(o, DataScalerProxy) and
                self.datascaler == o.datascaler and
                self.inverse == o.inverse)

    def transformer(self):
        return lambda a: self[a]

    def __getitem__(self, item):
        if item in (np.NINF, np.PINF):
            return item
        if item is None or isnan(item):
            return 0
        return (self.datascaler.transform
                if not self.inverse
                else self.datascaler.inverse_transform)(np.array(item).reshape(1, -1))[0, 0]

    def transform(self, x):
        return self.datascaler.transform(x.reshape(-1, 1)).ravel()

    @staticmethod
    def keys():
        return R

    @staticmethod
    def values():
        return R

    def __repr__(self):
        return '<DataScalerProxy, mean=%f, var=%s>' % (self.datascaler.mean, self.datascaler.variance)


class Numeric(Distribution):
    '''
    Wrapper class for numeric domains and distributions.
    '''

    values = Identity()
    labels = Identity()

    def __init__(self, quantile=None):
        super().__init__()
        self._quantile = quantile
        self.to_json = self.inst_to_json

    def __str__(self):
        return self.cdf.pfmt()

    def __getitem__(self, value):
        return self.p(value)

    def __eq__(self, o):
        if not issubclass(type(o), Numeric):
            return False
        return type(o).equiv(type(self))  and self._quantile == o._quantile

    def __hash__(self):
        return hash((type(self), self.values, self.labels, self._quantile))

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

    def sample(self, n):
        raise NotImplemented()

    def sample_one(self):
        raise NotImplemented()

    def _expectation(self):
        e = 0
        singular = True  # In case the CDF is jump fct the expectation is where the jump happens
        for i, f in zip(self.cdf.intervals, self.cdf.functions):
            if i.lower == np.NINF or i.upper == np.PINF:
                continue
            e += (self.cdf.eval(i.upper) - self.cdf.eval(i.lower)) * (i.upper + i.lower) / 2
            singular = False
        return e if not singular else i.lower

    def expectation(self):
        return self.labels[self._expectation()]

    def mpe(self):
        return max([(interval, function)
                    for interval, function in zip(self.cdf.intervals, self.cdf.functions)],
                   key=lambda x: x[1].m if isinstance(x[1], LinearFunction) else 0)[0]

    def fit(self, data, rows=None, col=None, **params):
        self._quantile = QuantileDistribution(**params)
        self._quantile.fit(data, rows=rows, col=col)
        return self

    def _p(self, value):
        if isinstance(value, numbers.Number) and np.isinf(self.pdf.eval(value)):
            return 0
        elif value.lower == value.upper and not value.isempty() and np.isinf(self.pdf.eval(value.lower)):
            return 1
        return ((self.cdf.eval(value.upper) if value.upper != np.PINF else 1.) -
                (self.cdf.eval(value.lower) if value.lower != np.NINF else 0.))

    def p(self, value):
        pass

    def kl_divergence(self, other):
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
        dist = type(self)(quantile=self._quantile.copy())
        dist.values = copy.copy(self.values)
        dist.labels = copy.copy(self.labels)
        return dist

    @staticmethod
    def merge(distributions, weights):
        if not all(distributions[0].__class__ == d.__class__ for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        return type(distributions[0])(QuantileDistribution.merge(distributions, weights))

    def update(self, dist, weight):
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

    def crop(self, interval):
        interval_ = interval.copy()
        interval_.lower = self.values[interval.lower]
        interval_.upper = self.values[interval.upper]
        return self._crop(interval_)

    @classmethod
    def type_to_json(cls):
        return {'type': 'numeric',
                'class': cls.__name__}

    def inst_to_json(self):
        return {'class': type(self).__name__,
                'quantile': self._quantile.to_json() if self._quantile is not None else None}

    to_json = type_to_json

    @staticmethod
    def from_json(data):
        return Numeric(quantile=QuantileDistribution.from_json(data['quantile']))

    @classmethod
    def type_from_json(cls, data):
        return cls

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
        std = ifnot(np.std([i.upper - i.lower for i in self.cdf.intervals[1:-1]]),
                    self.cdf.intervals[1].upper - self.cdf.intervals[1].lower) * 2
        bounds = np.array([self.cdf.intervals[0].upper - std / 2] +
                          [v.upper for v in self.cdf.intervals[:-2]] +
                          [self.cdf.intervals[-1].lower] +
                          [self.cdf.intervals[-1].lower + std / 2])

        bounds_ = np.array([self.labels[b] for b in bounds])

        ax.plot(bounds_,
                self.cdf.multi_eval(bounds),
                color='cornflowerblue',
                linestyle='dashed',
                label='Piecewise linear CDF from bounds',
                linewidth=2,
                markersize=12)

        ax.scatter(bounds_[1:-1],
                   self.cdf.multi_eval(bounds[1:-1]),
                   color='orange',
                   marker='o',
                   label='Piecewise Function limits')
        ax.legend()  # do we need a legend with only one plotted line?
        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()


class ScaledNumeric(Numeric):
    '''
    Scaled numeric distribution represented by mean and variance.
    '''

    scaler = None

    def __init__(self, quantile=None):
        super().__init__(quantile=quantile)

    @classmethod
    def type_to_json(cls):
        return {'type': 'scaled-numeric',
                'class': cls.__name__,
                'scaler': cls.scaler.to_json()}

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
        return cls(quantile=QuantileDistribution.from_json(data['quantile']))


class HashableOrderedDict(OrderedDict):
    '''
    Ordered dict that can be hashed.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, arg):
        return self[arg]

    def __hash__(self):
        return hash((tuple(self.values()), tuple(self.keys())))


class OrderedDictProxy:
    '''
    This is a proxy class that mimics the interface of a regular dict
    without inheriting from dict.
    '''

    def __init__(self, *args, **kwargs):
        self._dict = HashableOrderedDict(*args, **kwargs)
        self.values = self._dict.values
        self.keys = self._dict.keys

    def __repr__(self):
        return '<OrderedDictProxy #%d values=[%s]>' % (len(self), ';'.join(mapstr(self.keys())))

    def transformer(self):
        return lambda a: self._dict[a]

    def __getitem__(self, arg):
        return self._dict[arg]

    def __iter__(self):
        return iter(self._dict)

    def __call__(self, arg):
        return self._dict[arg]

    def __hash__(self):
        return hash((tuple(self._dict.values()), tuple(self._dict.keys())))

    def __len__(self):
        return len(self._dict)

    def __eq__(self, other):
        if not isinstance(other, OrderedDictProxy):
            return False
        for self_, other_ in zip(self._dict.items(), other._dict.items()):
            if self_[0] != other_[0] or self_[1] != other_[1]:
                return False
        return True


class Multinomial(Distribution):
    '''
    Abstract supertype of all symbolic domains and distributions.
    '''

    values = None
    labels = None

    def __init__(self, params=None):
        super().__init__()
        if params is not None:
            if not iterable(params):
                raise ValueError(f'Probabilities must be an iterable with {len(self.values)} elements, got {params}')
            if len(self.values) != len(params):
                raise ValueError('Number of values and probabilities must coincide.')
            self._params = np.array(params)
        else:
            self._params = None
        if not issubclass(type(self), Multinomial) or type(self) is Multinomial:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')
        self.to_json = self.inst_to_json

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(self):
        return len(self.values)

    def __contains__(self, item):
        return item in self.values

    @classmethod
    def equiv(cls, other):
        if not issubclass(other, Multinomial):
            return False
        return cls.__name__ == other.__name__ and cls.labels == other.labels and cls.values == other.values

    def __getitem__(self, value):
        return self.p(value)

    def __setitem__(self, label, p):
        self._params[self.values[label]] = p

    def __eq__(self, other):
        return type(self) is type(other) and (self.probabilities == other.probabilities).all()

    def __hash__(self):
        return hash((Multinomial, self.values.values(), self.labels.values(), self._params))

    def __str__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{";".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self.probabilities)])}]>'

    def __repr__(self):
        if self._p is None:
            return f'{self._cl}<p=n/a>'
        return f'\n{self._cl}<p=[\n{sepcomma.join([f" {v}={p:.3}" for v, p in zip(self.labels, self.probabilities)])}]>;'

    def sorted(self):
        return sorted([(p, l) for p, l in zip(self._params, self.labels.values())],
                      key=itemgetter(0), reverse=True)

    def copy(self):
        return type(self)(params=self._params)

    def p(self, labels):
        if not iterable(labels):
            raise TypeError('Argument must be iterable (got %s).' % type(labels))
        return self._p(self.values[label] for label in labels)

    def _p(self, values):
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) for v in i1):
            raise TypeError('All arguments must be integers.')
        return sum(self._params[v] for v in i2)

    def sample(self, n):
        '''Returns ``n`` sample `values` according to their respective probability'''
        return wsample(self.values, self._params, n)

    def sample_one(self):
        '''Returns one sample `value` according to its probability'''
        return wchoice(self.values, self._params)

    def sample_labels(self, n):
        '''Returns ``n`` sample `labels` according to their respective probability'''
        return [self.labels[i] for i in wsample(self.values, self._params, n)]

    def sample_one_label(self):
        '''Returns one sample `label` according to its probability'''
        return self.labels[wchoice(self.values, self._params)]

    def _expectation(self):
        '''Returns the value with the highest probability for this variable'''
        return max([(v, p) for v, p in zip(self.values.values(), self._params)], key=itemgetter(1))[0]

    def expectation(self):
        return self.labels[self._expectation()]

    def mpe(self):
        return self.expectation()

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

    def crop(self, incl_values=None, excl_values=None):
        '''
        Compute the posterior of the multinomial distribution.

        ``values`` and ``exclude`` are indices of the values (labels) that are admitted and/or excluded.
        '''
        incl_values_ = [self.values[v] for v in incl_values] if incl_values is not None else None
        excl_values_ = [self.values[v] for v in excl_values] if excl_values is not None else None
        return self._crop(incl_values_, excl_values_)

    def fit(self, data, rows=None, col=None):
        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(len(data))):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def update(self, dist, weight):
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight
        return self

    @staticmethod
    def merge(distributions, weights):
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')
        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError('Weights must sum to 1 (but is %s).' % sum(weights))
        params = np.zeros(distributions[0].n_values)
        for d, w in zip(distributions, weights):
            params += d.probabilities * w
        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')
        return type(distributions[0])(params=params)

    @classmethod
    def type_to_json(cls):
        return {'type': 'symbolic',
                'class': cls.__name__,
                'labels': list(cls.labels.values())}

    def inst_to_json(self):
        return {'class': type(self).__name__,
                'params': list(self._params)}

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        return SymbolicType(data['class'], data['labels'])

    @classmethod
    def from_json(cls, data):
        return cls(data['params'])

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
            ax.barh(x, probs, xerr=err, color='cornflowerblue', label='%', align='center')
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
            ax.bar(x, probs, yerr=err, color='cornflowerblue', label='%')
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


class Bool(Multinomial):
    '''
    Wrapper class for Boolean domains and distributions.
    '''

    values = OrderedDictProxy([(False, 0), (True, 1)])
    labels = OrderedDictProxy([(0, False), (1, True)])

    def __init__(self, params=None):
        if params is not None and not iterable(params):
            params = [1 - params, params]
        super().__init__(params=params)

    def __str__(self):
        if self.p is None:
            return f'{self._cl}<p=n/a>'
        return f'{self._cl}<p=[{",".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self._params)])}]>'

    def __setitem__(self, v, p):
        if not iterable(p):
            p = np.array([p, 1 - p])
        super().__setitem__(v, p)


def SymbolicType(name, labels):
    t = type(name, (Multinomial,), {})
    t.values = OrderedDictProxy([(lbl, int(val)) for val, lbl in zip(range(len(labels)), labels)])
    t.labels = OrderedDictProxy([(int(val), lbl) for val, lbl in zip(range(len(labels)), labels)])
    return t


def NumericType(name, values):
    t = type(name, (ScaledNumeric,), {})
    if values is not None:
        t.scaler = DataScaler(values)
        t.values = DataScalerProxy(t.scaler, inverse=False)
        t.labels = DataScalerProxy(t.scaler, inverse=True)
    return t


_DISTRIBUTION_TYPES = {
    'numeric': Numeric,
    'scaled-numeric': ScaledNumeric,
    'symbolic': Multinomial
}

_DISTRIBUTIONS = {
    'Numeric': Numeric,
    'ScaledNumeric': ScaledNumeric,
    'Multinomial': Multinomial
}

