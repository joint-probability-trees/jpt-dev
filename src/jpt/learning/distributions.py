import copy
import math
import numbers
import os
from operator import itemgetter

from matplotlib.backends.backend_pdf import PdfPages

import dnutils
from dnutils import first, out, ifnone, stop
from dnutils.stats import Gaussian as Gaussian_, _matshape

from scipy.stats import multivariate_normal, mvn, norm

import numpy as np
from numpy import iterable

import matplotlib.pyplot as plt

from jpt.sampling import wsample, wchoice

logger = dnutils.getlogger(name='GaussianLogger', level=dnutils.ERROR)


class Gaussian(Gaussian_):
    '''Extension of the Gaussian from dnutils.'''

    PRECISION = 1e-15

    def __init__(self, mean=None, cov=None, data=None, weights=None):
        '''
        Creates a new Gaussian distribution.
        :param mean:    the mean of the Gaussian. May be a scalar (univariante) or an array (multivariate).
        :param cov:     the covariance of the Gaussian. May be a scalar (univariate) or a matrix (multivariate).
        :param data:    if ``mean`` and ``cov`` are not provided, ``data`` may be a data set (matrix) from which
                        the parameters of the distribution are estimated.
        :param weight:  [optional] weights for the data points. The weight do not need to be normalized.
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
        :return:
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
        '''Update the distribution with new data points given in `data`.'''
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
        '''update the Gaussian distribution with a new data point `x` and weight `w`.'''
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
    """A Multivariate Gaussian distribution that can be incrementally updated with new samples
    """

    def __init__(self, mean=None, cov=None, data=None, ignore=-6000000):
        self.ignore = ignore
        super(MultiVariateGaussian, self).__init__(mean=mean, cov=cov, data=data)

    def cdf(self, intervals):
        """Computes the CDF for a multivariate normal distribution.

        :param intervals: the boundaries of the integral
        :type intervals: list of matcalo.utils.utils.Interval
        """
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], self.mean, self.cov))

    def pdf(self):
        var = multivariate_normal(mean=self.mean, cov=self.cov)
        return var.pdf

    @property
    def mvg(self):
        """Computes the multivariate Gaussian distribution.
        """
        return multivariate_normal(self.mean, self.cov, allow_singular=True)

    @property
    def dim(self):
        """Returns the dimension of the distribution.
        """
        if self._mean is None:
            raise ValueError('no dimensionality specified yet.')
        return len(self._mean) if hasattr(self.mean, '__len__') else 1

    @property
    def cov_(self):
        """Returns the covariance matrix for prettyprinting (precision .2).
        """
        return list([round(c, 2) for c in r] for r in self.cov) if hasattr(self.cov, '__len__') else round(self.cov, 2)

    @property
    def mean_(self):
        """Returns the mean vector for prettyprinting (precision .2).
        """
        return list([round(c, 2) for c in self.mean]) if hasattr(self.mean, '__len__') else round(self.mean, 2)

    def conditional(self, given):
        r"""Returns a distribution conditioning on the variables in ``given`` following the calculations described
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

        :param given: the variables the returned distribution conditions on (mapping indices to values or Intervals of values)
        :type given: dict
        """
        indices = sorted(list(given.keys()))
        k = self.dim - len(indices)
        a = np.array([given[i] for i in indices])

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
        """
        .. highlight:: python
        .. code-block:: python

            import sys
            self.dim==1
        """
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


class Distribution:

    values = None

    def __init__(self):
        pass

    def sample(self, n):
        raise NotImplementedError

    def sample_one(self):
        raise NotImplementedError

    def expectation(self):
        raise NotImplementedError

    def plot(self, name=None, directory='/tmp', pdf=False, view=False, **kwargs):
        '''Generates a plot of the distribution.

        :param name:        the name of the disribution (used for the generated filename)
        :type name:         str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :return:            None
        '''
        raise NotImplementedError

    def __hash__(self):
        return hash(self.values)


class Multinomial(Distribution):

    values = None

    def __init__(self, p):
        super().__init__()
        if not iterable(p):
            raise ValueError('Probabilities must be an iterable with at least 2 elements, got %s' % p)
        if len(self.values) != len(p):
            raise ValueError('Number of values and probabilities must coincide.')
        self._p = np.array(p)  # either probabilities or counters

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, p):
        self._p = p

    def sample(self, n):
        return wsample(self.values, self.p, n)

    def sample_one(self):
        return wchoice(self.values, self.p)

    def expectation(self):
        return max([(v, p) for v, p in zip(self.values, self.p)], key=itemgetter(1))

    def plot(self, name=None, directory='/tmp', pdf=False, view=False, horizontal=False):
        """

        :param name:        the name of the disribution (used for the generated filename)
        :type name:         str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :type horizontal:   bool
        :return:            None
        """
        # Only save figures, do not show
        if not view:
            plt.ioff()

        x = np.arange(len(self.values))  # the label locations
        width = 0.35  # the width of the bars
        err = [.02]*len(self.values)

        fig, ax = plt.subplots()
        ax.set_title(f'{name or f"Distribution of {self.__class__.__name__}"}')

        if horizontal:
            bars = ax.barh(x, self.p, width, xerr=err, color='cornflowerblue', label='%', align='center')

            ax.set_xlabel('%')
            ax.set_yticks(x)
            ax.set_yticklabels(self.values)

            ax.invert_yaxis()
            ax.set_xlim(left=0., right=1.)
        else:
            bars = ax.bar(x, self.p, width, yerr=err, color='cornflowerblue', label='%')

            ax.set_ylabel('%')
            ax.set_xticks(x)
            ax.set_xticklabels(self.values)

            ax.set_ylim(bottom=0., top=1.)

        ax.bar_label(bars, fmt='%.2f')
        fig.tight_layout()

        # save figure as PDF or PNG
        if pdf:
            logger.debug(f"Saving distributions plot to {os.path.join(directory, f'{name or self.__class__.__name__}.pdf')}")
            with PdfPages(os.path.join(directory, f'{name or self.__class__.__name__}.pdf')) as pdf:
                pdf.savefig(fig)
        else:
            logger.debug(f"Saving distributions plot to {os.path.join(directory, f'{name or self.__class__.__name__}.png')}")
            plt.savefig(os.path.join(directory, f'{name or self.__class__.__name__}.png'))

        if view:
            plt.show()

    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'Type mismatch. Can only add type {type(self)} but got {type(other)}.')
        m = type(self, ((self.p + other.p) / 2))
        m.values = self.values
        return m

    def __iadd__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'Type mismatch. Can only add type {type(self)} but got {type(other)}.')
        self._p = (self.p + other.p) / 2
        return self

    def __getitem__(self, value):
        return self.p[self.values.index(value)]

    def __setitem__(self, value, p):
        self.p[self.values.index(value)] = p

    def __eq__(self, other):
        return type(self) is type(other) and (self.p == other.p).all()

    def __hash__(self):
        return hash(f'{self.__class__.__name__}{self.values}')


class Histogram(Multinomial):

    values = Multinomial.values

    def __init__(self, p):
        super().__init__(p)
        self._d = sum(p)  # denominator (default 1)

    @Multinomial.p.getter
    def p(self):
        return self._p / self._d

    @property
    def d(self):
        return self._d

    def __setitem__(self, value, p):
        self._p[self.values.index(value)] = p
        self._d = sum(p)

    def expectation(self):
        return max([(v, p) for v, p in zip(self.values, self._p)], key=itemgetter(1))

    def plot(self, name=None, directory='/tmp', pdf=False, view=False, horizontal=False):
        """

        :param name:        the name of the disribution (used for the generated filename)
        :type name:         str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :type horizontal:   bool
        :return:            None
        """
        # Only save figures, do not show
        if not view:
            plt.ioff()

        x = np.arange(len(self.values))  # the label locations
        width = 0.35  # the width of the bars
        err = [.015]*len(self.values)

        fig, ax = plt.subplots()
        ax.set_title(f'{name or f"Distribution of {self.__class__.__name__}"}')

        if horizontal:
            ax2 = ax.twiny()

            bars = ax.barh(x, self.p, xerr=err, color='cornflowerblue', label='%', align='center')

            ax.set_xlabel('%')
            ax.set_yticks(x)
            ax.set_yticklabels(self.values)
            ax2.set_xlabel('count')

            ax.invert_yaxis()
            ax2.invert_yaxis()
            ax.set_xlim(left=0., right=1.)
            ax2.set_xlim(left=0., right=self.d)

        else:
            ax2 = ax.twinx()

            bars = ax.bar(x, self.p, yerr=err, color='cornflowerblue', label='%')

            ax.set_ylabel('%')
            ax.set_xticks(x)
            ax.set_xticklabels(self.values)
            ax2.set_ylabel('count')

            ax.set_ylim(bottom=0., top=1.)
            ax2.set_ylim(bottom=0., top=self.d)

        ax.bar_label(bars, labels=[f'{v} ({round(v/self.d*100, 2)}%)' for v in self._p], label_type='edge')
        fig.tight_layout()

        # save figure as PDF or PNG
        if pdf:
            logger.debug(f"Saving distributions plot to {os.path.join(directory, f'{name or self.__class__.__name__}.pdf')}")
            with PdfPages(os.path.join(directory, f'{name or self.__class__.__name__}.pdf')) as pdf:
                pdf.savefig(fig)
        else:
            logger.debug(f"Saving distributions plot to {os.path.join(directory, f'{name or self.__class__.__name__}.png')}")
            plt.savefig(os.path.join(directory, f'{name or self.__class__.__name__}.png'))

        if view:
            plt.show()

    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'Type mismatch. Can only add type {type(self)} but got {type(other)}')
        h = type(self)((self._p + other._p))
        # h.values = self.values
        return h

    def __iadd__(self, other):
        if type(self) is not type(other):
            raise TypeError(f'Type mismatch. Can only add type {type(self)} but got {type(other)}.')
        self._p += other._p
        self._d += other.d
        return self

    def __eq__(self, other):
        return super().__eq__(other) and self.d == other.d


class Bool(Multinomial):

    values = [True, False]

    def __init__(self, p):
        if not iterable(p):
            p = [p, 1 - p]
        super().__init__(p)

    def __getitem__(self, v):
        return self.p[v]

    def __setitem__(self, v, p):
        self.p[v] = p
        self.p[1 - v] = 1 - p
