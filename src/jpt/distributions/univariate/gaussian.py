import copy
import math
import numbers

import numpy as np
from dnutils import ifnone, first
from dnutils.stats import Gaussian as Gaussian_, _matshape
from matplotlib import pyplot as plt
from scipy.stats import norm, multivariate_normal

from jpt.base.utils import save_plot


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
        return multivariate_normal(self._mean, self._cov, allow_singular=True).pdf

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
                    return False
        return True

    def plot(
            self,
            dim: int = 2,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            pdf: bool = False,
            view: bool = False,
            save: bool = False
    ) -> None:
        """

        :param dim: if the distribution has a higher dimension than 1, the parameter `dim` allows to decide whether the
        plot is a 2-dimensional heatmap or a 3-dimensional surface plot
        :param title: the title of the plot
        :param fname: the name of the file if the plot is saved
        :param directory: the location of the file if the plot is stored
        :param pdf: whether to save the file as a pdf, if False, the file is stored as png. Default is False
        :param view: whether to show the generated plot
        :param save: whether to save the plot to a file
        :return:
        """
        fig, ax = plt.subplots(num=1, clear=True)

        if self.dim == 1:
            m = self.mean[0]
            v = self.cov[0][0]
            r = abs(2.5 * v)
            X = np.linspace(m - r, m + r, int(m + r - (m - r)))
            Z = np.array(self.pdf(X))

            ax.plot(X, Z)
            ax.set_xlabel('$x$')
            ax.set_ylabel(r'$p(x|\mu,\sigma)$')
        else:
            cmap = 'BuPu'  # viridis, Blues, PuBu, 0rRd, BuPu

            x = np.linspace(-2, 2, 30)
            y = np.linspace(-2, 2, 30)
            X, Y = np.meshgrid(x, y)

            xy = np.column_stack([X.flat, Y.flat])
            Z = self.pdf(xy)
            Z = Z.reshape(X.shape)

            if dim == 2:
                # generate heatmap
                c = ax.pcolormesh(X, Y, Z, cmap=cmap)

                # setting the limits of the plot to the limits of the data
                # ax.axis([xmin, xmax, ymin, ymax])
                fig.colorbar(c, ax=ax)

                ax.set_xlabel('$x$')
                ax.set_ylabel(r'$p(x|\mu,\sigma)$')
            else:
                # generate 3d-surface plot
                ax = plt.axes(projection='3d')
                ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')

                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                ax.set_zlabel(r'$p(x|\mu,\sigma)$')

        plt.title(title or 'Gaussian Distribution:\n$\mu=$' + str(self._mean) + ',\n$\sigma=$' + str(self._cov))

        if save:
            save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()
