'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''

from dnutils import first

from scipy.stats import multivariate_normal, mvn, norm

import numpy as np

import matplotlib.pyplot as plt

from .univariate import Gaussian

try:
    from ..base.intervals import __module__
    from ..base.quantiles import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from ..base.intervals import R, ContinuousSet
    from ..base.quantiles import QuantileDistribution, LinearFunction


# ----------------------------------------------------------------------------------------------------------------------

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
