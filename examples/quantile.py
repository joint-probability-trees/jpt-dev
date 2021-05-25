import pyximport
pyximport.install()

from jpt.base.intervals import ContinuousSet, INC, EXC


from jpt.base.quantiles import QuantileDistribution


import numpy as np
from matplotlib import pyplot as plt

from jpt.learning.distributions import Gaussian


def test_quantiles():
    gauss1 = Gaussian(-1, 2).sample(100)
    gauss2 = Gaussian(3, .5).sample(100)
    data = np.hstack([gauss1, gauss2])

    dist1 = QuantileDistribution(epsilon=1e-10)
    dist1.fit(gauss1)
    dist2 = QuantileDistribution(epsilon=1e-10)
    dist2.fit(gauss2)

    dist = QuantileDistribution.merge([dist1, dist2], [.5, .5])

    x = np.linspace(-7, 7, 500)
    cdf1 = dist1.cdf.multi_eval(x)
    cdf2 = dist2.cdf.multi_eval(x)
    cdf = dist.cdf.multi_eval(x)
    # pdf = dist.pdf.multi_eval(x)
    # ppf = dist.ppf.multi_eval(x)

    plt.scatter(data, np.zeros(data.shape[0]), label='Raw data')
    plt.plot(np.sort(data), np.cumsum(np.ones(data.shape[0]) / data.shape[0]), label='Combined data distribution')
    plt.plot(x, cdf1, label='CDF-1')
    plt.plot(x, cdf2, label='CDF-2')
    plt.plot(x, cdf, label='Combined CDF')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_quantiles()
