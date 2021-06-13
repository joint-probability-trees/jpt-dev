from operator import itemgetter

import pyximport
from dnutils import out

pyximport.install()

from jpt.learning.qreg import QReg


from jpt.base.intervals import ContinuousSet, INC, EXC


from jpt.base.quantiles import QuantileDistribution


import numpy as np
from matplotlib import pyplot as plt

from jpt.learning.distributions import Gaussian


def test_quantiles():
    gauss1 = Gaussian(-1, 2).sample(100)
    gauss2 = Gaussian(7, .5).sample(100)
    gauss3 = Gaussian(3, .1).sample(100)
    data = np.hstack([gauss1, gauss2, gauss3])

    x, counts = np.unique(data, return_counts=True)
    y = np.asarray(counts, dtype=np.float64)
    np.cumsum(y, out=np.asarray(y))
    n_samples = data.shape[0]
    for i in range(x.shape[0]):
        y[i] /= n_samples

    reg = QReg(eps=.01)
    reg.fit(np.array([x, y]).T, presort=0)
    plt.scatter(data, np.zeros(data.shape[0]), label='Raw data')

    points = np.array(reg.points)
    plt.scatter(points[:, 0], points[:, 1])

    dist1 = QuantileDistribution(epsilon=1e-10)
    dist1.fit(gauss1)
    dist2 = QuantileDistribution(epsilon=1e-10)
    dist2.fit(gauss2)
    dist3 = QuantileDistribution(epsilon=1e-10)
    dist3.fit(gauss3)

    dist = QuantileDistribution.merge([dist1, dist2, dist3], [.333, .333, .333])

    x = np.linspace(-7, 9, 500)
    cdf1 = dist1.cdf.multi_eval(x)
    cdf2 = dist2.cdf.multi_eval(x)
    cdf = dist.cdf.multi_eval(x)

    out('CDF:', dist.cdf.pfmt())

    # pdf = dist.pdf.multi_eval(x)
    ppf = dist.ppf.multi_eval(x)

    out('PPF:', dist.ppf.pfmt())

    out('PPF(1) =', dist.ppf.eval(1))
    out('PPF(0) =', dist.ppf.eval(0))

    plt.plot(np.sort(data), np.cumsum(np.ones(data.shape[0]) / data.shape[0]), label='Combined data distribution')
    plt.plot(x, cdf1, label='CDF-1')
    plt.plot(x, cdf2, label='CDF-2')
    plt.plot(x, cdf, label='Combined CDF')
    plt.plot(x, ppf, label='Percentile-point fct')
    plt.grid()
    plt.legend()
    plt.show()


def main(*args):
    test_quantiles()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
