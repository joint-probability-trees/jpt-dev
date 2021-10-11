import pyximport
import scipy.stats as stats

pyximport.install()

# from jpt.base.quantiles import QuantileDistribution
from operator import itemgetter

from dnutils import out

from jpt.learning.cdfreg import CDFRegressor


# from jpt.base.intervals import ContinuousSet, INC, EXC


from jpt.base.quantiles import QuantileDistribution


import numpy as np
from matplotlib import pyplot as plt

from jpt.learning.distributions import Gaussian


def qdata(data):
    data = np.sort(data)
    x, counts = np.unique(data, return_counts=True)
    y = np.asarray(counts, dtype=np.float64)
    np.cumsum(y, out=np.asarray(y))
    n_samples = data.shape[0]
    for i in range(x.shape[0]):
        y[i] /= n_samples
    return np.array([x, y])


def test_quantiles():
    gauss1 = Gaussian(-1, 1.5)
    g1data = gauss1.sample(100)
    gauss2 = Gaussian(3, .08)
    g2data = gauss2.sample(100)
    gauss3 = Gaussian(7, .5)
    g3data = gauss3.sample(100)
    data = np.hstack([sorted(g1data), sorted(g2data), sorted(g3data)])

    reg = CDFRegressor(eps=.01)
    reg.fit(qdata(data))

    # print(reg.cdf.pfmt())

    points = np.array(reg.support_points)

    fig, ax = plt.subplots()
    # ax.set_title("Quantiles")


    dist_all = QuantileDistribution(epsilon=.1)
    dist_all.fit(data.reshape(-1, 1), None, 0)


    # dist1 = QuantileDistribution(epsilon=1e-10)
    # dist1.fit(gauss1)
    # plt.scatter(gauss1, np.zeros(gauss1.shape[0]), label='Raw data')

    # reg.fit(np.array(qdata(gauss1)).T, presort=1)
    # points = np.array(reg.points)
    # plt.scatter(points[:, 0], points[:, 1])

    # dist2 = QuantileDistribution(epsilon=1e-10)
    # dist2.fit(gauss2)
    # plt.scatter(gauss2, np.zeros(gauss2.shape[0]), label='Raw data')

    # reg.fit(np.array(qdata(gauss2)).T, presort=1)
    # points = np.array(reg.points)
    # plt.scatter(points[:, 0], points[:, 1])

    # dist3 = QuantileDistribution(epsilon=1e-10)
    # dist3.fit(gauss3)
    # plt.scatter(gauss3, np.zeros(gauss3.shape[0]), label='Raw data')

    # reg.fit(np.array(qdata(gauss3)).T, presort=1)
    # points = np.array(reg.points)
    # plt.scatter(points[:, 0], points[:, 1])

    # dist = QuantileDistribution.merge([dist1, dist2, dist3], [.333, .333, .333])

    # x = np.linspace(-7, 9, 500)
    # cdf1 = dist_all.cdf.multi_eval(x)
    # cdf2 = dist_all.cdf.multi_eval(x)
    # cdf = dist_all.cdf.multi_eval(x)

    # out('CDF:', dist.cdf.pfmt())

    # pdf = dist_all.pdf.multi_eval(x)
    # ppf = dist_all.ppf.multi_eval(x)

    # out('PPF:', dist.ppf.pfmt())

    # out('PPF(1) =', dist.ppf.eval(1))
    # out('PPF(0) =', dist.ppf.eval(0))

    x = sorted(np.linspace(-4, 9, 300))
    ax.plot(points[:, 0], points[:, 1], label='PLF of CDF', marker='o', color='orange')
    ax.scatter(data, np.zeros(data.shape[0]), label='Raw data', marker='x', color='cornflowerblue')
    ax.scatter(np.sort(data), np.cumsum(np.ones(data.shape[0]) / data.shape[0]), label='Combined data distribution', marker='+', color='black')
    ax.plot(x, [gauss1.pdf(d) + gauss2.pdf(d) + gauss3.pdf(d) for d in x], label='Mixture of Gaussians', color='green')
    # plt.plot(x, cdf1, label='CDF-1')
    # plt.plot(x, cdf2, label='CDF-2')
    # plt.plot(x, cdf, label='Combined CDF')
    # plt.plot(x, pdf, label='Percentile-point fct')
    # ax.plot(x, dist_all.cdf.multi_eval(x), label='All CDF', linestyle='dashed')


    plt.grid()
    plt.legend()
    plt.show()


def main(*args):
    test_quantiles()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
