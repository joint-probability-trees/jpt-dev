import itertools
from itertools import repeat

import numpy as np
import pandas as pd
import pyximport
from dnutils import out
from matplotlib import pyplot as plt

pyximport.install()

import os
import pickle
import pprint

from quantiles import Quantiles
from intervals import ContinuousSet as Interval


def test_muesli():
    # f = os.path.join('data', 'human_muesli.pkl')

    # with open(f, 'rb') as fi:
    #     data = pickle.load(fi)

    # df = pd.DataFrame({'X': data[0] + 3, 'Y': data[1] + 2, 'Class': data[2]})
    # df.to_pickle('data/human_muesli.dat')
    df = pd.read_pickle('data/human_muesli.dat')

    # pprint.pprint(data)

    # data_ = np.array(sorted([float(x) for x in data.T[0]]))
    # print(data_)

    # quantiles = Quantiles(data[0], epsilon=.0001)
    # cdf = quantiles.cdf()
    # print(cdf.pfmt())

    # retrieve upper bounds of the first n-1 intervals and the lower bound of the last interval to generate data
    # for the cdf plot (for use in Distribution class where original data is not available)
    # bounds = np.array([v.upper for v in cdf.intervals[:-2]] + [cdf.intervals[-1].lower])

    # sample data from overal interval over all intervals
    # i = Interval(cdf.intervals[0].lower, cdf.intervals[-1].lower, left=cdf.intervals[0].left, right=cdf.intervals[-1].right)
    # samplefirst = cdf.intervals[0].sample(10)
    # sampled = np.array(sorted([i.sample()[0] for _ in range(20)]))
    # print('sampled', sampled)

    # d = np.array(sorted([[float(x) for x in rows] for rows in data.T[:-1]]))
    # print('data', d.T)

    fig, ax = plt.subplots()
    ax.set_title(f"Piecewise linear CDF")
    ax.set_xlabel('value')
    ax.set_ylabel('%')

    # ax.plot(data_, np.cumsum([1] * len(data_)) / len(data_), color='green', label='CumSum($\mathcal{D}$)', linewidth=2)
    # ax.plot(data_, cdf.multi_eval(data_), color='orange', linewidth=2, label='Piecewise fn from original data')
    icons = itertools.cycle(['o', 'x', '1'])
    for icon, clazz in zip(icons, df['Class'].unique()):
        samples = df[df['Class'] == clazz]
        ax.scatter(samples['X'], samples['Y'], color='black', marker=icon, label=clazz)
    ax.legend()
    # ax.plot(bounds, cdf.multi_eval(bounds), color='cornflowerblue', linestyle='dashed', linewidth=2, markersize=12, label='Piecewise fn from bounds')
    # ax.plot(sampled, cdf.multi_eval(sampled), color='red', linestyle='dotted', linewidth=2, label='Piecewise fn from original data')

    # ax2 = ax.twiny()
    # ax2.set_xlim(left=0., right=1.0)
    # ax2.plot(bounds, cdf.multi_eval(bounds), color='cornflowerblue', linestyle='dashed', linewidth=2, markersize=12, label='Piecewise fn from bounds')
    # ax.legend()

    plt.show()


if __name__ == '__main__':
    test_muesli()
