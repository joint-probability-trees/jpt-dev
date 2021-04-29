import numpy as np
import pyximport
from matplotlib import pyplot as plt

pyximport.install()

import os
import pickle
import pprint

from quantiles import Quantiles


def test_muesli():
    f = os.path.join('data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = pickle.load(fi)

    data = np.array(data)

    pprint.pprint(data)
    print(len(data))

    data_ = np.array(sorted([float(x) for x in data.T[0]]))
    print(data_)

    quantiles = Quantiles(data_, epsilon=.0001)
    cdf = quantiles.cdf()
    pdf = quantiles.pdf()
    plt.plot(data_, np.cumsum([1] * len(data_)) / len(data_), label='CumSum($\mathcal{D}$)', linewidth=2)
    plt.plot(data_, cdf.multi_eval(data_), linewidth=2, label='Piecewise linear PDF')
    plt.show()


if __name__ == '__main__':
    test_muesli()
