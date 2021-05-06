import numpy as np
import pyximport
from matplotlib import pyplot as plt

from dnutils import out
from jpt.learning.distributions import Numeric, SymbolicType
from jpt.learning.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

pyximport.install()

import os
import pickle
import pprint
from intervals import ContinuousSet as Interval

from quantiles import Quantiles


def test_muesli_quantile():
    f = os.path.join('data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = pickle.load(fi)

    quantiles = Quantiles(data[0], epsilon=.0001)
    cdf_ = quantiles.cdf()
    d = Numeric(cdf=cdf_)

    interval = Interval(-2.05, -2.0)
    p = d.p(interval)
    print('query', interval, p)

    print(d.cdf.pfmt())
    d.plot(name='Müsli Beispiel', view=True)


def muesli_tree():
    f = os.path.join('data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = pickle.load(fi)

    unique, counts = np.unique(data[2], return_counts=True)

    ObjectType = SymbolicType('ObjectType', unique)

    x = NumericVariable('X', Numeric)
    y = NumericVariable('Y', Numeric)
    o = SymbolicVariable('Object', ObjectType)

    jpt = JPT([x, y, o], name="Müslitree", min_samples_leaf=10)
    jpt.learn(list(zip(*data)))

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    jpt.plot(plotvars=[x, y, o])


if __name__ == '__main__':
    out('Müsli plot:')
    test_muesli_quantile()

    out('\nMüsli tree:')
    muesli_tree()

