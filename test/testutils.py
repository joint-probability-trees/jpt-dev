import os
import pickle

import pandas as pd

from jpt.trees import JPT
from jpt.distributions import Numeric
from jpt.base.intervals import ContinuousSet, INC, EXC
from jpt.base.functions import PiecewiseFunction, ConstantFunction

from jpt.distributions.qpd import QuantileDistribution


__path__, _ = os.path.split(__file__)


def gaussian_jpt() -> JPT:
    '''
    Returns a JPT with one single variable representing a Gaussian distribution.

    :return:
    '''
    return JPT.load(os.path.join(__path__, 'resources', 'gaussian-jpt.dat'))


def gaussian_numeric() -> Numeric:
    with open(os.path.join(__path__, 'resources', 'gaussian_100.dat'), 'rb') as f:
        return Numeric().fit(pickle.load(f).reshape(-1, 1))


def gaussian_data_1d() -> pd.DataFrame:
    with open(os.path.join(__path__, 'resources', 'gaussian_100.dat'), 'rb') as f:
        return pd.DataFrame.from_records(
            pickle.load(f).reshape(-1, 1),
            columns=['X']
        )


def uniform_numeric(a: float, b: float) -> Numeric:
    if b <= a:
        raise ValueError('Illegal interval: a = %s >= %s = b' % (a, b))
    return Numeric().set(
        QuantileDistribution.from_pdf(
            PiecewiseFunction.zero().overwrite_at(
                ContinuousSet(a, b, INC, EXC),
                ConstantFunction(1 / (b - a))
            )
        )
    )