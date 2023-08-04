import os
import pickle

from jpt import JPT
from jpt.distributions import Numeric

try:
    from jpt.distributions.quantile.quantiles import __module__
    from jpt.base.functions import __module__
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.intervals import ContinuousSet, INC, EXC
    from jpt.distributions.quantile.quantiles import QuantileDistribution
    from jpt.base.functions import PiecewiseFunction, ConstantFunction


def gaussian_jpt() -> JPT:
    '''
    Returns a JPT with one single variable representing a Gaussian distribution.

    :return:
    '''
    return JPT.load(os.path.join('resources', 'gaussian-jpt.dat'))


def gaussian_numeric() -> Numeric:
    with open(os.path.join('resources', 'gaussian_100.dat'), 'rb') as f:
        return Numeric().fit(pickle.load(f).reshape(-1, 1))


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