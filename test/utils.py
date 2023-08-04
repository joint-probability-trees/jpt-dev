import os
import pickle

from jpt import JPT
from jpt.distributions import Numeric


def gaussian_jpt() -> JPT:
    '''
    Returns a JPT with one single variable representing a Gaussian distribution.

    :return:
    '''
    return JPT.load(os.path.join('resources', 'gaussian-jpt.dat'))


def gaussian_numeric() -> Numeric:
    with open(os.path.join('resources', 'gaussian_100.dat'), 'rb') as f:
        return Numeric().fit(pickle.load(f).reshape(-1, 1))
