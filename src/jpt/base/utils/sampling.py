'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import random

from dnutils import first, out
import numpy as np


class RouletteWheelSampler:
    '''Roulette wheel proportional sampler'''

    def __init__(self, elements, weights, normalize=False):
        if len(elements) != len(weights):
            raise ValueError('Element vector and weight vector must have same lengths: %s vs. %s' % (len(elements), len(weights)))
        if normalize:
            s = np.sum(weights)
            weights = [w / s for w in weights]
        self._upperbounds = np.cumsum(weights)
        self._elements = elements

    def __getitem__(self, x):
        return self._elements[self.index(x)]

    def index(self, x):
        '''Returns the index of the element, which corresponds to the "roulette" field, ``x`` falls into.
        :param x:
        :return:
        '''
        if not (0 <= x <= self._upperbounds[-1]):
            raise IndexError('Indexes in the roulette wheel must be in [0, %s] (got %s)' % (self._upperbounds[-1], x))
        for i, bound in enumerate(self._upperbounds):
            if x <= bound:
                return i

    def sample(self, n=1):
        '''Sample ``n`` values from the the roulette wheel.
        :param n:
        :return:
        '''
        if not n:
            return []
        return [self[random.uniform(0, self._upperbounds[-1])] for _ in range(n)]

    def samplei(self, n=1):
        '''Same as ``sample()``, but returns a list of indices of selected elements.
        :param n:
        :return:
        '''
        idx = [self.index(random.uniform(0, self._upperbounds[-1])) for _ in range(n)]
        return [(self._elements[i], i) for i in idx]


def wchoice(population, weights):
    '''Choose one element from the ``population`` proportionally to their ``weights``.'''
    return first(wsample(population, weights, 1))


def wchoiced(dist):
    '''Choose from the dict ``dist`` one element from key set proportionally to the weights given as values'''
    keys = list(dist.keys())
    return wchoice(keys, [dist[k] for k in keys])


def wchoicei(population, weights):
    '''Choose one element from the ``population`` proportionally to their ``weights`` and return its index.'''
    return first(wsamplei(population, weights, 1))


def wsample(population, weights, k):
    '''Obtain a sample of the ``population`` of length ``k``.

    The probability of each element in ``population`` to be sampled is proportional to its weight in ``weights``
    vector. ``len(population)`` must equal to ``len(weights)``.

    :param population:
    :param weights:
    :param k:
    :return:
    '''
    return RouletteWheelSampler(population, weights).sample(k)


def wsamplei(population, weights, k):
    '''Equivalent to ``wsample``, but returns a tuples of the elements chosen and their index in the population.
    :param population:
    :param weights:
    :param k:
    :return:
    '''
    return RouletteWheelSampler(population, weights).samplei(k)
