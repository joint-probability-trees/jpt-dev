import pyximport
pyximport.install()
import math
from functools import reduce

import numpy as np
from numpy import iterable

from dnutils import ifnone

from jpt.base.intervals import ContinuousSet


class Conditional:

    def __init__(self, typ, conditionals):
        self.type = typ
        self.conditionals = conditionals
        self.p = {}

    def __getitem__(self, values):
        if not iterable(values):
            values = (values,)
        return self.p[tuple(values)]

    def __setitem__(self, evidence, dist):
        if not iterable(evidence):
            evidence = (evidence,)
        self.p[evidence] = dist

    def sample(self, evidence, n):
        if not iterable(evidence):
            evidence = (evidence,)
        return self.p[tuple(evidence)].sample(n)

    def sample_one(self, evidence):
        if not iterable(evidence):
            evidence = (evidence,)
        return self.p[tuple(evidence)].sample_one()


def mapstr(seq, format=None):
    '''Convert the sequence ``seq`` into a list of strings by applying ``str`` to each of its elements.'''
    return [format(e) for e in seq] if callable(format) else [ifnone(format, '%s') % (e,) for e in seq]


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable)


def tojson(obj):
    """Recursively generate a JSON representation of the object ``obj``."""
    if hasattr(obj, 'tojson'):
        return obj.tojson()
    if type(obj) in (list, tuple):
        return [tojson(e) for e in obj]
    elif isinstance(obj, dict):
        return {str(k): tojson(v) for k, v in obj.items()}
    return obj


def format_path(path):
    '''
    Returns a readable string representation of a conjunction of variable assignments,
    given by the dictionary ``path``.
    '''
    return ' ^ '.join([var.str(val, fmt='logic') for var, val in path.items()])


# ----------------------------------------------------------------------------------------------------------------------
# Entropy calculation

def entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p``.
    :param p:   the probabilities
    :type p:    [float] or {str:float}
    :return:
    '''
    if isinstance(p, dict):
        p = list(p.values())
    return abs(-sum([0 if p_i == 0 else math.log(p_i, 2) * p_i for p_i in p]))


def max_entropy(n):
    '''Compute the maximal entropy that a multinomial random variable with ``n`` states can have,
    i.e. the entropy value assuming a uniform distribution over the values.
    :param p:
    :return:
    '''
    return entropy([1 / n for _ in range(n)])


def rel_entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p`` normalized
    by the maximal entropy that a multinomial distribution of the dimensionality of ``p``
    can have.
    :type p: distribution'''
    if len(p) == 1:
        return 0
    return entropy(p) / max_entropy(len(p))


# ----------------------------------------------------------------------------------------------------------------------
# Gini index


def gini(p):
    '''Compute the Gini impurity for the distribution ``p``.'''
    if isinstance(p, dict):
        p = list(p.values())
    return np.mean([p_i * (1 - p_i) for p_i in p])


# ----------------------------------------------------------------------------------------------------------------------


class ClassPropertyDescriptor(object):

    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    '''
    This decorator allows to define class properties in the same way as normal object properties.

    https://stackoverflow.com/questions/5189699/how-to-make-a-class-property
    '''
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


# ----------------------------------------------------------------------------------------------------------------------


def list2interval(l):
    '''
    Converts a list representation of an interval to an instance of type
    '''
    lower, upper = l
    return ContinuousSet(np.NINF if lower in (np.NINF, -float('inf'), None) else np.float64(lower),
                         np.PINF if upper in (np.PINF, float('inf'), None) else np.float64(upper))


def normalized(dist, identity_on_zeros=False, allow_neg=False):
    '''Returns a modification of ``seq`` in which all elements sum to 1, but keep their proportions.'''
    if isinstance(dist, (list, np.ndarray)):
        dist_ = {i: p for i, p in enumerate(dist)}
    else:
        dist_ = dict(dist)
    signs = {k: np.sign(v) for k, v in dist_.items()}
    if not all(e >= 0 for e in dist_.values()) and not allow_neg:
        raise ValueError('Negative elements not allowed: %s' % np.array(list(dist_.values())))
    absvals = {k: abs(v) for k, v in dist_.items()}
    z = sum(absvals.values())
    if not z and not identity_on_zeros:
        raise ValueError('Not a proper distribution: %s' % dist)
    elif not z and identity_on_zeros:
        return [0] * len(dist)
    if isinstance(dist, dict):
        return {e: absvals[e] / z * signs[e] for e in dist_}
    else:
        rval = [absvals[e] / z * signs[e] for e in range(len(dist_))]
        if isinstance(dist, np.ndarray):
            return np.array(rval)
        return rval
