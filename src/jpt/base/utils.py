import logging
import numbers
import os
from _csv import QUOTE_MINIMAL, register_dialect, QUOTE_NONE, QUOTE_NONNUMERIC
from csv import Dialect

import math
from typing import Callable, Iterable, Any, Tuple

import numpy as np
import arff
import csv

from functools import reduce

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import iterable

from dnutils import ifnone, stop, out


try:
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.intervals import ContinuousSet


# ----------------------------------------------------------------------------------------------------------------------

def pairwise(seq: Iterable[Any]) -> Iterable[Tuple[Any, Any]]:
    '''Iterate over all consecutive pairs in ``seq``.'''
    for e in seq:
        if 'prev' in locals():
            yield prev, e
        prev = e


# ----------------------------------------------------------------------------------------------------------------------

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


def mapstr(seq: Iterable, fmt: Callable = None, limit: int = None):
    '''
    Convert the sequence ``seq`` into a list of strings by applying ``str`` to each of its elements.

    If a ``limit`` is passed, the resulting list of strings will be truncated to ``limit`` elements at
    most, and an additional three dots "..." in the middle.

    ``fmt`` is an optional formating function that is applied to every sequence element,
    defaulting to the ``str`` builtin function, if ``None``.
    '''
    fmt = ifnone(fmt, str)
    result = [fmt(e) for e in seq]
    if not limit or limit >= len(seq):
        return result
    return result[:max(limit // 2, 1)] + ['...'] + result[len(result) - limit // 2:]


def setstr(seq: Iterable, fmt: Callable = None, limit: int = None, sep: str = ', '):
    '''
    Return a string representing the given sequence ``seq`` as a set.

    If a ``limit`` is passed, the resulting list of strings will be truncated to ``limit`` elements at
    most, and an additional three dots "..." in the middle.

    ``fmt`` is an optional formating function that is applied to every sequence element,
    defaulting to the ``str`` builtin function, if ``None``.

    ``sep`` specifies the separator character to be used, defaulting to the comma.
    '''
    return sep.join(mapstr(seq, fmt=fmt, limit=limit))


def prod(it: Iterable[numbers.Number]):
    '''
    Compute the product of all elements in ``it``.
    '''
    return reduce(lambda x, y: x * y, it)


def none2nan(it: Iterable[numbers.Number]):
    '''
    Return a copy of the passed iterable ``it``, in which all occurrence
    of ``None`` have been replaced by ``np.nan``.
    '''
    return map(lambda x: ifnone(x, np.nan), it)


def to_json(obj):
    '''
    Recursively generate a JSON representation of the object ``obj``.

    Non-natively supported data types must provide a ``to_json()`` method that
    returns a representation that is in turn jsonifiable.
    '''
    if hasattr(obj, 'to_json'):
        return obj.to_json()
    if isinstance(obj, list):
        return [to_json(e) for e in obj]
    elif isinstance(obj, dict):
        return {str(k): to_json(v) for k, v in obj.items()}
    elif type(obj) is str or isinstance(obj, numbers.Number):
        return obj
    else:
        raise TypeError('Object of type %s is not jsonifiable.' % type(obj).__name__)
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
    :param n:
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
    return ContinuousSet(np.NINF if lower in (np.NINF, -float('inf'), None, ...) else np.float64(lower),
                         np.PINF if upper in (np.PINF, float('inf'), None, ...) else np.float64(upper))


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


class CSVDialect(Dialect):
    """Describe the usual properties of Excel-generated CSV files."""
    delimiter = ';'
    quotechar = '"'
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = QUOTE_NONNUMERIC

# excel:
#     delimiter = ','
#     quotechar = '"'
#     doublequote = True
#     skipinitialspace = False
#     lineterminator = '\r\n'
#     quoting = QUOTE_MINIMAL


register_dialect("csvdialect", CSVDialect)


def convert(k, v):
    v = ifnone(v, '')
    if k == 'transaction_dt':
        return str(v)
    try:
        v = int(v)
    except ValueError:
        try:
            v = float(v)
        except ValueError:
            v = str(v)
    return v


def arfftocsv(arffpath, csvpath):
    print(f'Loading arff file: {arffpath}\n')
    data = arff.load(open(arffpath, 'r'), encode_nominal=True)

    print(f'Writing to csv file: {csvpath}\n')
    with open(csvpath, 'w', newline='') as csvfile:
        fieldnames = [d[0] for d in data.get('attributes')]
        writer = csv.DictWriter(csvfile, dialect='csvdialect', fieldnames=fieldnames)
        writer.writeheader()

        for dp in data.get('data'):
            writer.writerow({k: convert(k, v) for k, v in zip(fieldnames, dp)})


def save_plot(fig,  directory, fname, fmt='pdf'):
    # save figure as PDF or PNG
    if fmt == 'pdf':
        logging.debug(
            f"Saving distributions plot to {os.path.join(directory, f'{fname}.pdf')}")
        with PdfPages(os.path.join(directory, f'{fname}.pdf')) as pdf:
            pdf.savefig(fig)
    else:
        logging.debug(
            f"Saving distributions plot to {os.path.join(directory, f'{fname}.png')}")
        plt.savefig(os.path.join(directory, f'{fname}.png'))

