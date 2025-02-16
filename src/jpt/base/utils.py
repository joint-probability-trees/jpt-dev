import heapq
import logging
import numbers
import os
from _csv import register_dialect, QUOTE_NONNUMERIC
from csv import Dialect

import math
from typing import Callable, Iterable, Any, Tuple, Set, List, Union, Dict

import numpy as np
import arff
import csv

from functools import reduce

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy import iterable

from dnutils import ifnone, ifnot

try:
    from .intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .intervals import ContinuousSet


# ----------------------------------------------------------------------------------------------------------------------
# Type definitions

Symbol = Union[str, int]
Collections = (list, set, tuple)


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


def mapstr(seq: Iterable, fmt: Callable = None, limit: int = None, ellipse: str = '...'):
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
    return result[:max(limit // 2, 1)] + [ellipse] + result[len(result) - limit // 2:]


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


def setstr_int(intset: Set[int], sep_inner: str = '', sep_outer: str = ', ') -> str:
    '''
    Return a prettyfied string representing the given set of integers.

    Consecutive numbers (e.g. 0, 1, 2) will be collapsed into a range representation "0...2".

    :param intset:      the set of integers to be formatted.
    :param sep_inner:   the separator character to be used for separating consecutive ("inner") numbers
    :param sep_outer:   the separator character to be used for separating non-consecutive ("outer") numbers
    '''
    elements = list(sorted(intset))
    chunks = []
    chunk = [elements[0]]
    for a, b in pairwise(elements):
        if b == a + 1:
            chunk.append(b)
        else:
            chunks.append(chunk)
            chunk = [b]
    chunks.append(chunk)
    chunks = [mapstr(c, limit=2, ellipse='...') for c in chunks]
    chunks = [
        sep_inner.join(c) if len(c) == 3 else sep_outer.join(c)
        for c in chunks
    ]
    return sep_outer.join([c for c in chunks])


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


def format_path(path, **kwargs):
    '''
    Returns a readable string representation of a conjunction of variable assignments,
    given by the dictionary ``path``. The ``kwargs`` are passed to the jpt.variables.Variable.str function to allow
    customized formatting.
    '''
    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'logic'
    return ' ^ '.join([var.str(val, **kwargs) for var, val in path.items()])


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

def list2interval(l: List[int]) -> ContinuousSet:
    '''
    Converts a list representation of an interval to an instance of type
    '''
    lower, upper = l
    return ContinuousSet(
        -np.inf if lower in (-np.inf, -float('inf'), None, ...) else np.float64(lower),
        np.inf if upper in (np.inf, float('inf'), None, ...) else np.float64(upper)
    )


def list2set(values: List[Symbol]) -> Set[str]:
    """
    Convert a list to a set.
    """
    return set(values)


def list2intset(bounds: List[int]) -> Set[int]:
    '''
    Convert a 2-element list specifying a lower and an upper bound into a
    integer set containing the admissible values of the corresponding interval
    '''
    if not len(bounds) == 2:
        raise ValueError(
            'Argument list must have length 2, got length %d.' % len(bounds)
        )

    # if bounds[0] < cls.lmin or bounds[1] > cls.lmax:
    #     raise ValueError(
    #         f'Argument must be in [%d, %d].' % (cls.lmin, cls.lmax)
    #     )

    return set(range(bounds[0], bounds[1] + 1))


def normalized(
        dist: Union[List[numbers.Real], np.ndarray, Dict[Any,numbers.Real]],
        identity_on_zeros: bool = False,
        allow_neg: bool = False,
        zeros: float = .0
) -> Union[List[numbers.Real], np.ndarray, Dict[Any,numbers.Real]]:
    '''
    Returns a modification of ``seq`` in which all elements sum to 1, but keep their proportions.

    ``dist`` can be either a list or numpy array, or a dict mapping from random events
    of any type to their respective value.

    :param dist:   the values to be normalized
    :param identity_on_zeros: if all values in dist are zero, the input argument is returned unchanged.
                              If ``False``, an error is raised in the case that the distribution cannot be
                              normalized.
    :param allow_neg:  determines whether negative values are allowed. If ``True``, negative values will be treated
                       as positive values for the normalization, but will keep their sign in the output.
    :param zeros:  in [0,1], if ``zeros`` is a non-negative, non-zero float, all zeros in the distribution
                   will be set to that fraction of the smallest non-zero value in the distribuition.
                   This parameter can be used to artificially circumvent strict-0 values, which may be prohibitive
                   in some cases.
    '''
    if isinstance(dist, (list, np.ndarray)):
        dist_ = {i: p for i, p in enumerate(dist)}
    elif isinstance(dist, dict):
        dist_ = dict(dist)
    else:
        raise TypeError(
            'Illegal type of distribution: %s.' % type(dist).__name__
        )

    if not all(e >= 0 for e in dist_.values()) and not allow_neg:
        raise ValueError(
            'Negative elements not allowed: %s' % np.array(list(dist_.values()))
        )

    signs = {
        k: ifnot(np.sign(v), 1) for k, v in dist_.items()
    }

    retvals = {
        k: abs(v) for k, v in dist_.items()
    }

    while 1:
        z = sum(retvals.values())
        if not z:
            if not identity_on_zeros:
                raise ValueError('Not a proper distribution: %s' % dist)
            if not z and identity_on_zeros:
                return dist.copy()

        retvals = {
            k: v / z for k, v in retvals.items()
        }

        if zeros and min(retvals.values()) == 0:
            abs_min = min([v for v in retvals.values() if v > .0]) * zeros
            retvals = {
                k: ifnot(v, abs_min) for k, v in retvals.items()
            }
            continue

        result = {
            k: v * signs[k] for k, v in retvals.items()
        }
        if not isinstance(dist, dict):
            result = [
                result[e]for e in range(len(dist_))
            ]
        if isinstance(dist, np.ndarray):
            return np.array(result, dtype=dist.dtype)
        return result


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


# ----------------------------------------------------------------------------------------------------------------------

def count(start: int = 0, inc: int = 1):
    value = start
    while 1:
        yield value
        value += inc


class Heap:
    '''
    Implementation of a heap wrapper inspired by

    https://stackoverflow.com/questions/8875706/heapq-with-custom-compare-predicate#8875823
    '''

    # ------------------------------------------------------------------------------------------------------------------

    class Iterator:

        def __init__(self, heap: 'Heap', reverse=False):
            self.heap = heap
            self._list_iterator = (reversed if reverse else iter)(self.heap._data)

        def __next__(self):
            return next(self._list_iterator)[2]

        def __iter__(self):
            return self

    # ------------------------------------------------------------------------------------------------------------------

    def __init__(
            self,
            data: Iterable[Any] = None,
            key: Callable = None,
            inc: int = 1
    ):
        self._key = ifnone(key, lambda x: x)
        self._index = 0
        self._inc = inc
        if data:
            self._data = [(self._key(item), i, item) for i, item in zip(count(0, self._inc), data)]
            self._index = len(self._data) * self._inc
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(
            self._data,
            (self._key(item), self._index, item)
        )
        self._index += self._inc

    def pop(self):
        return heapq.heappop(self._data)[2]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item: int):
        return self._data[item][2]

    def __delitem__(self, item: int):
        del self._data[item]

    def __bool__(self):
        return bool(self._data)

    def __repr__(self):
        return '<Heap %s>' % [item for _, _, item in self._data]

    def __iter__(self):
        return Heap.Iterator(self)

    def __reversed__(self):
        return Heap.Iterator(self, reverse=True)

    def index(self, item: Any) -> int:
        for idx, (_, _, i) in enumerate(self._data):
            if i == item:
                return idx
        else:
            raise ValueError(
                'Item %s not found.' % item
            )
