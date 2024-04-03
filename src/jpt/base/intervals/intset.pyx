# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False

__module__ = 'intset.pyx'

import numbers
from itertools import count

from .base cimport DTYPE_t, SIZE_t

cimport numpy as np
import numpy as np

from .unionset cimport UnionSet
from .base cimport NumberSet

from .base import (
    _CHAR_EMPTYSET,
    _CHAR_INTEGERS,
)

import re

from typing import Dict, Any, List, Set

# ----------------------------------------------------------------------------------------------------------------------

_Z = IntSet(np.NINF, np.PINF)


cdef SIZE_t is_int(DTYPE_t x):
    return round(x) == x


def _enumerate_ints():
    counter = 0
    yield counter
    while 1:
        counter += 1
        yield counter
        yield -counter


re_intset = re.compile(r'{(?P<lower>[-+]?\d+)?(?P<sep>\.\.)?(?P<upper>[-+]?\d+)?}')


cdef class IntSet(Interval):

    def __init__(self, DTYPE_t lower, DTYPE_t upper):
        if not np.isinf(lower) and round(lower) != lower:
            raise ValueError(
                'Lower bound if IntSet must be an integer, got %s' % lower
            )
        if not np.isinf(upper) and round(upper) != upper:
            raise ValueError(
                'Upper bound if IntSet must be an integer, got %s' % upper
            )
        self._lower = lower
        self._upper = upper

    def __getstate__(self):
        return (
            self._lower,
            self._upper
        )

    def __setstate__(self, state):
        self._lower, self._upper = state

    @staticmethod
    def parse(str s) -> IntSet:
        if s == _CHAR_EMPTYSET:
            return <IntSet> IntSet.emptyset()
        m = re_intset.match(s)
        if m is None:
            raise ValueError(
                f'Malformed integer set string: "{s}"'
            )
        m = m.groupdict()
        return IntSet(
            int(m['lower']) if m.get('lower') is not None else np.NINF,
            int(m['upper']) if m.get('upper') is not None else np.PINF
        )

    cpdef SIZE_t isempty(self):
        return self._lower > self._upper

    cpdef SIZE_t isninf(self):
        return np.isneginf(self.lower)

    cpdef SIZE_t ispinf(self):
        return np.isposinf(self.upper)

    @staticmethod
    cdef NumberSet _emptyset():
        return IntSet(np.PINF, np.NINF)

    @staticmethod
    def emptyset():
        return IntSet._emptyset()

    @staticmethod
    cdef Interval _allnumbers():
        return IntSet(np.NINF, np.PINF)

    @staticmethod
    def allnumbers():
        return IntSet._allnumbers()

    cpdef NumberSet copy(self):
        return IntSet(
            self.lower,
            self.upper
        )

    cpdef DTYPE_t size(self):
        if np.isneginf(self._lower) or np.isposinf(self._upper):
            return np.PINF
        elif self.isempty():
            return 0
        else:
            return self.upper - self.lower + 1

    def __eq__(self, other):
        if self.isempty() and other.isempty():
            return True
        elif isinstance(other, IntSet):
            return (
                self._lower == other._lower
                and self._upper == other._upper
            )
        elif isinstance(other, UnionSet):
            return other == self
        else:
            raise TypeError(
                'IntSet.__eq__() undefined for type %s' % type(other).__qualname__
            )

    @property
    def lower(self):
        return int(self._lower) if not np.isinf(self._lower) else self._lower

    @lower.setter
    def lower(self, l):
        self._lower = l

    @property
    def upper(self):
        return int(self._upper) if not np.isinf(self._upper) else self._upper

    @upper.setter
    def upper(self, u):
        self._upper = u

    cpdef DTYPE_t fst(self):
        return self.lower if not self.isempty() else np.nan

    @property
    def min(self):
        return self.lower if not self.isempty() else np.nan

    @property
    def max(self):
        return self.upper if not self.isempty() else np.nan

    def __iter__(self):
        if np.isneginf(self.lower) and np.isposinf(self.upper):
            return iter(_enumerate_ints())
        elif np.isneginf(self.lower) and not np.isposinf(self.upper):
            return iter(count(self.max, -1))
        elif not np.isneginf(self.lower) and np.isposinf(self.upper):
            return iter(count(self.min))
        else:
            return iter(range(self.min, self.max + 1))

    def __str__(self):
        if np.isneginf(self._lower) and np.isposinf(self._upper):
            return _CHAR_INTEGERS
        elif self.isempty():
            return _CHAR_EMPTYSET
        elif self.size() == 1:
            return f'{{{self.lower}}}'
        return (
            f'{{{self.lower if not np.isneginf(self._lower) else ""}'
            # f'{".." if self.size() > 2 else ","}'
            f'..'
            f'{self.upper if not np.isposinf(self._upper) else ""}}}'
        )

    def __repr__(self):
        return '<IntSet %s>' % str(self)

    cpdef SIZE_t contains_value(self, DTYPE_t x):
        if np.isinf(x):
            return x == self._lower or x == self._upper
        return is_int(x) and self._lower <= x <= self._upper

    cpdef SIZE_t issuperseteq(self, NumberSet other):
        if isinstance(other, IntSet):
            return self._lower <= other._lower and self._upper >= other._upper
        else:
            return other.issubseteq(self)

    cpdef SIZE_t issuperset(self, NumberSet other):
        if isinstance(other, IntSet):
            return (
                self._lower < other._lower and self._upper >= other._upper
                or self._lower <= other._lower and self._upper > other._upper
            )
        else:
            return other.issubset(self)

    def __contains__(self, item) -> bool:
        if isinstance(item, numbers.Number):
            return bool(self.contains_value(item))
        else:
            raise TypeError(
                'Object of type %s unsupported by IntSet.__contains__()' % type(item).__qualname__
            )

    cpdef SIZE_t intersects(self, NumberSet other):
        return not self.isdisjoint(other)

    cpdef SIZE_t isdisjoint(self, NumberSet other):
        cdef IntSet other_
        if isinstance(other, IntSet):
            other_ = <IntSet> other
            return self._upper < other_._lower or self._lower > other_._upper
        else:
            return other.isdisjoint(self)

    cpdef NumberSet intersection(self, NumberSet other):
        if isinstance(other, IntSet):
            return IntSet(
                max(self._lower, other._lower),
                min(self._upper, other._upper)
            )
        else:
            return other.intersection(self)

    def __hash__(self):
        return hash((
            IntSet,
            self._lower,
            self._upper
        )) if not self.isempty() else hash(frozenset())

    def to_json(self):
        return {
            'type': IntSet.__qualname__,
            'upper': self.upper,
            'lower': self.lower,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> IntSet:
        if 'type' in data and data.get('type', None) != IntSet.__qualname__:
            raise ValueError(
                f'Illegal type name: {data.get("type")}'
            )
        return IntSet(
            data['lower'],
            data['upper'],
        )

    @staticmethod
    def from_list(list l: List[int]) -> IntSet:
        if len(l) != 2:
            raise ValueError(
                'List must contain exactly two values (or ellipses "...") that specify the interval bounds'
            )
        return IntSet(
            l[0],
            l[1]
        )

    # noinspection PyTypeChecker
    @staticmethod
    def from_set(set numbers: Set[int]) -> NumberSet:
        return UnionSet([
            IntSet(n, n) for n in numbers
        ]).simplify(keep_type=False)

    cpdef IntSet complement(self):
        return _Z - self

    cpdef SIZE_t contiguous(self, Interval other):
        if isinstance(other, IntSet):
            return (
                other._lower == self._upper + 1
                or other._upper == self._lower - 1
            )
        else:
            raise TypeError(
                f'Operation .contiguous() undefined for argument of type {other.__class__.__qualname__}'
            )

    def __sub__(self, other):
        return self.difference(other)

    cpdef NumberSet difference(self, NumberSet other):
        if other.issuperseteq(self):
            return self.emptyset()
        if self.intersects(other):
            return UnionSet([
                IntSet(self.lower, other.lower - 1),
                IntSet(other.upper + 1, self.upper)
            ]).simplify()
        else:
            return self.copy()

    cpdef NumberSet union(self, NumberSet other):
        if self.intersects(other) or self.contiguous(<IntSet> other):
            return IntSet(
                min(self.lower, other.lower),
                max(self.upper, other.upper)
            )
        else:
            return UnionSet([
                self,
                other
            ])

    cpdef NumberSet xmirror(self):
        if self == _Z:
            return _Z
        elif self.isempty():
            return IntSet._emptyset()
        return IntSet(
            -self.upper,
            -self.lower
        )

    cpdef DTYPE_t[::1] _sample(self, SIZE_t k=1, DTYPE_t[::1] result=None):
        if self.isempty():
            raise ValueError('Cannot sample from an empty set.')

        if self.isinf():
            raise ValueError('Cannot sample uniformly from an infinite set.')

        elif result is None:
            result = np.ndarray(shape=k)

        if self.size() == 1:
            result[...] = np.ones(shape=k) * self._lower
        else:
            result[...] = np.random.randint(self.lower, self.upper, k)

        return result

    cpdef NumberSet simplify(self):
        return self.copy()

    EMPTY = IntSet.emptyset()
    ALL = _Z