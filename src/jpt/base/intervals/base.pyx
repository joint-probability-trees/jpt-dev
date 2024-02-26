# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False
__module__ = 'base.pyx'

from itertools import tee

from typing import Iterable, Any, Tuple, Dict

import numpy as np
cimport numpy as np


# ----------------------------------------------------------------------------------------------------------------------
# String and formatting constants

_CHAR_CUP = u'\u222A'
_CHAR_CAP = u'\u2229'
_CHAR_EMPTYSET = u'\u2205'
_CHAR_INFTY = '∞'
_CHAR_INTEGERS = 'ℤ'
_CHAR_REAL_NUMBERS = 'ℝ'

LEFT = 0
RIGHT = 1


# ----------------------------------------------------------------------------------------------------------------------

CLOSED = 2
HALFOPEN = 3
OPEN = 4

# NB: Do not remove this! Declaration of _INC and _EXC in base.pxd does not set their values!
_INC = 1
_EXC = 2

INC = np.int32(_INC)
EXC = np.int32(_EXC)

# ----------------------------------------------------------------------------------------------------------------------

cdef class NumberSet:
    """
    Abstract superclass for UnionSet and ContinuousSet.
    """

    @staticmethod
    cdef NumberSet _emptyset():
        return NumberSet()

    @staticmethod
    def emptyset() -> NumberSet:
        return NumberSet._emptyset()

    def __suppress_inheritance(self):
        if type(self) is not NumberSet:
            raise NotImplementedError(
                f'Method not implemented in type {type(self).__qualname__}'
            )

    def __getstate__(self):
        self.__suppress_inheritance()
        return ()

    def __setstate__(self, _):
        self.__suppress_inheritance()
        pass

    def __and__(self, other):
        raise NotImplementedError()

    def __sub__(self, other):
        return self.difference(other)

    def __or__(self, other):
        raise NotImplementedError()

    def __eq__(self, other):
        self.__suppress_inheritance()
        return other.isempty()

    def __bool__(self) -> bool:
        return not self.isempty()

    def __contains__(self, item) -> bool:
        return self.contains_value(item)

    cpdef SIZE_t contains_value(self, DTYPE_t x):
        self.__suppress_inheritance()
        return False

    cpdef SIZE_t issuperseteq(self, NumberSet other):
        self.__suppress_inheritance()
        return other.isempty()

    cpdef SIZE_t issuperset(self, NumberSet other):
        self.__suppress_inheritance()
        return False

    cpdef NumberSet union(self, NumberSet other):
        self.__suppress_inheritance()
        return other.copy()

    cpdef NumberSet difference(self, NumberSet other):
        self.__suppress_inheritance()
        return NumberSet.EMPTY

    cpdef SIZE_t isdisjoint(self, NumberSet other):
        self.__suppress_inheritance()
        return True

    cpdef SIZE_t intersects(self, NumberSet other):
        self.__suppress_inheritance()
        return False

    cpdef NumberSet intersection(self, NumberSet other):
        self.__suppress_inheritance()
        return NumberSet.EMPTY

    cpdef SIZE_t isempty(self):
        self.__suppress_inheritance()
        return True

    cpdef DTYPE_t size(self):
        self.__suppress_inheritance()
        return 0

    cpdef NumberSet copy(self):
        self.__suppress_inheritance()
        return NumberSet()

    cpdef DTYPE_t fst(self):
        self.__suppress_inheritance()
        return np.nan

    cpdef DTYPE_t lst(self):
        self.__suppress_inheritance()
        return np.nan

    @property
    def min(self):
        return self.fst()

    @property
    def max(self):
        return self.lst()

    cpdef NumberSet xmirror(self):
        self.__suppress_inheritance()
        return NumberSet.emptyset()

    cpdef SIZE_t isninf(self):
        self.__suppress_inheritance()
        return False

    cpdef SIZE_t ispinf(self):
        self.__suppress_inheritance()
        return False

    cpdef SIZE_t isinf(self):
        return self.isninf() or self.ispinf()

    cpdef np.ndarray[DTYPE_t] sample(self, SIZE_t k=1, DTYPE_t[::1] result=None):
           return np.array(self._sample(k, result))

    cpdef DTYPE_t[::1] _sample(self, SIZE_t k=1, DTYPE_t[::1] result=None):
        raise NotImplementedError()

    def __hash__(self):
        self.__suppress_inheritance()
        return hash(frozenset())


# ----------------------------------------------------------------------------------------------------------------------

from .contset import ContinuousSet, _R
from .intset import IntSet, _Z

cdef class Interval(NumberSet):

    def __suppress_inheritance(self):
        if type(self) is not Interval:
            raise NotImplementedError(
                f'Method not implemented in type {type(self).__qualname__}'
            )

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, l):
        self._lower = l

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, u):
        self._upper = u

    @staticmethod
    def parse(str s) -> Interval:
        s = s.strip()
        if s == _CHAR_EMPTYSET:
            return <Interval> Interval.emptyset()
        elif s == _CHAR_INTEGERS:
            return _Z
        elif s == _CHAR_REAL_NUMBERS:
            return _R
        else:
            try:
                return IntSet.parse(s)
            except ValueError:
                return ContinuousSet.parse(s)

    @staticmethod
    def from_json(data: Dict[str, Any]) -> Interval:
        itype = data.get('type')
        return {
            IntSet.__qualname__: IntSet,
            ContinuousSet.__qualname__: ContinuousSet
        }.get(itype, ContinuousSet).from_json(data)

    cpdef SIZE_t contiguous(self, Interval other):
        raise NotImplementedError()

    def __contains__(self, item):
        return self.contains_value(item)

    cpdef NumberSet complement(self):
        raise NotImplementedError()

    def __hash__(self):
        self.__suppress_inheritance()
        return hash(frozenset())

    cpdef SIZE_t contains_value(self, DTYPE_t value):
        self.__suppress_inheritance()
        return False

    @staticmethod
    cdef NumberSet _emptyset():
        return Interval()

    @staticmethod
    def emptyset() -> NumberSet:
        return Interval._emptyset()

    cpdef SIZE_t isempty(self):
        self.__suppress_inheritance()
        return True

    def __eq__(self, other):
        self.__suppress_inheritance()
        return other.isempty()

    cpdef NumberSet xmirror(self):
        self.__suppress_inheritance()
        return Interval.emptyset()

    cpdef SIZE_t isninf(self):
        self.__suppress_inheritance()
        return False if self.isempty() else np.isneginf(self._lower)

    cpdef SIZE_t ispinf(self):
        self.__suppress_inheritance()
        return False if self.isempty() else np.isposinf(self._upper)

    cpdef SIZE_t isinf(self):
        return self.isninf() or self.ispinf()

    def chop(self, points: Iterable[float]) -> Iterable[Interval]:
        if not points:
            return Interval.emptyset()
        elif self.isempty():
            raise ValueError(
                'Cannot chop an empty set'
            )
        else:
            raise NotImplementedError()

    @staticmethod
    cdef Interval _allnumbers():
        raise NotImplementedError()

    @staticmethod
    def allnumbers():
        raise NotImplementedError()