# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False

__module__ = 'contset.pyx'

import numbers
import traceback
from typing import List, Iterable, Dict, Any

from dnutils import ifnot, ifnone

from .base cimport (
    NumberSet,
    DTYPE_t,
    SIZE_t
)
from jpt.base.constants import eps

from .unionset import UnionSet

from .base import (
    RIGHT,
    LEFT,
)

import math

import numpy as np

from .base import _CHAR_EMPTYSET
from .base import INC, EXC

import re



# ----------------------------------------------------------------------------------------------------------------------

NOTATION_SQUARED = {
    LEFT: {
        INC: '[',
        EXC: ']'
    },
    RIGHT: {
        INC: ']',
        EXC: '['
    }
}

NOTATION_PARANTHESES = {
    LEFT: {
        INC: '[',
        EXC: '('
    },
    RIGHT: {
        INC: ']',
        EXC: ')'
    }
}

NOTATIONS = {
    'par': NOTATION_PARANTHESES,
    'sq': NOTATION_SQUARED
}

interval_notation = 'par'


# ----------------------------------------------------------------------------------------------------------------------

re_int = re.compile(
    r'(?P<ldelim>\(|\[|\])(?P<lval>.+),(?P<rval>.+)(?P<rdelim>\)|\]|\[)'
)

_R = ContinuousSet(np.NINF, np.PINF, EXC, EXC)

_INC = 1
_EXC = 2

CLOSED = 2
HALFOPEN = 3
OPEN = 4

# ----------------------------------------------------------------------------------------------------------------------

cdef class ContinuousSet(Interval):
    """
    Actual Interval representation. Wrapped by :class:`Interval` to allow more complex intervals with gaps.

    :Example:

        >>> from jpt.base.intervals import ContinuousSet
        >>> i1 = ContinuousSet.parse('[0,1]')
        >>> i2 = ContinuousSet.parse('[2,5]')
        >>> i3 = ContinuousSet.parse('[3,4]')
        >>> i1.isempty()
        False
        >>> i1.intersects(i2)
        False
        >>> i2.intersects(i3)
        True
        >>> i2.intersection(i3)
        <ContinuousSet=[3.0,4.0]>
        >>> print(i2.intersection(i3))
        [3.0,4.0]
        >>> i4 = i1.union(i2)
        >>> print(i4)
        <UnionSet=[<ContinuousSet [0.0,1.0]>; <ContinuousSet [2.0,3.0]>]>
        >>> i5 = i4.union(ContinuousSet('[0.5,3]'))
        >>> print(i5)
        [0.0,3.0]
    """

    def __init__(
            self,
            DTYPE_t lower,
            DTYPE_t upper,
            np.int32_t left = 0,
            np.int32_t right = 0
    ):
        if lower > upper:
            raise ValueError(
                'Lower bound must be lower or equal '
                'to upper bound. Got lower = %s > upper = %s.' % (lower, upper)
            )
        self._lower = lower
        self._upper = upper
        self.left = ifnot(left, _INC)
        self.right = ifnot(right, _INC)

    @property
    def lower(self):
        return self._lower

    @property
    def upper(self):
        return self._upper

    @lower.setter
    def lower(self, l):
        self._lower = l

    @upper.setter
    def upper(self, u):
        self._upper = u

    @staticmethod
    def fromstring(str s):
        """
        Parse a string to a ContinuousSet.
        Round brackets are open borders, rectangular brackets are closed borders
        :param s: The string to parse
        :return: The corresponding ContinuousSet
        """
        if s == _CHAR_EMPTYSET:
            return ContinuousSet._emptyset()
        interval = ContinuousSet(np.nan, np.nan)
        tokens = re_int.match(s.replace(" ", "").replace('∞', 'inf'))

        if tokens is None:
            raise ValueError('Malformed input string: "{}"'.format(interval))

        if tokens.group('ldelim') in ['(', ']']:
            interval.left = _EXC

        elif tokens.group('ldelim') == '[':
            interval.left = _INC

        else:
            raise ValueError('Illegal left delimiter {} in interval {}'.format(tokens.group('ldelim'),
                                                                               interval))

        if tokens.group('rdelim') in [')', '[']:
            interval.right = _EXC

        elif tokens.group('rdelim') == ']':
            interval.right = _INC

        else:
            raise ValueError('Illegal right delimiter {} in interval {}'.format(tokens.group('rdelim'),
                                                                                interval))

        try:
            interval.lower = <DTYPE_t> float(tokens.group('lval'))
            interval.upper = <DTYPE_t> float(tokens.group('rval'))

        except:
            traceback.print_exc()
            raise ValueError(
                'Illegal interval values {}, {} in interval {}'.format(
                tokens.group('lval'),
                    tokens.group('rval'),
                    interval
                )
            )

        return interval

    parse = ContinuousSet.fromstring

    @staticmethod
    def from_list(l: List[float]) -> ContinuousSet:
        '''
        Converts a list representation of an interval to an instance of type
        '''
        lower, upper = l
        return ContinuousSet(
            np.NINF if lower in (np.NINF, -float('inf'), None, ...) else np.float64(lower),
            np.PINF if upper in (np.PINF, float('inf'), None, ...) else np.float64(upper)
        )

    cpdef SIZE_t itype(self):
        """
        Get the interval type.
        :return: 2 for closed, 3 for half open, 4 for open
        """
        return self.right + self.left

    cpdef SIZE_t isempty(self):
        """
        Check if this interval is empty.
        :return: boolean describing if this is empty or not.
        """
        if self.lower > self.upper:
            return True
        if self.lower == self.upper:
            return not self.isclosed()
        return self.lower + eps >= self.upper and self.itype() == OPEN

    cpdef SIZE_t isclosed(self):
        """
        Check if this interval is closed.
        :return: boolean describing is this is closed or not.
        """
        return self.itype() == CLOSED

    cpdef SIZE_t isninf(self):
        """
        Check if this interval is infinite to the left (negative infty)
        :return:
        """
        return np.isinf(self.lower)

    cpdef SIZE_t ispinf(self):
        """
        Check if this interval is infinite to the right (positive infty)
        :return:
        """
        return np.isinf(self.upper)
    #
    cpdef SIZE_t isinf(self):
        """
        Check if this interval is infinite to the right (positive infty)
        :return:
        """
        return self.ispinf() or self.isninf()

    @staticmethod
    cdef NumberSet _emptyset():
        """
        Create an empty interval centered at 0.
        :return: An empty interval
        """
        return ContinuousSet(0, 0, EXC, EXC)

    @staticmethod
    def emptyset() -> ContinuousSet:
        return ContinuousSet._emptyset()

    @staticmethod
    cdef Interval _allnumbers():
        """
        Create a ContinuousSet that contains all numbers but infinity and -infinity
        :return: Infinitely big ContinuousSet
        """
        return ContinuousSet(np.NINF, np.PINF, _EXC, _EXC)

    @staticmethod
    def allnumbers():
        return ContinuousSet._allnumbers()

    cpdef DTYPE_t[::1] _sample(self, SIZE_t k=1, DTYPE_t[::1] result=None):
        """
        Draw from this interval ``k`` evenly distributed samples.
        :param k: The amount of samples
        :param result: optional array to write into
        :return: The drawn samples
        """

        if self.isempty():
            raise ValueError('Cannot sample from an empty set.')

        if self.isinf():
            raise ValueError('Cannot sample uniformly from an infinite set.')

        cdef DTYPE_t upper = self.upper if self.right == _INC else np.nextafter(self.upper, self.upper - 1)
        cdef DTYPE_t lower = self.lower if self.left == _INC else np.nextafter(self.lower, self.lower + 1)

        if result is None:
            result = np.random.uniform(
                max(np.finfo(np.float64).min, lower),
                min(np.finfo(np.float64).max, upper),
                k
            )

        else:
            result[...] = np.random.uniform(
                max(np.finfo(np.float64).min, lower),
                min(np.finfo(np.float64).max, upper),
                k
            )
        return result

    def any_point(self) -> numbers.Real:
        '''
        Returns an arbitrary point that lies in this ``ContinuousSet``.

        The difference to ``ContinuousSet.sample()`` is that ``any()`` does also return
        a value for infinite intervals and is deterministic.
        :return:
        '''
        if self.ispinf() and self.isninf():
            return 0
        elif self.isninf():
            return self.max - eps
        elif self.ispinf():
            return self.min + eps
        return (self.min + self.max) * .5

    cpdef DTYPE_t[::1] linspace(
            self,
            SIZE_t num,
            DTYPE_t default_step=1,
            DTYPE_t[::1] result=None
    ):
        """
        Create a linear space from lower to upper in this interval.
        Open and closed borders are treated the same.
        Can be numerical imprecise in the interval borders.
        If the interval is infinite a linear space is created with half of the ``num`` on the left
        side of 0 and the other half on the right side.

        :param num: the amount of steps
        :param default_step: the step-size
        :param result: optional array to write in
        :return: np.array containing the linear space from lower to upper
        """

        if self.isempty():
            raise IndexError('Cannot create linspace from an empty set.')

        cdef DTYPE_t start, stop
        cdef DTYPE_t[::1] samples

        if result is None:
            samples = np.ndarray(shape=num, dtype=np.float64)
        else:
            samples = result

        cdef DTYPE_t n
        cdef DTYPE_t space, val
        cdef np.int32_t alternate = 1

        if self.lower == np.NINF and self.upper == np.PINF:
            alternate = -1
            space = default_step
            start = 0

        elif self.lower == np.NINF:
            space = -default_step
            start = self.upper

        elif self.upper == np.PINF:
            space = default_step
            start = self.lower

        else:
            start = self.lower
            stop = self.upper
            if num > 1:
                n = <DTYPE_t> num - 1
                space = abs((stop - start)) / n

        val = start
        cdef np.int64_t i
        cdef np.int32_t pos = math.floor(num / 2) - (0 if num % 2 else 1)

        if num == 1:
            if alternate == -1:
                samples[0] = 0
            elif self.lower == np.NINF:
                samples[0] = self.upper
            elif self.upper == np.PINF:
                samples[0] = self.lower
            else:
                samples[0] = (stop + start) / 2

        else:
            for i in range(num - 1, -1, -1) if space < 0 else range(num):
                pos += <np.int32_t> round(i * (-1) ** (i + 1))
                samples[i if alternate == 1 else pos] = val if (alternate != -1 or i % 2) else -val
                if alternate != -1 or (not i % 2 or val == 0):
                    val += space

        if self.left == EXC and self.lower != np.NINF:
            samples[0] = np.nextafter(samples[0], samples[0] + 1)

        if self.right == EXC and self.upper != np.PINF:
            samples[-1] = np.nextafter(samples[-1], samples[-1] - 1)

        return samples

    cpdef NumberSet copy(self):
        """
        Return an exact copy of this interval.
        :return: the exact copy
        """
        return ContinuousSet(
            self._lower,
            self._upper,
            self.left,
            self.right
        )

    cpdef SIZE_t contains_value(self, DTYPE_t value):
        """
        Checks if ``value`` lies in interval
        :param value: the value
        :return: True if the value is in this interval, else False
        """
        return self.intersects(
            ContinuousSet(value, value)
        )

    cpdef SIZE_t issuperseteq(self, NumberSet other):
        if other.isempty():
            return True
        return (
            (self.min <= other.min and self.max >= other.max)
        )

    cpdef SIZE_t issuperset(self, NumberSet other):
        if other.isempty() and not self.isempty():
            return True
        return (
            (self.min < other.min and self.max >= other.max)
            or ((self.min <= other.min and self.max > other.max))
        )

    cpdef SIZE_t contains_interval(
            self,
            NumberSet other,
            int proper_containment=False
    ):
        """
        Checks if ``other`` lies in interval.

        :param other: the other interval
        :param proper_containment:  If ``proper_containment`` is ``True``, ``other`` needs to be properly surrounded by
        this set, i.e. the intersection of both needs to consists of two non-empty, disjoint and
        non-contiguous intervals.
        :return: True if the other interval is contained, else False
        """
        if self.isempty():
            return False
        # if isinstance(other, UnionSet):
        #     return all(
        #         [self.contains_interval(i, proper_containment=proper_containment) for i in other.intervals]
        #     )
        if self.lowermost() > other.lowermost() or self.uppermost() < other.uppermost():
            return False
        if self.lower == other.lower:
            if self.left == _EXC and other.left == _INC:
                return False
            elif self.left == other.left and proper_containment:
                return False
        if self.upper == other.upper:
            if self.right == _EXC and other.right == _INC:
                return False
            elif self.right == other.right and proper_containment:
                return False
        return True

    cpdef SIZE_t contiguous(self, Interval other):
        """
        Checks if this interval and the given interval are contiguous.
        :param other: the other interval
        :return: True if they are contiguous, else False
        """
        if not isinstance(other, ContinuousSet):
            raise TypeError(
                f'Operation ContinuousSet.contiguous() not supported on argument of type {other.__class__.__qualname__}'
            )
        if self.lower == other.upper and (other.right + self.left == HALFOPEN):
            return True
        if self.upper == other.lower and (self.right + other.left == HALFOPEN):
            return True
        return False

    cpdef SIZE_t intersects(self, NumberSet other):
        """
        Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.SInterval
        :returns True if the two intervals intersect, False otherwise
        :rtype: bool
        """
        if other.isempty() or self.isempty():
            return False
        if isinstance(other, UnionSet):
            return other.intersects(self)
        elif isinstance(other, ContinuousSet):
            if other.lower > self.upper or other.upper < self.lower:
                return False
            if self.lower == other.upper and (other.right == _EXC or self.left == _EXC):
                return False
            if self.upper == other.lower and (self.right == _EXC or other.left == _EXC):
                return False
        else:
            raise TypeError(
                f'Unsupprted argument type: {other.__class__.__qualname__}'
            )
        return True

    cpdef SIZE_t isdisjoint(self, NumberSet other):
        """
        Check if ``other`` and this are disjoint, i. e. do not intersect.
        :param other: the other NumberSet
        :return: True if they are disjoint, False if they intersect
        """
        return not self.intersects(other)

    cpdef NumberSet intersection(
            self,
            NumberSet other
    ):
        """
        Compute the intersection of this ``ContinuousSet`` and ``other``.

        The arguments ``left`` and ``right`` (both boolean) can be used to specify if
        the left or right end of the interval should be open or closed.

        :param other: The other ContinousSet
        :param left: Open/Close flag for the left border
        :param right: Open/Close flag for the right border
        :return:

        :Example:
        >>> from jpt.base.intervals import ContinuousSet, INC, EXC
        >>> i1 = ContinuousSet.parse('[0,1]')
        >>> i2 = ContinuousSet.parse('[0.5,1.5]')
        >>> i1.intersection(i2)
        <ContinuousSet=[0.500,1.000]>
        """
        if not self.intersects(other):
            return ContinuousSet._emptyset()
        if isinstance(other, ContinuousSet):
            result = ContinuousSet(
                max(self.lower, other.lower),
                min(self.upper, other.upper)
            )
            result.left = (
                max(self.left, other.left)
                if other.lower == self.lower
                else (
                    self.left
                    if self.lower > other.lower
                    else
                    other.left)
            )
            result.right = (
                max(self.right, other.right)
                if other.upper == self.upper
                else (
                    self.right
                    if self.upper < other.upper
                    else
                    other.right
                )
            )
        else:
            result = other.intersection(self)
        return result

    cpdef ContinuousSet intersection_with_ends(
            self,
            ContinuousSet other,
            int left = 0,
            int right = 0
    ):
        '''
        Like .intersection(), but supports the option to specify
        whether the ``left`` and ``right`` ends of the resulting interval
        shall be open or closed.

        Example:
        >>> i1.intersection_with_ends(i2, right=EXC)
            <ContinuousSet=[0.500,1.000[>
            >>> i1.intersection_with_ends(i2).upper
            1.0
            >>> i1.intersection_with_ends(i2, right=EXC).upper
            1.0000000000000002
        '''
        cdef ContinuousSet result = self.intersection(other)
        # if left == _INC and result.left == _EXC:
        #     result.left = _INC
        #     result.lower = np.nextafter(result.lower, result.lower + 1)
        # elif left == _EXC and result.left == _INC:
        #     result.left = _EXC
        #     result.lower = np.nextafter(result.lower, result.lower - 1)
        # if right == _INC and result.right == _EXC:
        #     result.right = _INC
        #     result.upper = np.nextafter(result.upper, result.upper - 1)
        # elif right == _EXC and result.right == _INC:
        #     result.right = _EXC
        #     result.upper = np.nextafter(result.upper, result.upper + 1)
        return result.ends(left, right)

    cpdef ContinuousSet boundaries(self, int left= 0, int right=0):
        cdef ContinuousSet result = self.copy()
        cdef lower_minus_eps = np.nextafter(result.lower, result.lower - 1.)
        cdef lower_plus_eps = np.nextafter(result.lower, result.lower + 1.)
        cdef upper_minus_eps = np.nextafter(result.upper, result.upper - 1.)
        cdef upper_plus_eps = np.nextafter(result.upper, result.upper + 1.)

        if result.left == _INC and left == _EXC:
            result.lower = lower_minus_eps
            result.left = _EXC
        elif result.left == _EXC and left == _INC:
            result.lower = lower_plus_eps
            result.left = _INC
        if result.right == _INC and right == _EXC:
            result.upper = upper_plus_eps
            result.right = _EXC
        elif result.right == _EXC and right == _INC:
            result.upper = upper_minus_eps
            result.right = _INC
        return result

    cpdef NumberSet union(self, NumberSet other):
        """
        Compute the union of this ``ContinuousSet`` and ``other``.
        :param other: The other ContinuousSet
        :return: The union of both sets as ContinuousSet if the union is contiguous or UnionSet if it is not.
        """
        if not self.intersects(other) and not self.contiguous(other):
            if self.isempty():
                return other.copy()
            return UnionSet([self]).union(UnionSet([other]))
        cdef np.int32_t left = (min(self.left, other.left) if self.lower == other.lower
                                else (self.left if self.lower < other.lower else other.left))
        cdef np.int32_t right = (min(self.right, other.right) if self.upper == other.upper
                                 else (self.right if self.upper > other.upper else other.right))
        return ContinuousSet(min(self.lower, other.lower), max(self.upper, other.upper), left, right)

    cpdef NumberSet difference(self, NumberSet other):
        """
        Compute the set difference of this ``ContinuousSet`` minus ``other``.
        :param other: the other NumberSet
        :return: difference of those sets as UnionSet
        """
        if other.isempty():
            return self.copy()
        if isinstance(other, UnionSet):
            return UnionSet([self]).difference(other)
        cdef NumberSet result
        if other.issuperseteq(self):
            return ContinuousSet._emptyset()

        elif self.contains_interval(other, proper_containment=True):
            result = UnionSet([
                ContinuousSet(
                    self.lower,
                    other.lower,
                    self.left,
                    _INC if other.left == _EXC else _EXC
                ),
                ContinuousSet(
                    other.upper,
                    self.upper,
                    _INC if other.right == _EXC else _EXC,
                    self.right
                )
            ])
            return result
        elif self.intersects(other):
            result = self.copy()
            if self.contains_value(other.uppermost()) and self.uppermost() != other.uppermost():
                result.lower = other.upper
                result.left = _INC if other.right == _EXC else _EXC
            elif self.contains_value(other.lowermost()) and self.lowermost() != other.lowermost():
                result.upper = other.lower
                result.right = _INC if other.left == _EXC else _EXC
            return result
        else:
            return self.copy()

    cpdef NumberSet complement(self):
        """
        Calculate the complement set of this interval.
        :return: Complement of this interval
        """
        return _R.difference(self)

    cpdef DTYPE_t size(self):
        """
        Alternative to __len__ but may return float (inf)
        :return: The amount of numbers in this ``ContinuousSet``
        """
        if self.isempty():
            return 0
        if self.lower == self.upper or np.nextafter(self.lower, self.upper) == self.upper and self.itype() == HALFOPEN:
            return 1
        return np.inf

    cpdef DTYPE_t fst(self):
        """
        Get the lowest value in this interval or nan if it's empty.
        :return: the lowest value as float
        """
        if self.isempty():
            return np.nan
        if self.lower != np.NINF:
            if self.left == _INC:
                return self.lower
            else:
                return np.nextafter(self.lower, self.lower + 1)
        else:
            # return np.finfo(np.float64).min
            return np.NINF

    cpdef DTYPE_t lst(self):
        if self.isempty():
            return np.nan
        if self.upper != np.PINF:
            if self.right == _INC:
                return self.upper
            else:
                return self.upper - eps
        else:
            # return np.finfo(np.float64).max
            return np.PINF

    cpdef DTYPE_t uppermost(self):
        """
        :return: The highest computer-representable value in this ``ContinuousSet``.
        """
        return self.upper if self.right == _INC else np.nextafter(self.upper, self.upper - 1)

    cpdef DTYPE_t lowermost(self):
        """
        :return: The lowest computer-representable value in this ``ContinuousSet``.
        """
        return self.lower if self.left == _INC else np.nextafter(self.lower, self.lower + 1)

    # @property
    # def min(self):
    #     return self.lowermost()
    #
    # @property
    # def max(self):
    #     return self.uppermost()

    @property
    def width(self):
        if self.isninf() or self.ispinf():
            return np.inf
        else:
            return self.max - self.min

    def chop(self, points: Iterable[float]) -> Iterable[ContinuousSet]:
        '''
        Return a sequence of contiguous intervals obtained by chopping the
        interval at hand into pieces at the locations ``points``.

        If a point does not lie within the interval, it is ignored.

        The individual chops are constructed by convention in a way such that they
        are open to their right side and the remainder of the interval is closed on its left side.
        i.e. the remainder of the interval contains the cutting point itself as an element.
        '''
        remainder = self.copy()
        for p in sorted(points):
            if p not in remainder:
                continue
            cut = ContinuousSet(remainder.lower, p, remainder.left, _EXC)
            remainder.lower = p
            remainder.left = _INC
            if cut:
                yield cut
            if not remainder:
                return
        if remainder:
            yield remainder

    cpdef ContinuousSet ends(self, int left = 0, int right = 0):
        cdef ContinuousSet result = self.copy()
        if left:
            if result.left == _INC and left == _EXC:
                result.lower = result.lower - eps if np.isfinite(result.lower) else result.lower
                result.left = _EXC
            if result.left == _EXC and left == _INC:
                result.lower = result.lower + eps if np.isfinite(result.lower) else result.lower
                result.left = _INC
        if right:
            if result.right == _INC and right == _EXC:
                result.upper = result.upper + eps if np.isfinite(result.upper) else result.upper
                result.right = _EXC
            elif result.right == _EXC and right == _INC:
                result.upper = result.upper - eps if np.isfinite(result.upper) else result.upper
                result.right = _INC
        return result

    cpdef NumberSet xmirror(self):
        '''
        Returns a modification of this ``ContinuousSet``, which has been mirrored at position ``x=0``.

        :return:
        '''
        result = self.copy()
        result.lower, result.upper = -result.upper, -result.lower
        result.left, result.right = result.right, result.left
        return result

    cpdef NumberSet simplify(self):
        return self.copy()

    def __round__(self, n: int = None):
        return ContinuousSet(
            round(self.lower, n),
            round(self.upper, n),
            self.left,
            self.right
        )

    def __eq__(self, other):
        if isinstance(other, UnionSet):
            return other == self
        return all((
            self.min == other.min,
            self.max == other.max
        )) or self.isempty() and other.isempty()

    @staticmethod
    def comparator(i1: ContinuousSet, i2: ContinuousSet) -> int:
        '''
        A comparator for sorting intervals on a total order.

        Intervals must be disjoint, otherwise a ``ValueError`` is raised.
        '''
        if i1.max < i2.min:
            return -1
        elif i2.max < i1.min:
            return 1
        else:
            raise ValueError(
                'Intervals must be disjoint, got %s and %s, intersecting at %s' % (
                    i1, i2, i1.intersection(i2)
                )
            )

    def __ne__(self, other):
        return not self == other

    def __or__(self, other: NumberSet) -> NumberSet:
        return self.union(other)

    def __and__(self, other: NumberSet) -> NumberSet:
        return self.intersection(other)

    def __sub__(self, other: NumberSet) -> NumberSet:
        return self.difference(other)

    def __str__(self):
        return self.pfmt()

    def range(self):
        """
        :return: The range of this interval as lower - upper.
        """
        return self.upper - self.lower

    def pfmt(self, number_format: str = None, notation: str = None) -> str:
        '''
        Return a pretty-formatted representation of this ``ContinuousSet``.

        :param number_format: the format string that is used to format the
                              numbers, defaults to "%s", i.e. the default string conversion.
                              May also be "%.3f" for 3-digit floating point representations, for instance.
        :param notation: Either "par" for paranthesis style format (i.e. squared brackets for
                         closed interval ends and parantheses for open interval ends, or
                         "sq" for squared brackets formatting style.
        :return:
        '''
        precision = ifnone(number_format, '%s')
        if self.isempty():
            return _CHAR_EMPTYSET
        if self.lower == self.upper and self.left == self.right == INC:
            return f'{{{precision % self.lower}}}'
        brackets = NOTATIONS[ifnone(notation, interval_notation)]
        return '{}{},{}{}'.format(
            brackets[LEFT][int(self.left)],
            '-∞' if self.lower == np.NINF else (precision % float(self.lower)),
            '∞' if self.upper == np.inf else (precision % float(self.upper)),
            brackets[RIGHT][int(self.right)]
        )

    def __repr__(self):
        return '<{}={}>'.format(
            self.__class__.__name__,
            '{}{},{}{}'.format(
                {INC: '[', EXC: '('}[int(self.left)],
                '-∞' if self.lower == np.NINF else ('%.3f' % self.lower),
                '∞' if self.upper == np.inf else ('%.3f' % self.upper),
                {INC: ']', EXC: ')'}[int(self.right)]
            )
        )

    def __hash__(self):
        return hash(
            (ContinuousSet, self.lower, self.upper, self.left, self.right)
        ) if not self.isempty() else hash(frozenset())

    def __getstate__(self):
        return self.lower, self.upper, self.left, self.right

    def __setstate__(self, x):
        self.lower, self.upper, self.left, self.right = x

    def to_json(self):
        return {
            'type': ContinuousSet.__qualname__,
            'upper': self.upper,
            'lower': self.lower,
            'left': self.left,
            'right': self.right
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> ContinuousSet:
        if 'type' in data and data.get('type') != ContinuousSet.__qualname__:
            raise ValueError(
                'Illegal type name: %s' % data.get("type")
            )
        return ContinuousSet(
            data['lower'],
            data['upper'],
            data['left'],
            data['right']
        )

    def ifin(self, x, else_=None):
        return x if x in self else else_

    EMPTY = ContinuousSet.emptyset()
    ALL = _R
