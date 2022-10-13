# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False
import numbers
import re
import traceback
from itertools import tee
from operator import attrgetter

import math
from typing import Iterable, Any, Tuple, List

import numpy as np
cimport numpy as np
cimport cython
from dnutils import ifnone, first, ifnot

__module__ = 'intervals.pyx'


# ----------------------------------------------------------------------------------------------------------------------

CLOSED = 2
HALFOPEN = 3
OPEN = 4

# NB: Do not remove this! Declaration of _INC and _EXC in intervals.pxd does not set their values!
_INC = 1
_EXC = 2

INC = np.int32(_INC)
EXC = np.int32(_EXC)

EMPTY = ContinuousSet(0, 0, _EXC, _EXC)
R = ContinuousSet(np.NINF, np.PINF, _EXC, _EXC)


# ----------------------------------------------------------------------------------------------------------------------

cdef class NumberSet:
    """
    Abstract superclass for RealSet and ContinuousSet.
    """
    def __getstate__(self):
        return ()

    def __setstate__(self, _):
        pass


_CUP = u'\u222A'
_CAP = u'\u2229'
_EMPTYSET = u'\u2205'


# ----------------------------------------------------------------------------------------------------------------------

def chop(seq: Iterable[Any]) -> Iterable[Tuple[Any, Iterable]]:
    """
    Returns pairs of the first element ("head") and the remainder
    ("tail") for all right subsequences of ``seq``
    :param seq: The sequence to yield from
    :return: Head and Tail of the sequence
    """
    it = iter(seq)
    try:
        head = next(it)
        it, tail = tee(it)
        yield head, tail
        yield from chop(it)
    except StopIteration:
        return


# ----------------------------------------------------------------------------------------------------------------------

@cython.final
cdef class RealSet(NumberSet):
    """
    Class for interval calculus providing basic interval manipulation, such as :func:`sample`,
    :func:`intersection` and :func:`union`. An Instance of this type actually represents a
    complex range of values (possibly with gaps) by wrapping around multiple continuous intervals
    (:class:`ContinuousSet`).
    A range of values with gaps can occur for example by unifying two intervals that do not intersect (e.g. [0, 1] and
    [3, 4]).

    .. note::
        Accepts the following string representations of types of intervals:
          - closed intervals ``[a,b]``
          - half-closed intervals ``]a,b] or (a,b] and [a,b[ or [a,b)``
          - open intervals ``]a,b[ or (a,b)``

        ``a`` and ``b`` can be of type int or float (also: scientific notation) or {+-} :math:`∞`
    """

    def __init__(RealSet self, intervals: str or List[ContinuousSet or str]=None):
        """
        Create a RealSet

        :param intervals: The List of intervals to create the RealSet from
        :type intervals: str, or List of ContinuousSet or str that can be parsed
        """

        # member for all intervals
        self.intervals = []
        if type(intervals) is str:
            intervals = [intervals]
        if intervals is not None:
            self.intervals = []
            for i in intervals:
                if type(i) is str:
                    i = ContinuousSet.parse(i)
                self.intervals.append(i)

    def __contains__(self, value: numbers.Number) -> bool:
        """
        Checks if ``value`` lies in interval
        :param value: The value to check
        :return: Rather the value is in this RealSet or not
        """
        if not isinstance(value, numbers.Number):
            raise TypeError('Containment check unimplemented fo object of type %s.' % type(value).__name__)
        return any([value in i for i in self.intervals])

    def __repr__(self):
        return '<{}=[{}]>'.format(self.__class__.__name__, '; '.join([repr(i) for i in self.intervals]))

    def __str__(self):
        if self.isempty():
            return _EMPTYSET
        return (' %s ' % _CUP).join([str(i) for i in self.intervals])

    def __eq__(self, other):
        if self.isempty() and other.isempty():
            return True
        self_ = self.simplify()
        other_ = other.simplify() if isinstance(other, RealSet) else other
        if isinstance(other_, ContinuousSet) and isinstance(self_, ContinuousSet):
            return self_ == other_
        elif isinstance(other_, RealSet) and isinstance(self_, RealSet):
            return self_.intervals == other_.intervals
        return False

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        tmp = sorted(self.intervals, key=attrgetter('right'))
        tmp = sorted(self.intervals, key=attrgetter('upper'))
        tmp = sorted(self.intervals, key=attrgetter('left'))
        return hash((RealSet, tuple(sorted(tmp, key=attrgetter('lower')))))

    def __setstate__(self, state):
        self.intervals = state

    def __getstate__(self):
        return self.intervals

    cpdef DTYPE_t size(RealSet self):
        """
        This size of this ``RealSet``.
        
        The size of a ``RealSet`` is the sum of the sizes of its constituents.
        
        Size refers to the number of values that are possible. For any range that allows more than
        one value this is infinite.
        
        :return: float
        """
        cdef DTYPE_t s = 0
        cdef int i

        # simplify set, s. t. singular values are only counted once
        simplified = self.simplify()
        if isinstance(simplified, ContinuousSet):
            return simplified.size()

        for i in range(len(simplified.intervals)):
            s += simplified.intervals[i].size()

        return s

    @staticmethod
    def emptyset():
        """
        Create an empty ``RealSet``
        :return: empty RealSet
        """
        return RealSet()

    cpdef DTYPE_t[::1] sample(RealSet self, np.int32_t n=1, DTYPE_t[::1] result=None):
        """
        Chooses an element from self.intervals proportionally to their sizes, then returns a uniformly sampled
        value from that Interval.
        :param n: The amount of samples to generate
        :param result: None or an array to write into
        :returns: value(s) from the represented value range
        :rtype: float

        """
        if self.isempty():
            raise IndexError('Cannot sample from an empty set.')
        cdef ContinuousSet i_
        cdef DTYPE_t[::1] weights = np.array([abs(i_.upper - i_.lower)
                                              for i_ in self.intervals if i_.size()], dtype=np.float64)
        if sum(weights) == 0:
            weights[...] = 1.
        cdef DTYPE_t[::1] upperbounds = np.cumsum(weights)
        if result is None:
            result = np.ndarray(shape=n, dtype=np.float64)
        cdef int i, j
        cdef DTYPE_t resval, bound
        for i in range(n):
            resval = np.random.uniform(0, min([np.finfo(np.float64).max, upperbounds[-1]]))
            for j, bound in enumerate(upperbounds):
                if resval <= bound:
                    self.intervals[j].sample(result=result[i:i+1])
                    break
        return result

    cpdef inline np.int32_t contains_value(RealSet self, DTYPE_t value):
        """
        Checks if `value` lies in this RealSet
        :param value: The value to check
        :return: Rather the value is in this RealSet or not
        """
        cdef ContinuousSet s
        for s in self.intervals:
            if s.contains_value(value):
                return True
        return False

    cpdef inline np.int32_t contains_interval(RealSet self, ContinuousSet other):
        """
        Checks if ``value`` lies in interval
        :param other: The other interval
        :return: bool
        """
        cdef ContinuousSet s
        for s in self.intervals:
            if s.contains_interval(other):
                return True
        return False

    cpdef inline np.int32_t isempty(RealSet self):
        """
        Checks whether this RealSet is empty or not.

        :returns: True if this is interval is empty, i.e. does not contain any values, False otherwise
        :rtype: bool

        :Example:

        >>> RealSet(['[0,1]']).isempty()
        False
        >>> RealSet(']1,1]').isempty()
        True

        """
        cdef ContinuousSet s
        for s in self.intervals:
            if not s.isempty():
                return False
        return True

    cpdef inline DTYPE_t fst(RealSet self):
        """
        Get the lowest value
        :return: the lowest value as number
        :rtype: numbers.Number
        """
        return min([i.fst() for i in self.intervals])

    cpdef inline np.int32_t intersects(RealSet self, NumberSet other):
        """
        Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: NumberSet
        :returns: True if the two intervals intersect, False otherwise
        :rtype: bool
        """
        if isinstance(other, ContinuousSet):
            other = RealSet([other])
        cdef ContinuousSet s, s_
        for s in self.intervals:
            for s_ in other.intervals:
                if s.intersects(s_):
                    return True
        return False

    cpdef inline np.int32_t isdisjoint(RealSet self, NumberSet other):
        """
        Checks whether the this interval is disjoint with ``other``.
        Inverse methode of ``NumberSet.intersects``
        :param other: The other NumberSet
        :return: True if the two sets are disjoint, False otherwise
        :rtype: bool
        """
        return not self.intersects(other)

    cpdef inline NumberSet intersection(RealSet self, NumberSet other):
        """
        Computes the intersection of this value range with ``other``.

        :param other: the other NumberSet
        :returns: the intersection of this interval with ``other``
        :rtype: RealSet
        """
        cdef RealSet other_
        if isinstance(other, ContinuousSet):
            other_ = RealSet([other])
        else:
            other_ = other
        cdef RealSet result = RealSet()
        for subset_i in self.intervals:
            for subset_j in other_.intervals:
                result.intervals.append(subset_j.intersection(subset_i))
        return result.simplify()

    cpdef inline NumberSet simplify(RealSet self):
        """        
        Constructs a new simplified modification of this ``RealSet`` instance, in which the
        subset intervals are guaranteed to be non-overlapping and non-empty.
        
        In the case that the resulting set comprises only a single ``ContinuousSet``,
        that ``ContinuousSet`` is returned instead.
        """
        intervals = []
        tail = list(self.intervals)
        while tail:
            head, tail_ = first(chop(tail))
            if not head.isempty():
                intervals.append(head)
            tail = []
            for subset in tail_:
                diff = subset.difference(head)
                if isinstance(diff, ContinuousSet):
                    tail.append(diff)
                else:
                    tail.extend(diff.intervals)
        if not intervals:
            return EMPTY
        head, tail = first(chop(sorted(intervals, key=attrgetter('upper'))))
        intervals = [head]
        for subset in tail:
            if intervals[-1].contiguous(subset):
                intervals[-1].upper = subset.upper
                intervals[-1].right = subset.right
            else:
                intervals.append(subset)
        if len(intervals) == 1:
            return first(intervals)
        return RealSet(intervals)

    cpdef inline RealSet copy(RealSet self):
        """
        Return a deep copy of this real-valued set.
        :return: copy of this RealSet
        """
        return RealSet([i.copy() for i in self.intervals])

    cpdef inline NumberSet union(RealSet self, NumberSet other):
        '''
        Compute the union set of this ``RealSet`` and ``other``.
        '''
        if isinstance(other, ContinuousSet):
            other = RealSet([other])
        return RealSet(self.intervals + other.intervals).simplify()

    cpdef inline NumberSet difference(RealSet self, NumberSet other):
        if isinstance(other, ContinuousSet):
            other = RealSet([other])
        intervals = []
        q = list(self.intervals)
        while q:
            s = q.pop(0)
            for subset in other.intervals:
                diff = s.difference(subset)
                if isinstance(diff, ContinuousSet):
                    s = diff
                else:
                    q.insert(0, diff.intervals[1])
                    s = diff.intervals[0]
            intervals.append(s)
        return RealSet(intervals).simplify()

    cpdef inline NumberSet complement(RealSet self):
        return RealSet([R]).difference(self).simplify()


# ----------------------------------------------------------------------------------------------------------------------

re_int = re.compile(r'(?P<ldelim>\(|\[|\])(?P<lval>.+),(?P<rval>.+)(?P<rdelim>\)|\]|\[)')


@cython.final
cdef class ContinuousSet(NumberSet):
    '''
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
        <RealSet=[<ContinuousSet [0.0,1.0]>; <ContinuousSet [2.0,3.0]>]>
        >>> i5 = i4.union(ContinuousSet('[0.5,3]'))
        >>> print(i5)
        [0.0,3.0]
    '''

    def __init__(ContinuousSet self,
                 DTYPE_t lower=np.nan,
                 DTYPE_t upper=np.nan,
                 np.int32_t left=0,
                 np.int32_t right=0):
        if lower > upper:
            raise ValueError('Lower bound must not be smaller than upper bound (%s > %s)' % (lower, upper))
        self.lower = lower
        self.upper = upper
        self.left = ifnot(left, _INC)
        self.right = ifnot(right, _INC)

    @staticmethod
    def fromstring(str s):
        if s == _EMPTYSET:
            return EMPTY
        interval = ContinuousSet()
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
            interval.lower = np.float64(tokens.group('lval'))
            interval.upper = np.float64(tokens.group('rval'))

        except:
            traceback.print_exc()
            raise ValueError('Illegal interval values {}, {} in interval {}'.format(tokens.group('lval'),
                                                                                    tokens.group('rval'),
                                                                                    interval))

        return interval

    parse = ContinuousSet.fromstring

    cpdef inline np.int32_t itype(ContinuousSet self):
        return self.right + self.left

    cpdef inline np.int32_t isempty(ContinuousSet self):
        if self.lower > self.upper:
            return False
        if self.lower == self.upper:
            return not self.isclosed()
        return np.nextafter(self.lower, self.upper) == self.upper and self.itype() == OPEN

    cpdef inline np.int32_t isclosed(ContinuousSet self):
        return self.itype() == CLOSED

    cpdef inline ContinuousSet emptyset(ContinuousSet self):
        return ContinuousSet(0, 0, _EXC, _EXC)

    cpdef inline ContinuousSet allnumbers(ContinuousSet self):
        return ContinuousSet(np.NINF, np.inf, _EXC, _EXC)

    cpdef DTYPE_t[::1] sample(ContinuousSet self, np.int32_t k=1, DTYPE_t[::1] result=None):
        '''
        Draw from this interval ``k`` evenly distributed samples.
        '''
        cdef DTYPE_t upper = self.upper if self.right == _INC else np.nextafter(self.upper, self.upper - 1)
        cdef DTYPE_t lower = self.lower if self.left == _INC else np.nextafter(self.lower, self.lower + 1)

        if result is None:
            result = np.random.uniform(max(np.finfo(np.float64).min, lower),
                                       min(np.finfo(np.float64).max, upper), k)

        else:
            result[...] = np.random.uniform(max(np.finfo(np.float64).min, lower),
                                            min(np.finfo(np.float64).max, upper), k)
        return result

    cpdef DTYPE_t[::1] linspace(ContinuousSet self,
                                     np.int32_t num,
                                     DTYPE_t default_step=1,
                                     DTYPE_t[::1] result=None):
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
                pos += i * (-1) ** (i + 1)
                samples[i if alternate == 1 else pos] = val if (alternate != -1 or i % 2) else -val
                if alternate != -1 or (not i % 2 or val == 0):
                    val += space

        if self.left == EXC and self.lower != np.NINF:
            samples[0] = np.nextafter(samples[0], samples[0] + 1)

        if self.right == EXC and self.upper != np.PINF:
            samples[-1] = np.nextafter(samples[-1], samples[-1] - 1)

        return samples

    cpdef inline ContinuousSet copy(ContinuousSet self):
        '''Return an exact copy of this interval.'''
        return ContinuousSet(self.lower, self.upper, self.left, self.right)

    cpdef inline np.int32_t contains_value(ContinuousSet self, DTYPE_t value):
        '''Checks if ``value`` lies in interval'''
        return self.intersects(ContinuousSet(value, value))

    cpdef inline np.int32_t contains_interval(ContinuousSet self,
                                              ContinuousSet other,
                                              int proper_containment=False):
        '''Checks if ``other`` lies in interval.
        
        If ``proper_containment`` is ``True``, ``other`` needs to be properly surrounded by
        this set, i.e. the intersection of both needs to consists of two non-empty, disjoint and 
        non-contiguous intervals.'''
        if self.lower > other.lower or self.upper < other.upper:
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

    cpdef inline np.int32_t contiguous(ContinuousSet self, ContinuousSet other):
        if self.lower == other.upper and (other.right + self.left == HALFOPEN):
            return True
        if self.upper == other.lower and (self.right + other.left == HALFOPEN):
            return True
        return False

    cpdef inline np.int32_t intersects(ContinuousSet self, NumberSet other):
        '''
        Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.SInterval
        :returns True if the two intervals intersect, False otherwise
        :rtype: bool
        '''
        if isinstance(other, RealSet):
            return other.intersects(self)
        if other.lower > self.upper or other.upper < self.lower:
            return False
        if self.lower == other.upper and (other.right == _EXC or self.left == _EXC):
            return False
        if self.upper == other.lower and (self.right == _EXC or other.left == _EXC):
            return False
        return True

    cpdef inline np.int32_t isdisjoint(ContinuousSet self, NumberSet other):
        '''Equivalent to ``not self.intersects(other)'''
        return not self.intersects(other)

    cpdef inline ContinuousSet intersection(ContinuousSet self,
                                            ContinuousSet other,
                                            int left=0,
                                            int right=0):
        '''
        Compute the intersection of this ``ContinuousSet`` and ``other``.
        
        The arguments ``left`` and ``right`` (both boolean) can be used to specify if
        the left or right end of the interval should be open or closed.
        
        :Example:
        >>> from jpt.base.intervals import ContinuousSet, INC, EXC
        >>> i1 = ContinuousSet.parse('[0,1]')
        >>> i2 = ContinuousSet.parse('[0.5,1.5]')
        >>> i1.intersection(i2)
        <ContinuousSet=[0.500,1.000]>
        >>> i1.intersection(i2, right=EXC)
        <ContinuousSet=[0.500,1.000[>
        >>> i1.intersection(i2).upper
        1.0
        >>> i1.intersection(i2, right=EXC).upper
        1.0000000000000002
        '''
        if not self.intersects(other):
            return self.emptyset()
        result = ContinuousSet(max(self.lower, other.lower), min(self.upper, other.upper))
        result.left = (max(self.left, other.left) if other.lower == self.lower
                       else (self.left if self.lower > other.lower else other.left))
        result.right = (max(self.right, other.right) if other.upper == self.upper
                        else (self.right if self.upper < other.upper else other.right))
        if left == _INC and result.left == _EXC:
            result.left = _INC
            result.lower = np.nextafter(result.lower, result.lower + 1)
        elif left == _EXC and result.left == _INC:
            result.left = _EXC
            result.lower = np.nextafter(result.lower, result.lower - 1)
        if right == _INC and result.right == _EXC:
            result.right = _INC
            result.upper = np.nextafter(result.upper, result.upper - 1)
        elif right == _EXC and result.right == _INC:
            result.right = _EXC
            result.upper = np.nextafter(result.upper, result.upper + 1)
        return result

    cpdef inline ContinuousSet boundaries(ContinuousSet self, int left= 0, int right=0):
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

    cpdef inline NumberSet union(ContinuousSet self, ContinuousSet other):
        '''
        Compute the union of this ``ContinuousSet`` and ``other``.
        '''
        if not self.intersects(other) and not self.contiguous(other):
            if self.isempty():
                return other.copy()
            return RealSet([self]).union(RealSet([other]))
        cdef np.int32_t left = (min(self.left, other.left) if self.lower == other.lower
                                else (self.left if self.lower < other.lower else other.left))
        cdef np.int32_t right = (min(self.right, other.right) if self.upper == other.upper
                                 else (self.right if self.upper > other.upper else other.right))
        return ContinuousSet(min(self.lower, other.lower), max(self.upper, other.upper), left, right)

    cpdef inline NumberSet difference(ContinuousSet self, NumberSet other):
        '''
        Compute the set difference of this ``ContinuousSet`` minus ``other``.
        '''
        if isinstance(other, RealSet):
            return RealSet([self]).difference(other)
        cdef NumberSet result
        if other.contains_interval(self):
            return self.emptyset()
        elif self.contains_interval(other, proper_containment=True):
            result = RealSet([
                ContinuousSet(self.lower, other.lower, self.left, _INC if other.left == _EXC else _EXC),
                ContinuousSet(other.upper, self.upper, _INC if other.left == _EXC else _EXC, self.right)
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

    cpdef inline NumberSet complement(ContinuousSet self):
        '''Return the complement set of this interval.'''
        return R.difference(self)

    cpdef inline DTYPE_t size(ContinuousSet self):
        '''Alternative to __len__ but may return float (inf)'''
        if self.isempty():
            return 0
        if self.lower == self.upper or np.nextafter(self.lower, self.upper) == self.upper and self.itype() == HALFOPEN:
            return 1
        return np.inf

    cpdef inline DTYPE_t fst(ContinuousSet self):
        if self.isempty():
            return np.nan
        if self.lower != np.NINF:
            if self.left == _INC:
                return self.lower
            else:
                return np.nextafter(self.lower, self.lower + 1)
        else:
            return np.finfo(np.float64).min

    cpdef inline DTYPE_t uppermost(ContinuousSet self):
        '''
        Return the smallest computer-representable value in this ``ContinuousSet``.
        '''
        return self.upper if self.right == _INC else np.nextafter(self.upper, self.upper - 1)

    cpdef inline DTYPE_t lowermost(ContinuousSet self):
        '''
        Return the biggest computer-representable value in this ``ContinuousSet``.
        '''
        return self.lower if self.left == _INC else np.nextafter(self.lower, self.lower + 1)

    def __contains__(self, x):
        try:
            if isinstance(x, ContinuousSet):
                return self.contains_interval(x)
            else:
                return self.contains_value(x)
        except TypeError:
            pass
        raise ValueError('Invalid data type: %s' % type(x).__name__)

    def __eq__(self, other):
        if isinstance(other, RealSet):
            return other == self
        return (self.lower == other.lower
                and self.upper == other.upper
                and self.left == other.left
                and self.right == other.right)  # hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return self.pfmt()

    def range(self):
        return self.upper - self.lower

    def pfmt(self, fmtstr=None):
        precision = ifnone(fmtstr, '%s')
        if self.isempty():
            return _EMPTYSET
        if self.lower == self.upper and self.left == self.right == INC:
            return f'{{{precision % self.lower}}}'
        return '{}{},{}{}'.format({INC: '[', EXC: ']'}[int(self.left)],
                                  '-∞' if self.lower == np.NINF else (precision % float(self.lower)),
                                  '∞' if self.upper == np.inf else (precision % float(self.upper)),
                                  {INC: ']', EXC: '['}[int(self.right)])

    def __repr__(self):
        return '<{}={}>'.format(self.__class__.__name__,
                                '{}{},{}{}'.format({INC: '[', EXC: ']'}[int(self.left)],
                                                   '-∞' if self.lower == np.NINF else ('%.3f' % self.lower),
                                                   '∞' if self.upper == np.inf else ('%.3f' % self.upper),
                                                   {INC: ']', EXC: '['}[int(self.right)]))

    def __bool__(self):
        return self.size() != 0

    def __hash__(self):
        return hash((ContinuousSet, self.lower, self.upper, self.left, self.right))

    def __getstate__(self):
        return self.lower, self.upper, self.left, self.right

    def __setstate__(self, x):
        self.lower, self.upper, self.left, self.right = x

    def to_json(self):
        return {'upper': self.upper,
                'lower': self.lower,
                'left': self.left,
                'right': self.right}

    @staticmethod
    def from_json(data):
        return ContinuousSet(data['lower'],
                             data['upper'],
                             data['left'],
                             data['right'])
