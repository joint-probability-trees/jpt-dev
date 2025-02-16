# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False
__module__ = 'intervals.pyx'

import numbers
import re
import traceback
from functools import cmp_to_key
from itertools import tee
from operator import attrgetter

import math
from typing import Iterable, Any, Tuple, List, Dict

import numpy as np
cimport numpy as np
cimport cython
from dnutils import ifnone, first, ifnot

from jpt.base.constants import eps

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
R = ContinuousSet(-np.inf, np.inf, _EXC, _EXC)


# ----------------------------------------------------------------------------------------------------------------------

cdef class NumberSet:
    """
    Abstract superclass for RealSet and ContinuousSet.
    """
    def __getstate__(self):
        return ()

    def __setstate__(self, _):
        pass

    def __and__(self, other):
        raise NotImplementedError()

    def __or__(self, other):
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------------------------------------
# String and formatting constants

_CUP = u'\u222A'
_CAP = u'\u2229'
_EMPTYSET = u'\u2205'
_INFTY = '∞'

LEFT = 0
RIGHT = 1

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
        self.intervals: List[ContinuousSet] = []
        if type(intervals) is str:
            intervals = [intervals]
        if intervals is not None:
            self.intervals = []
            for i in intervals:
                if type(i) is str:
                    i = ContinuousSet.parse(i)
                else:
                    i = i.copy()
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

    def __bool__(self):
        return not self.isempty()

    def __hash__(self):
        tmp = sorted(self.intervals, key=attrgetter('right'))
        tmp = sorted(self.intervals, key=attrgetter('upper'))
        tmp = sorted(self.intervals, key=attrgetter('left'))
        return hash((RealSet, tuple(sorted(tmp, key=attrgetter('lower')))))

    def __round__(self, n: int):
        return RealSet(
            [round(i, n) for i in self.intervals]
        )

    def __setstate__(self, state):
        self.intervals = state

    def __getstate__(self):
        return self.intervals

    def any_point(self) -> numbers.Real:
        for i in self.intervals:
            if not i.isempty():
                return i.any_point()

    def to_json(self) -> Dict[str, Any]:
        return {
            'intervals': i.to_json() for i in self.intervals
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> RealSet:
        return RealSet(
            intervals=[ContinuousSet.from_json(d) for d in data['intervals']]
        )

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

    cpdef inline np.int32_t contains_interval(RealSet self, NumberSet other):
        """
        Checks if ``value`` lies in interval
        :param other: The other interval
        :return: bool
        """
        if isinstance(other, RealSet):
            return all([self.contains_interval(i) for i in other.intervals])
        cdef ContinuousSet s
        for s in self.intervals:
            if s.contains_interval(other):
                return True
        return False

    cpdef inline np.int32_t isninf(RealSet self):
        """
        Check if this ``RealSet`` is infinite to the left (negative infty).
        :return: 
        """
        for i in self.intervals:
            if i.isninf():
                return True
        return False

    cpdef inline np.int32_t ispinf(RealSet self):
        """
        Check if this ``RealSet`` is infinite to the right (positive infty).
        :return: 
        """
        for i in self.intervals:
            if i.ispinf():
                return True
        return False

    cpdef inline np.int32_t isinf(RealSet self):
        """
        Check if this ``RealSet`` is infinite to the right OR the left (negative OR positive infty).
        :return: 
        """
        for i in self.intervals:
            if i.ispinf() or i. isninf():
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
        :return: the lowest value as number or math.nan if there are no values in this RealSet
        :rtype: numbers.Number
        """
        valid_intervals = [i.fst() for i in self.intervals if not i.isempty()]
        if len(valid_intervals) > 0:
            return min(valid_intervals)
        else:
            return math.nan

    @property
    def min(self) -> numbers.Real:
        return self.fst()

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

    def intersections(self, other: RealSet) -> RealSet:
        '''
        Compute a ``RealSet`` whose individual interval constituents contain
        all pairwise intersections of this ``RealSet``'s constituents and the ``other``
        ``RealSet``'s. The result is sorted but not simplified in the sense that
        contiguous sub-intervals are not merged.
        '''
        intervals_ = [i1.intersection(i2) for i1 in self.intervals for i2 in other.intervals]
        intervals = []
        closed = set()
        for i in intervals_:
            if i.isempty() or i in closed:
                continue
            intervals.append(i)
            closed.add(i)
        return RealSet(
            intervals=list(
                sorted(
                    intervals,
                    key=cmp_to_key(ContinuousSet.comparator)
                )
            )
        )

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
        if other.isempty():
            return self.copy()
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

    def chop(self, points: Iterable[float]) -> Iterable[ContinuousSet]:
        for i in self.simplify().intervals:
            yield from i.chop(points)

    cpdef inline RealSet xmirror(self):
        '''
        Returns a modification of this ``RealSet``, which has been mirrored at position x=0.  
        :return: 
        '''
        return RealSet(
            list(
                reversed(
                    [i.xmirror() for i in self.intervals]
                )
            )
        )

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)


# ----------------------------------------------------------------------------------------------------------------------

re_int = re.compile(r'(?P<ldelim>\(|\[|\])(?P<lval>.+),(?P<rval>.+)(?P<rdelim>\)|\]|\[)')


@cython.final
cdef class ContinuousSet(NumberSet):
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
        <RealSet=[<ContinuousSet [0.0,1.0]>; <ContinuousSet [2.0,3.0]>]>
        >>> i5 = i4.union(ContinuousSet('[0.5,3]'))
        >>> print(i5)
        [0.0,3.0]
    """

    def __init__(
            ContinuousSet self,
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
        self.lower = lower
        self.upper = upper
        self.left = ifnot(left, _INC)
        self.right = ifnot(right, _INC)

    @staticmethod
    def fromstring(str s):
        """
        Parse a string to a ContinuousSet.
        Round brackets are open borders, rectangular brackets are closed borders
        :param s: The string to parse
        :return: The corresponding ContinuousSet
        """
        if s == _EMPTYSET:
            return EMPTY
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
            raise ValueError('Illegal interval values {}, {} in interval {}'.format(tokens.group('lval'),
                                                                                    tokens.group('rval'),
                                                                                    interval))

        return interval

    parse = ContinuousSet.fromstring

    cpdef inline np.int32_t itype(ContinuousSet self):
        """
        Get the interval type.
        :return: 2 for closed, 3 for half open, 4 for open
        """
        return self.right + self.left

    cpdef inline np.int32_t isempty(ContinuousSet self):
        """
        Check if this interval is empty.
        :return: boolean describing if this is empty or not.
        """
        if self.lower > self.upper:
            return False
        if self.lower == self.upper:
            return not self.isclosed()
        return self.lower + eps >= self.upper and self.itype() == OPEN

    cpdef inline np.int32_t isclosed(ContinuousSet self):
        """
        Check if this interval is closed.
        :return: boolean describing is this is closed or not.
        """
        return self.itype() == CLOSED

    cpdef inline np.int32_t isninf(ContinuousSet self):
        """
        Check if this interval is infinite to the left (negative infty)
        :return: 
        """
        return np.isinf(self.lower)

    cpdef inline np.int32_t ispinf(ContinuousSet self):
        """
        Check if this interval is infinite to the right (positive infty)
        :return: 
        """
        return np.isinf(self.upper)

    cpdef inline np.int32_t isinf(ContinuousSet self):
        """
        Check if this interval is infinite to the right (positive infty)
        :return: 
        """
        return self.ispinf() or self.isninf()

    @staticmethod
    cdef inline ContinuousSet c_emptyset():
        """
        Create an empty interval centered at 0.
        :return: An empty interval
        """
        return ContinuousSet(0, 0, _EXC, _EXC)

    @staticmethod
    def emptyset():
        """
        Python method for creating an empty interval
        :return: An empty interval
        """
        return ContinuousSet.c_emptyset()

    @staticmethod
    cdef inline ContinuousSet c_allnumbers():
        """
        Create a ContinuousSet that contains all numbers but infinity and -infinity
        :return: Infinitely big ContinuousSet
        """
        return ContinuousSet(-np.inf, np.inf, _EXC, _EXC)

    @staticmethod
    def allnumbers():
        return ContinuousSet.c_allnumbers()

    cpdef np.ndarray[np.float64_t] sample(ContinuousSet self, np.int32_t k=1, DTYPE_t[::1] result=None):
        return np.array(self._sample(k, result))

    cpdef DTYPE_t[::1] _sample(ContinuousSet self, np.int32_t k=1, DTYPE_t[::1] result=None):
        """
        Draw from this interval ``k`` evenly distributed samples.
        :param k: The amount of samples
        :param result: optional array to write into
        :return: The drawn samples
        """

        if self.isempty():
            raise IndexError('Cannot sample from an empty set.')

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

    cpdef DTYPE_t[::1] linspace(ContinuousSet self,
                                     np.int32_t num,
                                     DTYPE_t default_step=1,
                                     DTYPE_t[::1] result=None):
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

        if self.lower == -np.inf and self.upper == np.inf:
            alternate = -1
            space = default_step
            start = 0

        elif self.lower == -np.inf:
            space = -default_step
            start = self.upper

        elif self.upper == np.inf:
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
        cdef np.int32_t pos = <np.int32_t> math.floor(num / 2) - (0 if num % 2 else 1)

        if num == 1:
            if alternate == -1:
                samples[0] = 0
            elif self.lower == -np.inf:
                samples[0] = self.upper
            elif self.upper == np.inf:
                samples[0] = self.lower
            else:
                samples[0] = (stop + start) / 2

        else:
            for i in range(num - 1, -1, -1) if space < 0 else range(num):
                pos += <np.int32_t> round(i * (-1) ** (i + 1))
                samples[i if alternate == 1 else pos] = val if (alternate != -1 or i % 2) else -val
                if alternate != -1 or (not i % 2 or val == 0):
                    val += space

        if self.left == EXC and self.lower != -np.inf:
            samples[0] = np.nextafter(samples[0], samples[0] + 1)

        if self.right == EXC and self.upper != np.inf:
            samples[-1] = np.nextafter(samples[-1], samples[-1] - 1)

        return samples

    cpdef inline ContinuousSet copy(ContinuousSet self):
        """
        Return an exact copy of this interval.
        :return: the exact copy
        """
        return ContinuousSet(self.lower, self.upper, self.left, self.right)

    cpdef inline np.int32_t contains_value(ContinuousSet self, DTYPE_t value):
        """
        Checks if ``value`` lies in interval
        :param value: the value
        :return: True if the value is in this interval, else False
        """
        return self.intersects(ContinuousSet(value, value))

    cpdef inline np.int32_t contains_interval(
            ContinuousSet self,
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
        if isinstance(other, RealSet):
            return all(
                [self.contains_interval(i, proper_containment=proper_containment) for i in other.intervals]
            )
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

    cpdef inline np.int32_t contiguous(ContinuousSet self, ContinuousSet other):
        """
        Checks if this interval and the given interval are contiguous.
        :param other: the other interval
        :return: True if they are contiguous, else False
        """
        if self.lower == other.upper and (other.right + self.left == HALFOPEN):
            return True
        if self.upper == other.lower and (self.right + other.left == HALFOPEN):
            return True
        return False

    cpdef inline np.int32_t intersects(ContinuousSet self, NumberSet other):
        """
        Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.SInterval
        :returns True if the two intervals intersect, False otherwise
        :rtype: bool
        """
        if other.isempty():
            return False
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
        """
        Check if ``other`` and this are disjoint, i. e. do not intersect.
        :param other: the other NumberSet
        :return: True if they are disjoint, False if they intersect 
        """
        return not self.intersects(other)

    cpdef inline ContinuousSet intersection(ContinuousSet self,
                                            ContinuousSet other,
                                            int left=0,
                                            int right=0):
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
        >>> i1.intersection(i2, right=EXC)
        <ContinuousSet=[0.500,1.000[>
        >>> i1.intersection(i2).upper
        1.0
        >>> i1.intersection(i2, right=EXC).upper
        1.0000000000000002
        """
        if not self.intersects(other):
            return ContinuousSet.c_emptyset()
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
        """
        Compute the union of this ``ContinuousSet`` and ``other``.
        :param other: The other ContinuousSet 
        :return: The union of both sets as ContinuousSet if the union is contiguous or RealSet if it is not.
        """
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
        """
        Compute the set difference of this ``ContinuousSet`` minus ``other``.
        :param other: the other NumberSet 
        :return: difference of those sets as RealSet
        """
        if other.isempty():
            return self.copy()
        if isinstance(other, RealSet):
            return RealSet([self]).difference(other)
        cdef NumberSet result
        if other.contains_interval(self):
            return ContinuousSet.c_emptyset()

        elif self.contains_interval(other, proper_containment=True):
            result = RealSet([
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

    cpdef inline NumberSet complement(ContinuousSet self):
        """
        Calculate the complement set of this interval.
        :return: Complement of this interval
        """
        return R.difference(self)

    cpdef inline DTYPE_t size(ContinuousSet self):
        """
        Alternative to __len__ but may return float (inf)
        :return: The amount of numbers in this ``ContinuousSet``
        """
        if self.isempty():
            return 0
        if self.lower == self.upper or np.nextafter(self.lower, self.upper) == self.upper and self.itype() == HALFOPEN:
            return 1
        return np.inf

    cpdef inline DTYPE_t fst(ContinuousSet self):
        """
        Get the lowest value in this interval or nan if it's empty.
        :return: the lowest value as float
        """
        if self.isempty():
            return np.nan
        if self.lower != -np.inf:
            if self.left == _INC:
                return self.lower
            else:
                return np.nextafter(self.lower, self.lower + 1)
        else:
            return np.finfo(np.float64).min

    cpdef inline DTYPE_t uppermost(ContinuousSet self):
        """
        :return: The highest computer-representable value in this ``ContinuousSet``.
        """
        return self.upper if self.right == _INC else np.nextafter(self.upper, self.upper - 1)

    cpdef inline DTYPE_t lowermost(ContinuousSet self):
        """
        :return: The lowest computer-representable value in this ``ContinuousSet``.
        """
        return self.lower if self.left == _INC else np.nextafter(self.lower, self.lower + 1)

    @property
    def min(self):
        return self.lowermost()

    @property
    def max(self):
        return self.uppermost()

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

    cpdef inline NumberSet simplify(self):
        return self.copy()

    cpdef inline ContinuousSet ends(self, int left=-1, int right=-1):
        cdef ContinuousSet result = self.copy()
        if left != -1:
            if result.left == _INC and left == _EXC:
                result.lower = result.lower - eps if np.isfinite(result.lower) else result.lower
                result.left = _EXC
            if result.left == _EXC and left == _INC:
                result.lower = result.lower + eps if np.isfinite(result.lower) else result.lower
                result.left = _INC
        if right != -1:
            if result.right == _INC and right == _EXC:
                result.upper = result.upper + eps if np.isfinite(result.upper) else result.upper
                result.right = _EXC
            elif result.right == _EXC and right == _INC:
                result.upper = result.upper - eps if np.isfinite(result.upper) else result.upper
                result.right = _INC
        return result

    cpdef inline ContinuousSet xmirror(self):
        '''
        Returns a modification of this ``ContinuousSet``, which has been mirrored at position ``x=0``.
        
        :return: 
        '''
        result = self.copy()
        result.lower, result.upper = -result.upper, -result.lower
        result.left, result.right = result.right, result.left
        return result

    def __round__(self, n: int = None):
        return ContinuousSet(
            round(self.lower, n),
            round(self.upper, n),
            self.left,
            self.right
        )

    def __contains__(self, x):
        try:
            if isinstance(x, ContinuousSet):
                return self.contains_interval(x)
            elif isinstance(x, RealSet):
                return all([self.contains_interval(i) for i in x.intervals])
            else:
                return self.contains_value(x)
        except TypeError:
            pass
        raise ValueError('Invalid data type: %s' % type(x).__name__)

    def __eq__(self, other):
        if isinstance(other, RealSet):
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
            return _EMPTYSET
        if self.lower == self.upper and self.left == self.right == INC:
            return f'{{{precision % self.lower}}}'
        brackets = NOTATIONS[ifnone(notation, interval_notation)]
        return '{}{},{}{}'.format(
            brackets[LEFT][int(self.left)],
            '-∞' if self.lower == -np.inf else (precision % float(self.lower)),
            '∞' if self.upper == np.inf else (precision % float(self.upper)),
            brackets[RIGHT][int(self.right)]
        )

    def __repr__(self):
        return '<{}={}>'.format(
            self.__class__.__name__,
            '{}{},{}{}'.format(
                {INC: '[', EXC: '('}[int(self.left)],
                '-∞' if self.lower == -np.inf else ('%.3f' % self.lower),
                '∞' if self.upper == np.inf else ('%.3f' % self.upper),
                {INC: ']', EXC: ')'}[int(self.right)]
            )
        )

    def __bool__(self):
        return not self.isempty()

    def __hash__(self):
        return hash(
            (ContinuousSet, self.lower, self.upper, self.left, self.right)
        )

    def __getstate__(self):
        return self.lower, self.upper, self.left, self.right

    def __setstate__(self, x):
        self.lower, self.upper, self.left, self.right = x

    def to_json(self):
        return {
            'upper': self.upper,
            'lower': self.lower,
            'left': self.left,
            'right': self.right
        }

    @staticmethod
    def from_json(data):
        return ContinuousSet(
            data['lower'],
            data['upper'],
            data['left'],
            data['right']
        )
