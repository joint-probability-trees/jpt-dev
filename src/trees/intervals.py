import numbers
from operator import attrgetter

from numpy import random
import re
import traceback

import numpy as np

import dnutils

logger = dnutils.getlogger(name='IntervalLogger', level=dnutils.ERROR)

INC = 1
EXC = 2
CLOSED = 2
HALFOPEN = 3
OPEN = 4


class Interval:
    """Wrapper class for intervals providing convenience functions such as :func:`sample`, :func:`intersects` and
    :func:`union`. An Instance of this type actually represents a complex range of values (possibly with gaps) by
    wrapping around multiple intervals (:class:`matcalo.utils.utils.SInterval`). Overlapping SIntervals will be
    merged.
    A range of values with gaps can occur for example by unifying two intervals that do not intersects (e.g. [0, 1] and
    [3, 4]).

    .. note::
        Accepts the following string representations of types of intervals:
          - closed intervals ``[a,b]``
          - half-closed intervals ``]a,b] or (a,b] and [a,b[ or [a,b)``
          - open intervals ``]a,b[ or (a,b)``

        ``a`` and ``b`` can be of type int or float (also: scientific notation) or {+-} :math:`∞`

    :Example:

        >>> from trees.intervals import Interval
        >>> i1 = Interval.fromstring('[0,1]')
        >>> i2 = Interval.fromstring('[2,5]')
        >>> i3 = Interval.fromstring('[3,4]')
        >>> i1.isempty()
        False
        >>> i1.intersects(i2)
        False
        >>> i2.intersects(i3)
        True
        >>> i2.intersection(i3)
        <Interval=[<SInterval=[3.0,3.0]>]>
        >>> print(i2.intersection(i3))
        [3.0,3.0]
        >>> i4 = i1.union(i2)
        >>> print(i4)
        [[0.0,1.0]; [2.0,3.0]]
        >>> i5 = i4.union(Interval.fromstring('[0.5,3]'))
        >>> print(i5)
        [0.0,5.0]

    """

    def __init__(self, lower, upper, left=INC, right=INC):
        self._intervals = [SInterval(lower, upper, left=left, right=right)]

    def __contains__(self, value):
        """Checks if `value` lies in interval"""
        return any([value in i for i in self._intervals])

    def __repr__(self):
        return '<{}=[{}]>'.format(__class__.__name__, '; '.join([repr(i) for i in self._intervals]))

    def __str__(self):
        return '{}{}{}'.format('[' if len(self.intervals) > 1 else '', '; '.join([str(i) for i in self.intervals]), ']' if len(self.intervals) > 1 else '')

    def __eq__(self, other):
        if self.isempty() and other.isempty():
            return True
        elif isinstance(other, SInterval) and len(self.intervals) == 1:
            return other == self.intervals[0]
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((Interval, tuple(sorted(self._intervals, key=attrgetter('lower')))))

    def size(self):
        return sum([i.size() for i in self._intervals])

    @staticmethod
    def fromstring(interval):
        i = Interval(np.nan, np.nan)
        i._intervals = [SInterval.fromstring(interval)]
        return i

    @staticmethod
    def emptyinterval():
        return Interval(0, 0, EXC, EXC)

    @staticmethod
    def r_interval():
        """Represents ℝ"""
        return Interval(-np.inf, np.inf, EXC, EXC)

    @property
    def intervals(self):
        return self._intervals

    @intervals.setter
    def intervals(self, ints):
        self._intervals = ints

    @property
    def lower(self):
        """The lower bound.

        :returns: the lowermost bound of the value range
        :rtype: float
        """
        return np.min([i.lower for i in self._intervals])

    @property
    def upper(self):
        """The upper bound.

        :returns: the uppermost bound of the value range
        :rtype: float
        """
        return np.min([i.upper for i in self._intervals])

    def sample(self):
        """Chooses an element from self.intervals proportionally to their sizes, then returns a uniformly sampled
        value from that Interval.

        :returns: a value from the represented value range
        :rtype: float
        """
        weights = [abs(i.upper - i.lower) for i in self._intervals]

        # normalize if none of the weights is (-)infinity
        if np.inf not in weights and -np.inf not in weights:
            weights = weights / np.linalg.norm(weights, ord=1)
        upperbounds = np.cumsum(weights)
        resval = random.uniform(0, np.min([np.finfo(np.float64).max, upperbounds[-1]]))
        for i, bound in enumerate(upperbounds):
            if resval <= bound:
                return self._intervals[i].sample()

    def contains(self, value):
        """Checks if ``value`` lies in interval.

        :param value: The element to be checked if it lies in the interval.
        :type value: float
        :return: True if the value lies in the interval, False otherwise
        :rtype: bool
        """
        return any([value in i for i in self._intervals])

    def contains_value(self, value):
        """Checks if ``value`` lies in interval"""
        for s in self.intervals:
            if s.contains_value(value):
                return True
        return False

    def contains_interval(self, other):
        """Checks if ``value`` lies in interval"""
        for s in self.intervals:
            if s.contains_interval(other):
                return True
        return False

    def isempty(self):
        """Checks whether this interval contains values.

        :returns: True if this is interval is empty, i.e. does not contain any values, False otherwise
        :rtype: bool

        :Example:

            >>> Interval.fromstring('[0,1]').isempty()
            False
            >>> Interval.fromstring(']1,1]').isempty()
            True

        """
        return all([i.isempty() for i in self._intervals])

    def intersects(self, other):
        """Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.Interval
        :returns: True if the two intervals intersects, False otherwise
        :rtype: bool

        """
        return any([i.intersects(o) for i in self._intervals for o in other.intervals])

    def intersection(self, other):
        """Computes the intersection of this value range with ``other``.

        :param other: the other value range Interval
        :type other: matcalo.utils.utils.Interval
        :returns: the intersection of this interval with ``other``
        :rtype: matcalo.utils.utils.Interval
        """
        nint = Interval(0, 0, EXC, EXC)
        if not self.intersects(other):
            nint.intervals = [self.emptyinterval()]
            return nint
        q = self.intervals

        ivals = []
        for jdx, current in enumerate(other.intervals):
            for idx, ival in enumerate(q):
                if current.intersects(ival):
                    ivals.append(current.intersection(ival))
                    q = q[idx:]
                else:
                    ivals.append(current)

        nint._intervals = sorted(ivals, key=attrgetter('lower'))
        return nint

    def union(self, other):
        """Unifies this value range with ``other``.

        :param other: the other value range Interval
        :type other: matcalo.utils.utils.Interval
        :returns: the union of this interval with ``other``
        :rtype: matcalo.utils.utils.Interval
        """
        nint = Interval(0, 0, EXC, EXC)
        q = self._intervals + other.intervals
        ivals = []
        while q:
            current = q.pop(0)
            for idx, ival in enumerate(ivals):
                if current.intersects(ival) or current.contiguous(ival):
                    ivals = ivals[:idx]
                    q.append(current.union(ival))
                    break
            else:
                ivals.append(current)
        nint._intervals = sorted(ivals, key=attrgetter('lower'))
        return nint


class SInterval:
    """Actual Interval representation. Wrapped by :class:`Interval` to allow more complex intervals with gaps.

    .. seealso:: :class:`Interval`
    """

    def __init__(self, lower=np.nan, upper=np.nan, left=INC, right=INC):
        self.lower = lower
        self.upper = upper
        self.left = left
        self.right = right

    @staticmethod
    def fromstring(interval):
        i = SInterval(np.nan, np.nan)

        tokens = re.match(r'(?P<ldelim>\(|\[|\])(?P<lval>.+),(?P<rval>.+)(?P<rdelim>\)|\]|\[)', interval.replace(" ", "").replace('∞', 'inf'))
        if tokens is None:
            raise Exception('something went wrong with input {}'.format(interval))
        if tokens.group('ldelim') in ['(', ']']:
            i.left = EXC
        elif tokens.group('ldelim') == '[':
            i.left = INC
        else:
            raise Exception('Illegal left delimiter {} in interval {}'.format(tokens.group('ldelim'), interval))
        if tokens.group('rdelim') in [')', '[']:
            i.right = EXC
        elif tokens.group('rdelim') == ']':
            i.right = INC
        else:
            raise Exception('Illegal right delimiter {} in interval {}'.format(tokens.group('rdelim'), interval))
        try:
            i.lower = np.float64(tokens.group('lval'))
            i.upper = np.float64(tokens.group('rval'))
        except:
            traceback.print_exc()
            raise Exception('Illegal interval values {}, {} in interval {}'.format(tokens.group('lval'), tokens.group('rval'), interval))
        return i

    def itype(self):
        return self.right + self.left

    def isempty(self):
        return self.lower >= self.upper and (EXC == self.left or self.right == EXC)

    def isclosed(self):
        return self.itype() == OPEN

    def sample(self, k=1):
        s = random.uniform(np.max([np.finfo(np.float32).min, self.lower]), np.min([np.finfo(np.float32).max, self.upper]), k)
        # make sure to not sample open bounds
        if not self.isclosed():
            for i in range(s.shape[0]):
                while s[i] == self.lower and self.left == EXC or s == self.upper and self.right == EXC:
                    s[i] = random.uniform(np.max([np.finfo(np.float32).min, self.lower]), np.min([np.finfo(np.float32).max, self.upper]))
        return s

    def contains_value(self, value):
        """Checks if ``value`` lies in interval"""
        return self.intersects(SInterval(value, value))

    def contains_interval(self, other):
        """Checks if ``value`` lies in interval"""
        if self.lower > other.lower or self.upper < other.upper:
            return False
        if self.lower == other.lower and self.left == EXC:
            return False
        if self.upper == other.upper and self.right == EXC:
            return False
        return True

    def contiguous(self, other):
        if self.lower == other.upper and (other.right + self.left == HALFOPEN):
            return True
        if self.upper == other.lower and (self.right + other.left == HALFOPEN):
            return True
        return False

    def intersects(self, other):
        """Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.SInterval
        :returns: True if the two intervals intersects, False otherwise
        :rtype: bool
        """
        if other.lower > self.upper or other.upper < self.lower:
            return False
        if self.lower == other.upper and (other.right == EXC or self.left == EXC):
            return False
        if self.upper == other.lower and (self.right == EXC or other.left == EXC):
            return False
        return True

    def intersection(self, other):
        if not self.intersects(other):
            return Interval.emptyinterval()
        left = np.max([self.left, other.left]) if self.lower == other.lower else (self.left if self.lower > other.lower else other.left)
        right = np.max([self.right, other.right]) if self.upper == other.upper else (self.right if self.upper < other.upper else other.right)
        return SInterval(np.max([self.lower, other.lower]), np.min([self.upper, other.upper]), left, right)

    def union(self, other):
        if not self.intersects(other) and not self.contiguous(other):
            i = Interval(np.nan, np.nan)
            i.intervals = sorted([self, other], key=lambda x: x.lower)
            return i
        left = np.min([self.left, other.left]) if self.lower == other.lower else (self.left if self.lower < other.lower else other.left)
        right = np.min([self.right, other.right]) if self.upper == other.upper else (self.right if self.upper > other.upper else other.right)
        return SInterval(np.min([self.lower, other.lower]), np.max([self.upper, other.upper]), left=left, right=right)

    def size(self):
        if self.isempty():
            return 0
        if self.lower == self.upper:
            return 1
        return np.inf

    def __contains__(self, x):
        if isinstance(x, SInterval):
            return self.contains_interval(x)
        elif isinstance(x, numbers.Number):
            return self.contains_value(x)
        else:
            raise ValueError('Invalid data type: %s' % type(x))

    def __eq__(self, other):
        return self.lower == other.lower and self.upper == other.upper and self.left == other.left and self.right == other.right  # hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return '{}{},{}{}'.format('[' if self.left == INC else ']', '-∞' if self.lower == -np.inf else f'{self.lower:.2f}', '∞' if self.upper == np.inf else f'{self.upper:.2f}', ']' if self.right == INC else '[')

    def __repr__(self):
        return '<{}={}>'.format(__class__.__name__, str(self))

    def __bool__(self):
        return self.size() != 0

    def __hash__(self):
        return hash((SInterval, self.lower, self.upper, self.left, self.right))
