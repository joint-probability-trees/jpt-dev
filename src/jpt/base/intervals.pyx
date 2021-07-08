# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False
import re
import traceback
from operator import attrgetter

import math
import numpy as np
cimport numpy as np
cimport cython

cdef class NumberSet:

    def __getstate__(self):
        return ()

    def __setstate__(self, _):
        pass


_CUP = u'\u222A'
_CAP = u'\u2229'
_EMPTYSET = u'\u2205'


@cython.final
cdef class RealSet(NumberSet):
    """Wrapper class for intervals providing convenience functions such as :func:`sample`, :func:`intersect` and
    :func:`union`. An Instance of this type actually represents a complex range of values (possibly with gaps) by
    wrapping around multiple intervals (:class:`matcalo.utils.utils.SInterval`). Overlapping SIntervals will be
    merged.
    A range of values with gaps can occur for example by unifying two intervals that do not intersect (e.g. [0, 1] and
    [3, 4]).

    .. note::
        Accepts the following string representations of types of intervals:
          - closed intervals ``[a,b]``
          - half-closed intervals ``]a,b] or (a,b] and [a,b[ or [a,b)``
          - open intervals ``]a,b[ or (a,b)``

        ``a`` and ``b`` can be of type int or float (also: scientific notation) or {+-} :math:`∞`

    :Example:

    >>> from fta.learning.intervals import RealSet
    >>> i1 = ContinuousSet.fromstring('[0,1]')
    >>> i2 = ContinuousSet.fromstring('[2,5]')
    >>> i3 = ContinuousSet.fromstring('[3,4]')
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
    >>> i5 = i4.union(Interval('[0.5,3]'))
    >>> print(i5)
    [0.0,5.0]

    """

    def __init__(RealSet self, ContinuousSet interval=None):
        if interval is None:
            self.intervals = []
        else:
            self.intervals = [interval]

    def __contains__(self, value):
        '''Checks if `value` lies in interval'''
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
        elif isinstance(other, ContinuousSet) and len(self.intervals) == 1:
            return other == self.intervals[0]
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((RealSet, tuple(sorted(self.intervals, key=attrgetter('lower')))))

    cpdef DTYPE_t size(RealSet self):
        cdef DTYPE_t s = 0
        cdef int i
        for i in range(len(self.intervals)):
            s += self.intervals[i].size()
        return s

    @staticmethod
    def emptyset():
        return RealSet()

    cpdef DTYPE_t[::1] sample(RealSet self, np.int32_t n=1, DTYPE_t[::1] result=None):
        '''Chooses an element from self.intervals proportionally to their sizes, then returns a uniformly sampled
        value from that Interval.

        :returns: a value from the represented value range
        :rtype: float
        '''
        if self.isempty():
            raise IndexError('Cannot sample from an empty set.')
        cdef ContinuousSet i_
        cdef DTYPE_t[::1] weights = np.array([abs(i_.upper - i_.lower) for i_ in self.intervals if i_.size()], dtype=np.float64)
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
        '''Checks if ``value`` lies in interval'''
        cdef ContinuousSet s
        for s in self.intervals:
            if s.contains_value(value):
                return True
        return False

    cpdef inline np.int32_t contains_interval(RealSet self, ContinuousSet other):
        '''Checks if ``value`` lies in interval'''
        cdef ContinuousSet s
        for s in self.intervals:
            if s.contains_interval(other):
                return True
        return False

    cpdef inline np.int32_t isempty(RealSet self):
        '''Checks whether this interval contains values.

        :returns: True if this is interval is empty, i.e. does not contain any values, False otherwise
        :rtype: bool

        :Example:

        >>> Interval('[0,1]').isempty()
        False
        >>> Interval(']1,1]').isempty()
        True

        '''
        cdef ContinuousSet s
        for s in self.intervals:
            if not s.isempty():
                return False
        return True

    cpdef inline DTYPE_t fst(RealSet self):
        return min([i.fst() for i in self.intervals])

    cpdef inline np.int32_t intersects(RealSet self, RealSet other):
        '''Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.Interval
        :returns: True if the two intervals intersect, False otherwise
        :rtype: bool

        '''
        cdef ContinuousSet s, s_
        for s in self.intervals:
            for s_ in other.intervals:
                if s.intersects(s_):
                    return True
        return False

    cpdef inline RealSet intersection(RealSet self, RealSet other):
        '''Computes the intersection of this value range with ``other``.

        :param other: the other value range Interval
        :type other: matcalo.utils.utils.Interval
        :returns: the intersection of this interval with ``other``
        :rtype: matcalo.utils.utils.Interval
        '''
        cdef RealSet nint = RealSet(ContinuousSet(np.NINF, np.inf, _EXC, _EXC))
        cdef ContinuousSet current, ival
        cdef list q = self.intervals
        cdef list ivals = []
        cdef int idx
        for current in other.intervals:
            for idx, ival in enumerate(q):
                if current.intersects(ival):
                    ivals.append(current.intersection(ival))
                    q = q[idx:]
                else:
                    ivals.append(current)
        nint.intervals = ivals
        return nint

    cpdef inline RealSet union(RealSet self, RealSet other) except +:
        '''Unifies this value range with ``other``.

        :param other: the other value range Interval
        :type other: matcalo.utils.utils.Interval
        :returns: the union of this interval with ``other``
        :rtype: matcalo.utils.utils.Interval
        '''
        cdef RealSet nint = RealSet()
        cdef ContinuousSet current, ival, tmpi
        cdef list q = self.intervals + (other.intervals if other else [])
        cdef list ivals = []
        cdef int idx
        while q:
            current = q.pop(0)
            if current.isempty():
                continue
            for idx, ival in enumerate(ivals):
                if current.intersects(ival) or current.contiguous(ival):
                    q.extend(ivals[idx+1:])
                    ivals = ivals[:idx]
                    q.append(current.union(ival))
                    break
            else:
                ivals.append(current)
        nint.intervals = ivals
        return nint

    cpdef inline RealSet difference(RealSet self, RealSet other):
        pass

    cpdef inline RealSet complement(RealSet self):
        pass


re_int = re.compile(r'(?P<ldelim>\(|\[|\])(?P<lval>.+),(?P<rval>.+)(?P<rdelim>\)|\]|\[)')

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


@cython.final
cdef class ContinuousSet(NumberSet):
    '''
    Actual Interval representation. Wrapped by :class:`Interval` to allow more complex intervals with gaps.

    .. seealso:: :class:`Interval`
    '''

    def __init__(ContinuousSet self,
                 DTYPE_t lower=np.nan,
                 DTYPE_t upper=np.nan,
                 np.int32_t left=_INC,
                 np.int32_t right=_INC):
        self.lower = lower
        self.upper = upper
        self.left = left
        self.right = right

    @staticmethod
    def fromstring(str s):
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

    cpdef inline np.int32_t itype(ContinuousSet self):
        return self.right + self.left

    cpdef inline np.int32_t isempty(ContinuousSet self):
        if self.lower >= self.upper:
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
        # make sure to not sample open bounds
        # cdef np.int32_t i
        # if not self.isclosed():
        #     for i in s.shape[0]:
        #         while s[i] == self.lower and self.left == _EXC or s[i] == self.upper and self.right == _EXC:
        #             s[i] = np.random.uniform(np.max(np.finfo(np.float64).min, self.lower), min(np.finfo(np.float64).max, self.upper))
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

    cpdef inline np.int32_t contains_interval(ContinuousSet self, ContinuousSet other):
        '''Checks if ``value`` lies in interval'''
        if self.lower > other.lower or self.upper < other.upper:
            return False
        if self.lower == other.lower and self.left == _EXC:
            return False
        if self.upper == other.upper and self.right == _EXC:
            return False
        return True

    cpdef inline np.int32_t contiguous(ContinuousSet self, ContinuousSet other):
        if self.lower == other.upper and (other.right + self.left == HALFOPEN):
            return True
        if self.upper == other.lower and (self.right + other.left == HALFOPEN):
            return True
        return False

    cpdef inline np.int32_t intersects(ContinuousSet self, ContinuousSet other):
        '''Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: matcalo.utils.utils.SInterval
        :returns True if the two intervals intersect, False otherwise
        :rtype: bool
        '''
        if other.lower > self.upper or other.upper < self.lower:
            return False
        if self.lower == other.upper and (other.right == _EXC or self.left == _EXC):
            return False
        if self.upper == other.lower and (self.right == _EXC or other.left == _EXC):
            return False
        return True

    cpdef inline ContinuousSet intersection(ContinuousSet self, ContinuousSet other):
        if not self.intersects(other):
            return self.emptyset()

        cdef np.int32_t left = max(self.left, other.left) if self.lower == other.lower else (self.left if self.lower > other.lower else other.left)
        cdef np.int32_t right = max(self.right, other.right) if self.upper == other.upper else (self.right if self.upper < other.upper else other.right)
        return ContinuousSet(max(self.lower, other.lower), min(self.upper, other.upper), left, right)

    cpdef inline NumberSet union(ContinuousSet self, ContinuousSet other):
        if not self.intersects(other) and not self.contiguous(other):
            if self.isempty():
                return other.copy()
            return RealSet(self).union(RealSet(other))
        cdef np.int32_t left = min(self.left, other.left) if self.lower == other.lower else (self.left if self.lower < other.lower else other.left)
        cdef np.int32_t right = min(self.right, other.right) if self.upper == other.upper else (self.right if self.upper > other.upper else other.right)
        return ContinuousSet(min(self.lower, other.lower), max(self.upper, other.upper), left, right)

    cpdef inline NumberSet difference(ContinuousSet self, ContinuousSet other):
        cdef NumberSet result
        if self.contains_interval(other) and not (self.lower == other.lower and self.left == other.left or
                                                  self.upper == other.upper and self.right == other.right):
            result = RealSet()
            result.intervals.append(ContinuousSet(self.lower, other.lower, self.left, _INC if other.left == _EXC else _EXC))
            result.intervals.append(ContinuousSet(other.upper, self.upper, INC if other.left == _EXC else _EXC, self.right))
            return result
        elif self.intersects(other):
            result = self.copy()
            if other.contains_interval(self):
                return self.emptyset()
            elif self.lower <= other.upper <= self.upper:
                result.lower = other.upper
                result.left = _INC if other.right == _EXC else _EXC
            elif self.upper >= other.lower >= self.lower:
                result.upper = other.lower
                result.right = _INC if other.left == _EXC else _EXC
            return result
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
        return self.lower == other.lower and self.upper == other.upper and self.left == other.left and self.right == other.right  # hash(self) == hash(other)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        if self.isempty():
            return _EMPTYSET
        if self.lower == self.upper and self.left == self.right == INC:
            return '[%.3f]' % self.lower
        return '{}{},{}{}'.format({INC: '[', EXC: ']'}[int(self.left)],
                                  '-∞' if self.lower == np.NINF else ('%.3f' % self.lower),
                                  '∞' if self.upper == np.inf else ('%.3f' % self.upper),
                                  {INC: ']', EXC: '['}[int(self.right)])

    def __repr__(self):
        return '<{}={}>'.format(self.__class__.__name__, '{}{},{}{}'.format({INC: '[', EXC: ']'}[int(self.left)],
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
