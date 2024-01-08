# cython: infer_types=True
# cython: language_level=3
# cython: cdivision=True
# cython: wraparound=True
# cython: boundscheck=False
# cython: nonecheck=False

__module__ = 'unionset.pyx'

from functools import cmp_to_key
from operator import attrgetter

from dnutils import first

from .base cimport DTYPE_t, SIZE_t

from .contset import ContinuousSet
from .contset cimport ContinuousSet

from .intset import IntSet
from .intset cimport IntSet

from .base import _CHAR_EMPTYSET, _CHAR_CUP, chop, EXC, INC
from .base cimport Interval

from typing import List, Dict, Any, Iterable
import numbers

cimport numpy as np
import numpy as np

_EMPTY = RealSet([])


# ----------------------------------------------------------------------------------------------------------------------

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

    def __init__(self, intervals: str or List[Interval or str]=None):
        """
        Create a RealSet

        :param intervals: The List of intervals to create the RealSet from
        :type intervals: str, or List of ContinuousSet or str that can be parsed
        """
        # member for all intervals
        self.intervals: List[Interval] = []
        if type(intervals) is str:
            intervals = [intervals]
        if intervals is not None:
            self.intervals = []
            for i in intervals:
                if type(i) is str:
                    i = Interval.parse(i)
                else:
                    i = i.copy()
                self.intervals.append(i)
        self._check_interval_types()

    @property
    def dtype(self):
        return first(self.intervals, type, type)

    @staticmethod
    def emptyset():
        return RealSet._emptyset()

    def _check_interval_types(self):
        clazz = None
        for i in self.intervals:
            if clazz is None:
                clazz = type(i)
            else:
                if not isinstance(i, clazz):
                    raise TypeError(
                        f'All intervals in a RealSet must be of the same '
                        f'``Interval`` substype, got {i.__class__.__qualname__}'
                    )
        return 0

    cpdef SIZE_t contains_value(self, DTYPE_t value):
        if not isinstance(value, numbers.Number):
            raise TypeError(
                'Containment check unimplemented fo object of type %s.' % type(value).__name__
            )
        return any([i.contains_value(value) for i in self.intervals])

    def __repr__(self):
        return '<{}=[{}]>'.format(
            self.__class__.__name__, '; '.join([repr(i) for i in self.intervals])
        )

    def __str__(self):
        if self.isempty():
            return _CHAR_EMPTYSET
        return (' %s ' % _CHAR_CUP).join([str(i) for i in self.intervals])

    def __eq__(self, other):
        if self.isempty() and other.isempty():
            return True
        self_ = self.simplify()
        other_ = other.simplify() if isinstance(other, RealSet) else other

        if isinstance(other_, Interval) and isinstance(self_, Interval):
            return self_ == other_
        elif isinstance(other_, RealSet) and isinstance(self_, RealSet):
            return self_.intervals == other_.intervals

        return False

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        return not self.isempty()

    def __hash__(self):
        return hash((
            RealSet,
            frozenset(self.intervals)
        ))

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
            'intervals': {
                'type': i.__class__.__qualname__,
                'data': i.to_json()
            } for i in self.intervals
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'RealSet':
        intervals = []
        for d in data['intervals']:
            if 'type' in d:
                clazz = {'ContinousSet': ContinuousSet, 'IntSet': IntSet}[d['type']]
                d_ = d['data']
            else:
                clazz = ContinuousSet
                d_ = d
            intervals.append(
                clazz.from_json(d_)
            )
        return RealSet(
            intervals
        )

    cpdef DTYPE_t size(self):
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
        if isinstance(simplified, Interval):
            return simplified.size()
        cdef RealSet r = simplified
        for i in range(len(r.intervals)):
            s += r.intervals[i].size()

        return s

    @staticmethod
    cdef NumberSet _emptyset():
        return _EMPTY.copy()

    cpdef DTYPE_t[::1] _sample(self, SIZE_t n=1, DTYPE_t[::1] result=None):
        """
        Chooses an element from self.intervals proportionally to their sizes, then returns a uniformly sampled
        value from that Interval.
        :param n: The amount of samples to generate
        :param result: None or an array to write into
        :returns: value(s) from the represented value range
        :rtype: float

        """
        if self.isempty():
            raise ValueError('Cannot sample from an empty set.')
        cdef Interval i_
        cdef DTYPE_t[::1] weights = np.array(
            [abs(i_.upper - i_.lower) for i_ in self.intervals if np.isfinite(i_.size())],
            dtype=np.float64
        )
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

    cpdef SIZE_t issuperseteq(self, NumberSet other):
        """
        Checks if ``value`` lies in interval
        :param other: The other interval
        :return: bool
        """
        if isinstance(other, Interval):
            other = RealSet([other])
        return (other - self).isempty()

    cpdef SIZE_t isninf(self):
        """
        Check if this ``RealSet`` is infinite to the left (negative infty).
        :return:
        """
        for i in self.intervals:
            if i.isninf():
                return True
        return False

    cpdef SIZE_t ispinf(self):
        """
        Check if this ``RealSet`` is infinite to the right (positive infty).
        :return:
        """
        for i in self.intervals:
            if i.ispinf():
                return True
        return False

    cpdef SIZE_t isinf(self):
        """
        Check if this ``RealSet`` is infinite to the right OR the left (negative OR positive infty).
        :return:
        """
        for i in self.intervals:
            if i.ispinf() or i. isninf():
                return True
        return False

    cpdef SIZE_t isempty(self):
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
        cdef Interval s
        for s in self.intervals:
            if not s.isempty():
                return False
        return True

    cpdef DTYPE_t fst(self):
        """
        Get the lowest value
        :return: the lowest value as number or math.nan if there are no values in this RealSet
        :rtype: numbers.Number
        """
        valid_intervals = [i.fst() for i in self.intervals if i]
        if len(valid_intervals) > 0:
            return min(valid_intervals)
        else:
            return np.nan

    cpdef DTYPE_t lst(self):
        """
        Get the lowest value
        :return: the lowest value as number or math.nan if there are no values in this RealSet
        :rtype: numbers.Number
        """
        valid_intervals = [i.lst() for i in self.intervals if i]
        if len(valid_intervals) > 0:
            return max(valid_intervals)
        else:
            return np.nan

    cpdef SIZE_t intersects(self, NumberSet other):
        """
        Checks whether the this interval intersects with ``other``.

        :param other: the other interval
        :type other: NumberSet
        :returns: True if the two intervals intersect, False otherwise
        :rtype: bool
        """
        if isinstance(other, Interval):
            other = RealSet([other])
        cdef Interval s, s_
        for s in self.intervals:
            for s_ in other.intervals:
                if s.intersects(s_):
                    return True
        return False

    cpdef SIZE_t isdisjoint(self, NumberSet other):
        """
        Checks whether the this interval is disjoint with ``other``.
        Inverse methode of ``NumberSet.intersects``
        :param other: The other NumberSet
        :return: True if the two sets are disjoint, False otherwise
        :rtype: bool
        """
        return not self.intersects(other)

    cpdef NumberSet intersection(self, NumberSet other):
        """
        Computes the intersection of this value range with ``other``.

        :param other: the other NumberSet
        :returns: the intersection of this interval with ``other``
        :rtype: RealSet
        """
        cdef RealSet other_
        if isinstance(other, Interval):
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
        if other.dtype is not ContinuousSet:
            raise TypeError(
                'Method intersections() is not applicible to RealSet of type %s' % other.dtype
            )
        intervals_ = [
            i1.intersection_with_ends(i2, left=INC, right=EXC)
            for i1 in self.intervals
            for i2 in other.intervals
        ]
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

    cpdef NumberSet simplify(self, SIZE_t keep_type=False):
        """
        Constructs a new simplified modification of this ``RealSet`` instance, in which the
        subset intervals are guaranteed to be non-overlapping and non-empty.

        In the case that the resulting set comprises only a single ``ContinuousSet``,
        that ``ContinuousSet`` is returned instead.
        """
        intervals = []
        tail = [i.copy() for i in self.intervals]
        while tail:
            head, tail_ = first(chop(tail))
            if not head.isempty():
                intervals.append(head)
            tail = []
            for subset in tail_:
                diff = subset.difference(head)
                if isinstance(diff, Interval):
                    tail.append(diff)
                else:
                    tail.extend(diff.intervals)
        if not intervals:
            return _EMPTY.copy()

        head, tail = first(
            chop(sorted(intervals, key=attrgetter('upper')))
        )
        intervals = [head]
        for subset in tail:
            if intervals[-1].contiguous(subset):
                intervals[-1].upper = subset.upper
                if isinstance(intervals[-1], ContinuousSet):
                    intervals[-1].right = subset.right
            else:
                intervals.append(subset)
        if not keep_type and len(intervals) == 1:
            return first(intervals)
        return RealSet(intervals)

    cpdef NumberSet copy(self):
        """
        Return a deep copy of this real-valued set.
        :return: copy of this RealSet
        """
        return RealSet(
            [i.copy() for i in self.intervals]
        )

    cpdef NumberSet union(self, NumberSet other):
        '''
        Compute the union set of this ``RealSet`` and ``other``.
        '''
        if isinstance(other, Interval):
            other = RealSet([other])
        cdef RealSet other_ = <RealSet> other
        return RealSet(
            self.intervals + other_.intervals
        ).simplify()

    cpdef NumberSet difference(self, NumberSet other):
        self._check_interval_types()
        if other.isempty() or self.isdisjoint(other):
            return self.copy()
        if isinstance(other, Interval):
            other = RealSet([other])
        intervals = []
        q = [i.copy() for i in self.intervals]
        cdef NumberSet diff
        while q:
            s = q.pop(0)
            for subset in other.intervals:
                diff = s.difference(subset)
                if isinstance(diff, Interval):
                    s = diff
                elif isinstance(diff, RealSet):
                    q.insert(0, diff.intervals[1])
                    s = diff.intervals[0]
                else:
                    raise TypeError('This should not happen.')
            intervals.append(s)
        return RealSet(intervals).simplify(keep_type=True)

    # cpdef inline NumberSet complement(self):
    #     return RealSet([R]).difference(self).simplify()

    def chop(self, points: Iterable[float]) -> Iterable[ContinuousSet]:
        for i in self.simplify(keep_type=True).intervals:
            yield from i.chop(points)

    cpdef NumberSet xmirror(self):
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
        ) if not self.isempty() else RealSet.EMPTY

    def __and__(self, other):
        return self.intersection(other)

    def __or__(self, other):
        return self.union(other)

    EMPTY = RealSet.emptyset()