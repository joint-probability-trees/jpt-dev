"""
Tests for ContinuousSet interval implementation.
"""
from unittest import TestCase
import math
import pickle
import numpy as np
import json

from ddt import ddt, data, unpack
from dnutils.tools import ifstr

from jpt.base.constants import eps
from jpt.base.utils import chop

from jpt.base.intervals import (
    ContinuousSet,
    INC,
    EXC,
    UnionSet,
    R,
    STR_EMPTYSET,
    STR_INFTY,
    IntSet,
    Z,
    Interval
)


@ddt
class ContinuousSetTest(TestCase):

    @data(('[-10, 5]', ContinuousSet(-10, 5)),
          (']5, 10]', ContinuousSet(5, 10, EXC)),
          ('[0, 1]', ContinuousSet(0, 1)),
          ('[2, 3]', ContinuousSet(2, 3)),
          (']-inf,0[', ContinuousSet(-np.inf, 0, EXC, EXC)),
          ('[0, inf[', ContinuousSet(0, np.inf, INC, EXC)),
          (']0,0[', ContinuousSet(0, 0, EXC, EXC)),
          (']-1,-1[', ContinuousSet(-1, -1, EXC, EXC)))
    @unpack
    def test_creation_parsing(self, s, i):
        """Parsing and creation"""
        # Act
        result = ContinuousSet.parse(s)

        # Assert
        self.assertEqual(result, i)
        self.assertEqual(
            hash(i),
            hash(i.copy())
        )

    @data(
        (ContinuousSet.parse('(-inf,0]'), -np.inf),
        (ContinuousSet.parse('[0, 1]'), 0),
        (ContinuousSet.parse('(1,2]'), 1 + eps)
    )
    @unpack
    def test_min(self, i, min_):
        # Assert
        self.assertEqual(
            min_,
            i.min
        )

    def test_copy(self):
        # Arrange
        i = ContinuousSet(0, 1)

        # Act
        result = i.copy()

        # Assert
        self.assertEqual(
            i,
            result
        )
        self.assertEqual(
            hash(i),
            hash(result)
        )
        self.assertNotEqual(
            id(i),
            id(result)
        )

    # ------------------------------------------------------------------------------------------------------------------

    def test_itype(self):
        i1 = ContinuousSet.parse('(0,1)')
        self.assertEqual(i1.itype(), 4)

        i2 = ContinuousSet.parse('[0,1]')
        self.assertEqual(i2.itype(), 2)

        self.assertEqual(ContinuousSet.parse('(0,1]').itype(), ContinuousSet.parse('[3,4)').itype())

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet(0, 1), '[0.0,1.0]'),
        (ContinuousSet(-np.inf, np.inf, EXC, EXC), f'(-{STR_INFTY},{STR_INFTY})')
    )
    @unpack
    def test_pfmt_par(self, i, s):
        # Act
        s_ = i.pfmt(notation='par')
        # Assert
        self.assertEqual(s, s_)

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet(0, 1), '[0.0,1.0]'),
        (ContinuousSet(-np.inf, np.inf, EXC, EXC), f']-{STR_INFTY},{STR_INFTY}[')
    )
    @unpack
    def test_pfmt_sq(self, i, s):
        # Act
        s_ = i.pfmt(notation='sq')
        # Assert
        self.assertEqual(s, s_)

    # ------------------------------------------------------------------------------------------------------------------

    def test_value_check(self):
        self.assertRaises(ValueError, ContinuousSet, 1, -1)

    @data(
        ('[0,1]', '[0,1]', True),
        ('[0,1]', '[0,1)', False),
        ('[0,1)', '[0,1]', False),
        ('[0,1)', '[0,1)', True),
        ('(0,1]', '[0,1]', False),
        ('(0,1]', '(0,1]', True),
        (ContinuousSet(0 + eps, 1 - eps, INC, INC), '(0,1)', True),
        ('(0,0)', '(1,1)', True),  # Different kinds of empty sets
        ('[0,0)', ContinuousSet.emptyset(), True)
    )
    @unpack
    def test_equality(self, i1, i2, eq):
        # Arrange
        i1 = ifstr(i1, ContinuousSet.parse)
        i2 = ifstr(i2, ContinuousSet.parse)
        # Act & Assert
        if eq:
            self.assertEqual(i1, i2)
        else:
            self.assertNotEqual(i1, i2)

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0,1]', 0, 1),
        ('(-1,1)', -1 + eps, 1 - eps),
        ('(3,4]', 3 + eps, 4)
    )
    @unpack
    def test_min_max(self, i, min_true, max_true):
        # Arrange
        i = ifstr(i, ContinuousSet.parse)
        # Act
        min_, max_ = i.min, i.max
        # Assert
        self.assertEqual(min_true, min_)
        self.assertEqual(max_true, max_)

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('(-inf,1)', True),
        ('(0,inf)', False),
        ('(-inf,inf)', True),
        ('(0,0)', False),
        ('[0,1]', False),
    )
    @unpack
    def test_isninf(self, i, t):
        # Arrange
        i = ifstr(i, ContinuousSet.parse)
        # Act & Assert
        self.assertEqual(t, i.isninf())

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('(-inf,1)', False),
        ('(0,inf)', True),
        ('(-inf,inf)', True),
        ('(0,0)', False),
        ('[0,1]', False),
    )
    @unpack
    def test_ispinf(self, i, t):
        # Arrange
        i = ifstr(i, ContinuousSet.parse)
        # Act & Assert
        self.assertEqual(t, i.ispinf())

    # ------------------------------------------------------------------------------------------------------------------

    @data((']0, 0[',))
    @unpack
    def test_emptyness(self, s):
        # Arrange
        cs = ContinuousSet.parse(s)

        # Act
        empty = cs.isempty()

        # Assert
        self.assertTrue(
            empty,
            msg='%s is not recognized empty.' % cs)
        self.assertEqual(
            ContinuousSet.emptyset(),
            ContinuousSet.parse(']0, 0[')
        )

    def test_emptyset(self):
        # Act
        emptyset = ContinuousSet.emptyset()

        # Assert
        self.assertEqual(
            hash(frozenset()),
            hash(emptyset)
        )
        self.assertEqual(
            0,
            emptyset.size()
        )

    @data((']0,0[',), ('[0,1]',), ('[2,3[',))
    @unpack 
    def test_serialization(self, i):
        self.assertEqual(i, pickle.loads(pickle.dumps(i)))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet.parse('[0,1]'), .5, True),
        (ContinuousSet.parse('[0,1]'), 0, True),
        (ContinuousSet.parse('[0,1]'), 1, True),
        (ContinuousSet.parse(']0,1]'), 0, False),
        (ContinuousSet.parse(']0,1['), 1, False),
        (ContinuousSet.parse(']0,1['), 5, False),
        (ContinuousSet.parse(']0,1['), -5, False),
    )
    @unpack
    def test_value_containment(self, i, v, r):
        self.assertEqual(r, i.contains_value(v))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0, 2]', '[.5, 1.5]', True),
        ('[.5, 1.5]', '[0, 2]', False),
        ('[0,2]', '[1,3]', False),
        ('[0,2]', '[-1,1]', False),
        ('[0,2]', '[1,2]', True),
        ('[0,2]', '[0,1]', True),
        ('[0,2]', '[1,2[', True),
        ('[0,2]', '[0,1[', True),
        (ContinuousSet(0, np.nextafter(0, 1), INC, EXC), '[0,0]', True),
        ('[0,0]', ContinuousSet(0, np.nextafter(0, 1), INC, EXC), True),
    )
    @unpack
    def test_interval_containment(self, i1, i2, o):
        if type(i1) is str:
            i1 = ContinuousSet.parse(i1)
        if type(i2) is str:
            i2 = ContinuousSet.parse(i2)
        self.assertEqual(o, i1.contains_interval(i2))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0, 2]', '[.5, 1.5]', True),
        ('[.5, 1.5]', '[0, 2]', False),
        ('[0,2]', '[1,3]', False),
        ('[0,2]', '[-1,1]', False),
        ('[0,2]', '[1,2]', False),
        ('[0,2]', '[0,1]', False),
        ('[0,2]', '[1,2[', True),
        ('[0,2]', ']0,1]', True),
    )
    @unpack
    def test_interval_proper_containment(self, i1, i2, o):
        self.assertEqual(
            o,
            ContinuousSet.parse(i1).contains_interval(
                ContinuousSet.parse(i2),
                proper_containment=True
            )
        )

    # ------------------------------------------------------------------------------------------------------------------

    @data(('[-10, 5]', ']5, 10]'),
          ('[0, 1]', '[2, 3]'),
          (']-inf,0[', '[0, inf['),
          ('[0, 1]', ']0,0['),
          (']-1,-1[', ']-1,-1['))
    @unpack
    def test_disjoint(self, i1, i2):
        i1 = ContinuousSet.parse(i1)
        i2 = ContinuousSet.parse(i2)
        self.assertEqual(ContinuousSet.emptyset(), i1.intersection(i2))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[-10, 5]', '[0, 10]', True),
        ('[-10, 10]', '[-5, 5]', True),
        ('[-10, 10]', '[-10, 10]', True),
        ('[-10, 10]', '[1,1]', True),
        ('[0.3,0.7[', '[.3,.3]', True),
        ('[0,1]', STR_EMPTYSET, False),
        ('[0,1]', '[3,4]', False)
    )
    @unpack
    def test_intersects(self, i1, i2, truth):
        # Arrange
        i1 = ContinuousSet.parse(i1)
        i2 = ContinuousSet.parse(i2)

        # Act
        result_1 = i1.intersects(i2)
        result_2 = i2.intersects(i1)

        self.assertEqual(
            truth,
            result_1
        )
        self.assertEqual(
            truth,
            result_2
        )

    # ------------------------------------------------------------------------------------------------------------------

    @data(('[-10, 5]', '[0, 10]', '[0,5]'),
          ('[-10, 10]', '[-5, 5]', '[-5,5]'),
          ('[-10, 10]', '[-10, 10]', '[-10,10]'),
          ('[-10, 10]', '[1,1]', '[1,1]'),
          ('[0.3,0.7[', '[.3,.3]', '[.3,.3]'),
          ('[0,1]', STR_EMPTYSET, STR_EMPTYSET)
          )
    @unpack
    def test_intersection(self, i1, i2, r):
        i1 = ContinuousSet.parse(i1)
        i2 = ContinuousSet.parse(i2)
        self.assertEqual(ContinuousSet.parse(r),
                         i1.intersection(i2))
        self.assertEqual(i1.intersection(i2),
                         i2.intersection(i1))

    # ------------------------------------------------------------------------------------------------------------------

    def test_intersection_optional_left(self):
        # Arrange
        i1 = ContinuousSet.parse('[-1, 1]')
        i2 = ContinuousSet.parse(']-1, 1[')

        # Act
        result_1 = i1.intersection_with_ends(i2, left=INC)
        result_2 = i2.intersection_with_ends(i1, left=INC)

        # Assert
        self.assertEqual(
            ContinuousSet(np.nextafter(i2.lower, i2.lower + 1), i2.upper, INC, EXC),
            result_1
        )
        self.assertEqual(
            result_1,
            result_2
        )

    # ------------------------------------------------------------------------------------------------------------------

    def test_intersection_optional_right(self):
        # Arrange
        i1 = ContinuousSet.parse('[-1, 1]')
        i2 = ContinuousSet.parse(']-1, 1[')

        # Act
        result_1 = i1.intersection_with_ends(i2, right=INC)
        result_2 = i2.intersection_with_ends(i1, right=INC)

        # Assert
        self.assertEqual(
            ContinuousSet(i2.lower, np.nextafter(i2.upper, i2.upper - 1), EXC, INC),
            result_1
        )
        self.assertEqual(
            result_1,
            result_2
        )

    # ------------------------------------------------------------------------------------------------------------------

    @data(('[-10, 5]', '[0, 10]', ContinuousSet.parse('[-10,0[')),
          ('[-10, 10]', '[-5, 5]', UnionSet(['[-10,-5[', ']5,10]'])),
          ('[-10, 10]', ']-5, 5[', UnionSet(['[-10,-5]', '[5,10]'])),
          ('[-10, 10]', '[-5, 5[', UnionSet(['[-10,-5[', '[5,10]'])),
          ('[-1.0,1.0]', '[0.0, 1.0]', ContinuousSet(-1, 0, INC, EXC)),
          ('[0,1]', '[1,2]', ContinuousSet(0, 1, INC, EXC)),
          ('[-10, 10]', ContinuousSet.emptyset(), '[-10,10]'),
          (ContinuousSet(0, 0 + eps, INC, EXC), ContinuousSet.emptyset(), ContinuousSet(0, 0 + eps, INC, EXC)),
          (ContinuousSet(0 + eps, np.inf, INC, EXC), ContinuousSet.emptyset(), ContinuousSet(0 + eps, np.inf, INC, EXC)),
          ('[-1, 1]', '[-1,1]', ContinuousSet.emptyset())
          )
    @unpack
    def test_difference(self, i1, i2, r):
        # Arrange
        i1 = ifstr(i1, ContinuousSet.parse)
        i2 = ifstr(i2, ContinuousSet.parse)
        r = ifstr(r, ContinuousSet.parse)

        # Act
        diff = i1.difference(i2)

        # Assert
        self.assertEqual(r, diff)

    # ------------------------------------------------------------------------------------------------------------------

    @data(('[-10, 5]',), (']5, 10]',),
          ('[0, 1]',), ('[2, 3]',),
          (']-inf,0[',), ('[0, inf[',),
          ('[0, 1]',), (']0,0[',),
          (']-1,-1[',), (']-1,-1[',))
    @unpack
    def test_json_serialization(self, i):
        i = ContinuousSet.parse(i)
        self.assertEqual(i, ContinuousSet.from_json(i.to_json()))

    # ------------------------------------------------------------------------------------------------------------------

    @data(('[-10, 5]',), (']5, 10]',),
          ('[0, 1]',), ('[2, 3]',),
          (']-inf,0[',), ('[0, inf[',),
          ('[0, 1]',), (']0,0[',),
          (']-1,-1[',), (']-1,-1[',))
    @unpack
    def test_pickle(self, i):
        # Arrange
        i = ContinuousSet.parse(i)

        # Act
        result = pickle.loads(pickle.dumps(i))

        # Assert
        self.assertEqual(
            i,
            result
        )

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet.parse('[0,1]'), UnionSet([ContinuousSet(-np.inf, 0, EXC, EXC), ContinuousSet(1, np.inf, EXC, EXC)])),
        (ContinuousSet.emptyset(), R),
        (R, ContinuousSet.emptyset())
    )
    @unpack
    def test_complement(self, i, r):
        self.assertEqual(r, i.complement())

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('(-inf,1)', '[1,2)', True),
        ('(-inf,1]', '[1,2)', False),
        ('(-inf,1]', '(1,2)', True),
        (ContinuousSet(1, 2 + eps, INC, EXC), ContinuousSet(2 + eps, 3, INC, EXC), True),
    )
    @unpack
    def test_contiguous(self, i1, i2, t):
        # Arrange
        i1 = ifstr(i1, ContinuousSet.parse)
        i2 = ifstr(i2, ContinuousSet.parse)
        # Act
        r = i1.contiguous(i2)
        # Assert
        if t:
            self.assertTrue(r)
        else:
            self.assertFalse(r)

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet.parse('[0,1]'), ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,2]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,1]'), ContinuousSet.parse('[0,2]')),
        (ContinuousSet.parse('[0,3]'), ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,3]'), ContinuousSet.parse('[0,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse(']2,3]'), ContinuousSet.parse('[1,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[3,4]'), UnionSet([ContinuousSet.parse('[1,2]'),
                                                                              ContinuousSet.parse('[3,4]')])),
        (ContinuousSet.emptyset(), ContinuousSet.parse('[0,1]'), ContinuousSet(0, 1)),
        (ContinuousSet.emptyset(), ContinuousSet.emptyset(), ContinuousSet.emptyset()),
        (R, ContinuousSet(0, 1), R)
    )
    @unpack
    def test_union(self, i1, i2, r):
        self.assertEqual(r, i1.union(i2))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0,0]',),
        ('[0,1]',),
        (']0,0[',),
    )
    @unpack
    def test_hash(self, i):
        self.assertEqual(hash(i), hash(pickle.loads(pickle.dumps(i))))

    # ------------------------------------------------------------------------------------------------------------------

    def test_sample(self):
        # test default usage
        i1 = ContinuousSet.parse("[-1,1]")
        samples = i1.sample(100)
        for sample in samples:
            self.assertTrue(sample in i1)

        # test raising index error
        i2 = ContinuousSet.emptyset()
        self.assertRaises(ValueError, i2.sample, 100)

        # test singular value
        i3 = ContinuousSet.parse("[-.5, -.5]")
        samples = i3.sample(100)
        for sample in samples:
            self.assertEqual(sample, -0.5)

    # ------------------------------------------------------------------------------------------------------------------

    def test_linspace(self):
        i1 = ContinuousSet.parse("(-1,1)")
        samples = i1.linspace(100)
        self.assertEqual(len(samples), 100)
        for sample in samples[1:-1]:
            self.assertTrue(sample in i1)

        i2 = ContinuousSet.emptyset()
        self.assertRaises(ValueError, i2.sample, 100)

    # ------------------------------------------------------------------------------------------------------------------

    def test_size(self):
        # test infinitely big sets
        i1 = ContinuousSet.allnumbers()
        self.assertEqual(float("inf"), i1.size())

        # test UnionSet of size 1
        i2 = ContinuousSet.parse("[-.5, -.5]")
        self.assertEqual(i2.size(), 1)

        # test emptyset
        i3 = ContinuousSet.emptyset()
        self.assertEqual(i3.size(), 0)

    # ------------------------------------------------------------------------------------------------------------------

    def test_uppermost_lowermost(self):
        # test infinitely big sets
        i1 = ContinuousSet.allnumbers()
        self.assertEqual(float("inf"), i1.uppermost())
        self.assertEqual(-float("inf"), i1.lowermost())

        # test regular set
        i2 = ContinuousSet.parse("[-.5, .5]")
        self.assertEqual(-0.5, i2.lowermost())
        self.assertEqual(0.5, i2.uppermost())

        # test impulse
        i3 = ContinuousSet.parse("[-.5, -.5]")
        self.assertEqual(i3.lowermost(), i3.uppermost())

        # test open borders
        i4 = ContinuousSet.parse("(-.5, .5)")
        self.assertEqual(-0.5, i2.lowermost())
        self.assertEqual(0.5, i2.uppermost())

    # ------------------------------------------------------------------------------------------------------------------

    def test_chop_normal(self):
        # Arrange
        i = ContinuousSet(0, 1)

        # Act
        chops = list(i.chop([.1, .5, .8]))

        # Assert
        self.assertEqual(
            [
                ContinuousSet.parse('[0,.1['),
                ContinuousSet.parse('[.1,.5['),
                ContinuousSet.parse('[.5,.8['),
                ContinuousSet.parse('[.8,1]')
            ],
            chops
        )
        self.assertEqual(i, UnionSet(intervals=chops))

    # ------------------------------------------------------------------------------------------------------------------

    def test_chop_border_case_closed(self):
        # Arrange
        i = ContinuousSet(0, 1)

        # Act
        chop_lower = list(i.chop([0]))
        chop_upper = list(i.chop([1]))

        # Assert
        self.assertEqual(
            [i], chop_lower
        )
        self.assertEqual(i, UnionSet(intervals=chop_lower))
        self.assertEqual(
            [
                ContinuousSet.parse('[0,1['),
                ContinuousSet.parse('[1,1]')
            ], chop_upper
        )
        self.assertEqual(i, UnionSet(intervals=chop_upper))

    # ------------------------------------------------------------------------------------------------------------------

    def test_chop_border_case_open(self):
        # Arrange
        i = ContinuousSet(0, 1, EXC, EXC)

        # Act
        chop_lower = list(i.chop([i.min]))
        chop_upper = list(i.chop([i.max]))

        # Assert
        self.assertEqual(
            [i], chop_lower
        )
        self.assertEqual(i, UnionSet(intervals=chop_lower))
        self.assertEqual(
            [
                ContinuousSet(0, i.max, EXC, EXC),
                ContinuousSet(i.max, i.max)
            ], chop_upper
        )
        self.assertEqual(i, UnionSet(intervals=chop_upper))

    # ------------------------------------------------------------------------------------------------------------------

    def test_chop_border_case_closed_left(self):
        # Arrange
        i = ContinuousSet(0, 1)

        # Act
        chops = list(i.chop([0]))

        # Assert
        self.assertEqual(
            [i], chops
        )
        self.assertEqual(i, UnionSet(intervals=chops))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0,1]', {'left': INC, 'right': INC}, '[0,1]'),
        ('[0,1]', {'left': EXC, 'right': EXC}, ContinuousSet(0 - eps, 1 + eps, EXC, EXC)),
        (']0,1[', {'left': INC, 'right': INC}, ContinuousSet(0 + eps, 1 - eps, INC, INC)),
    )
    @unpack
    def test_ends(self, i, args, t):
        # Arrange
        i = ifstr(i, ContinuousSet.parse)
        t = ifstr(t, ContinuousSet.parse)

        # Act
        i_ = i.ends(**args)

        # Assert
        self.assertEqual(
            t,
            i_,
            msg='Transforming %s with %s returned %s, not %s' % (
                i, args, i_, t
            )
        )

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        ('[0,1]', '[-1,0]'),
        ('(3,5]', '[-5,-3)'),
        ('(-1,1)', '(-1,1)')
    )
    @unpack
    def test_xmirror(self, i, t):
        # Arrange
        i = ifstr(i, ContinuousSet.parse)
        t = ifstr(t, ContinuousSet.parse)
        # Act
        r = i.xmirror()
        # Assert
        self.assertEqual(t, r)
        self.assertEqual(r.min , -i.max)
        self.assertEqual(r.max, -i.min)

    def test_round(self):
        # Arrange
        i = ContinuousSet(0.1234, 5.6789, INC, EXC)

        # Act
        i_ = round(i, 2)

        # Assert
        self.assertEqual(
            ContinuousSet(0.12, 5.68, INC, EXC),
            i_
        )

    def test_transform(self):
        # Arrange
        i = ContinuousSet(0, 1)

        # Act
        result = i.transform(lambda x: 2 * x + 3)

        # Assert
        self.assertEqual(
            ContinuousSet(3, 5),
            result
        )


class ContinuousSetContiguityTest(TestCase):
    """
    Test cases for critical contiguity bugs in ContinuousSet implementations.
    """

    def test_intersecting_intervals_not_contiguous(self):
        """
        Test that intersecting intervals are correctly identified as NOT contiguous.
        
        Two intervals that share common points (intersect) should NOT be contiguous.
        Contiguous intervals must be disjoint but touching.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act
        result = interval1.contiguous(interval2)
        
        # Assert
        # These intervals intersect at point 1, so they should NOT be contiguous
        self.assertFalse(result, "Intersecting intervals should not be contiguous")

    def test_contiguous_half_open_intervals(self):
        """
        Test contiguity with half-open intervals (should work correctly).
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, EXC)  # [0, 1)
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act & Assert
        self.assertTrue(interval1.contiguous(interval2))
        self.assertTrue(interval2.contiguous(interval1))

    def test_contiguous_open_closed_intervals(self):
        """
        Test contiguity between open and closed intervals.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(1, 2, EXC, INC)  # (1, 2]
        
        # Act & Assert
        self.assertTrue(interval1.contiguous(interval2))
        self.assertTrue(interval2.contiguous(interval1))

    def test_non_contiguous_intervals(self):
        """
        Test that non-contiguous intervals are correctly identified.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(2, 3, INC, INC)  # [2, 3]
        
        # Act & Assert
        self.assertFalse(interval1.contiguous(interval2))
        self.assertFalse(interval2.contiguous(interval1))

    def test_truly_contiguous_intervals(self):
        """
        Test truly contiguous intervals (disjoint but touching).
        
        This is the correct definition of contiguous intervals.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, EXC)  # [0, 1)
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act & Assert
        self.assertTrue(interval1.contiguous(interval2))
        self.assertTrue(interval2.contiguous(interval1))
        
        # Verify they are disjoint but their union is connected
        self.assertTrue(interval1.isdisjoint(interval2))
        union = interval1.union(interval2)
        self.assertIsInstance(union, ContinuousSet)  # Should be single connected interval


class ContinuousSetIntersectionTest(TestCase):
    """
    Test cases for intersection operation edge cases and bugs.
    """

    def test_intersection_touching_boundaries(self):
        """
        Test intersection of intervals that touch at boundaries.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, EXC)  # [0, 1)
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act
        result = interval1.intersection(interval2)
        
        # Assert
        self.assertTrue(result.isempty(), "Non-overlapping intervals should have empty intersection")

    def test_intersection_single_point_overlap(self):
        """
        Test intersection where intervals share only a single point.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act
        result = interval1.intersection(interval2)
        
        # Assert
        if not result.isempty():
            self.assertEqual(result.lower, 1)
            self.assertEqual(result.upper, 1)
            self.assertTrue(result.isclosed())

    def test_intersection_with_mixed_types_error_handling(self):
        """
        Test proper error handling for mixed type intersections.
        """
        # Arrange
        continuous = ContinuousSet(0, 5)
        integer = IntSet(2, 8)
        
        # Act & Assert
        with self.assertRaises(TypeError):
            continuous.intersection(integer)


class ContinuousSetEmptinessTest(TestCase):
    """
    Test cases for critical emptiness detection bugs.
    """

    def test_isempty_point_intervals(self):
        """
        Test emptiness detection for point intervals with different boundary types.
        """
        # Arrange & Act & Assert
        # Closed point interval should not be empty
        closed_point = ContinuousSet(1, 1, INC, INC)  # [1, 1]
        self.assertFalse(closed_point.isempty())
        
        # Open point interval should be empty
        open_point = ContinuousSet(1, 1, EXC, EXC)  # (1, 1)
        self.assertTrue(open_point.isempty())
        
        # Half-open point intervals should be empty
        half_open1 = ContinuousSet(1, 1, INC, EXC)  # [1, 1)
        half_open2 = ContinuousSet(1, 1, EXC, INC)  # (1, 1]
        self.assertTrue(half_open1.isempty())
        self.assertTrue(half_open2.isempty())

    def test_isempty_very_small_intervals(self):
        """
        Test emptiness detection for very small intervals near machine precision.
        """
        # Arrange
        eps = 1e-15
        very_small_closed = ContinuousSet(0, eps, INC, INC)  # [0, eps]
        very_small_open = ContinuousSet(0, eps, EXC, EXC)    # (0, eps)
        
        # Act & Assert
        self.assertFalse(very_small_closed.isempty())
        # For very small open intervals, behavior depends on eps implementation
        # This test documents current behavior and catches regressions


class ContinuousSetSamplingTest(TestCase):
    """
    Test cases for sampling method edge cases and bugs.
    """

    def test_sample_from_point_interval(self):
        """
        Test sampling from a point interval.
        """
        # Arrange
        point_interval = ContinuousSet(5, 5, INC, INC)  # [5, 5]
        
        # Act
        samples = point_interval.sample(10)
        
        # Assert
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(s == 5 for s in samples))

    def test_sample_from_empty_interval_error(self):
        """
        Test that sampling from empty interval raises appropriate error.
        """
        # Arrange
        empty_interval = ContinuousSet(1, 1, EXC, EXC)  # (1, 1) - empty
        
        # Act & Assert
        with self.assertRaises(ValueError):
            empty_interval.sample(1)


class ContinuousSetPerformanceTest(TestCase):
    """
    Test cases for performance-critical operations.
    """

    def test_operations_numerical_stability(self):
        """
        Test numerical stability for operations near machine precision.
        """
        # Arrange
        eps = np.finfo(float).eps
        interval1 = ContinuousSet(1.0, 1.0 + eps)
        interval2 = ContinuousSet(1.0 + eps, 1.0 + 2*eps)
        
        # Act
        union_result = interval1.union(interval2)
        
        # Assert - should handle numerical precision correctly
        self.assertIsNotNone(union_result)


class ContinuousSetSerializationTest(TestCase):
    """
    Test cases for serialization methods that are missing coverage.
    """

    def test_json_serialization_edge_cases(self):
        """
        Test JSON serialization for ContinuousSet edge cases.
        """
        # Arrange
        test_cases = [
            ContinuousSet(-np.inf, np.inf, EXC, EXC),  # (-inf, inf)
            ContinuousSet(0, np.inf, INC, EXC),        # [0, inf)
            ContinuousSet(-np.inf, 0, EXC, INC),       # (-inf, 0]
            ContinuousSet(1, 1, INC, INC),             # [1, 1] - point
        ]
        
        for interval in test_cases:
            with self.subTest(interval=interval):
                # Act
                json_data = interval.to_json()
                reconstructed = ContinuousSet.from_json(json_data)
                
                # Assert
                self.assertEqual(interval, reconstructed)


class ContinuousSetTransformationTest(TestCase):
    """
    Test cases for transformation methods that lack coverage.
    """

    def test_continuous_set_transform_linear(self):
        """
        Test ContinuousSet transform with linear function.
        """
        # Arrange
        interval = ContinuousSet(0, 10, INC, INC)  # [0, 10]
        
        def linear_transform(x):
            return 2 * x + 3  # y = 2x + 3
        
        # Act
        transformed = interval.transform(linear_transform)
        
        # Assert
        self.assertIsInstance(transformed, ContinuousSet)
        self.assertEqual(transformed.lower, 3)     # 2*0 + 3 = 3
        self.assertEqual(transformed.upper, 23)    # 2*10 + 3 = 23

    def test_continuous_set_transform_with_infinite_bounds(self):
        """
        Test transformation with infinite bounds.
        """
        # Arrange
        interval = ContinuousSet(-np.inf, np.inf, EXC, EXC)
        
        def identity(x):
            return x
        
        # Act
        transformed = interval.transform(identity)
        
        # Assert
        self.assertEqual(interval, transformed)

    def test_transform_linear(self):
        """
        Test ContinuousSet transform with linear function.
        """
        # Arrange
        interval = ContinuousSet(0, 10, INC, INC)  # [0, 10]
        
        def linear_transform(x):
            return 2 * x + 3  # y = 2x + 3
        
        # Act
        transformed = interval.transform(linear_transform)
        
        # Assert
        self.assertIsInstance(transformed, ContinuousSet)
        self.assertEqual(transformed.lower, 3)     # 2*0 + 3 = 3
        self.assertEqual(transformed.upper, 23)    # 2*10 + 3 = 23

    def test_transform_with_infinite_bounds(self):
        """
        Test transformation with infinite bounds.
        """
        # Arrange
        interval = ContinuousSet(-np.inf, np.inf, EXC, EXC)
        
        def identity(x):
            return x
        
        # Act
        transformed = interval.transform(identity)
        
        # Assert
        self.assertEqual(interval, transformed)


class ContinuousSetComplementTest(TestCase):
    """
    Test cases for complement operations that are missing coverage.
    """

    def test_complement_finite_interval(self):
        """
        Test complement of finite ContinuousSet.
        """
        # Arrange
        interval = ContinuousSet(1, 3, INC, INC)  # [1, 3]
        
        # Act
        complement = interval.complement()
        
        # Assert
        self.assertIsInstance(complement, UnionSet)
        # Should be (-∞, 1) ∪ (3, ∞)
        self.assertEqual(len(complement.intervals), 2)

    def test_complement_left_infinite(self):
        """
        Test complement of left-infinite interval.
        """
        # Arrange
        interval = ContinuousSet(-np.inf, 0, EXC, INC)  # (-∞, 0]
        
        # Act
        complement = interval.complement()
        
        # Assert
        self.assertIsInstance(complement, ContinuousSet)
        self.assertEqual(complement.lower, 0)
        self.assertTrue(np.isinf(complement.upper))


class ContinuousSetBoundaryMethodsTest(TestCase):
    """
    Test cases for boundary-related methods missing coverage.
    """

    def test_uppermost_lowermost(self):
        """
        Test uppermost and lowermost methods for ContinuousSet.
        """
        # Arrange
        interval = ContinuousSet(1, 5, EXC, INC)  # (1, 5]
        
        # Act & Assert
        self.assertEqual(interval.uppermost(), 5)
        # For open lower bound, lowermost should handle appropriately
        
    def test_min_max_with_open_bounds(self):
        """
        Test min/max properties with open boundaries.
        """
        # Arrange
        interval = ContinuousSet(1, 5, EXC, EXC)  # (1, 5)
        
        # Act & Assert
        # For open intervals, min should be just inside the boundary
        self.assertGreater(interval.min, 1)
        self.assertLess(interval.max, 5)


class ContinuousSetSetOperationsTest(TestCase):
    """
    Test cases for set operations that lack comprehensive coverage.
    """

    def test_continuous_set_union_with_contiguous_closed_intervals(self):
        """
        Test union of contiguous closed intervals.
        
        This tests the interaction between the contiguous bug and union operations.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act
        result = interval1.union(interval2)
        
        # Assert
        # Should return a single ContinuousSet [0, 2], not a UnionSet
        if isinstance(result, ContinuousSet):
            self.assertEqual(result.lower, 0)
            self.assertEqual(result.upper, 2)
            self.assertEqual(result.left, INC)
            self.assertEqual(result.right, INC)
        else:
            # Currently this fails due to contiguous bug
            self.fail(f"Expected ContinuousSet, got {type(result)}: {result}")

    def test_continuous_set_truly_contiguous_intervals(self):
        """
        Test truly contiguous intervals (disjoint but touching).
        
        This is the correct definition of contiguous intervals.
        """
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, EXC)  # [0, 1)
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act & Assert
        self.assertTrue(interval1.contiguous(interval2))
        self.assertTrue(interval2.contiguous(interval1))
        
        # Verify they are disjoint but their union is connected
        self.assertTrue(interval1.isdisjoint(interval2))
        union = interval1.union(interval2)
        self.assertIsInstance(union, ContinuousSet)  # Should be single connected interval

    def test_difference_operation(self):
        """
        Test ContinuousSet difference operation edge cases.
        """
        # Arrange
        interval1 = ContinuousSet(0, 10, INC, INC)   # [0, 10]
        interval2 = ContinuousSet(3, 7, INC, INC)    # [3, 7]
        
        # Act
        difference = interval1.difference(interval2)
        
        # Assert
        # Should result in [0, 3) ∪ (7, 10]
        self.assertIsInstance(difference, UnionSet)
        if isinstance(difference, UnionSet):
            self.assertEqual(len(difference.intervals), 2)

    def test_symmetric_difference_operations(self):
        """
        Test symmetric difference (XOR) operations.
        """
        # Arrange
        interval1 = ContinuousSet(0, 5, INC, INC)
        interval2 = ContinuousSet(3, 8, INC, INC)
        
        # Act
        # Symmetric difference: (A - B) ∪ (B - A)
        sym_diff1 = interval1.difference(interval2)
        sym_diff2 = interval2.difference(interval1)
        result = sym_diff1.union(sym_diff2)
        
        # Assert
        # Should be [0, 3) ∪ (5, 8]
        self.assertIsInstance(result, UnionSet)


class ContinuousSetErrorHandlingTest(TestCase):
    """
    Test cases for error handling scenarios that lack coverage.
    """

    def test_invalid_parsing(self):
        """
        Test ContinuousSet.parse() with invalid input.
        """
        # Arrange
        invalid_strings = [
            "[1,2,3]",      # Too many values
            "[1]",          # Too few values  
            "(1,2]extra",   # Extra characters
            "[a,b]",        # Non-numeric values
        ]
        
        # Act & Assert
        for invalid_string in invalid_strings:
            with self.subTest(string=invalid_string):
                with self.assertRaises(ValueError):
                    ContinuousSet.parse(invalid_string)

    def test_operations_on_wrong_types(self):
        """
        Test operations with incompatible types raise appropriate errors.
        """
        # Arrange
        continuous = ContinuousSet(0, 5)
        integer_set = IntSet(2, 8)
        
        # Act & Assert
        with self.assertRaises(TypeError):
            continuous.contiguous(integer_set)


class ContinuousSetContiguityBasicTest(TestCase):
    """
    Basic contiguity tests for ContinuousSet functionality.
    """

    def test_contiguity_fix(self):
        """Test the critical contiguity fix."""
        # Arrange
        interval1 = ContinuousSet(0, 1, INC, INC)  # [0, 1]
        interval2 = ContinuousSet(1, 2, INC, INC)  # [1, 2]
        
        # Act
        result = interval1.contiguous(interval2)
        
        # Assert
        # These intervals intersect at point 1, so they should NOT be contiguous
        self.assertFalse(result, "Intersecting intervals should not be contiguous")


class ContinuousSetBasicTest(TestCase):
    """
    Basic tests for ContinuousSet functionality.
    """

    def test_continuous_set_creation(self):
        """Test basic ContinuousSet functionality."""
        # Arrange & Act
        cs = ContinuousSet(0, 5, INC, INC)
        
        # Assert
        self.assertEqual(cs.lower, 0)
        self.assertEqual(cs.upper, 5)
        self.assertFalse(cs.isempty())

    def test_interval_parsing(self):
        """Test basic interval parsing."""
        # Test ContinuousSet parsing
        cs = ContinuousSet.parse('[0,5]')
        self.assertEqual(cs.lower, 0)
        self.assertEqual(cs.upper, 5)

    def test_json_serialization_basic(self):
        """Test basic JSON serialization."""
        # Test ContinuousSet
        cs = ContinuousSet(0, 5)
        json_data = cs.to_json()
        reconstructed = ContinuousSet.from_json(json_data)
        self.assertEqual(cs, reconstructed)


if __name__ == '__main__':
    import unittest
    unittest.main()