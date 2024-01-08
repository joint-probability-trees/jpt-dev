from unittest import TestCase

import math
import pickle

from ddt import ddt, data, unpack
import numpy as np

from dnutils.tools import ifstr

from constants import eps
from intervals.base import chop

from intervals import (
    ContinuousSet,
    INC,
    EXC,
    RealSet,
    R,
    STR_EMPTYSET,
    STR_INFTY,
    IntSet,
    Z,
    Interval
)


class UtilsTest(TestCase):

    def test_chop(self):
        truth = [
            (0, [1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (1, [2, 3, 4, 5, 6, 7, 8, 9]),
            (2, [3, 4, 5, 6, 7, 8, 9]),
            (3, [4, 5, 6, 7, 8, 9]),
            (4, [5, 6, 7, 8, 9]),
            (5, [6, 7, 8, 9]),
            (6, [7, 8, 9]),
            (7, [8, 9]),
            (8, [9]),
            (9, [])]
        result = []
        for h, t in chop(list(range(10))):
            result.append((h, list(t)))
        self.assertEqual(truth, result)
        self.assertEqual([], list(chop([])))


@ddt
class ContinuousSetTest(TestCase):

    @data(('[-10, 5]', ContinuousSet(-10, 5)),
          (']5, 10]', ContinuousSet(5, 10, EXC)),
          ('[0, 1]', ContinuousSet(0, 1)),
          ('[2, 3]', ContinuousSet(2, 3)),
          (']-inf,0[', ContinuousSet(np.NINF, 0, EXC, EXC)),
          ('[0, inf[', ContinuousSet(0, np.PINF, INC, EXC)),
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
        (ContinuousSet.parse('(-inf,0]'), np.NINF),
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
        (ContinuousSet(np.NINF, np.PINF, EXC, EXC), f'(-{STR_INFTY},{STR_INFTY})')
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
        (ContinuousSet(np.NINF, np.PINF, EXC, EXC), f']-{STR_INFTY},{STR_INFTY}[')
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
        ('[0,0)', ContinuousSet.EMPTY, True)
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

    @data(']0, 0[',)
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

    @data(']0,0[', '[0,1]', '[2,3[')
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
        self.assertEqual(ContinuousSet.EMPTY, i1.intersection(i2))

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
          ('[-10, 10]', '[-5, 5]', RealSet(['[-10,-5[', ']5,10]'])),
          ('[-10, 10]', ']-5, 5[', RealSet(['[-10,-5]', '[5,10]'])),
          ('[-10, 10]', '[-5, 5[', RealSet(['[-10,-5[', '[5,10]'])),
          ('[-1.0,1.0]', '[0.0, 1.0]', ContinuousSet(-1, 0, INC, EXC)),
          ('[0,1]', '[1,2]', ContinuousSet(0, 1, INC, EXC)),
          ('[-10, 10]', ContinuousSet.EMPTY, '[-10,10]'),
          (ContinuousSet(0, 0 + eps, INC, EXC), ContinuousSet.EMPTY, ContinuousSet(0, 0 + eps, INC, EXC)),
          (ContinuousSet(0 + eps, np.PINF, INC, EXC), ContinuousSet.EMPTY, ContinuousSet(0 + eps, np.PINF, INC, EXC)),
          ('[-1, 1]', '[-1,1]', ContinuousSet.EMPTY)
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
    def test_serialization(self, i):
        i = ContinuousSet.parse(i)
        self.assertEqual(i, ContinuousSet.from_json(i.to_json()))

    # ------------------------------------------------------------------------------------------------------------------

    @data(
        (ContinuousSet.parse('[0,1]'), RealSet([ContinuousSet(np.NINF, 0, EXC, EXC), ContinuousSet(1, np.PINF, EXC, EXC)])),
        (ContinuousSet.EMPTY, R),
        (R, ContinuousSet.EMPTY)
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
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[3,4]'), RealSet([ContinuousSet.parse('[1,2]'),
                                                                              ContinuousSet.parse('[3,4]')])),
        (ContinuousSet.EMPTY, ContinuousSet.parse('[0,1]'), ContinuousSet(0, 1)),
        (ContinuousSet.EMPTY, ContinuousSet.EMPTY, ContinuousSet.EMPTY),
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

        # test RealSet of size 1
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
        self.assertEqual(i, RealSet(intervals=chops))

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
        self.assertEqual(i, RealSet(intervals=chop_lower))
        self.assertEqual(
            [
                ContinuousSet.parse('[0,1['),
                ContinuousSet.parse('[1,1]')
            ], chop_upper
        )
        self.assertEqual(i, RealSet(intervals=chop_upper))

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
        self.assertEqual(i, RealSet(intervals=chop_lower))
        self.assertEqual(
            [
                ContinuousSet(0, i.max, EXC, EXC),
                ContinuousSet(i.max, i.max)
            ], chop_upper
        )
        self.assertEqual(i, RealSet(intervals=chop_upper))

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
        self.assertEqual(i, RealSet(intervals=chops))

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

    @data(
        (ContinuousSet(0, 1), ContinuousSet(0, 1), True),
        (ContinuousSet(0, 1), ContinuousSet(0, 1, INC, EXC), True),
        (ContinuousSet(0, 1), ContinuousSet(1, 2), False),
        (ContinuousSet(0, 1, INC, EXC), ContinuousSet(0, 1), False),
        (ContinuousSet(0, 1), ContinuousSet.EMPTY, True),
        (ContinuousSet.EMPTY, ContinuousSet.EMPTY, True),
        (ContinuousSet.EMPTY, ContinuousSet(0, 1), False)
    )
    @unpack
    def test_issuperseteq(self, superset, subset, truth):
        # Act
        issuperseteq = superset.issuperseteq(subset)

        # Arrange
        self.assertEqual(
            truth,
            issuperseteq
        )

    @data(
        (ContinuousSet(0, 1), ContinuousSet(0, 1), False),
        (ContinuousSet(0, 1), ContinuousSet(0, 1, INC, EXC), True),
        (ContinuousSet(0, 1), ContinuousSet(1, 2), False),
        (ContinuousSet(0, 1, INC, EXC), ContinuousSet(0, 1), False),
        (ContinuousSet(0, 1), ContinuousSet.EMPTY, True),
        (ContinuousSet.EMPTY, ContinuousSet.EMPTY, False),
        (ContinuousSet.EMPTY, ContinuousSet(0, 1), False)
    )
    @unpack
    def test_issuperset(self, superset, subset, truth):
        # Act
        issuperset = superset.issuperset(subset)

        # Arrange
        self.assertEqual(
            truth,
            issuperset
        )


@ddt
class RealSetContinuousTest(TestCase):

    @data(
        (['[0, 1]', '[2, 3]'], RealSet([ContinuousSet(0, 1), ContinuousSet(2, 3)])),
    )
    @unpack
    def test_creation(self, i, o):
        s = RealSet(i)
        self.assertEqual(s, o)

    def test_copy(self):
        # Arrange
        r = RealSet([
            ContinuousSet(0, 1),
            ContinuousSet(2, 3)
        ])

        # Act
        result = r.copy()

        # Assert
        self.assertEqual(
            r,
            result,
        )
        self.assertNotEqual(
            id(r),
            id(result)
        )

    @data(
        ([], True),
        ([']0,0['], True),
        (['[1,1]'], False),
        ([']-inf,inf['], False)
    )
    @unpack
    def test_isempty(self, i, o):
        self.assertEqual(o, RealSet(i).isempty())

    @data(
        (RealSet(['[0,1]']), RealSet(['[0,1]']), True),
        (RealSet(['[0,1]']), RealSet(['[0,1[']), False),
    )
    @unpack
    def test_equality(self, i1, i2, o):
        return self.assertEqual(o, i1 == i2)

    @data(
        (['[0,1]', '[2,3]'], RealSet([']1,2[']), ContinuousSet.EMPTY),
        (['[0,1]', '[2,3]'], RealSet(['[0,1]', '[2,3]']), RealSet(['[0,1]', '[2,3]'])),
        (['[0,1]', '[2,3]'], ContinuousSet.EMPTY, ContinuousSet.EMPTY),
    )
    @unpack
    def test_intersection(self, i1, i2, o):
        self.assertEqual(o, RealSet(i1).intersection(i2))

    @data(
        (RealSet(['[-1,1]', '[-.5,.5]']), ContinuousSet(-1, 1)),
        (RealSet(['[-1,1]', '[.5,2]']), ContinuousSet(-1, 2)),
        (RealSet(['[0,1]', '[2,3]']), RealSet(['[0,1]', '[2,3]'])),
        (RealSet(['[0,1]', '[1,2]']), ContinuousSet(0, 2)),
        (RealSet(['[0,1]', ']1,3]']), ContinuousSet(0, 3)),
        (RealSet([']1,3]', '[0,1]']), ContinuousSet(0, 3)),
        (RealSet([ContinuousSet.EMPTY]), ContinuousSet.EMPTY),
    )
    @unpack
    def test_simplify(self, i, o):
        self.assertEqual(o, i.simplify())

    @data(
        (RealSet(['[-1,1]', '[-.5,.5]']), RealSet(['[0,1[']), RealSet(['[-1,0[', '[1,1]'])),
        (RealSet(['[-1,1]', '[-.5,.5]']), RealSet(['[0,1]']), ContinuousSet.parse('[-1,0[')),
        (RealSet(['[0,1]', '[1,2]', '[2,3]']), ContinuousSet(1, 2), RealSet(['[0,1[', ']2,3]'])),
        (RealSet([ContinuousSet.EMPTY]), ContinuousSet.EMPTY, RealSet.EMPTY),
    )
    @unpack
    def test_difference(self, i1, i2, o):
        self.assertEqual(o, i1.difference(i2))

    @data(
        (RealSet(['[0,1]', '[2,3]']),
         RealSet(['[-1.5,-1]', '[-3,-2]']),
         RealSet(['[-1.5,-1]', '[-3,-2]', '[0,1]', '[2,3]'])),
    )
    @unpack
    def test_union(self, i1, i2, r):
        self.assertEqual(r, i1.union(i2))

    def test_serialization(self):
        i1 = RealSet([']0,1[', ']2,3['])
        i2 = RealSet([']2,3[', ']0,1['])
        i3 = RealSet([']2,3[', ']0,1]'])
        self.assertEqual(i1, pickle.loads(pickle.dumps(i1)))
        self.assertEqual(i2, pickle.loads(pickle.dumps(i2)))
        self.assertEqual(i3, pickle.loads(pickle.dumps(i3)))
        self.assertEqual(i1, pickle.loads(pickle.dumps(i2)))
        self.assertNotEqual(i1, pickle.loads(pickle.dumps(i3)))

    def test_hash(self):
        i1 = RealSet([']0,1[', ']2,3['])
        i2 = RealSet([']2,3[', ']0,1['])
        i3 = RealSet([']2,3[', ']0,1]'])
        self.assertEqual(hash(i1), hash(i2))
        self.assertEqual(hash(i1), hash(pickle.loads(pickle.dumps(i1))))
        self.assertEqual(hash(i2), hash(pickle.loads(pickle.dumps(i2))))
        self.assertNotEqual(hash(i1), hash(i3))

    def test_size(self):
        # test infinitely big sets
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertEqual(float("inf"), r1.size())

        # test RealSet of size 1
        r2 = RealSet(["[-.5, -.5]"])
        self.assertEqual(r2.size(), 1)

        # test multiple but same single values
        r3 = RealSet(["[-.5, -.5]", "[-.5, -.5]"])
        self.assertEqual(r3.size(), 1)

        # test multiple but different values
        r4 = RealSet(["[-.5, -.5]", "[-.6, -.6]"])
        self.assertEqual(r4.size(), 2)

        # test emptyset
        r5 = RealSet.emptyset()
        self.assertEqual(r5.size(), 0)

    def test_sample(self):
        # test default usage
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        samples = r1.sample(100)
        for sample in samples:
            self.assertTrue(sample in r1)

        # test raising index error
        r2 = RealSet.emptyset()
        self.assertRaises(ValueError, r2.sample, 100)

        # test singular value
        r3 = RealSet(["[-.5, -.5]"])
        samples = r3.sample(100)
        for sample in samples:
            self.assertEqual(sample, -0.5)

    def test_contains_value(self):
        # test infinitely big sets
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.contains_value(.5))
        self.assertTrue(r1.contains_value(0.75))
        self.assertFalse(r1.contains_value(10))

        # test singular values
        r4 = RealSet(["[-.5, -.5]", "[-.6, -.6]"])
        self.assertTrue(r4.contains_value(-.5))
        self.assertTrue(r4.contains_value(-.6))
        self.assertFalse(r4.contains_value(-.55))

        # test emptyset
        r5 = RealSet.emptyset()
        self.assertFalse(r5.contains_value(1))

    def test_issuperseteq(self):
        # test regular case
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.issuperseteq(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r1.issuperseteq(ContinuousSet(-2, 0.25)))
        self.assertTrue(r1.issuperseteq(ContinuousSet.EMPTY))

        # test empty set
        r2 = RealSet.EMPTY
        self.assertFalse(r2.issuperseteq(ContinuousSet(-0.25, 0.25)))
        self.assertTrue(r2.issuperseteq(ContinuousSet.EMPTY))

        # test singular value
        r3 = RealSet(["[-.5, -.5]"])
        self.assertTrue(r3.issuperseteq(ContinuousSet(-0.5, -0.5)))
        self.assertTrue(r3.issuperseteq(ContinuousSet.EMPTY))

    def test_fst(self):
        # test default case
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertEqual(r1.fst(), -1.)

        # test set containing empty set
        r2 = RealSet([ContinuousSet(0, 0, 2, 2)])
        self.assertTrue(math.isnan(r2.fst()))

        # test empty set
        r3 = RealSet.emptyset()
        self.assertTrue(math.isnan(r3.fst()))

    @data(
        (RealSet(['(-inf,1]', '[3,inf)']), True, True, True),
        (RealSet(['(-inf,2]', '[3,5]']), True, False, True),
        (RealSet(['[2,inf)', '[3,inf)']), False, True, True),
        (RealSet(['[-1,-1]', '[4,7]']), False, False, False),
        (RealSet.EMPTY, False, False, False)
    )
    @unpack
    def test_infinity(self, i, isninf, ispinf, isinf):
        # Act
        isninf_, ispinf_, isinf_ = (
            i.isninf(),
            i.ispinf(),
            i.isinf()
        )

        # Assert
        self.assertEqual(
            isninf,
            isninf_
        )
        self.assertEqual(
            ispinf,
            ispinf_
        )
        self.assertEqual(
            isinf,
            isinf_
        )

    def test_intersects(self):
        # this also tests isdisjoint
        # test regular case
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.intersects(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r1.intersects(ContinuousSet(-7, -6)))
        self.assertFalse(r1.intersects(ContinuousSet.EMPTY))

        # test empty set
        r2 = RealSet.emptyset()
        self.assertFalse(r2.intersects(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r2.intersects(ContinuousSet.EMPTY))

        # test singular value
        r3 = RealSet(["[-.5, -.5]"])
        self.assertTrue(r3.intersects(ContinuousSet(-0.5, -0.5)))
        self.assertFalse(r3.intersects(ContinuousSet.EMPTY))

    def test_chop(self):
        # Arrange
        r = RealSet([']0,1]', '[2,3['])

        # Act
        chops = r.chop([.5, 1.2, 2.5])

        # Assert
        self.assertEqual(
            [
                ContinuousSet(0, .5, EXC, EXC),
                ContinuousSet(.5, 1, INC, INC),
                ContinuousSet(2, 2.5, INC, EXC),
                ContinuousSet(2.5, 3, INC, EXC),
            ],
            list(chops)
        )

    def test_xmirror(self):
        # Arrange
        s = RealSet(['(-1,1]', '[2,3)'])
        # Act
        s_ = s.xmirror()
        # Assert
        self.assertEqual(
            RealSet([
                '(-3,-2]', '[-1,1)'
            ]),
            s_
        )

    def test_round(self):
        # Arrange
        i = RealSet([
            ContinuousSet(0.1234, 5.6789, INC, EXC),
            ContinuousSet(3.456, 7.89, INC, EXC),
        ])

        # Act
        i_ = round(i, 1)

        # Assert
        self.assertEqual(
            RealSet([
                ContinuousSet(0.1, 5.7, INC, EXC),
                ContinuousSet(3.5, 7.9, INC, EXC),
            ]),
            i_
        )

    @data(
        R, '[0,1]', '(-inf,-1)', '[0,inf]'
    )
    def test_any_point(self, i):
        i = ifstr(i, ContinuousSet.parse)
        p = i.any_point()
        self.assertTrue(i.contains_value(p))


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class RealSetIntegerTest(TestCase):

    @data(
        (['{0..1}', '{2..3}'], RealSet([IntSet(0, 1), IntSet(2, 3)])),
    )
    @unpack
    def test_creation(self, i, o):
        # Act
        s = RealSet(i)

        # Assert
        self.assertEqual(s, o)

    def test_creation_with_integrity_violation(self):
        # Arrange
        cset = ContinuousSet(0, 1)
        iset = IntSet(2, 3)

        # Act & Assert
        self.assertRaises(
            TypeError,
            RealSet,
            [cset, iset]
        )

    def test_copy(self):
        # Arrange
        r = RealSet([IntSet(0, 1), IntSet(3, 4)])

        # Act
        result = r.copy()

        # Assert
        self.assertEqual(
            r,
            result
        )
        self.assertNotEqual(
            id(r),
            id(result)
        )

    @data(
        ([], True),
        ([IntSet.emptyset()], True),
        (['{1..1}'], False),
        (['{..}'], False)
    )
    @unpack
    def test_isempty(self, i, truth):
        # Arrange
        rs = RealSet(i)

        # Act
        isempty = rs.isempty()

        # Assert
        self.assertEqual(
            truth,
            isempty
        )

    @data(
        (RealSet(['{0..1}']), RealSet(['{0..1}']), True),
        (RealSet(['{0..1}']), RealSet(['{0..2}']), False),
        (RealSet(['{0..2}']), RealSet(['{0..1}', '{2..2}']), True)
    )
    @unpack
    def test_equality(self, i1, i2, truth):
        # Act
        eq = i1 == i2

        # Assert
        self.assertEqual(
            truth,
            eq
        )

    @data(
        (RealSet(['{-1..0}', '{3..4}']), RealSet(['{1..2}']), RealSet.emptyset()),
        (RealSet(['{0..1}', '{2..3}']), RealSet(['{0..1}', '{2..3}']), IntSet(0, 3)),
        (RealSet(['{0..1}', '{2..3}']), RealSet.EMPTY, RealSet.EMPTY),
    )
    @unpack
    def test_intersection(self, i1, i2, o):
        # Act
        intersection = i1.intersection(i2)

        # Assert
        self.assertEqual(
            o,
            intersection
        )

    @data(
        (RealSet(['{-1..1}', '{0..0}']), IntSet(-1, 1)),
        (RealSet(['{-1..1}', '{0..2}']), IntSet(-1, 2)),
        (RealSet(['{0..0}', '{2..3}']), RealSet(['{0..0}', '{2..3}'])),
        (RealSet(['{0..1}', '{2..3}']), IntSet(0, 3)),
        (RealSet(['{0..5}', '{1..3}']), IntSet(0, 5)),
        (RealSet([IntSet.emptyset()]), RealSet.emptyset()),
    )
    @unpack
    def test_simplify(self, i, truth):
        # Act
        simplified = i.simplify()

        # Assert
        self.assertEqual(
            truth,
            simplified
        )

    @data(
        (RealSet(['{0..5}', '{1..3}']), RealSet([IntSet(0, 5)])),
    )
    @unpack
    def test_simplify_with_keep_type(self, i, truth):
        # Act
        simplified = i.simplify(keep_type=True)

        # Assert
        self.assertEqual(
            truth,
            simplified
        )

    @data(
        (RealSet(['{-2..2}', '{-1..1}']), RealSet(['{0..1}']), RealSet(['{-2..-1}', '{2..2}'])),
        (RealSet(['[-1,1]', '[-.5,.5]']), RealSet(['[0,1]']), ContinuousSet.parse('[-1,0[')),
        (RealSet(['[0,1]', '[1,2]', '[2,3]']), ContinuousSet(1, 2), RealSet(['[0,1[', ']2,3]'])),
        (RealSet([IntSet.emptyset()]), RealSet.emptyset(), RealSet.emptyset()),
    )
    @unpack
    def test_difference(self, i1, i2, o):
        self.assertEqual(o, i1.difference(i2))

    @data(
        (RealSet(['[0,1]', '[2,3]']),
         RealSet(['[-1.5,-1]', '[-3,-2]']),
         RealSet(['[-1.5,-1]', '[-3,-2]', '[0,1]', '[2,3]'])),
    )
    @unpack
    def test_union(self, i1, i2, r):
        self.assertEqual(r, i1.union(i2))

    def test_serialization(self):
        i1 = RealSet([']0,1[', ']2,3['])
        i2 = RealSet([']2,3[', ']0,1['])
        i3 = RealSet([']2,3[', ']0,1]'])
        self.assertEqual(i1, pickle.loads(pickle.dumps(i1)))
        self.assertEqual(i2, pickle.loads(pickle.dumps(i2)))
        self.assertEqual(i3, pickle.loads(pickle.dumps(i3)))
        self.assertEqual(i1, pickle.loads(pickle.dumps(i2)))
        self.assertNotEqual(i1, pickle.loads(pickle.dumps(i3)))

    def test_hash(self):
        i1 = RealSet([']0,1[', ']2,3['])
        i2 = RealSet([']2,3[', ']0,1['])
        i3 = RealSet([']2,3[', ']0,1]'])
        self.assertEqual(hash(i1), hash(i2))
        self.assertEqual(hash(i1), hash(pickle.loads(pickle.dumps(i1))))
        self.assertEqual(hash(i2), hash(pickle.loads(pickle.dumps(i2))))
        self.assertNotEqual(hash(i1), hash(i3))

    @data(
        (RealSet(['{..1}', '{3..}']), True, True, True),
        (RealSet(['{..2}', '{3..5}']), True, False, True),
        (RealSet(['{2..}', '{3..}']), False, True, True),
        (RealSet(['{-1..-1}', '{4..7}']), False, False, False),
        (RealSet.EMPTY, False, False, False)
    )
    @unpack
    def test_infinity(self, i, isninf, ispinf, isinf):
        # Act
        isninf_, ispinf_, isinf_ = (
            i.isninf(),
            i.ispinf(),
            i.isinf()
        )

        # Assert
        self.assertEqual(
            isninf,
            isninf_
        )
        self.assertEqual(
            ispinf,
            ispinf_
        )
        self.assertEqual(
            isinf,
            isinf_
        )

    def test_size(self):
        # test infinitely big sets
        r1 = RealSet(['{..0}', '{5..}'])
        self.assertEqual(float("inf"), r1.size())

        # test RealSet of size 1
        r2 = RealSet(["{0..0}"])
        self.assertEqual(r2.size(), 1)

        # test multiple but same single values
        r3 = RealSet(["{0..0}", "{0..0}"])
        self.assertEqual(r3.size(), 1)

        # test multiple but different values
        r4 = RealSet(["{0..0}", "{1..1}"])
        self.assertEqual(r4.size(), 2)

        # test emptyset
        r5 = RealSet.emptyset()
        self.assertEqual(r5.size(), 0)

    def test_sample(self):
        # Arrange
        # test default usage
        r1 = RealSet(['{-2..2}', '{-1..1}'])
        # test singular value
        r2 = RealSet(["{1..1}"])
        # test raising index error
        r3 = RealSet.emptyset()

        # Act
        samples1 = r1.sample(100)
        samples2 = r2.sample(100)

        # Assert
        for sample in samples1:
            self.assertTrue(sample in r1)

        for sample in samples2:
            self.assertEqual(1, sample)

        self.assertRaises(ValueError, r3.sample, 100)


    def test_contains_value(self):
        # test infinitely big sets
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.contains_value(.5))
        self.assertTrue(r1.contains_value(0.75))
        self.assertFalse(r1.contains_value(10))

        # test singular values
        r4 = RealSet(["[-.5, -.5]", "[-.6, -.6]"])
        self.assertTrue(r4.contains_value(-.5))
        self.assertTrue(r4.contains_value(-.6))
        self.assertFalse(r4.contains_value(-.55))

        # test emptyset
        r5 = RealSet.emptyset()
        self.assertFalse(r5.contains_value(1))

    def test_issuperseteq(self):
        # test regular case
        r1 = RealSet(['{-3..3}', '{-2..2}'])
        self.assertTrue(r1.issuperseteq(IntSet(-1, 1)))
        self.assertFalse(r1.issuperseteq(IntSet(-4, 1)))
        self.assertTrue(r1.issuperseteq(IntSet.EMPTY))

        # test empty set
        r2 = RealSet.EMPTY
        self.assertFalse(r2.issuperseteq(IntSet(-1, 1)))
        self.assertTrue(r2.issuperseteq(IntSet.EMPTY))

        # test singular value
        r3 = RealSet(["{-1..-1}"])
        self.assertTrue(r3.issuperseteq(IntSet(-1, -1)))
        self.assertTrue(r3.issuperseteq(IntSet.EMPTY))

    def test_min(self):
        # test default case
        r1 = RealSet(['{-2..2}', '{-1..1}'])
        self.assertEqual(
            -2,
            r1.fst()
        )

        # test set containing empty set
        r2 = RealSet([IntSet.EMPTY])
        self.assertTrue(
            math.isnan(r2.fst())
        )

        # test empty set
        r3 = RealSet.EMPTY
        self.assertTrue(
            math.isnan(r3.fst())
        )

    @data(
        (RealSet(['{0..1}', '{3..4}']), 0, True),
        (RealSet(['{0..1}', '{3..4}']), 2, False),
        (RealSet(['{0..1}', '{3..4}']), 4, True),
        (RealSet(['{..0}', '{5..}']), -10, True),
        (RealSet(['{..0}', '{5..}']), 0, True),
        (RealSet(['{..0}', '{5..}']), 1, False),
        (RealSet(['{..0}', '{5..}']), 10, True),
    )
    @unpack
    def test_contains_value(self, interval, value, truth):
        # Act
        contains = interval.contains_value(value)

        # Assert
        self.assertEqual(
            truth,
            contains
        )


    @data(
        (RealSet(['{-3..3}', '{-2..2}']), IntSet(-1, 1), True),
        (RealSet(['{-2..2}', '{-1..1}']), IntSet(-7, -6), False),
        (RealSet(['{-2..2}', '{-1..1}']), RealSet.emptyset(), False),
        (RealSet.emptyset(), IntSet(-1, 1), False),
        (RealSet.emptyset(), IntSet.emptyset(), False),
        (RealSet(["{-1..-1}"]), IntSet(-1, -1), True),
        (RealSet(["{-1..-1}"]), RealSet.emptyset(), False)
    )
    @unpack
    def test_intersects_and_isdisjoint(self, i1, i2, truth):
        # Act
        intersects = i1.intersects(i2)
        isdisjoint = i1.isdisjoint(i2)

        # Assert
        self.assertEqual(
            truth,
            intersects
        )
        self.assertEqual(
            not truth,
            isdisjoint
        )

    def test_chop(self):
        # Arrange
        r = RealSet([']0,1]', '[2,3['])

        # Act
        chops = r.chop([.5, 1.2, 2.5])

        # Assert
        self.assertEqual(
            [
                ContinuousSet(0, .5, EXC, EXC),
                ContinuousSet(.5, 1, INC, INC),
                ContinuousSet(2, 2.5, INC, EXC),
                ContinuousSet(2.5, 3, INC, EXC),
            ],
            list(chops)
        )

    @data(
        (RealSet(['{-1..0}', '{2..3}']), RealSet(['{-3..-2}', '{0..1}'])),
        (RealSet.EMPTY, RealSet.EMPTY),
    )
    @unpack
    def test_xmirror(self, rset, truth):
        # Act
        mirrored = rset.xmirror()

        # Assert
        self.assertEqual(
            truth,
            mirrored
        )

    def test_round(self):
        # Arrange
        i = RealSet([
            ContinuousSet(0.1234, 5.6789, INC, EXC),
            ContinuousSet(3.456, 7.89, INC, EXC),
        ])

        # Act
        i_ = round(i, 1)

        # Assert
        self.assertEqual(
            RealSet([
                ContinuousSet(0.1, 5.7, INC, EXC),
                ContinuousSet(3.5, 7.9, INC, EXC),
            ]),
            i_
        )

    @data(
        R, '[0,1]', '(-inf,-1)', '[0,inf]'
    )
    def test_any_point(self, i):
        i = ifstr(i, ContinuousSet.parse)
        p = i.any_point()
        self.assertTrue(i.contains_value(p))


# ----------------------------------------------------------------------------------------------------------------------

class ContinuousSetOperatorTest(TestCase):

    def test_intersection(self):
        # Arrange
        i1 = ContinuousSet(0, 2)
        i2 = ContinuousSet(1, 3)

        # Act
        intersect = i1 & i2

        # Assert
        self.assertEqual(ContinuousSet(1, 2), intersect)

    def test_union(self):
        # Arrange
        i1 = ContinuousSet(0, 1)
        i2 = ContinuousSet(2, 3)

        # Act
        union = i1 | i2

        # Assert
        self.assertEqual(RealSet(['[0, 1]', '[2,3]']), union)

    def test_diff(self):
        # Arrange
        i1 = ContinuousSet(0, 3)
        i2 = ContinuousSet(1, 2)

        # Act
        diff = i1 - i2

        # Assert
        self.assertEqual(RealSet(['[0, 1[', ']2,3]']), diff)


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class IntSetTest(TestCase):

    def test_constructor(self):
        # Act
        i1 = IntSet(0, 1)
        i2 = IntSet(np.NINF, np.PINF)

        # Assert
        self.assertIsInstance(
            i1.lower,
            int
        )
        self.assertIsInstance(
            i1.upper,
            int
        )
        self.assertEqual(
            np.NINF,
            i2.lower
        )
        self.assertEqual(
            np.PINF,
            i2.upper
        )
        self.assertRaises(
            ValueError,
            IntSet,
            .5,
            2
        )
        self.assertRaises(
            ValueError,
            IntSet,
            1.,
            2.5
        )

    def test_copy(self):
        # Arrange
        i = IntSet(0, 2)

        # Act
        copy = i.copy()

        # Assert
        self.assertEqual(
            i,
            copy
        )
        self.assertNotEqual(
            id(i),
            id(copy)
        )

    def test_emptyset(self):
        empty = IntSet.emptyset()

        # Assert
        self.assertEqual(
            hash(empty),
            hash(frozenset())
        )
        self.assertEqual(
            empty,
            IntSet.emptyset()
        )

    @data(
        ('{0..2}', IntSet(0, 2)),
        ('{..0}', IntSet(np.NINF, 0)),
        ('{5..}', IntSet(5, np.PINF)),
        ('{..}', Z)
    )
    @unpack
    def test_parse(self, s, t):
        # Act
        result = IntSet.parse(s)

        # Assert
        self.assertEqual(
            t,
            result
        )


    def test_str(self):
        # Arrange
        i1 = IntSet(np.NINF, np.PINF)
        i2 = IntSet(np.NINF, 3)
        i3 = IntSet(-1, np.PINF)

        # Act
        s1 = str(i1)
        s2 = str(i2)
        s3 = str(i3)

        # Assert
        self.assertEqual(
            'ℤ',
            s1
        )
        self.assertEqual(
            '{..3}',
            s2
        )
        self.assertEqual(
            '{-1..}',
            s3
        )

    def test_hash(self):
        self.assertEqual(
            hash(IntSet(0, -1)),
            hash(IntSet(-1, -2))
        )
        self.assertEqual(
            hash(frozenset()),
            hash(IntSet(0, -1))
        )
        self.assertNotEqual(
            hash(frozenset()),
            hash(IntSet(0, 1))
        )


    @data(
        (IntSet(0, 1), IntSet(2, 3), True),
        (IntSet(0, 1), IntSet(1, 2), False),
        (IntSet(0, 1), IntSet(3, 4), False)
    )
    @unpack
    def test_contiguous(self, i1, i2, truth):
        # Act
        result = i1.contiguous(i2)
        symmetric = i2.contiguous(i1)

        # Assert
        self.assertEqual(
            truth,
            result
        )
        self.assertEqual(
            truth,
            symmetric
        )

    @data(
        (Z, -1, True),
        (Z, 1.2, False),
        (Z, np.NINF, True),
        (Z, np.PINF, True),
        (IntSet(np.NINF, 4), 4, True),
        (IntSet(np.NINF, 4), 5, False),
        (IntSet(-1, np.PINF), -1, True),
        (IntSet(-1, np.PINF), -2, False),
        (IntSet(0, -1), 0, False)
    )
    @unpack
    def test_contains_value(self, i, n, t):
        self.assertEqual(
            t,
            i.contains_value(n)
        )

    @data(
        (Z, IntSet(0, 1), True),
        (IntSet(0, 1), IntSet(0, 1), True),
        (IntSet(0, 1), IntSet(1, 1), True),
        (IntSet(0, 1), IntSet(1, 2), False),
        (IntSet(2, 4), IntSet(-1, 0), False),
        (IntSet(0, 1), IntSet(1, 2), False),
        (IntSet(0, 1), IntSet.emptyset(), True),
        (IntSet.EMPTY, IntSet.EMPTY, True)
    )
    @unpack
    def test_superseteq(self, i1, i2, t):
        self.assertEqual(
            t,
            i1.issuperseteq(i2)
        )

    @data(
        (Z, IntSet(0, 1), True),
        (IntSet(0, 1), IntSet(0, 1), False),
        (IntSet(0, 1), IntSet(1, 1), True),
        (IntSet(0, 1), IntSet(1, 2), False),
        (IntSet(2, 4), IntSet(-1, 0), False),
        (IntSet(0, 1), IntSet(1, 2), False),
        (IntSet(0, 1), IntSet.emptyset(), True)
    )
    @unpack
    def test_superset(self, i1, i2, t):
        self.assertEqual(
            t,
            i1.issuperset(i2)
        )

    @data(

    )
    @unpack
    def test_isempty(self, i, t):
        self.assertEquals(
            t,
            i.isempty()
        )

    @data(

    )
    @unpack
    def test_eq(self, i1, i2, t):
        if t:
            self.assertEqual(
                i1,
                i2,
            )
        else:
            self.assertNotEqual(
                i1,
                i2
            )

    @data(

    )
    @unpack
    def test_isdisjoint(self, i1, i2, t):
        if t:
            self.assertTrue(
                i1.isdisjoint(i2),
            )
            self.assertTrue(
                i2.isdisjoint(i1),
            )
        else:
            self.assertFalse(
                i1.isdisjoint(i2)
            )
            self.assertFalse(
                i2.isdisjoint(i1)
            )
    @data(
        (IntSet(0, -1), IntSet(-2, -3), IntSet.EMPTY),
        (IntSet(0, 3), IntSet(1, 2), IntSet(1, 2)),
        (IntSet(0, 3), IntSet(-1, 1), IntSet(0, 1)),
        (IntSet(0, 3), IntSet(2, 4), IntSet(2, 3)),
        (IntSet(0, 1), IntSet(-1, 4), IntSet(0, 1)),
        (Z, IntSet(0, 2), IntSet(0, 2)),
    )
    @unpack
    def test_intersection(self, i1, i2, t):
        self.assertEqual(
            t,
            i1.intersection(i2)
        )
        self.assertEqual(
            t,
            i2.intersection(i1)
        )

    def test_iter_all_ints(self):
        # Arrange
        i = iter(Z)

        # Act
        result = [next(i) for _ in range(10)]

        # Assert
        self.assertEqual(
            [0, 1, -1, 2, -2, 3, -3, 4, -4, 5],
            result
        )

    def test_iter_neg_inf(self):
        # Arrange
        int_set = IntSet(np.NINF, 3)
        int_iter = iter(int_set)

        # Act
        result = [next(int_iter) for _ in range(10)]

        # Assert
        self.assertEqual(
            [3, 2, 1, 0, -1, -2, -3, -4, -5, -6],
            result
        )

    def test_iter_pos_inf(self):
        # Arrange
        int_set = IntSet(-1, np.PINF)
        int_iter = iter(int_set)

        # Act
        result = [next(int_iter) for _ in range(10)]

        # Assert
        self.assertEqual(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            result
        )

    def test_iter_finite(self):
        # Arrange
        int_set = IntSet(-1, 3)
        int_iter = iter(int_set)

        # Act
        result = list(int_iter)

        # Assert
        self.assertEqual(
            [-1, 0, 1, 2, 3],
            result
        )

    @data(
        (IntSet(0, 2), IntSet(2, 3), IntSet(0, 3)),
        (IntSet(0, 1), IntSet(3, 4), RealSet([IntSet(0, 1), IntSet(3, 4)]))
    )
    @unpack
    def test_union(self, i1, i2, truth):
        # Act
        result = i1.union(i2)

        # Assert
        self.assertEqual(
            truth,
            result
        )

    @data(
        (IntSet(0, 1), IntSet(2, 3), IntSet(0, 1)),
        (IntSet(0, 5), IntSet(2, 3), RealSet([IntSet(0, 1), IntSet(4, 5)])),
        (IntSet(0, 4), IntSet(2, 6), IntSet(0, 1)),
        (IntSet(0, 4), IntSet(-2, 0), IntSet(1, 4)),
        (IntSet(0, 1), IntSet(-1, 3), IntSet.emptyset()),
    )
    @unpack
    def test_difference(self, i1, i2, truth):
        # Act
        result = i1.difference(i2)

        # Assert
        self.assertEqual(
            truth,
            result
        )
        self.assertEqual(
            i1,
            i1,
            IntSet.emptyset()
        )
        self.assertEqual(
            i2,
            i2,
            IntSet.emptyset()
        )

    @data(
        (IntSet(0, 1), 2),
        (IntSet(np.PINF, np.NINF), 0),
        (IntSet(np.NINF, 0), np.PINF)
    )
    @unpack
    def test_size(self, interval, truth):
        # Act
        size = interval.size()

        # Assert
        self.assertEqual(
            truth,
            size
        )

        # i2 = IntSet(np.NINF, 0)
        # i3 = IntSet(1, np.PINF)
        # i4 = IntSet.ALL
        # i5 = IntSet.EMPTY
    @data(
        ('{0..1}', False, False, False),
        ('{..0}', True, False, True),
        ('{1..}', False, True, True),
        ('{..}', True, True, True),
        (u'\u2205', False, False, False)
    )
    @unpack
    def test_ininity(self, interval, isninf, ispinf, isinf):
        # Arrange
        interval = IntSet.parse(interval)

        # Act
        isninf_, ispinf_, isinf_ = interval.isninf(), interval.ispinf(), interval.isinf()

        # Assert
        self.assertEqual(
            isninf,
            isninf_
        )
        self.assertEqual(
            ispinf,
            ispinf_
        )
        self.assertEqual(
            isinf,
            isinf_
        )

    def test_xmirror(self):
        # Arrange
        i = IntSet(-1, 3)

        # Act
        i_ = i.xmirror()

        # Assert
        self.assertEqual(
            IntSet(-3, 1),
            i_
        )

    def test_from_set(self):
        # Arrange
        numbers = {1, 3, 4}
        numbers_contiguous = {1, 2, 3}

        # Act
        realset = IntSet.from_set(numbers)
        intset = IntSet.from_set(numbers_contiguous)

        # Assert
        self.assertEqual(
            RealSet(['{1..1}', '{3..4}']),
            realset
        )
        self.assertEqual(
            IntSet(1, 3),
            intset
        )

    def test_serialization(self):
        # Arrange
        intset_finite = IntSet(1, 2)
        intset_ninf = IntSet(np.NINF, 0)
        intset_pinf = IntSet(0, np.PINF)
        intset_inf = Z

        # Act
        intset_finite_ = IntSet.from_json(intset_finite.to_json())
        intset_ninf_ = IntSet.from_json(intset_ninf.to_json())
        intset_pinf_ = IntSet.from_json(intset_pinf.to_json())
        intset_inf_ = IntSet.from_json(intset_inf.to_json())

        # Assert
        self.assertEqual(
            intset_finite,
            intset_finite_
        )
        self.assertEqual(
            intset_ninf,
            intset_ninf_
        )
        self.assertEqual(
            intset_pinf,
            intset_pinf_
        )
        self.assertEqual(
            Z,
            intset_inf_
        )

# ----------------------------------------------------------------------------------------------------------------------

@ddt
class IntervalTest(TestCase):

    @data(
        ('ℝ', R),
        ('ℤ', Z),
        ('∅', Interval.emptyset()),
        ('{1..9}', IntSet(1, 9)),
        ('(1,9]', ContinuousSet(1, 9, EXC, INC))
    )
    @unpack
    def test_parse(self, s, t):
        # Act
        result = Interval.parse(s)

        # Assert
        self.assertEqual(
            t,
            result
        )

    def test_serialization(self):
        # Arrange
        intset = IntSet(0, 1)
        contset = ContinuousSet(0, 1)

        # Act
        intset_ = Interval.from_json(intset.to_json())
        contset_ = Interval.from_json(contset.to_json())

        # Assert
        self.assertIsInstance(
            intset_,
            IntSet
        )
        self.assertEqual(
            intset,
            intset_
        )
        self.assertIsInstance(
            contset_,
            ContinuousSet
        )
        self.assertEqual(
            contset,
            contset_
        )