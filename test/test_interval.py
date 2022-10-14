import math
import pickle

from ddt import ddt, data, unpack
import numpy as np
import unittest

from dnutils import out

try:
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.intervals import ContinuousSet, INC, EXC, EMPTY, RealSet, chop, R, _EMPTYSET


class UtilsTest(unittest.TestCase):

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
class IntervalTest(unittest.TestCase):

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
        self.assertEqual(ContinuousSet.parse(s), i)
        self.assertEqual(hash(i), hash(i.copy()))

    def test_itype(self):
        i1 = ContinuousSet.parse('(0,1)')
        self.assertEqual(i1.itype(), 4)

        i2 = ContinuousSet.parse('[0,1]')
        self.assertEqual(i2.itype(), 2)

        self.assertEqual(ContinuousSet.parse('(0,1]').itype(), ContinuousSet.parse('[3,4)').itype())

    def test_value_check(self):
        self.assertRaises(ValueError, ContinuousSet, 1, -1)

    def test_lowermost(self):
        self.assertEqual(np.nextafter(1, 2), ContinuousSet.parse(']1,2]').lowermost())

    @data(']0, 0[',)
    def test_emptyness(self, s):
        self.assertTrue(ContinuousSet.parse(s).isempty(),
                        msg='%s is not recognized empty.' % s)
        self.assertEqual(ContinuousSet.emptyset_p(), ContinuousSet.parse(']0, 0['))

    @data(']0,0[', '[0,1]', '[2,3[')
    @unpack
    def test_serialization(self, i):
        self.assertEqual(i, pickle.loads(pickle.dumps(i)))

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

    @data(
        ('[0, 2]', '[.5, 1.5]', True),
        ('[.5, 1.5]', '[0, 2]', False),
        ('[0,2]', '[1,3]', False),
        ('[0,2]', '[-1,1]', False),
        ('[0,2]', '[1,2]', True),
        ('[0,2]', '[0,1]', True),
        ('[0,2]', '[1,2[', True),
        ('[0,2]', '[0,1[', True),
    )
    @unpack
    def test_interval_containment(self, i1, i2, o):
        self.assertEqual(o, ContinuousSet.parse(i1).contains_interval(ContinuousSet.parse(i2)))

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
        self.assertEqual(o, ContinuousSet.parse(i1).contains_interval(ContinuousSet.parse(i2),
                                                                      proper_containment=True))

    @data(('[-10, 5]', ']5, 10]'),
          ('[0, 1]', '[2, 3]'),
          (']-inf,0[', '[0, inf['),
          ('[0, 1]', ']0,0['),
          (']-1,-1[', ']-1,-1['))
    @unpack
    def test_disjoint(self, i1, i2):
        i1 = ContinuousSet.parse(i1)
        i2 = ContinuousSet.parse(i2)
        self.assertEqual(EMPTY, i1.intersection(i2))

    @data(('[-10, 5]', '[0, 10]', '[0,5]'),
          ('[-10, 10]', '[-5, 5]', '[-5,5]'),
          ('[-10, 10]', '[-10, 10]', '[-10,10]'),
          ('[-10, 10]', '[1,1]', '[1,1]'),
          ('[0.3,0.7[', '[.3,.3]', '[.3,.3]'),
          ('[0,1]', _EMPTYSET, _EMPTYSET)
          )
    @unpack
    def test_intersection(self, i1, i2, r):
        i1 = ContinuousSet.parse(i1)
        i2 = ContinuousSet.parse(i2)
        self.assertEqual(ContinuousSet.parse(r),
                         i1.intersection(i2))
        self.assertEqual(i1.intersection(i2),
                         i2.intersection(i1))

    def test_intersection_optional_left(self):
        i1 = ContinuousSet.parse('[-1, 1]')
        i2 = ContinuousSet.parse(']-1, 1[')
        self.assertEqual(ContinuousSet(np.nextafter(i2.lower, i2.lower + 1), i2.upper, INC, EXC),
                         i1.intersection(i2, left=INC))
        self.assertEqual(i1.intersection(i2, left=INC),
                         i2.intersection(i1, left=INC))

    def test_intersection_optional_right(self):
        i1 = ContinuousSet.parse('[-1, 1]')
        i2 = ContinuousSet.parse(']-1, 1[')
        self.assertEqual(ContinuousSet(i2.lower, np.nextafter(i2.upper, i2.upper - 1), EXC, INC),
                         i1.intersection(i2, right=INC))
        self.assertEqual(i1.intersection(i2, right=INC),
                         i2.intersection(i1, right=INC))

    @data(('[-10, 5]', '[0, 10]', ContinuousSet.parse('[-10,0[')),
          ('[-10, 10]', '[-5, 5]', RealSet(['[-10,-5[', ']5,10]'])),
          ('[-10, 10]', ']-5, 5[', RealSet(['[-10,-5]', '[5,10]'])),
          ('[-1.0,1.0]', '[0.0, 1.0]', ContinuousSet(-1, 0, INC, EXC)),
          ('[0,1]', '[1,2]', ContinuousSet(0, 1, INC, EXC))
          # ('[-10, 10]', '[-10, 10]', '[-10,10]'),
          # ('[-10, 10]', '[1,1]', '[1,1]')
          )
    @unpack
    def test_difference(self, i1, i2, r):
        self.assertEqual(r, ContinuousSet.parse(i1).difference(ContinuousSet.parse(i2)))

    @data(('[-10, 5]',), (']5, 10]',),
          ('[0, 1]',), ('[2, 3]',),
          (']-inf,0[',), ('[0, inf[',),
          ('[0, 1]',), (']0,0[',),
          (']-1,-1[',), (']-1,-1[',))
    @unpack
    def test_serialization(self, i):
        i = ContinuousSet.parse(i)
        self.assertEqual(i, ContinuousSet.from_json(i.to_json()))

    @data(
        (ContinuousSet.parse('[0,1]'), RealSet([ContinuousSet(np.NINF, 0, EXC, EXC),
                                                ContinuousSet(1, np.PINF, EXC, EXC)])),
        (EMPTY, R),
        (R, EMPTY)
    )
    @unpack
    def test_complement(self, i, r):
        self.assertEqual(r, i.complement())

    @data(
        (ContinuousSet.parse('[0,1]'), ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,2]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,1]'), ContinuousSet.parse('[0,2]')),
        (ContinuousSet.parse('[0,3]'), ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[0,3]'), ContinuousSet.parse('[0,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse(']2,3]'), ContinuousSet.parse('[1,3]')),
        (ContinuousSet.parse('[1,2]'), ContinuousSet.parse('[3,4]'), RealSet([ContinuousSet.parse('[1,2]'),
                                                                              ContinuousSet.parse('[3,4]')])),
        (EMPTY, ContinuousSet.parse('[0,1]'), ContinuousSet(0, 1)),
        (EMPTY, EMPTY, EMPTY),
        (R, ContinuousSet(0, 1), R)
    )
    @unpack
    def test_union(self, i1, i2, r):
        self.assertEqual(r, i1.union(i2))

    @data(
        ('[0,0]',),
        ('[0,1]',),
        (']0,0[',),
    )
    @unpack
    def test_hash(self, i):
        self.assertEqual(hash(i), hash(pickle.loads(pickle.dumps(i))))

    def test_sample(self):
        # test default usage
        i1 = ContinuousSet.parse("[-1,1]")
        samples = i1.sample(100)
        for sample in samples:
            self.assertTrue(sample in i1)

        # test raising index error
        i2 = ContinuousSet.emptyset_p()
        self.assertRaises(IndexError, i2.sample, 100)

        # test singular value
        i3 = ContinuousSet.parse("[-.5, -.5]")
        samples = i3.sample(100)
        for sample in samples:
            self.assertEqual(sample, -0.5)

    def test_linspace(self):
        i1 = ContinuousSet.parse("(-1,1)")
        samples = i1.linspace(100)
        self.assertEqual(len(samples), 100)
        for sample in samples[1:-1]:
            self.assertTrue(sample in i1)

        i2 = ContinuousSet.emptyset_p()
        self.assertRaises(IndexError, i2.sample, 100)


@ddt
class RealSetTest(unittest.TestCase):

    @data(
        (['[0, 1]', '[2, 3]'], RealSet([ContinuousSet(0, 1), ContinuousSet(2, 3)])),
    )
    @unpack
    def test_creation(self, i, o):
        s = RealSet(i)
        self.assertEqual(s, o)

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
        (['[0,1]', '[2,3]'], RealSet([']1,2[']), EMPTY),
        (['[0,1]', '[2,3]'], RealSet(['[0,1]', '[2,3]']), RealSet(['[0,1]', '[2,3]'])),
        (['[0,1]', '[2,3]'], EMPTY, EMPTY),
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
        (RealSet([EMPTY]), EMPTY),
    )
    @unpack
    def test_simplify(self, i, o):
        self.assertEqual(o, i.simplify())

    @data(
        (RealSet(['[-1,1]', '[-.5,.5]']), RealSet(['[0,1[']), RealSet(['[-1,0[', ']1,1]'])),
        (RealSet(['[-1,1]', '[-.5,.5]']), RealSet(['[0,1]']), ContinuousSet.parse('[-1,0[')),
        (RealSet(['[0,1]', '[1,2]', '[2,3]']), ContinuousSet(1, 2), RealSet(['[0,1[', ']2,3]'])),
        (RealSet([EMPTY]), EMPTY, EMPTY),
    )
    @unpack
    def test_difference(self, i1, i2, o):
        self.assertEqual(o, i1.difference(i2))

    @data(
        (RealSet('[1,2]'), RealSet([']-inf,1[', ']2,inf[']))
    )
    @unpack
    def test_complement(self, i, o):
        self.assertEqual(o, i.complement())

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
        self.assertRaises(IndexError, r2.sample, 100)

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

    def test_contains_interval(self):
        # test regular case
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.contains_interval(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r1.contains_interval(ContinuousSet(-2, 0.25)))
        self.assertTrue(r1.contains_interval(ContinuousSet(0, 0, 2, 2)))

        # test empty set
        r2 = RealSet.emptyset()
        self.assertFalse(r2.contains_interval(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r2.contains_interval(ContinuousSet(0, 0, 2, 2)))

        # test singular value
        r3 = RealSet(["[-.5, -.5]"])
        self.assertTrue(r3.contains_interval(ContinuousSet(-0.5, -0.5)))
        self.assertFalse(r3.contains_interval(ContinuousSet(0, 0, 2, 2)))

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

    def test_intersects(self):
        # test regular case
        r1 = RealSet(['[-1,1]', '[-.5,.5]'])
        self.assertTrue(r1.intersects(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r1.intersects(ContinuousSet(-7, -6)))
        self.assertTrue(r1.intersects(ContinuousSet(0, 0, 2, 2)))

        # test empty set
        r2 = RealSet.emptyset()
        self.assertFalse(r2.intersects(ContinuousSet(-0.25, 0.25)))
        self.assertFalse(r2.intersects(ContinuousSet(0, 0, 2, 2)))

        # test singular value
        r3 = RealSet(["[-.5, -.5]"])
        self.assertTrue(r3.intersects(ContinuousSet(-0.5, -0.5)))
        self.assertFalse(r3.intersects(ContinuousSet(0, 0, 2, 2)))

