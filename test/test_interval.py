from ddt import ddt, data, unpack
import numpy as np
import unittest

try:
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.intervals import ContinuousSet, INC, EXC, EMPTY


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
        '''Parsing and creation'''
        self.assertEqual(ContinuousSet.parse(s), i)
        self.assertEqual(hash(i), hash(i.copy()))

    def test_value_check(self):
        self.assertRaises(ValueError, ContinuousSet, 1, -1)

    @data(']0, 0[',)
    def test_emptyness(self, s):
        self.assertTrue(ContinuousSet.parse(s).isempty(),
                        msg='%s is not recognized empty.' % s)

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
          ('[-10, 10]', '[1,1]', '[1,1]'))
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

    @data(('[-10, 5]',), (']5, 10]',),
          ('[0, 1]',), ('[2, 3]',),
          (']-inf,0[',), ('[0, inf[',),
          ('[0, 1]',), (']0,0[',),
          (']-1,-1[',), (']-1,-1[',))
    @unpack
    def test_serialization(self, i):
        i = ContinuousSet.parse(i)
        self.assertEqual(i, ContinuousSet.from_json(i.to_json()))
