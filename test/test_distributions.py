from unittest import TestCase

import numpy as np

from jpt.learning.distributions import SymbolicType, OrderedDictProxy, Multinomial


class MultinomialTest(TestCase):
    '''Test functions of the multinomial distributions'''

    # the 2nd component is the relevant one / the last point is to be ignored
    # then, the distribution is 5 / 10, 3 / 10, 2 / 10
    DATA = np.array([[1, 0, 8], [1, 0, 8], [1, 0, 8], [1, 1, 8], [1, 1, 8],
                     [1, 2, 8], [1, 0, 8], [1, 1, 8], [1, 2, 8], [1, 0, 8], [1, 0, 8]])

    def setUp(self) -> None:
        self.DistABC = SymbolicType('TestTypeString', labels=['A', 'B', 'C'])
        self.Dist123 = SymbolicType('TestTypeInt', labels=[1, 2, 3, 4, 5])

    def test_creation(self):
        '''Test the creation of the distributions'''
        DistABC = self.DistABC
        Dist123 = self.Dist123

        self.assertTrue(issubclass(DistABC, Multinomial))
        self.assertTrue(issubclass(Dist123, Multinomial))

        self.assertEqual(DistABC.values, OrderedDictProxy([('A', 0), ('B', 1), ('C', 2)]))
        self.assertEqual(DistABC.labels, OrderedDictProxy([(0, 'A'), (1, 'B'), (2, 'C')]))

        self.assertEqual(Dist123.values, OrderedDictProxy([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]))
        self.assertEqual(Dist123.labels, OrderedDictProxy([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]))

    def test_fit(self):
        '''Fitting of multinomial distributions'''
        DistABC = self.DistABC
        Dist123 = self.Dist123

        probs = [1 / 3] * 3
        d1 = DistABC(params=probs)

        self.assertRaises(ValueError, Dist123, params=probs)

        self.assertIsInstance(d1, DistABC)
        self.assertEqual(list(d1._params), probs)

        d1.fit(MultinomialTest.DATA,
               rows=np.array(list(range(MultinomialTest.DATA.shape[0] - 1)), dtype=np.int32),
               col=1)

        self.assertAlmostEqual(d1.p('A'), 5 / 10, 15)
        self.assertAlmostEqual(d1.p('B'), 3 / 10, 15)
        self.assertAlmostEqual(d1.p('C'), 2 / 10, 15)

        self.assertAlmostEqual(d1._p(0), 5 / 10, 15)
        self.assertAlmostEqual(d1._p(1), 3 / 10, 15)
        self.assertAlmostEqual(d1._p(2), 2 / 10, 15)

    def test_serialization(self):
        '''(De-)Serialization of Multinomials'''
        pass
