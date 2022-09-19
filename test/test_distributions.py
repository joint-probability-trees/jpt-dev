import json
import pickle
from unittest import TestCase

import numpy as np

from jpt.distributions.utils import OrderedDictProxy, DataScaler

try:
    from jpt.base.functions import __module__
    from jpt.distributions.quantile.quantiles import __module__
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import PiecewiseFunction, LinearFunction
    from jpt.distributions.quantile.quantiles import QuantileDistribution
    from jpt.base.intervals import ContinuousSet, EXC, INC


from jpt.base.errors import Unsatisfiability
from jpt.distributions import SymbolicType, Multinomial, NumericType, Gaussian, Numeric, \
    Distribution


class MultinomialTest(TestCase):
    '''Test functions of the multinomial distributions'''

    # the 2nd component is the relevant one / the last point is to be ignored
    # then, the distribution is 5 / 10, 3 / 10, 2 / 10
    DATA = np.array([[1, 0, 8], [1, 0, 8], [1, 0, 8], [1, 1, 8], [1, 1, 8],
                     [1, 2, 8], [1, 0, 8], [1, 1, 8], [1, 2, 8], [1, 0, 8], [1, 0, 8]], dtype=np.float64)

    def setUp(self) -> None:
        self.DistABC = SymbolicType('TestTypeString', labels=['A', 'B', 'C'])
        self.Dist123 = SymbolicType('TestTypeInt', labels=[1, 2, 3, 4, 5])

    def test_creation(self):
        '''Test the creation of the distributions'''
        DistABC = self.DistABC
        Dist123 = self.Dist123

        self.assertTrue(DistABC.equiv(DistABC))
        self.assertTrue(Dist123.equiv(Dist123))
        self.assertFalse(DistABC.equiv(Dist123))

        self.assertEqual(DistABC.n_values, 3)
        self.assertEqual(Dist123.n_values, 5)

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
        d1 = DistABC().set(params=probs)

        self.assertRaises(ValueError, Dist123().set, params=probs)

        self.assertIsInstance(d1, DistABC)
        self.assertEqual(list(d1._params), probs)

        d1.fit(MultinomialTest.DATA,
               rows=np.array(list(range(MultinomialTest.DATA.shape[0] - 1)), dtype=np.int32),
               col=1)

        self.assertAlmostEqual(d1.p({'A'}), 5 / 10, 15)
        self.assertAlmostEqual(d1.p({'B'}), 3 / 10, 15)
        self.assertAlmostEqual(d1.p({'C'}), 2 / 10, 15)

        self.assertAlmostEqual(d1._p({0}), 5 / 10, 15)
        self.assertAlmostEqual(d1._p({1}), 3 / 10, 15)
        self.assertAlmostEqual(d1._p({2}), 2 / 10, 15)

    def test_inference(self):
        '''Posterior, MPE, Expectation'''
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1/2, 1/4, 1/4])

        self.assertEqual(d1.expectation(), 'A')
        self.assertEqual(d1.mpe(), 'A')
        self.assertEqual(d1.crop(excl_values=['B']), DistABC().set([2/3, 0, 1/3]))
        self.assertEqual(d1.crop(incl_values=['A', 'B']), DistABC().set([2 / 3, 1 / 3, 0]))
        self.assertRaises(Unsatisfiability, d1.crop, excl_values=['A', 'B', 'C'])
        self.assertEqual(d1.crop(), d1)

    def test_domain_serialization(self):
        '''(De-)Serialization of Multinomial domains'''
        DistABC = self.DistABC
        Dist123 = self.Dist123

        DistABC_ = Distribution.type_from_json(DistABC.type_to_json())
        Dist123_ = Distribution.type_from_json(Dist123.type_to_json())

        self.assertTrue(DistABC_.equiv(DistABC))
        self.assertTrue(Dist123_.equiv(Dist123))

    def test_distributions_serialization(self):
        '''(De-)Serialziation of Multinomial distributions'''
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        Distribution.type_from_json(DistABC.type_to_json())
        d2 = Distribution.from_json(json.loads(json.dumps(d1.to_json())))
        self.assertEqual(d1, d2)

    def test_distribution_manipulation(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        d2 = DistABC().set(params=[0, .5, .5])
        d3 = DistABC.merge([d1, d2], weights=[.5, .5])
        d1.update(d2, .5)

        self.assertEqual(d3, DistABC().set(params=[.25, .375, .375]))
        self.assertEqual(d1, d3)
        self.assertEqual(d1.update(d2, 0), d1)

    def test_kldiv_equality(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        d2 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        self.assertEqual(d1.kl_divergence(d2), 0)
        self.assertEqual(d1.kl_divergence(d1), 0)
        self.assertEqual(0, DistABC().set(params=[1, 0, 0]).kl_divergence(DistABC().set(params=[1, 0, 0])))

    def test_kldiv_inequality(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        d2 = DistABC().set(params=[.25, .5, .25])
        self.assertEqual(0.1875, d1.kl_divergence(d2))

    def test_kldiv_extreme_inequality(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1, 0, 0])
        d2 = DistABC().set(params=[0, .5, .5])
        self.assertEqual(1, d1.kl_divergence(d2))

    def test_kldiv_type(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        self.assertRaises(TypeError, d1.kl_divergence, Numeric().fit(np.array([[1], [2], [3]],
                                                                              dtype=np.float64),
                                                                     col=0))

    def test_value_conversion(self):
        DistABC = self.DistABC
        self.assertEqual(0, DistABC.label2value('A'))
        self.assertEqual(1, DistABC.label2value('B'))
        self.assertEqual(2, DistABC.label2value('C'))
        self.assertEqual('A', DistABC.value2label(0))
        self.assertEqual('B', DistABC.value2label(1))
        self.assertEqual('C', DistABC.value2label(2))
        self.assertEqual({0, 2}, DistABC.label2value({'A', 'C'}))
        self.assertEqual({'C', 'B'}, DistABC.value2label({2, 1}))


# ----------------------------------------------------------------------------------------------------------------------

class NumericTest(TestCase):
    '''Test class for ``Numeric`` distributions'''

    GAUSSIAN = None

    @classmethod
    def setUp(cls) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            cls.GAUSSIAN = pickle.load(f)
        cls.DistGauss = NumericType('Uniform', values=NumericTest.GAUSSIAN)

    def test_hash(self):
        self.assertNotEqual(hash(Numeric), hash(self.DistGauss))

    def test_creation(self):
        '''The creation of the numeric distributions'''
        DistGauss = self.DistGauss

        # After the scaling, the values must have zero mean and std dev 1
        gauss = Gaussian(data=[[DistGauss.values[d]] for d in NumericTest.GAUSSIAN])
        self.assertAlmostEqual(gauss.mean[0], .0, 5)
        self.assertAlmostEqual(gauss.var[0], 1, 1)

        self.assertAlmostEqual(DistGauss.values.mean, .5, 1)
        self.assertAlmostEqual(DistGauss.values.scale, .6, 1)

    def test_domain_serialization(self):
        DistGauss = self.DistGauss
        self.assertTrue(DistGauss.equiv(DistGauss.type_from_json(DistGauss.type_to_json())))

    def test_fit(self):
        d = Numeric().fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        self.assertEqual(d.cdf, PiecewiseFunction.from_dict({']-∞,0.0[': 0,
                                                             '[0.0,1.0[': '1x',
                                                             '[1.0,∞[': 1}))

    def test_distribution_serialization(self):
        d = Numeric().fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        self.assertEqual(d, Distribution.from_json(d.to_json()))

    def test_manipulation(self):
        DistGauss = self.DistGauss
        data = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        d = DistGauss().fit(data, col=0)
        self.assertEqual(d.expectation(), .5)

        ground_truth = PiecewiseFunction.from_dict({ContinuousSet(np.NINF, DistGauss.values[.1], EXC, EXC): 0,
                                                    ContinuousSet(DistGauss.values[.1], DistGauss.values[.9], INC, EXC):
                                                        LinearFunction.from_points((DistGauss.values[.1], .0),
                                                                                   (DistGauss.values[.9], 1.)),
                                                    ContinuousSet(DistGauss.values[.9], np.PINF, INC, EXC): 1})
        f1 = d.crop(ContinuousSet(.1, .9, EXC, EXC)).cdf.round(10)
        f2 = ground_truth.round(10)
        self.assertEqual(f1, f2)

    def test_kldiv_equality(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        dist1 = DistGauss().fit(data1, col=0)
        self.assertEqual(0, dist1.kl_divergence(dist1))

    def test_kldiv_inequality(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        data2 = np.array([DistGauss.values[l] for l in np.linspace(.5, 1.5, 20)]).reshape(-1, 1)
        dist1 = DistGauss().fit(data1, col=0)
        dist2 = DistGauss().fit(data2, col=0)
        self.assertEqual(0.25, dist1.kl_divergence(dist2))

    def test_kldiv_inequality_extreme(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        data2 = np.array([DistGauss.values[l] for l in np.linspace(5, 10, 20)]).reshape(-1, 1)
        dist1 = DistGauss().fit(data1, col=0)
        dist2 = DistGauss().fit(data2, col=0)
        self.assertEqual(1, dist1.kl_divergence(dist2))

    def test_kldiv_type(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        d1 = DistGauss().fit(data1, col=0)
        self.assertRaises(TypeError, d1.kl_divergence, ...)

    def test_value_conversion(self):
        DistGauss = self.DistGauss
        self.assertEqual(0, DistGauss.value2label(DistGauss.label2value(0)))
        self.assertEqual(ContinuousSet(0, 1),
                         DistGauss.value2label(DistGauss.label2value(ContinuousSet(0, 1))))

    def test_label_inference(self):
        return
        raise NotImplementedError()

    def test_value_inference(self):
        return
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------------------------------------

class DataScalerTest(TestCase):

    DATA = None

    @classmethod
    def setUpClass(cls) -> None:
        DataScalerTest.DATA = Gaussian(5, 10).sample(10000)

    def test_transformation(self):
        scaler = DataScaler()
        scaler.fit(DataScalerTest.DATA)
        # self.assertAlmostEqual(5, scaler.mean, places=0)
        # self.assertAlmostEqual(10, scaler.variance, places=2)

        # Test single transformation
        for x in DataScalerTest.DATA:
            self.assertAlmostEqual(x, scaler.inverse_transform(scaler.transform(x)), 5)

        # Test bulk transformation
        data_ = scaler.transform(DataScalerTest.DATA)
        for x_, x in zip(data_, DataScalerTest.DATA):
            self.assertAlmostEqual(x, scaler.inverse_transform(x_), 5)

