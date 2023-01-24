import numbers

import json
import pickle
from unittest import TestCase

import numpy as np
from dnutils import out

from jpt.distributions.univariate import IntegerType, Integer
from jpt.distributions.utils import OrderedDictProxy, DataScaler

try:
    from jpt.base.functions import __module__
    from jpt.distributions.quantile.quantiles import __module__
    from jpt.base.intervals import __module__, R
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import PiecewiseFunction, LinearFunction
    from jpt.distributions.quantile.quantiles import QuantileDistribution
    from jpt.base.intervals import ContinuousSet, EXC, INC, RealSet


from jpt.base.errors import Unsatisfiability
from jpt.distributions import SymbolicType, Multinomial, NumericType, Gaussian, Numeric, \
    Distribution, ScaledNumeric


class MultinomialDistributionTest(TestCase):
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

        d1._fit(MultinomialDistributionTest.DATA,
                rows=np.array(list(range(MultinomialDistributionTest.DATA.shape[0] - 1)), dtype=np.int32),
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

        self.assertEqual(d1.expectation(), {'A'})
        self.assertEqual(d1.mpe(), (0.5, {'A'}))
        self.assertEqual(d1.crop([0, 2]), DistABC().set([2/3, 0, 1/3]))
        self.assertEqual(d1.crop([0, 1]), DistABC().set([2 / 3, 1 / 3, 0]))
        self.assertRaises(Unsatisfiability, d1.crop, restriction=[])
        # self.assertEqual(d1.crop(), d1)

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
        self.assertRaises(
            TypeError,
            d1.kl_divergence,
            Numeric()._fit(np.array([[1], [2], [3]], dtype=np.float64), col=0))

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

class NumericDistributionTest(TestCase):
    '''Test class for ``Numeric`` distributions'''

    GAUSSIAN = None

    @classmethod
    def setUp(cls) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            cls.GAUSSIAN = pickle.load(f)
        cls.DistGauss = NumericType('Uniform', values=NumericDistributionTest.GAUSSIAN)

    def test_hash(self):
        self.assertNotEqual(hash(Numeric), hash(self.DistGauss))

    def test_creation(self):
        '''The creation of the numeric distributions'''
        DistGauss = self.DistGauss

        # After the scaling, the values must have zero mean and std dev 1
        gauss = Gaussian(data=[[DistGauss.values[d]] for d in NumericDistributionTest.GAUSSIAN])
        self.assertAlmostEqual(gauss.mean[0], .0, 5)
        self.assertAlmostEqual(gauss.var[0], 1, 1)

        self.assertAlmostEqual(DistGauss.values.mean, .5, 1)
        self.assertAlmostEqual(DistGauss.values.scale, .6, 1)

    def test_domain_serialization(self):
        DistGauss = self.DistGauss
        self.assertTrue(DistGauss.equiv(DistGauss.type_from_json(DistGauss.type_to_json())))

    def test_fit(self):
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        self.assertEqual(d.cdf, PiecewiseFunction.from_dict({
            ']-∞,0.0[': 0,
            '[0.0,1.0000000000000002[': '1x',
            '[1.0000000000000002,∞[': 1
        }))

    def test_distribution_serialization(self):
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        self.assertEqual(d, Distribution.from_json(d.to_json()))

    def test_manipulation(self):
        DistGauss = self.DistGauss
        data = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        d = DistGauss()._fit(data, col=0)
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
        dist1 = DistGauss()._fit(data1, col=0)
        self.assertEqual(0, dist1.kl_divergence(dist1))

    def test_kldiv_inequality(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        data2 = np.array([DistGauss.values[l] for l in np.linspace(.5, 1.5, 20)]).reshape(-1, 1)
        dist1 = DistGauss()._fit(data1, col=0)
        dist2 = DistGauss()._fit(data2, col=0)
        self.assertEqual(np.nextafter(0.25, 1), dist1.kl_divergence(dist2))

    def test_kldiv_inequality_extreme(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        data2 = np.array([DistGauss.values[l] for l in np.linspace(5, 10, 20)]).reshape(-1, 1)
        dist1 = DistGauss()._fit(data1, col=0)
        dist2 = DistGauss()._fit(data2, col=0)
        self.assertEqual(1, dist1.kl_divergence(dist2))

    def test_kldiv_type(self):
        DistGauss = self.DistGauss
        data1 = np.array([DistGauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        d1 = DistGauss()._fit(data1, col=0)
        self.assertRaises(TypeError, d1.kl_divergence, ...)

    def test_value_conversion(self):
        DistGauss = self.DistGauss
        self.assertEqual(0, DistGauss.value2label(DistGauss.label2value(0)))
        self.assertEqual(ContinuousSet(0, 1),
                         DistGauss.value2label(DistGauss.label2value(ContinuousSet(0, 1))))
        self.assertEqual(RealSet(['[0, 1]', '[2,3]']),
                         DistGauss.value2label(DistGauss.label2value(RealSet(['[0, 1]', '[2,3]']))))

    def _test_label_inference(self):
        return
        raise NotImplementedError()

    def test_value_inference_normal(self):
        '''Inference under "normal" circumstances.'''
        dist = Numeric().set(params=QuantileDistribution.from_cdf(PiecewiseFunction.from_dict(
            {']-inf,0[': 0,
             '[0,1[': LinearFunction(1, 0),
             '[1,inf[': 1}
        )))
        self.assertEqual(0, dist._p(-1))
        self.assertEqual(0, dist._p(.5))
        self.assertEqual(0, dist._p(2))
        self.assertEqual(1, dist._p(R))
        self.assertEqual(.5, dist._p(ContinuousSet.parse('[0,.5]')))

    def test_value_inference_sinularity(self):
        '''PDF has a singularity like a Dirac impulse function.'''
        dist = Numeric().set(params=QuantileDistribution.from_cdf(PiecewiseFunction.from_dict(
            {']-inf,0.0[': 0,
             '[0.0,inf[': 1}
        )))
        self.assertEqual(0, dist._p(ContinuousSet.parse(']-inf,0[')))
        self.assertEqual(1, dist._p(ContinuousSet.parse('[0,inf[')))
        self.assertEqual(1, dist._p(0))


# ----------------------------------------------------------------------------------------------------------------------

class IntegerDistributionTest(TestCase):

    def test_set(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()

        # Act
        fair_dice.set([1 / 6] * 6)

        # Assert
        self.assertTrue(
            (np.array([1 / 6] * 6, dtype=np.float64) == fair_dice.probabilities).all(),
        )

    def test_fit(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice: Integer = dice()
        data = np.array(
            [[13, 1, 2],
             [14, 2, -1],
             [17, 3, -5],
             [18, 4, 20],
             [100, 5, 19],
             [-8, 6, -1]],
            dtype=np.float64
        )

        # Act
        fair_dice.fit(data, None, 1)

        # Assert
        self.assertTrue(
            (np.array([1 / 6] * 6, dtype=np.float64) == fair_dice.probabilities).all(),
        )

    def test_sampling(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        samples = list(fair_dice.sample(100))
        sample = fair_dice.sample_one()

        # Assert
        for s in samples:
            self.assertIsInstance(s, numbers.Integral)
            self.assertGreaterEqual(s, 1)
            self.assertLessEqual(s, 6)
        self.assertEqual(100, len(samples))

        self.assertGreaterEqual(sample, 1)
        self.assertLessEqual(sample, 6)
        self.assertIsInstance(sample, numbers.Integral)

    def test_expectation(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        _e = fair_dice._expectation()
        e = fair_dice.expectation()

        # Assert
        self.assertEqual(3.5, e)
        self.assertEqual(2.5, _e)

    def test_inference(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        p_singular_label = fair_dice.p(6)
        p_set_label = fair_dice.p({4, 5, 6})
        p_singular_value = fair_dice._p(0)
        p_set_values = fair_dice._p({0, 1, 2})

        # Assert
        self.assertEqual(1 / 6, p_singular_label)
        self.assertEqual(1 / 6, p_singular_value)
        self.assertEqual(3 / 6, p_set_label)
        self.assertEqual(3 / 6, p_set_values)

        self.assertRaises(ValueError, fair_dice.p, 0)
        self.assertRaises(ValueError, fair_dice.p, 7)

    def test_crop(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        biased_dice = fair_dice.crop({2, 3})
        _biased_dice = fair_dice._crop({0, 1})

        # Assert
        self.assertEqual(list([0, .5, .5, 0, 0, 0]), list(biased_dice.probabilities))
        self.assertEqual(list([.5, .5, 0, 0, 0, 0]), list(_biased_dice.probabilities))
        self.assertRaises(Unsatisfiability, biased_dice.crop, {1})

    def test_mpe(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        biased_dice = dice()
        biased_dice.set([0/6, 1/6, 2/6, 1/6, 1/6, 1/6])

        # Act
        p_fair, fair_mpe = fair_dice.mpe()
        _p_fair, _fair_mpe = fair_dice._mpe()
        p_biased, biased_mpe = biased_dice.mpe()
        _p_biased, _biased_mpe = biased_dice._mpe()

        # Assert
        self.assertEqual(set(range(1, 7)), fair_mpe)
        self.assertEqual(1/6, p_fair)
        self.assertEqual(set(range(0, 6)), _fair_mpe)
        self.assertEqual(1 / 6, _p_fair)

        self.assertEqual({3}, biased_mpe)
        self.assertEqual(2 / 6, p_biased)
        self.assertEqual({2}, _biased_mpe)
        self.assertEqual(2 / 6, _p_biased)

    def test_merge(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)
        biased_dice = dice()
        biased_dice.set([0 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 1 / 6])

        # Act
        merged = dice.merge(distributions=[fair_dice, biased_dice], weights=[.5, .5])

        # Assert
        self.assertEqual([.5/6, 1/6, 1.5/6, 1/6, 1/6, 1/6], list(merged.probabilities))

    def test_serialization(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        dice_ = Distribution.type_from_json(dice.to_json())
        fair_dice_ = Distribution.from_json(fair_dice.to_json())

        # Assert
        self.assertTrue(dice.equiv(dice_))
        self.assertEqual(fair_dice_, fair_dice)


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


# ----------------------------------------------------------------------------------------------------------------------

class TypeGeneratorTest(TestCase):

    def test_numeric_type(self):
        self.assertRaises(ValueError, NumericType, 'BlaType', [None, 1, 2])
        self.assertRaises(ValueError, NumericType, 'BlaType', [1, 2, float('inf')])
        self.assertTrue(issubclass(NumericType('bla', [1, 2, 3, 4]), ScaledNumeric))

    def test_integer_type(self):
        self.assertRaises(ValueError, IntegerType, 'Bla', 3, 2)
        t = IntegerType('Months', 1, 12)
        self.assertEqual(list(range(0, 12)), list(t.values.values()))
        self.assertEqual(list(range(1, 13)), list(t.labels.values()))
        self.assertEqual(1, t.lmin)
        self.assertEqual(12, t.lmax)
        self.assertEqual(0, t.vmin)
        self.assertEqual(11, t.vmax)

    def test_symbolic_type(self):
        t = SymbolicType('Object', labels=['Bowl', 'Spoon', 'Cereal'])
        self.assertEqual('Object', t.__qualname__)
        self.assertEqual(['Bowl', 'Spoon', 'Cereal'], list(t.labels.values()))