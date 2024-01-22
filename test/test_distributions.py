import json
import numbers
import pickle
from typing import Type
from unittest import TestCase

import numpy as np
import scipy.stats
from ddt import data, unpack, ddt
from dnutils.tools import ifstr

from jpt import SymbolicVariable
from jpt.base.constants import eps
from jpt.distributions.univariate import IntegerType, Integer
from jpt.distributions.utils import OrderedDictProxy, DataScaler
from utils import uniform_numeric

try:
    from jpt.base.functions import __module__
    from jpt.distributions.quantile.quantiles import __module__
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.functions import PiecewiseFunction, LinearFunction, ConstantFunction
    from jpt.distributions.quantile.quantiles import QuantileDistribution
    from jpt.base.intervals import ContinuousSet, EXC, INC, RealSet, R

from jpt.base.errors import Unsatisfiability
from jpt.distributions import SymbolicType, Multinomial, NumericType, Gaussian, Numeric, \
    Distribution, ScaledNumeric, Bool


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

    def test_label2value(self):
        '''Test the conversion from label to value space'''
        # Arrange
        ABC = self.DistABC
        # Act
        value_singular = ABC.label2value('A')
        value_set = ABC.label2value({'A', 'B', 'C'})
        value_list = ABC.label2value(['B', 'C'])
        value_tuple = ABC.label2value(('A', 'B', 'C'))
        # Assert
        self.assertEqual(0, value_singular)
        self.assertEqual({0, 1, 2}, value_set)
        self.assertEqual((0, 1, 2), value_tuple)
        self.assertEqual([1, 2], value_list)
        self.assertRaises(
            ValueError,
            ABC.label2value,
            'D'
        )

    def test_value2label(self):
        '''Test the conversion from value to label space'''
        # Arrange
        ABC = self.DistABC
        # Act
        label_singular = ABC.value2label(1)
        label_set = ABC.value2label({0, 1, 2})
        label_list = ABC.value2label([0, 1, 2])
        label_tuple = ABC.value2label((0, 1, 2))
        # Assert
        self.assertEqual('B', label_singular)
        self.assertEqual({'A', 'B', 'C'}, label_set)
        self.assertEqual(('A', 'B', 'C'), label_tuple)
        self.assertEqual(['A', 'B', 'C'], label_list)
        self.assertRaises(
            ValueError,
            ABC.value2label,
            'D'
        )

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

    def test_crop(self):
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        result1 = abc.crop({'A', 'C'})
        result2 = abc.crop('B')

        # Assert
        self.assertEqual([2 / 3, 0, 1 / 3], list(result1.probabilities))
        self.assertEqual([0, 1, 0], list(result2.probabilities))
        self.assertRaises(Unsatisfiability, abc.crop, ())

    def test__crop(self):
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        result1 = abc._crop({0, 2})
        result2 = abc._crop(1)

        # Assert
        self.assertEqual([2 / 3, 0, 1 / 3], list(result1.probabilities))
        self.assertEqual([0, 1, 0], list(result2.probabilities))
        self.assertRaises(Unsatisfiability, abc._crop, ())

    def test_mpe_uniform(self):
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 3, 1 / 3, 1 / 3])

        # Act
        result_uniform = abc.mpe()

        # Assert
        self.assertEqual(({'A', 'B', 'C'}, 1 / 3), result_uniform)

    def test_expectation(self):
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        result = abc.expectation()

        # Assert
        self.assertEqual({'A'}, result)

    def test_inference(self):
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        result_singular = abc.p('A')
        result_set = abc.p({'A', 'C'})
        result_list_duplicate = abc.p(['A', 'C', 'C'])

        # Assert
        self.assertEqual(.5, result_singular)
        self.assertEqual(.75, result_set)
        self.assertEqual(.75, result_list_duplicate)

    def test_domain_serialization(self):
        '''(De-)Serialization of Multinomial domains'''
        DistABC = self.DistABC
        Dist123 = self.Dist123

        DistABC_ = Distribution.from_json(DistABC.type_to_json())
        Dist123_ = Distribution.from_json(Dist123.type_to_json())

        self.assertTrue(DistABC_.equiv(DistABC))
        self.assertTrue(Dist123_.equiv(Dist123))

    def test_distributions_serialization(self):
        '''(De-)Serialziation of Multinomial distributions'''
        # Arrange
        DistABC = self.DistABC
        d = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        d_type = Distribution.from_json(DistABC.type_to_json())
        d_inst = d_type.from_json(json.loads(json.dumps(d.to_json())))

        # Assert
        self.assertEqual(d, d_inst)
        self.assertTrue(DistABC.equiv(d_type))

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

    def test_plot(self):
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        d1.plot(
            view=False,
            horizontal=True
        )

    def test_plot_coin(self):
        fr = SymbolicVariable('BiasedCoin', Bool)
        d1 = fr.distribution().set(5/12.)
        d1.plot(
            view=False,
            horizontal=True
        )

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

    def test_mpe(self):
        # Arrange
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        state, likelhood = d1.mpe()

        # Assert
        self.assertEqual(0.5, likelhood)
        self.assertEqual({"A"}, state)

    def test_k_mpe(self):
        # Arrange
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        k_mpe = d1.k_mpe(3)

        # Assert
        self.assertEqual(
            [({'A'}, 1/2), ({'B', 'C'}, 1/4)],
            k_mpe
        )

    def test_jaccard_identity(self):
        d1 = self.DistABC().set([.1, .4, .5])
        jacc = Multinomial.jaccard_similarity(d1, d1)
        self.assertEqual(1., jacc)

    def test_jaccard_disjoint(self):
        d1 = self.DistABC().set([0., 0., 1.])
        d2 = self.DistABC().set([1., 0., 0.])
        jacc = Multinomial.jaccard_similarity(d1, d2)
        self.assertEqual(0., jacc)

    def test_jaccard_overlap(self):
        d1 = self.DistABC().set([.1, .4, .5])
        d2 = self.DistABC().set([.2, .4, .4])

        jacc = Multinomial.jaccard_similarity(d1, d2)
        self.assertAlmostEqual(9/11, jacc, places=8)

    def test_jaccard_symmetry(self):
        d1 = self.DistABC().set([.1, .4, .5])
        d2 = self.DistABC().set([.2, .4, .4])
        jacc1 = Multinomial.jaccard_similarity(d1, d2)
        jacc2 = Multinomial.jaccard_similarity(d2, d1)
        self.assertEqual(jacc1, jacc2)


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class NumericDistributionTest(TestCase):
    '''Test class for ``Numeric`` distributions'''

    GAUSSIAN = None

    @classmethod
    def setUp(cls) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            cls.GAUSSIAN = pickle.load(f)
        cls.DistGauss: Type[Numeric] = NumericType(
            'Normal',
            values=NumericDistributionTest.GAUSSIAN
        )

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

    def test_copy(self):
        # Arrange
        a, b = 2, 3
        uniform = uniform_numeric(a, b)

        # Act
        uniform_ = uniform.copy()

        # Assert
        self.assertEqual(uniform, uniform_)
        self.assertNotEqual(id(uniform_), id(uniform))

    def test_domain_serialization(self):
        DistGauss = self.DistGauss
        self.assertTrue(
            DistGauss.equiv(
                DistGauss.type_from_json(
                    DistGauss.type_to_json()
                )
            )
        )

    def test_fit(self):
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        self.assertEqual(d.cdf, PiecewiseFunction.from_dict({
            ']-∞,0.0[': 0,
            '[0.0,1.0000000000000002[': '1x',
            '[1.0000000000000002,∞[': 1
        }))

    def test_distribution_serialization(self):
        # Arrange
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)

        # Act
        d_type = Distribution.from_json(json.loads(json.dumps(type(d).to_json())))
        d_inst = d_type.from_json(json.loads(json.dumps(d.to_json())))

        # Assert
        self.assertTrue(Numeric.equiv(d_type))
        self.assertEqual(d, d_inst)

    def test_crop(self):
        # Arrange
        Gauss: Type[Numeric] = self.DistGauss
        data = np.array([Gauss.values[l] for l in np.linspace(0, 1, 20)]).reshape(-1, 1)
        dist = Gauss()._fit(data, col=0)
        # Act
        cdf = dist._crop(
            ContinuousSet(Gauss.values[.1], Gauss.values[.9], EXC, EXC)
        ).cdf
        # Assert
        ground_truth = PiecewiseFunction.from_dict({
            '(-inf,%s)' % Gauss.values[.1]: 0,
            '[%s,%s)' % (Gauss.values[.1], Gauss.values[.9]):
                LinearFunction.from_points(
                    (Gauss.values[.1], .0),
                    (Gauss.values[.9], 1.)
                ),
            '[%s,inf)' % Gauss.values[.9]: 1
        })
        self.assertAlmostEqual(.5, dist.expectation(), places=10)
        self.assertEqual(ground_truth.round(10), cdf.round(10))

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

    def test_value2label(self):
        # Arrange
        Gauss = self.DistGauss

        # Act
        label_scalar = Gauss.value2label(0)
        label_interval = Gauss.value2label(
            ContinuousSet(0, 1)
        )

        # Assert
        self.assertAlmostEqual(.5, label_scalar, places=2)
        self.assertEqual(
            0,
            Gauss.value2label(
                Gauss.label2value(0)
            )
        )
        self.assertIsInstance(
            label_interval,
            ContinuousSet
        )
        self.assertEqual(
            ContinuousSet(0.5, 1.1),
            round(label_interval, 1)
        )
        self.assertEqual(
            ContinuousSet(0, 1),
            Gauss.value2label(
                Gauss.label2value(
                    ContinuousSet(0, 1)
                )
            )
        )

    def test_label2value(self):
        # Arrange
        Gauss = self.DistGauss

        # Act
        value_realset = Gauss.label2value(
            RealSet(['[0,1]', '[2,3]'])
        )
        value_scalar = Gauss.label2value(.5)
        value_interval = Gauss.label2value(
            ContinuousSet(0, 1)
        )

        # Assert
        self.assertEqual(
            RealSet([
                ContinuousSet(-.9, .9),
                ContinuousSet(2.6, 4.3)
            ]),
            round(value_realset, 1)
        )
        self.assertAlmostEqual(
            0,
            value_scalar,
            places=2
        )
        self.assertEqual(
            ContinuousSet(-.9, .9),
            round(value_interval, 1)
        )

    def test_mpe(self):
        np.random.seed(69)
        d = Numeric()._fit(np.random.normal(size=(100, 1)), col=0)
        likelihood, state = d.mpe()
        self.assertEqual(max(f.value for f in d.pdf.functions), likelihood)

    def test_k_mpe(self):
        # Arrange
        d = Numeric(precision=0)._fit(
            np.array([[1.], [2.5], [3.]]),
            col=0
        )

        # Act
        k_mpe = d.k_mpe(3)

        # Assert
        self.assertEqual(
            [
                (ContinuousSet(2.5, 3), 1.),
                (ContinuousSet(1, 2.5, INC, EXC), 1/3)
            ],
            k_mpe
        )

    def _test_label_inference(self):
        raise NotImplementedError()

    @data(
        (-1, 0),
        (.5, 0),
        (2, 0),
        (R, 1),
        ('[0,.5]', .5),
        ('(-inf, .5]', .5),
        ('[.5,inf)', .5),
        ('[.5,.5]', 0),
        (RealSet([
            '[0,.25]', '[.75,1]'
        ]), .5)
    )
    @unpack
    def test_inference(self, query, truth):
        '''Inference under "normal" circumstances.'''
        # Arrange
        dist = uniform_numeric(0, 1)
        query = ifstr(query, ContinuousSet.parse)

        # Act
        result = dist._p(query)

        # Assert
        self.assertEqual(
            truth,
            result
        )

    def test_value_inference_singularity(self):
        '''PDF has a singularity like a Dirac impulse function.'''
        dist = Numeric().set(
            params=QuantileDistribution.from_cdf(
                PiecewiseFunction.from_dict({
                    ']-inf,0.0[': 0,
                    '[0.0,inf[': 1
                })
            )
        )
        self.assertEqual(0, dist._p(ContinuousSet.parse(']-inf,0[')))
        self.assertEqual(1, dist._p(ContinuousSet.parse('[0,inf[')))
        self.assertEqual(1, dist._p(0))

    def test_sampling(self):
        """Sampling from a distribution"""
        data = np.random.normal(0, 1, 100).reshape(-1, 1)
        p = QuantileDistribution()
        p.fit(data, np.arange(100), 0)
        pdf = p.pdf
        samples = p.sample(100)
        self.assertTrue(all([pdf.eval(v) > 0 for v in samples]))

    def test_moments(self):
        np.random.seed(69)
        data = np.random.normal(0, 1, 100000).reshape(-1, 1)
        distribution = Numeric()
        distribution._fit(data, np.arange(len(data)), 0)

        data_mean = np.average(data)
        dist_mean = distribution.moment(1, 0)
        self.assertAlmostEqual(data_mean, dist_mean, delta=0.01)

        # be aware the empirical moments and qpd moments diverge
        for order in range(2, 4):
            empirical_moment = scipy.stats.moment(data, order)[0]
            dist_moment = distribution.moment(order, dist_mean)
            self.assertAlmostEqual(empirical_moment, dist_moment, delta=np.power(0.9, -order))

    def test_jaccard_identity(self):
        d1 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0, 1),
                    ConstantFunction(1)
                )
            )
        )

        jacc = Numeric.jaccard_similarity(d1, d1)
        self.assertEqual(1., jacc)

    def test_jaccard_disjoint(self):
        d1 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0, 1),
                    ConstantFunction(1)
                )
            )
        )

        d2 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(2, 3),
                    ConstantFunction(1)
                )
            )
        )

        jacc = Numeric.jaccard_similarity(d1, d2)
        self.assertEqual(0., jacc)

    def test_plot(self):
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        d.plot(
            view=False,
            title="Fancy Title",
            xlabel='my value',
            # color="#800080",
            color="#8000804D",
            # color='rgb(128, 0, 128)'
            # color='rgba(128, 0, 128, 179)'
        )

    def test_jaccard_overlap(self):
        d1 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0, 1),
                    ConstantFunction(1)
                )
            )
        )
        d2 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0.5, 1.5),
                    ConstantFunction(1)
                )
            )
        )

        jacc = Numeric.jaccard_similarity(d1, d2)
        self.assertAlmostEqual(1 / 3, jacc, places=8)

    def test_jaccard_symmetry(self):
        d1 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0, 1),
                    ConstantFunction(1)
                )
            )
        )
        d2 = Numeric().set(
            params=QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0.5, 1.5),
                    ConstantFunction(1)
                )
            )
        )

        jacc1 = Numeric.jaccard_similarity(d1, d2)
        jacc2 = Numeric.jaccard_similarity(d2, d1)
        self.assertEqual(jacc1, jacc2)

    def test_jaccard_singularity(self):
        # Arrange
        d1 = Numeric().set(
            QuantileDistribution.from_pdf(
                PiecewiseFunction
                .zero()
                .overwrite_at(
                    ContinuousSet(0, 0 + eps, INC, EXC),
                    ConstantFunction(np.inf)
                )
            )
        )
        d2 = d1.copy()

        # Act
        similarity = Numeric.jaccard_similarity(d1, d2)

        # Assert
        self.assertEqual(
            1,
            similarity
        )

    def test_add(self):
        # Arrange
        x = uniform_numeric(-1, 1)
        y = uniform_numeric(-1, 1)
        # Act
        z = (x + y)

        x.plot(view=False)
        y.plot(view=False)
        z.plot(view=False)
        # Assert
        self.assertAlmostEqual(
            x.expectation() + y.expectation(),
            z.expectation(),
            places=10
        )
        self.assertEqual(
            PiecewiseFunction.from_dict({
                '(-∞,-2.0)': 0,
                '[-2.0,2.0000000000000004)': .25,
                '[2.0000000000000004,∞)': 0
            }),
            z.pdf
        )

    def test__expectation_uniform(self):
        # Arrange
        uniform = uniform_numeric(1, 2)

        # Act
        expectation = uniform._expectation()

        # Assert
        self.assertEqual(
            1.5,
            expectation
        )
        self.assertEqual(
            expectation,
            uniform.expectation()
        )

    def test__expectation_2uniform(self):
        # Arrange
        dist = Numeric().set(
            QuantileDistribution.merge([
                uniform_numeric(1, 2),
                uniform_numeric(2, 4)
            ])
        )

        # Act
        expectation = dist._expectation()

        # Assert
        self.assertEqual(
            2.25,
            expectation
        )
        self.assertEqual(
            expectation,
            dist.expectation()
        )

    def test__moment_uniform(self):
        # Arrange
        a, b = 2, 3
        uniform = uniform_numeric(a, b)

        # Act
        moment_0 = uniform._moment(0, 0)  # 1
        moment_1_raw = uniform._moment(1, 0)  # Expectation
        moment_1_central = uniform._moment(1, moment_1_raw)  # 0
        moment_2_central = uniform._moment(2, moment_1_raw)  # Variance

        # Assert
        self.assertEqual(
            1,
            moment_0
        )
        self.assertEqual(
            (a + b) / 2,
            moment_1_raw
        )
        self.assertEqual(
            0,
            moment_1_central
        )
        self.assertEqual(
            1 / 12 * (b - a) ** 2,
            moment_2_central
        )

    def test_mpe(self):
        # Arrange
        dist = Numeric().set(
            QuantileDistribution.merge([
                uniform_numeric(0, 2),
                uniform_numeric(2, 3),
                uniform_numeric(3, 5),
            ])
        )

        # Act
        mpe_state, likelihood = dist.mpe()

        # Assert
        self.assertEqual(
            (round(1 / 3, 10), ContinuousSet.parse('[2,3)')),
            (round(likelihood, 10), mpe_state)
        )


# ----------------------------------------------------------------------------------------------------------------------

class IntegerDistributionTest(TestCase):

    Die = IntegerType('Dice', 1, 6)

    def test_value2label(self):
        # Arrange
        Die = self.Die

        # Act
        value_scalar = Die.value2label(0)
        value_set = Die.value2label({0})
        value_list = Die.value2label([0, 5])
        value_tuple = Die.value2label((1,))

        # Assert
        self.assertEqual(1, value_scalar)
        self.assertEqual({1}, value_set)
        self.assertEqual([1, 6], value_list)
        self.assertEqual((2,), value_tuple)
        self.assertRaises(
            ValueError,
            Die.value2label,
            6
        )

    def test_label2value(self):
        # Arrange
        Die = self.Die

        # Act
        label_scalar = Die.label2value(1)
        label_set = Die.label2value({1})
        label_list = Die.label2value([1, 6])
        label_tuple = Die.label2value((2,))

        # Assert
        self.assertEqual(0, label_scalar)
        self.assertEqual({0}, label_set)
        self.assertEqual([0, 5], label_list)
        self.assertEqual((1,), label_tuple)
        self.assertRaises(ValueError, Die.label2value, 0)

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
        p_duplicate_labels = fair_dice.p([1, 2, 2])

        # Assert
        self.assertEqual(1 / 6, p_singular_label)
        self.assertEqual(1 / 6, p_singular_value)
        self.assertEqual(3 / 6, p_set_label)
        self.assertEqual(3 / 6, p_set_values)
        self.assertEqual(2 / 6, p_duplicate_labels)

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
        biased_dice.set([0 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 1 / 6])

        # Act
        fair_mpe, p_fair = fair_dice.mpe()
        _fair_mpe, _p_fair = fair_dice._mpe()
        biased_mpe, p_biased = biased_dice.mpe()
        _biased_mpe, _p_biased = biased_dice._mpe()

        # Assert
        self.assertEqual(set(range(1, 7)), fair_mpe)
        self.assertEqual(1 / 6, p_fair)
        self.assertEqual(set(range(0, 6)), _fair_mpe)
        self.assertEqual(1 / 6, _p_fair)

        self.assertEqual({3}, biased_mpe)
        self.assertEqual(2 / 6, p_biased)
        self.assertEqual({2}, _biased_mpe)
        self.assertEqual(2 / 6, _p_biased)

    def test_k_mpe(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)
        biased_dice = dice()
        biased_dice.set([0 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 1 / 6])

        # Act
        fair_k_mpe = fair_dice.k_mpe(3)
        biased_k_mpe = biased_dice.k_mpe(3)

        # Assert
        self.assertEqual(
            fair_k_mpe,
            [({1, 2, 3, 4, 5, 6}, 1 / 6)]
        )

        self.assertEqual(
        [({3}, 0.3333333333333333), ({2, 4, 5, 6}, 0.16666666666666666)],
            biased_k_mpe
        )

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
        self.assertEqual([.5 / 6, 1 / 6, 1.5 / 6, 1 / 6, 1 / 6, 1 / 6], list(merged.probabilities))

    def test_serialization(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        dice_type = Distribution.from_json(dice.to_json())
        fair_dice_inst = dice.from_json(fair_dice.to_json())

        # Assert
        self.assertTrue(dice.equiv(dice_type))
        self.assertEqual(fair_dice_inst, fair_dice)

    def test_moment(self):
        data = np.random.randint(0, 10, size=(1000, 1))

        distribution = IntegerType("test", 0, 10)()
        distribution._fit(data, np.arange(len(data)), 0)

        data_mean = np.average(data)
        dist_mean = distribution.moment(1, 0)
        self.assertAlmostEqual(data_mean, dist_mean, delta=0.01)

        # be aware the empirical moments and qpd moments diverge
        for order in range(2, 4):
            empirical_moment = scipy.stats.moment(data, order)[0]
            dist_moment = distribution.moment(order, dist_mean)
            self.assertAlmostEqual(empirical_moment, dist_moment, delta=0.01)

    def test_jaccard_identity(self):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([1 / 6] * 6)
        jacc = Integer.jaccard_similarity(d1, d1)
        self.assertEqual(1., jacc)

    def test_jaccard_disjoint(self):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([0., 0., 0., 0., 0., 1.])
        d2 = dice().set([1., 0., 0., 0., 0., 0.])
        jacc = Integer.jaccard_similarity(d1, d2)
        self.assertEqual(0., jacc)

    def test_jaccard_overlap(self):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([2/6, 0/6, 1/6, 1/6, 1/6, 1/6])
        d2 = dice().set([0/6, 2/6, 1/6, 1/6, 1/6, 1/6])
        jacc = Integer.jaccard_similarity(d1, d2)
        self.assertEqual(.5, jacc)

    def test_jaccard_symmetry(self):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([2/6, 0/6, 1/6, 1/6, 1/6, 1/6])
        d2 = dice().set([0/6, 2/6, 1/6, 1/6, 1/6, 1/6])
        jacc1 = Integer.jaccard_similarity(d1, d2)
        jacc2 = Integer.jaccard_similarity(d2, d1)
        self.assertEqual(jacc1, jacc2)

    def test_add(self):
        pos = IntegerType('Pos', 0, 6)
        posx = pos()
        posx.set([0, 0, 1, 0, 0, 0, 0])

        delta = IntegerType('Delta', -1, 1)
        deltax = delta()
        deltax.set([0, 0, 1])

        sumpos = posx.add(deltax)

        self.assertEqual(list(sumpos.labels.values()), [-1, 0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(list(sumpos.values.values()), [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(list(sumpos.probabilities), [0, 0, 0, 0, 1, 0, 0, 0, 0])

    def test_add_bernoulli(self):
        coin = IntegerType('Coin', 0, 1)
        d1 = coin().set([1 / 2, 1 / 2])

        sumpos = d1.add(d1)

        res = []
        for _, l in sumpos.items():
            res.append(scipy.special.binom(2, l) * d1.p(1)**l * (1-d1.p(1))**(2-l))

        self.assertEqual([0, 1, 2], list(sumpos.labels.values()))
        self.assertEqual([0, 1, 2], list(sumpos.values.values()))
        self.assertEqual(res, list(sumpos.probabilities))

        d1.plot(
            view=False,
            color="rgb(0,104,180)"
        )

        sumpos.plot(
            view=False,
            color="rgb(0,104,180)"
        )


    def test_plot(self):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([1 / 6] * 6)
        d1.plot(
            title="Test",
            view=False,
            horizontal=False
        )

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
