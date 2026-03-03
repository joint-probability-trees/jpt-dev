import json
import numbers
import os
import pickle
from typing import Type
from unittest import TestCase

import numpy as np
import scipy.stats
from ddt import data, unpack, ddt
from dnutils.tools import ifstr

from jpt.base.constants import eps
from jpt.distributions.univariate.numeric import (
    NumericValueToLabelMap,
    NumericLabelToValueMap
)
from test.testutils import uniform_numeric, RESOURCES

from jpt.distributions.qpd import QuantileDistribution

from jpt.base.functions import (
    PiecewiseFunction,
    LinearFunction,
    ConstantFunction
)
from jpt.base.intervals import (
    ContinuousSet,
    EXC,
    INC,
    UnionSet,
    R
)

from jpt.base.errors import Unsatisfiability
from jpt.distributions.univariate import IntegerType
from jpt.distributions import (
    SymbolicType,
    NumericType,
    Gaussian,
    Numeric,
    Distribution,
    ScaledNumeric
)


# ----------------------------------------------------------------------

@ddt
class NumericDistributionTest(TestCase):
    '''Test class for ``Numeric`` distributions'''

    GAUSSIAN = None

    @classmethod
    def setUp(cls) -> None:
        with open(os.path.join(RESOURCES, 'gaussian_100.dat'), 'rb') as f:
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

    @data("matplotlib", "plotly", None)
    def test_plot_gaussian1d(self, engine):
        DistGauss = self.DistGauss
        gauss = Gaussian(data=[[DistGauss.values[d]] for d in NumericDistributionTest.GAUSSIAN])
        gauss.plot(engine, view=False)

    @data("matplotlib", "plotly", None)
    def test_plot_gaussian2d(self, engine):
        g = Gaussian([0, 0], [ [1, 3/5], [3/5, 2]])
        g.plot(engine, view=False, dim=2)

    @data("matplotlib", "plotly", None)
    def test_plot_gaussian3d(self, engine):
        g = Gaussian([0, 0], [ [1, 3/5], [3/5, 2]])
        g.plot(engine, view=False, dim=3)

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
            UnionSet(['[0,1]', '[2,3]'])
        )
        value_scalar = Gauss.label2value(.5)
        value_interval = Gauss.label2value(
            ContinuousSet(0, 1)
        )

        # Assert
        self.assertEqual(
            UnionSet([
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
        (UnionSet([
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

    @data("matplotlib", "plotly")
    def test_plot(self, engine):
        d = Numeric()._fit(np.linspace(0, 1, 20).reshape(-1, 1), col=0)
        d.plot(
            engine=engine,
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

    @data("matplotlib", "plotly")
    def test_add(self, engine):
        # Arrange
        x = uniform_numeric(-1, 1)
        y = uniform_numeric(-1, 1)
        # Act
        z = (x + y)

        x.plot(engine=engine, view=False)
        y.plot(engine=engine, view=False)
        z.plot(engine=engine, view=False)
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

    def test_sub(self):
        # Arrange
        x = uniform_numeric(-2, 2)
        y = uniform_numeric(-1, 1)

        # Act
        z = (x - y)

        # x.plot(view=False,title='x')
        # y.plot(view=False,title='y')
        # z.plot(view=False,title='z')

        # Assert
        self.assertAlmostEqual(
            x.expectation() - y.expectation(),
            z.expectation(),
            places=10
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

    def test_entropy(self):
        # Arrange
        numeric = uniform_numeric(0, 1)

        # Act
        entropy = numeric.entropy()

        # Assert
        self.assertEqual(
            0,
            entropy
        )


# ----------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------

class DataScalerTest(TestCase):
    DATA = None

    @classmethod
    def setUpClass(cls) -> None:
        DataScalerTest.DATA = Gaussian(5, 10).sample(10000)

    def test_singular_values_transformation(self):
        # Arrange
        value2label = NumericValueToLabelMap()
        value2label.fit(DataScalerTest.DATA)
        label2value = NumericLabelToValueMap()
        label2value.fit(DataScalerTest.DATA)

        for x in DataScalerTest.DATA:
            # Act
            single_value_result = value2label[label2value[x]]

            # Assert
            self.assertAlmostEqual(
                x,
                single_value_result,
                5
            )

    def test_bulk_values_transformation(self):
        # Arrange
        value2label = NumericValueToLabelMap()
        value2label.fit(DataScalerTest.DATA)
        label2value = NumericLabelToValueMap()
        label2value.fit(DataScalerTest.DATA)

        # Act
        values = value2label.transform(
            label2value.transform(DataScalerTest.DATA)
        )

        # Assert
        self.assertEqual(
            list(DataScalerTest.DATA.round(5)),
            list(values.round(5)),
        )


# ----------------------------------------------------------------------------------------------------------------------

class TypeGeneratorTest(TestCase):

    def test_numeric_type(self):
        self.assertRaises(ValueError, NumericType, 'BlaType', [None, 1, 2])
        self.assertRaises(ValueError, NumericType, 'BlaType', [1, 2, float('inf')])
        self.assertTrue(issubclass(NumericType('bla', [1, 2, 3, 4]), ScaledNumeric))

    def test_integer_type(self):
        # Act
        t_1 = IntegerType('Months', 1, 12)
        t_2 = IntegerType('Unbounded')

        # Assert
        self.assertEqual(
            (1, 12),
            (t_1.min, t_1.max)
        )
        self.assertEqual(
            (-np.inf, np.inf),
            (t_2.min, t_2.max)
        )
        self.assertRaises(ValueError, IntegerType, 'Bla', 3, 2)

    def test_symbolic_type(self):
        t = SymbolicType('Object', labels=['Bowl', 'Spoon', 'Cereal'])
        self.assertEqual('Object', t.__qualname__)
        self.assertEqual(['Bowl', 'Spoon', 'Cereal'], list(t.labels))
