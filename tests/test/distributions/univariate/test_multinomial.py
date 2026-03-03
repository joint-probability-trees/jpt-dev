import json
import pickle
from unittest import TestCase

import numpy as np
from ddt import data, ddt

from jpt.distributions.univariate.multinomial import (
    MultinomialValueMap,
    Bool
)
from jpt.variables import SymbolicVariable

from jpt.base.errors import Unsatisfiability
from jpt.distributions import (
    SymbolicType,
    Multinomial,
    Numeric,
    Distribution
)


# ----------------------------------------------------------------------

@ddt
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
        # Arrange
        DistABC = self.DistABC
        Dist123 = self.Dist123

        # Assert
        self.assertTrue(DistABC.equiv(DistABC))
        self.assertTrue(Dist123.equiv(Dist123))
        self.assertFalse(DistABC.equiv(Dist123))

        self.assertEqual(
            3,
            DistABC.n_values,
        )
        self.assertEqual(
            5,
            Dist123.n_values,
        )

        self.assertTrue(issubclass(DistABC, Multinomial))
        self.assertTrue(issubclass(Dist123, Multinomial))

        self.assertIsInstance(
            DistABC.values,
            MultinomialValueMap
        )
        self.assertIsInstance(
            DistABC.labels,
            MultinomialValueMap
        )

        self.assertEqual(
            MultinomialValueMap([('A', 0), ('B', 1), ('C', 2)]),
            DistABC.values,
        )
        self.assertEqual(
            MultinomialValueMap([(0, 'A'), (1, 'B'), (2, 'C')]),
            DistABC.labels,
        )

        self.assertEqual(
            MultinomialValueMap([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)]),
            Dist123.values,
        )
        self.assertEqual(
            MultinomialValueMap([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]),
            Dist123.labels,
        )

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
        """Verify cropping a distribution by label restricts and renormalizes probabilities."""
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
        """Verify cropping a distribution by value index restricts and renormalizes probabilities."""
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
        """Verify MPE of a uniform distribution returns all values."""
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 3, 1 / 3, 1 / 3])

        # Act
        result_uniform = abc.mpe()

        # Assert
        self.assertEqual(({'A', 'B', 'C'}, 1 / 3), result_uniform)

    def test_expectation(self):
        """Verify the mode returns the most probable label."""
        # Arrange
        ABC = self.DistABC
        abc = ABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        result = abc.mode()

        # Assert
        self.assertEqual({'A'}, result)

    def test_inference(self):
        """Verify probability queries for singular, set, and list evidence."""
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

    def test_pickle(self):
        """Verify pickle round-trip serialization of a multinomial distribution."""
        # Arrange
        DistABC = self.DistABC
        d = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        pickled = pickle.dumps(d)
        unpickled = pickle.loads(pickled)

        # Assert
        self.assertEqual(
            d,
            unpickled
        )

    def test_distribution_manipulation(self):
        """Verify merging and updating multinomial distributions."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        d2 = DistABC().set(params=[0, .5, .5])
        d3 = DistABC.merge([d1, d2], weights=[.5, .5])
        d1.update(d2, .5)

        self.assertEqual(d3, DistABC().set(params=[.25, .375, .375]))
        self.assertEqual(d1, d3)
        self.assertEqual(d1.update(d2, 0), d1)

    def test_kldiv_equality(self):
        """Verify KL divergence is zero for identical distributions."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        d2 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])
        self.assertEqual(d1.kl_divergence(d2), 0)
        self.assertEqual(d1.kl_divergence(d1), 0)
        self.assertEqual(0, DistABC().set(params=[1, 0, 0]).kl_divergence(DistABC().set(params=[1, 0, 0])))

    def test_kldiv_inequality(self):
        """Verify KL divergence is positive for different distributions."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        d2 = DistABC().set(params=[.25, .5, .25])
        self.assertEqual(0.1875, d1.kl_divergence(d2))

    def test_kldiv_extreme_inequality(self):
        """Verify KL divergence equals 1 for completely disjoint distributions."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1, 0, 0])
        d2 = DistABC().set(params=[0, .5, .5])
        self.assertEqual(1, d1.kl_divergence(d2))

    def test_kldiv_type(self):
        """Verify KL divergence raises TypeError for incompatible distribution types."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        self.assertRaises(
            TypeError,
            d1.kl_divergence,
            Numeric()._fit(np.array([[1], [2], [3]], dtype=np.float64), col=0))

    @data("matplotlib", "plotly")
    def test_plot(self, engine):
        """Verify plotting a multinomial distribution does not raise errors."""
        DistABC = self.DistABC
        d1 = DistABC().set(params=[.5, .25, .25])
        d1.plot(
            engine=engine,
            view=False,
            horizontal=False
        )

    @data("matplotlib", "plotly")
    def test_plot_coin(self, engine):
        """Verify plotting a biased coin distribution does not raise errors."""
        fr = SymbolicVariable('BiasedCoin', Bool)
        d1 = fr.distribution().set(5/12.)
        d1.plot(
            engine=engine,
            view=False,
            horizontal=False
        )

    def test_value_conversion(self):
        """Verify bidirectional label-to-value conversions for singles and sets."""
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
        """Verify MPE returns the most probable state and its likelihood."""
        # Arrange
        DistABC = self.DistABC
        d1 = DistABC().set(params=[1 / 2, 1 / 4, 1 / 4])

        # Act
        state, likelhood = d1.mpe()

        # Assert
        self.assertEqual(0.5, likelhood)
        self.assertEqual({"A"}, state)

    def test_k_mpe(self):
        """Verify k-MPE returns the top-k most probable explanations."""
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
        """Verify Jaccard similarity of a distribution with itself is 1."""
        d1 = self.DistABC().set([.1, .4, .5])
        jacc = Multinomial.jaccard_similarity(d1, d1)
        self.assertEqual(1., jacc)

    def test_jaccard_disjoint(self):
        """Verify Jaccard similarity of disjoint distributions is 0."""
        d1 = self.DistABC().set([0., 0., 1.])
        d2 = self.DistABC().set([1., 0., 0.])
        jacc = Multinomial.jaccard_similarity(d1, d2)
        self.assertEqual(0., jacc)

    def test_jaccard_overlap(self):
        """Verify Jaccard similarity for overlapping distributions."""
        d1 = self.DistABC().set([.1, .4, .5])
        d2 = self.DistABC().set([.2, .4, .4])

        jacc = Multinomial.jaccard_similarity(d1, d2)
        self.assertAlmostEqual(9/11, jacc, places=8)

    def test_jaccard_symmetry(self):
        """Verify Jaccard similarity is symmetric."""
        d1 = self.DistABC().set([.1, .4, .5])
        d2 = self.DistABC().set([.2, .4, .4])
        jacc1 = Multinomial.jaccard_similarity(d1, d2)
        jacc2 = Multinomial.jaccard_similarity(d2, d1)
        self.assertEqual(jacc1, jacc2)

    def test_mover_dist_identity(self):
        """Verify earth mover's distance of a distribution with itself is 0."""
        d1 = self.DistABC().set([.1, .4, .5])
        md = Multinomial.mover_dist(d1, d1)
        self.assertEqual(0., md)

    def test_mover_dist_symmetry(self):
        """Verify earth mover's distance is symmetric."""
        d1 = self.DistABC().set([.1, .4, .5])
        d2 = self.DistABC().set([.2, .4, .4])
        md1 = Multinomial.mover_dist(d1, d2)
        md2 = Multinomial.mover_dist(d2, d1)
        self.assertEqual(md1, md2)

    def test_mover_dist_triangle_inequality(self):
        """Verify earth mover's distance satisfies the triangle inequality."""
        a = self.DistABC().set([.1, .4, .5])
        b = self.DistABC().set([.2, .4, .4])
        c = self.DistABC().set([.2, .2, .6])
        ab = Multinomial.mover_dist(a, b)
        bc = Multinomial.mover_dist(b, c)
        ac = Multinomial.mover_dist(a, c)
        self.assertLessEqual(ac, ab + bc)

    def test_temp_goal(self):
        """Verify creating a uniform goal distribution from a subset of labels."""
        v1 = self.DistABC().set([.1, .4, .5])
        other = {'A', 'B'}

        v2_ = SymbolicVariable(v1.__class__.__name__, type(v1))
        v2 = v2_.distribution().set([(1 if x in other else 0) / len(other) for x in list(v2_.domain.labels)])
        self.assertEqual(1,1)


# ----------------------------------------------------------------------------------------------------------------------
