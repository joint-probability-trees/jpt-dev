import json
import numbers
import os
import pickle
from unittest import TestCase

import numpy as np
import scipy.stats
from ddt import data, unpack, ddt

from jpt.base.constants import eps
from jpt.distributions.univariate import IntegerType, Integer
from jpt.distributions.univariate.integer import (
    IntegerMap,
    IntegerValueToLabelMap,
    IntegerLabelToValueMap
)

from jpt.base.intervals import (
    ContinuousSet,
    EXC,
    INC,
    UnionSet,
    IntSet,
    Z
)

from jpt.base.errors import Unsatisfiability
from jpt.distributions import Distribution


# ----------------------------------------------------------------------

class IntegerValueMapTest(TestCase):

    def test_getitem(self):
        # Arrange
        z = IntegerValueToLabelMap()
        halfopen_pos = IntegerValueToLabelMap(lmin=-2)
        halfopen_neg = IntegerValueToLabelMap(lmax=3)
        closed = IntegerValueToLabelMap(lmin=-2, lmax=3)

        # Act
        result_z = z[0], z[-5], z[100]
        result_halfopen_pos = halfopen_pos[0], halfopen_pos[2], halfopen_pos[12]
        result_halfopen_neg = halfopen_neg[-13], halfopen_neg[-3], halfopen_neg[0]
        result_closed = closed[0], closed[2], closed[5]

        # Assert
        self.assertEqual(
            (0, -5, 100),
            result_z
        )
        self.assertEqual(
            (-2, 0, 10),
            result_halfopen_pos
        )
        self.assertRaises(
            ValueError,
            halfopen_pos.__getitem__,
            -1
        )
        self.assertEqual(
            (-10, 0, 3),
            result_halfopen_neg
        )
        self.assertRaises(
            ValueError,
            halfopen_neg.__getitem__,
            1
        )
        self.assertEqual(
            (-2, 0, 3),
            result_closed
        )
        self.assertRaises(
            ValueError,
            closed.__getitem__,
            -1
        )
        self.assertRaises(
            ValueError,
            closed.__getitem__,
            6
        )

    def test_iter_all_ints(self):
        # Arrange
        int_map = IntegerValueToLabelMap()
        int_iter = iter(int_map)

        # Act
        result = [next(int_iter) for _ in range(10)]

        # Assert
        self.assertEqual(
            [0, 1, -1, 2, -2, 3, -3, 4, -4, 5],
            result
        )

    def test_iter_neg_inf(self):
        # Arrange
        int_map = IntegerValueToLabelMap(lmax=3)
        int_iter = iter(int_map)

        # Act
        result = [next(int_iter) for _ in range(10)]

        # Assert
        self.assertEqual(
            [3, 2, 1, 0, -1, -2, -3, -4, -5, -6],
            result
        )

    def test_iter_pos_inf(self):
        # Arrange
        int_map = IntegerValueToLabelMap(lmin=-1)
        int_iter = iter(int_map)

        # Act
        result = [next(int_iter) for _ in range(10)]

        # Assert
        self.assertEqual(
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            result
        )

    def test_iter_finite(self):
        # Arrange
        int_map = IntegerValueToLabelMap(lmin=-1, lmax=3)
        int_iter = iter(int_map)

        # Act
        result = list(int_iter)

        # Assert
        self.assertEqual(
            [-1, 0, 1, 2, 3],
            result
        )

    def test_len_finite(self):
        # Arrange
        int_map = IntegerMap(-1, 1)

        # Act
        result = len(int_map)

        # Assert
        self.assertEqual(
            3,
            result
        )


class IntegerLabelMapTest(TestCase):

    def test_getitem(self):
        # Arrange
        z = IntegerLabelToValueMap()
        halfopen_pos = IntegerLabelToValueMap(lmin=-2)
        halfopen_neg = IntegerLabelToValueMap(lmax=3)
        closed = IntegerLabelToValueMap(lmin=-2, lmax=3)

        # Act
        result_z = z[0], z[-5], z[100]
        result_halfopen_pos = halfopen_pos[-2], halfopen_pos[0], halfopen_pos[10]
        result_halfopen_neg = halfopen_neg[-10], halfopen_neg[0], halfopen_neg[3]
        result_closed = closed[-2], closed[0], closed[3]

        # Assert
        self.assertEqual(
            (0, -5, 100),
            result_z
        )
        self.assertEqual(
            (0, 2, 12),
            result_halfopen_pos
        )
        self.assertRaises(
            ValueError,
            halfopen_pos.__getitem__,
            -3
        )
        self.assertEqual(
            (-13, -3, 0),
            result_halfopen_neg
        )
        self.assertRaises(
            ValueError,
            halfopen_neg.__getitem__,
            4
        )
        self.assertEqual(
            (0, 2, 5),
            result_closed
        )
        self.assertRaises(
            ValueError,
            closed.__getitem__,
            -3
        )
        self.assertRaises(
            ValueError,
            closed.__getitem__,
            4
        )

    def test_intset(self):
        # Arrange
        z = IntegerLabelToValueMap()
        halfopen_pos = IntegerLabelToValueMap(lmin=-2)
        halfopen_neg = IntegerLabelToValueMap(lmax=3)
        closed = IntegerLabelToValueMap(lmin=-2, lmax=3)

        # Act
        intset_z = z.as_set()
        intset_halfopen_pos = halfopen_pos.as_set()
        intset_halfopen_neg = halfopen_neg.as_set()
        intset_closed = closed.as_set()

        # Assert
        self.assertEqual(
            Z,
            intset_z
        )
        self.assertEqual(
            IntSet(0, np.inf),
            intset_halfopen_pos
        )
        self.assertEqual(
            IntSet(-np.inf, 0),
            intset_halfopen_neg
        )
        self.assertEqual(
            IntSet(0, 5),
            intset_closed
        )


# ----------------------------------------------------------------------------------------------------------------------

@ddt
class IntegerDistributionTest(TestCase):

    Die = IntegerType('Dice', 1, 6)

    def test_value2label(self):
        # Arrange
        Die = self.Die
        intset = IntSet(0, 2)
        realset = UnionSet([IntSet(0, 1), IntSet(4, 5)])

        # Act
        value_scalar = Die.value2label(0)
        value_set = Die.value2label({0})
        value_list = Die.value2label([0, 5])
        value_tuple = Die.value2label((1,))
        value_intset = Die.value2label(intset)
        value_realset = Die.value2label(realset)

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
        self.assertEqual(
            IntSet(1, 3),
            value_intset
        )
        self.assertEqual(
            UnionSet([IntSet(1, 2), IntSet(5, 6)]),
            value_realset
        )

    def test_label2value(self):
        # Arrange
        Die = self.Die
        intset = IntSet(1, 3)
        realset = UnionSet([IntSet(1, 2), IntSet(5, 6)])

        # Act
        label_scalar = Die.label2value(1)
        label_set = Die.label2value({1})
        label_list = Die.label2value([1, 6])
        label_tuple = Die.label2value((2,))
        label_intset = Die.label2value(intset)
        label_realset = Die.label2value(realset)

        # Assert
        self.assertEqual(0, label_scalar)
        self.assertEqual({0}, label_set)
        self.assertEqual([0, 5], label_list)
        self.assertEqual((1,), label_tuple)
        self.assertRaises(ValueError, Die.label2value, 0)
        self.assertEqual(
            IntSet(0, 2),
            label_intset
        )
        self.assertEqual(
            UnionSet([IntSet(0, 1), IntSet(4, 5)]),
            label_realset
        )

    def test_set_finite(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice_set_array = dice()
        fair_dice_set_dict = dice()

        # Act
        fair_dice_set_array.set([1 / 6] * 6)
        fair_dice_set_dict.set({1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6})

        # Assert
        self.assertRaises(
            ValueError,
            fair_dice_set_array.set,
            [1 / 6] * 7
        )
        self.assertEqual(
            {i: 1 / 6 for i in range(0, 6)},
            fair_dice_set_array.probabilities
        )
        self.assertEqual(
            {i: 1 / 6 for i in range(0, 6)},
            fair_dice_set_dict.probabilities
        )

    def test_set_infinite(self):
        # Arrange
        unbounded = IntegerType('Dice')
        dist = unbounded()

        # Act
        dist.set({0: 1 / 3, 1: 2 / 3})

        # Assert
        self.assertRaises(
            ValueError,
            dist.set,
            [1 / 6] * 6
        )
        self.assertRaises(
            ValueError,
            dist.set,
            {0: 2 / 3, 1: 2 / 3}
        )
        self.assertEqual(
            {0: 1 / 3, 1: 2 / 3},
            dist.probabilities
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
            {i: 1 / 6 for i in range(0, 6)},
            fair_dice.probabilities,
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
        p_intset_labels = fair_dice.p(IntSet(1, 3))
        p_intset_values = fair_dice._p(IntSet(0, 2))
        p_realset_labels = fair_dice.p(UnionSet(['{1..2}', '{5..6}']))
        p_realset_values = fair_dice._p(UnionSet(['{0..1}', '{4..5}']))

        # Assert
        self.assertEqual(1 / 6, p_singular_label)
        self.assertEqual(1 / 6, p_singular_value)
        self.assertEqual(3 / 6, p_set_label)
        self.assertEqual(3 / 6, p_set_values)
        self.assertEqual(2 / 6, p_duplicate_labels)
        self.assertEqual(3 / 6, p_intset_labels)
        self.assertEqual(3 / 6, p_intset_values)
        self.assertEqual(4 / 6, p_realset_labels)
        self.assertEqual(4 / 6, p_realset_values)

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
        self.assertEqual(
            {1: 0.5, 2: 0.5},
            biased_dice.probabilities
        )
        self.assertEqual(
            {0: 0.5, 1: 0.5},
            _biased_dice.probabilities
        )
        self.assertRaises(
            Unsatisfiability,
            biased_dice.crop,
            {1}
        )

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
        self.assertEqual(IntSet(1, 6), fair_mpe)
        self.assertEqual(1 / 6, p_fair)
        self.assertEqual(IntSet(0, 5), _fair_mpe)
        self.assertEqual(1 / 6, _p_fair)

        self.assertEqual(IntSet.from_set({3}), biased_mpe)
        self.assertEqual(2 / 6, p_biased)
        self.assertEqual(IntSet.from_set({2}), _biased_mpe)
        self.assertEqual(2 / 6, _p_biased)

    def test_k_mpe(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)
        biased_dice = dice()
        biased_dice.set([0 / 6, 1 / 6, 2 / 6, 1 / 6, 1 / 6, 1 / 6])

        # Act
        fair_k_mpe = list(fair_dice.k_mpe(3))
        biased_k_mpe = list(biased_dice.k_mpe(3))

        # Assert
        self.assertEqual(
            [(IntSet(1, 6), 1 / 6)],
            fair_k_mpe
        )

        self.assertEqual(
        [(IntSet.from_set({3}), 0.3333333333333333), (IntSet.from_set({2, 4, 5, 6}), 0.16666666666666666)],
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
        merged = dice.merge(
            distributions=[fair_dice, biased_dice],
            weights=[.5, .5]
        )

        # Assert
        self.assertEqual(
            {0: .5 / 6, 1: 1 / 6, 2: 1.5 / 6, 3: 1 / 6, 4: 1 / 6, 5: 1 / 6},
            merged.probabilities
        )

    def test_serialization(self):
        # Arrange
        dice = IntegerType('Dice', 1, 6)
        fair_dice = dice()
        fair_dice.set([1 / 6] * 6)

        # Act
        dice_type = Distribution.from_json(
            json.loads(
                json.dumps(
                    dice.to_json()
                )
            )
        )
        fair_dice_inst = dice.from_json(
            json.loads(
                json.dumps(
                    fair_dice.to_json()
                )
            )
        )

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
        # Arrange
        pos = IntegerType('Pos', 0, 6)
        posx = pos()
        posx.set([0, 0, 1, 0, 0, 0, 0])

        delta = IntegerType('Delta', -1, 1)
        deltax = delta()
        deltax.set([0, 0, 1])

        # Act
        sumpos = posx.add(deltax)

        # Assert
        self.assertEqual(
            (-1, 7),
            (sumpos.min, sumpos.max)
        )
        self.assertEqual(
            {4: 1},
            sumpos.probabilities
        )
        self.assertEqual(
            1,
            sumpos.p(3)
        )

    @data("matplotlib", "plotly")
    def test_add_bernoulli(self, engine):
        coin = IntegerType('Coin', 0, 1)
        d1 = coin().set([1 / 2, 1 / 2])

        sumpos = d1.add(d1)

        res = []
        for _, l in sumpos.items():
            res.append(scipy.special.binom(2, l) * d1.p(1)**l * (1-d1.p(1))**(2-l))

        self.assertEqual(
            (0, 2),
            (sumpos.min, sumpos.max)
        )
        self.assertEqual(
            {0: 0.25, 1: 0.5, 2: 0.25},
            sumpos.probabilities
        )

        d1.plot(
            engine=engine,
            view=False,
            color="rgb(0,104,180)"
        )

        sumpos.plot(
            engine=engine,
            view=False,
            color="rgb(0,104,180)"
        )

    def test_items_finite(self):
        # Arrange
        dist = IntegerType('TEST_FINITE',0, 2)()
        dist.set([.5, .5, 0])

        # Act
        items_exhaustive = dist.items(exhaustive=True)
        items_nonexhaustive = dist.items(exhaustive=False)

        # Assert
        self.assertEqual(
            [(0, 0.5), (1, 0.5), (2, 0)],
            list(items_exhaustive)
        )
        self.assertEqual(
            [(0, 0.5), (1, 0.5)],
            list(items_nonexhaustive)
        )

    def test_items_infinite(self):
        # Arrange
        dist = IntegerType('TEST_INFINITE')()
        dist.set({0: .5, 1: .5, 2: 0})

        # Act
        items_nonexhaustive = dist.items(exhaustive=False)

        # Assert
        self.assertEqual(
            [(0, 0.5), (1, 0.5)],
            list(items_nonexhaustive)
        )
        # self.assertRaises(
        #     ValueError,
        #     lambda: list(
        #         dist.items(exhaustive=True)
        #     )
        # )

    @data("matplotlib", "plotly")
    def test_plot(self, engine):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([1 / 6] * 6)
        d1.plot(
            engine=engine,
            title="Test",
            view=False,
            horizontal=False
        )

    @data("matplotlib", "plotly")
    def test_plot2(self, engine):
        dice = IntegerType('Dice', 1, 6)
        d1 = dice().set([1/6, 2/6, 3/6, 0, 0, 0])
        d1.plot(
            engine=engine,
            title="Test",
            view=False,
            horizontal=True
        )

# ----------------------------------------------------------------------------------------------------------------------
