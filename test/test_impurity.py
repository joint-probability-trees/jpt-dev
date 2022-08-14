import os
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt import trees
from jpt.distributions import SymbolicType, Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable

try:
    from jpt.learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.learning.impurity import Impurity


class ImpurityTest(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.read_csv(os.path.join('..', 'examples', 'data', 'restaurant.csv'))

        # declare variable types
        PatronsType = SymbolicType('Patrons', ['Some', 'Full', 'None'])
        PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
        FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
        WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

        # create variables
        cls.al = SymbolicVariable('Alternatives', Bool)
        cls.ba = SymbolicVariable('Bar', Bool)
        cls.fr = SymbolicVariable('Friday', Bool)
        cls.hu = SymbolicVariable('Hungry', Bool)
        cls.pa = SymbolicVariable('Patrons', PatronsType)
        cls.pr = SymbolicVariable('Price', PriceType)
        cls.ra = SymbolicVariable('Rain', Bool)
        cls.re = SymbolicVariable('Reservation', Bool)
        cls.fo = SymbolicVariable('Food', FoodType)
        cls.we = SymbolicVariable('WaitEstimate', WaitEstType)
        cls.wa = SymbolicVariable('WillWait', Bool)

        cls.variables = [cls.al, cls.ba, cls.fr, cls.hu, cls.pa, cls.pr, cls.ra, cls.re, cls.fo, cls.we, cls.wa]

    def test_symbolic(self):
        jpt = JPT(variables=ImpurityTest.variables, targets=[ImpurityTest.wa])

        data = jpt._preprocess_data(ImpurityTest.data)

        impurity = Impurity(jpt)
        impurity.min_samples_leaf = max(1, jpt.min_samples_leaf)
        impurity.setup(data, np.array(list(range(data.shape[0]))))
        impurity.compute_best_split(0, data.shape[0])

        self.assertNotEqual(impurity.best_var, -1)
        self.assertIs(ImpurityTest.variables[impurity.best_var], ImpurityTest.pa)
        self.assertEqual([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 2, 2],
                         list(np.asarray(impurity.feat, dtype=np.int32)))
        self.assertEqual({0, 2, 5, 7}, set(impurity.indices[:4]))
        self.assertEqual({1, 3, 4, 8, 9, 11}, set(impurity.indices[4:10]))
        self.assertEqual({6, 10}, set(impurity.indices[10:]))

