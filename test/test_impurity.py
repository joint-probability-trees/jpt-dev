import logging
import os
from unittest import TestCase

import numpy as np
import pandas as pd

from jpt import trees
from jpt.learning.distributions import SymbolicType, Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable

try:
    from jpt.learning.impurity import Impurity
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
    from jpt.learning.impurity import Impurity


class ImpurityTest(TestCase):

    @staticmethod
    def setUpClass() -> None:
        ImpurityTest.data = pd.read_csv(os.path.join('../', 'examples', 'data', 'restaurant.csv'))

        # declare variable types
        PatronsType = SymbolicType('Patrons', ['Some', 'Full', 'None'])
        PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
        FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
        WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

        # create variables
        ImpurityTest.al = SymbolicVariable('Alternatives', Bool)
        ImpurityTest.ba = SymbolicVariable('Bar', Bool)
        ImpurityTest.fr = SymbolicVariable('Friday', Bool)
        ImpurityTest.hu = SymbolicVariable('Hungry', Bool)
        ImpurityTest.pa = SymbolicVariable('Patrons', PatronsType)
        ImpurityTest.pr = SymbolicVariable('Price', PriceType)
        ImpurityTest.ra = SymbolicVariable('Rain', Bool)
        ImpurityTest.re = SymbolicVariable('Reservation', Bool)
        ImpurityTest.fo = SymbolicVariable('Food', FoodType)
        ImpurityTest.we = SymbolicVariable('WaitEstimate', WaitEstType)
        ImpurityTest.wa = SymbolicVariable('WillWait', Bool)

        ImpurityTest.variables = [ImpurityTest.al, ImpurityTest.ba, ImpurityTest.fr, ImpurityTest.hu,
                                  ImpurityTest.pa, ImpurityTest.pr, ImpurityTest.ra, ImpurityTest.re,
                                  ImpurityTest.fo, ImpurityTest.we, ImpurityTest.wa]

    def test_symbolic(self):
        JPT.logger.setLevel(logging.DEBUG)
        jpt = JPT(variables=ImpurityTest.variables, targets=[ImpurityTest.wa])
        data = jpt._preprocess_data(ImpurityTest.data)
        trees._data = data
        impurity = Impurity(jpt)
        impurity.min_samples_leaf = max(1, jpt.min_samples_leaf)
        impurity.setup(data, np.array(list(range(data.shape[0]))))
        impurity.compute_best_split(0, data.shape[0])
        self.assertNotEqual(impurity.best_var, -1)
        self.assertIs(ImpurityTest.variables[impurity.best_var], ImpurityTest.pa)
        jpt.learn(ImpurityTest.data)
