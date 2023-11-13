import datetime
import unittest
from typing import List

import numpy as np
import pandas
import pandas as pd
import random_events.variables
from random_events.variables import Symbolic, Integer

import jpt.probabilistic_circuit


class VariableTestCase(unittest.TestCase):
    variable: jpt.probabilistic_circuit.ScaledContinuous = jpt.probabilistic_circuit.ScaledContinuous('x', 2, 3)

    def test_encode(self):
        self.assertEqual(self.variable.encode(2), 0)
        self.assertEqual(self.variable.encode(5), 1)
        self.assertEqual(self.variable.encode(0), -2 / 3)

    def test_decode(self):
        self.assertEqual(self.variable.decode(0), 2)
        self.assertEqual(self.variable.decode(1), 5)
        self.assertEqual(self.variable.decode(-2 / 3), 0)


class JPTCreationTestCase(unittest.TestCase):
    variables: List[jpt.probabilistic_circuit.Variable] = tuple(sorted(
        [Symbolic("symbol", ("a", "b", "c")), Integer("integer", (1, 2, 4)),
         jpt.probabilistic_circuit.ScaledContinuous("real", 0, 1)]))

    def test_features_only(self):
        model = jpt.probabilistic_circuit.JPT(self.variables, features=self.variables[:2])
        self.assertEqual(model.targets, (self.variables[2],))
        self.assertEqual(model.features, self.variables[:2])

    def test_targets_only(self):
        model = jpt.probabilistic_circuit.JPT(self.variables, targets=self.variables[:2])
        self.assertEqual(model.targets, self.variables[:2])
        self.assertEqual(model.features, (self.variables[2],))

    def test_neither_features_nor_targets(self):
        model = jpt.probabilistic_circuit.JPT(self.variables)
        self.assertEqual(model.features, self.variables)
        self.assertEqual(model.targets, self.variables)

    def test_both_features_and_targets(self):
        model = jpt.probabilistic_circuit.JPT(self.variables, targets=self.variables[:2], features=self.variables[1:])
        self.assertEqual(model.targets, self.variables[:2])
        self.assertEqual(model.features, self.variables[1:])


class InferFromDataFrameTestCase(unittest.TestCase):

    data: pd.DataFrame

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        data = pd.DataFrame()
        data["real"] = np.random.normal(2, 4, 100)
        data["integer"] = np.concatenate((np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50)))
        data["symbol"] = np.random.randint(0, 4, 100).astype(str)
        cls.data = data

    def test_types(self):
        self.assertEqual(self.data.dtypes[0], float)
        self.assertEqual(self.data.dtypes[1], int)
        self.assertEqual(self.data.dtypes[2], object)

    def test_infer_from_dataframe_with_scaling(self):
        real, integer, symbol = jpt.probabilistic_circuit.infer_variables_from_dataframe(self.data)
        self.assertEqual(real.name, "real")
        self.assertIsInstance(real, jpt.probabilistic_circuit.ScaledContinuous)
        self.assertEqual(integer.name, "integer")
        self.assertEqual(symbol.name, "symbol")
        self.assertLess(real.minimal_distance, 1.)

    def test_infer_from_dataframe_without_scaling(self):
        real, integer, symbol = jpt.probabilistic_circuit.infer_variables_from_dataframe(self.data,
                                                                                         scale_continuous_types=False)
        self.assertNotIsInstance(real, jpt.probabilistic_circuit.ScaledContinuous)

    def test_unknown_type(self):
        df = pandas.DataFrame()
        df["time"] = [datetime.datetime.now()]
        with self.assertRaises(ValueError):
            jpt.probabilistic_circuit.infer_variables_from_dataframe(df)


class JPTTestCase(unittest.TestCase):

    data: pd.DataFrame
    real: jpt.probabilistic_circuit.ScaledContinuous
    integer: random_events.variables.Integer
    symbol: random_events.variables.Symbolic
    model: jpt.probabilistic_circuit.JPT

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)
        data = pd.DataFrame()
        data["real"] = np.random.normal(2, 4, 100)
        data["integer"] = np.concatenate((np.random.randint(low=0, high=4, size=50), np.random.randint(7, 10, 50)))
        data["symbol"] = np.random.randint(0, 4, 100).astype(str)
        cls.data = data
        cls.real, cls.integer, cls.symbol = jpt.probabilistic_circuit.infer_variables_from_dataframe(cls.data)
        cls.model = jpt.probabilistic_circuit.JPT([cls.real, cls.integer, cls.symbol])

    def test_preprocess_data(self):
        preprocessed_data = self.model.preprocess_data(self.data)
        mean = preprocessed_data[:, 1].mean()
        std = preprocessed_data[:, 1].std(ddof=1)
        self.assertEqual(self.real.mean, mean)
        self.assertEqual(self.real.std, std)

        # assert that this does not throw exceptions
        for variable, column in zip(self.model.variables, preprocessed_data.T):
            if isinstance(variable, random_events.variables.Discrete):
                variable.decode_many(column.astype(int))
            else:
                variable.decode_many(column)



if __name__ == '__main__':
    unittest.main()
