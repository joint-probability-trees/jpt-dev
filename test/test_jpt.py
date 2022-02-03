import pickle
from unittest import TestCase

from jpt.trees import JPT
from jpt.variables import NumericVariable


class JPTTest(TestCase):

    def setUp(self) -> None:
        with open('resources/gaussian_100.dat', 'rb') as f:
            self.data = pickle.load(f)

    def test_serialization(self):
        '''(de)serialization of JPTs'''

        var = NumericVariable('X')
        jpt = JPT([var], min_samples_leaf=.1)
        jpt.learn(self.data.reshape(-1, 1))
        jpt.plot(view=True)

        self.assertEqual(jpt, JPT.from_json(jpt.to_json()))

