import os
import sys
import unittest
from unittest import TestCase

from matplotlib import pyplot as plt

try:
    import regression
    import restaurant
    import gaussians
except ModuleNotFoundError:
    sys.path.append(os.path.join('..', 'examples'))

plt.switch_backend('agg')

# noinspection PyMethodMayBeStatic
class ExamplesTestRun(TestCase):

    def test_regression(self):
        import regression
        regression.main(visualize=False)

    def test_restaurant_auto_sample(self):
        import restaurant
        restaurant.restaurant_auto_sample(visualize=False)

    def test_restaurant_manual_sample(self):
        import restaurant
        restaurant.restaurant_manual_sample(visualize=False)

    def test_gaussians(self):
        import gaussians
        gaussians.main(verbose=False)

    def test_mnist(self):
        import mnist
        mnist.main(visualize=False)

    def test_muesli(self):
        import muesli
        muesli.main(visualize=False)
