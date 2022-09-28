import os
import sys
from unittest import TestCase

try:
    import regression
    import restaurant
    import gaussians
except ModuleNotFoundError:
    sys.path.append(os.path.join('..', 'examples'))


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
