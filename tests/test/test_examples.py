import sys
from unittest import TestCase

from matplotlib import pyplot as plt
from test.testutils import EXAMPLES_DIR

try:
    import regression
    import restaurant
    import gaussians
except ModuleNotFoundError:
    sys.path.append(EXAMPLES_DIR)

plt.switch_backend('agg')


# noinspection PyMethodMayBeStatic
class ExamplesTestRun(TestCase):

    def test_regression(self):
        import regression
        regression.main(visualize=False)

    def test_restaurant(self):
        import restaurant
        restaurant.main(visualize=False)

    def test_gaussians(self):
        import gaussians
        gaussians.main(visualize=False)

    def test_mnist(self):
        import mnist
        mnist.main(visualize=False)

    def test_muesli(self):
        import muesli
        muesli.main(visualize=False)

    def test_alarm(self):
        import alarm
        alarm.main(visualize=False)

    def test_tourism(self):
        import tourism
        tourism.main(visualize=False)

    def test_abalone(self):
        import abalone
        abalone.main(visualize=False)
