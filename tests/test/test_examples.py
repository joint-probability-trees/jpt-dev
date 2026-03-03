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
        """Verify the regression example runs without errors."""
        import regression
        regression.main(visualize=False)

    def test_restaurant(self):
        """Verify the restaurant example runs without errors."""
        import restaurant
        restaurant.main(visualize=False)

    def test_gaussians(self):
        """Verify the gaussians example runs without errors."""
        import gaussians
        gaussians.main(visualize=False)

    def test_mnist(self):
        """Verify the MNIST example runs without errors."""
        import mnist
        mnist.main(visualize=False)

    def test_muesli(self):
        """Verify the muesli example runs without errors."""
        import muesli
        muesli.main(visualize=False)

    def test_alarm(self):
        """Verify the alarm example runs without errors."""
        import alarm
        alarm.main(visualize=False)

    def test_tourism(self):
        """Verify the tourism example runs without errors."""
        import tourism
        tourism.main(visualize=False)

    def test_abalone(self):
        """Verify the abalone example runs without errors."""
        import abalone
        abalone.main(visualize=False)
