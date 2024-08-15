from jpt.base.pyximporter import pyx_import

pyx_import(
    '.quantiles'
)

from .quantiles import QuantileDistribution
