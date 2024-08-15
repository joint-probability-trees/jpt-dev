try:
    from .quantiles import __module__
except ModuleNotFoundError:
    from jpt.base import pyximporter
    pyximporter.install()
finally:
    from .quantiles import QuantileDistribution
