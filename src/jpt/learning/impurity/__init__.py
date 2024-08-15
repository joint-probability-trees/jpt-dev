
try:
    from .impurity import __module__
except ModuleNotFoundError:
    from jpt.base import pyximporter
    pyximporter.install()
finally:
    from .impurity import Impurity
