
__module__ = 'functions'

try:
    from .func import __module__
except ModuleNotFoundError:
    from jpt.base.utils import pyximporter
    pyximporter.install()
finally:
    from .func import (
        LinearFunction,
        QuadraticFunction,
        ConstantFunction,
        Undefined,
        Function,
        PiecewiseFunction,
        PLFApproximator
    )
