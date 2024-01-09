
__module__ = 'functions'

try:
    from .func import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
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