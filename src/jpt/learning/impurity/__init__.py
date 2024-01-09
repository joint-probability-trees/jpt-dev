try:
    from .impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .impurity import Impurity
