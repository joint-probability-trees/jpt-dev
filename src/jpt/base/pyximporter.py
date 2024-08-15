import importlib
import inspect
import multiprocessing
import os
import sys

from portalocker import Lock
from pyximport._pyximport3 import PyxImportLoader

multiprocessing.Condition

class ConcurrentPyxImportLoader(PyxImportLoader):

    def create_module(self, spec):
        with Lock(
            os.path.join(str(self.path))
        ):
            if spec.name in sys.modules:
                return sys.modules[spec.name]
            return super().create_module(spec)


def enable_concurrency():
    import pyximport
    from pyximport import _pyximport3
    _pyximport3.PyxImportLoader = ConcurrentPyxImportLoader
    pyximport.install()


def pyx_import(*modules):
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    package = mod.__name__
    for module in modules:
        _pyx_import(module, package)


# ----------------------------------------------------------------------------------------------------------------------

def _pyx_import(
        module: str,
        package: str
):
    try:
        return importlib.import_module(module, package)
    except ModuleNotFoundError:
        import pyximport
        pyximport.install()
        return importlib.import_module(module, package)
