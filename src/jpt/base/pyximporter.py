import os
import sys

try:
    from pyximport._pyximport3 import PyxImportLoader
except ModuleNotFoundError:
    pass
else:
    from portalocker import Lock

    class ConcurrentPyxImportLoader(PyxImportLoader):

        def create_module(self, spec):
            with Lock(
                os.path.join(str(self.path))
            ):
                if spec.name in sys.modules:
                    return sys.modules[spec.name]
                return super().create_module(spec)

    # ----------------------------------------------------------------------------------------------------------------------

    def install(
            pyximport=True,
            pyimport=False,
            build_dir=None,
            build_in_temp=True,
            setup_args=None,
            reload_support=False,
            load_py_module_on_import_failure=False,
            inplace=False,
            language_level=None,
            enable_concurrency=True,
    ):
        import pyximport as pyximport_mod

        if enable_concurrency:
            from pyximport import _pyximport3
            _pyximport3.PyxImportLoader = ConcurrentPyxImportLoader

        pyximport_mod.install(
            pyximport=pyximport,
            pyimport=pyimport,
            build_dir=build_dir,
            build_in_temp=build_in_temp,
            setup_args=setup_args,
            reload_support=reload_support,
            load_py_module_on_import_failure=load_py_module_on_import_failure,
            inplace=inplace,
            language_level=language_level,
        )


# def pyx_import(*modules):
#     frm = inspect.stack()[1]
#     mod = inspect.getmodule(frm[0])
#     package = mod.__name__
#     for module in modules:
#         _pyx_import(module, package)


# ----------------------------------------------------------------------------------------------------------------------

# def _pyx_import(
#         module: str,
#         package: str
# ):
#     try:
#         return importlib.import_module(module, package)
#     except ModuleNotFoundError:
#         import pyximport
#         pyximport.install()
#         return importlib.import_module(module, package)
