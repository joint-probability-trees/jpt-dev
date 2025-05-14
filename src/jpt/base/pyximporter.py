import os
import sys
import traceback


try:
    import cython, pyximport
    if cython.__version__ <= '3.0':
        pyximport_mod = pyximport._pyximport3
    else:
        import pyximport as pyximport_mod

except ModuleNotFoundError:
    traceback.print_exc()
else:
    from portalocker import Lock

    class ConcurrentPyxImportLoader(pyximport_mod.PyxImportLoader):

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
        if enable_concurrency:
            pyximport_mod.PyxImportLoader = ConcurrentPyxImportLoader

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
