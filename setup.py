import os

from setuptools import setup


# ------------------------------------------------------
# Skip Cython compilation for development installs
# (pyximport will handle JIT compilation at runtime).
# Usage: JPT_NO_CYTHON=1 pip install -e ".[dev]"

if not os.environ.get('JPT_NO_CYTHON'):
    import numpy
    from Cython.Build import cythonize

    basedir = 'src'

    pyxfiles = [
        "jpt/base/cutils/cutils.pyx",
        "jpt/base/functions/func.pyx",
        "jpt/base/intervals/base.pyx",
        "jpt/base/intervals/intset.pyx",
        "jpt/base/intervals/contset.pyx",
        "jpt/base/intervals/unionset.pyx",
        "jpt/distributions/qpd/quantiles.pyx",
        "jpt/distributions/qpd/cdfreg.pyx",
        "jpt/distributions/qpd/vwcdfreg.pyx",
        "jpt/learning/impurity/impurity.pyx",
    ]

    os.environ['CPATH'] = numpy.get_include()

    ext_modules = cythonize(
        [os.path.join(basedir, f) for f in pyxfiles],
        language='c++',
        include_path=[numpy.get_include()]
    )
else:
    ext_modules = []

setup(ext_modules=ext_modules)
