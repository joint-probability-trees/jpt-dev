import sys
import os
from distutils.extension import Extension
from setuptools import setup

import numpy
from Cython.Build import cythonize


# ----------------------------------------------------------------------------------------------------------------------
# Set up the C++ extensions for compilation


basedir = 'src'

pyxfiles = [
    "jpt/base/cutils.pyx",
    "jpt/base/functions.pyx",
    "jpt/base/intervals.pyx",
    "jpt/distributions/quantile/quantiles.pyx",
    "jpt/distributions/quantile/cdfreg.pyx",
    "jpt/learning/impurity.pyx",
]

# ----------------------------------------------------------------------------------------------------------------------
# We set the CPATH variable because the "include_dir" argument doesn't seem to work properly

_numpy_include_dir = numpy.get_include()
os.environ['CPATH'] = _numpy_include_dir
print('Setting CPATH environment variable to', _numpy_include_dir)

# ----------------------------------------------------------------------------------------------------------------------
# Compile the C++ extensions

compiled = cythonize(
    [os.path.join(basedir, f) for f in pyxfiles],
    language='c++',
    include_path=[numpy.get_include()]
)

# ----------------------------------------------------------------------------------------------------------------------

setup(
    ext_modules=compiled,
    include_dirs=[numpy.get_include()]
)
