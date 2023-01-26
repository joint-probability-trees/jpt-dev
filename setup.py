import sys
import os
from distutils.extension import Extension
from setuptools import setup

import numpy
from Cython.Build import cythonize


sys.path.insert(0, 'src')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


__version__ = read('version')


def requirements():
    with open('requirements.txt', 'r') as f:
        return [_.strip() for _ in f.readlines() if _.strip()]


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

extensions = [
    Extension("cutils", sources=[os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("intervals", sources=[os.path.join(basedir, pyxfiles[1]), os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("functions", sources=[os.path.join(basedir, pyxfiles[1]), os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("quantiles", sources=[os.path.join(basedir, pyxfiles[2]), os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("cdfreg", sources=[os.path.join(basedir, pyxfiles[3]), os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++"),
    Extension("impurity", sources=[os.path.join(basedir, pyxfiles[4]), os.path.join(basedir, pyxfiles[0])],
              include_dirs=[numpy.get_include()],
              extra_compile_args=["-O3"],
              language="c++")
]


# ----------------------------------------------------------------------------------------------------------------------
# We set the CPATH variable because the "include_dir" argument doesn't seem to work properly


_numpy_include_dir = numpy.get_include()
os.environ['CPATH'] = _numpy_include_dir
print('Setting CPATH environment variable to', _numpy_include_dir)


# ----------------------------------------------------------------------------------------------------------------------
# Compile the C++ extensions

compiled = cythonize([os.path.join(basedir, f) for f in pyxfiles],
                     language='c++',
                     include_path=[numpy.get_include()])

# ----------------------------------------------------------------------------------------------------------------------

setup(
    name='jpt',
    packages=['jpt', 'jpt.learning', 'jpt.base', 'jpt.distributions', 'jpt.distributions.quantile'],
    package_dir={'': 'src'},
    ext_modules=compiled,
    zip_safe=False,
    version=__version__,
    install_requires=requirements(),
)
