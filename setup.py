import sys
import os
from distutils.extension import Extension
from setuptools import setup

import numpy
from Cython.Build import cythonize


sys.path.insert(0, 'src')


__description__ = '''Joint Probability Trees (short JPTs) are a formalism for learning of and reasoning about joint probability
distributions, which is tractable for practical applications. JPTs support both symbolic and subsymbolic variables in a single
hybrid model, and they do not rely on prior knowledge about variable dependencies or families of distributions.
JPT representations build on tree structures that partition the problem space into relevant subregions that are elicited
from the training data instead of postulating a rigid dependency model prior to learning. Learning and reasoning scale
linearly in JPTs, and the tree structure allows white-box reasoning about any posterior probability :math:`P(Q\mid E)`,
such that interpretable explanations can be provided for any inference result. This documentation introduces the
code base of the ``pyjpt`` library, which is implemented in Python/Cython, and showcases the practical
applicability of JPTs in high-dimensional heterogeneous probability spaces, making it
a promising alternative to classic probabilistic

## Documentation
The documentation is hosted on readthedocs.org [here](https://joint-probability-trees.readthedocs.io/en/latest/).'''


def read_version(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read().strip()
    except FileNotFoundError:
        return '0.0.0'


__version__ = read_version(os.path.join('src', 'jpt', '.version'))


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

compiled = cythonize(
    [os.path.join(basedir, f) for f in pyxfiles],
    language='c++',
    include_path=[numpy.get_include()]
)

# ----------------------------------------------------------------------------------------------------------------------

setup(
    name='pyjpt',
    packages=[
        'jpt',
        'jpt.learning',
        'jpt.base',
        'jpt.distributions',
        'jpt.distributions.quantile',
        'jpt.distributions.univariate'
    ],
    package_dir={'': 'src'},
    ext_modules=compiled,
    zip_safe=False,
    version=__version__,
    install_requires=requirements(),
    long_description=__description__,
    package_data={'jpt': ['.version']},
    include_package_data=True,
    extras_require={'mlflow': ['mlflow >= 2.5.0']}
)
