import sys
import os
from distutils.extension import Extension
from setuptools import setup, find_packages

sys.path.insert(0, 'src')


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


__version__ = read('version')


def requirements():
    with open('requirements.txt', 'r') as f:
        return [_.strip() for _ in f.readlines() if _.strip()]


# ----------------------------------------------------------------------------------------------------------------------
# Some helper functions to manage the dependencies


# def apt_get_install(packages):
#     subprocess.check_call(['apt-get', "install", "-y"] + packages)


# def install(package, upgrade=False):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package] + ['-U'] if upgrade else [])


# ----------------------------------------------------------------------------------------------------------------------
# Install the dependencies on Ubuntu/Debian Systems

# try:
#     with open('/etc/os-release', 'r') as f:
#         os_info = {k: v for k, v in [l.split('=') for l in f.readlines()]}
#
#     if os_info.get('NAME', 'N/A') in {'Ubuntu', 'Debian'}:
#         apt_get_install(['glibc-source',
#                          'libatlas-base-dev',
#                          'gfortran',
#                          'libxslt1-dev',
#                          'libxml2-dev',
#                          'graphviz'])
#
#         install('pip', upgrade=True)

# except FileNotFoundError:
#     print('We are running on Windows or Linux distribution cannot be determined.')

# process = subprocess.Popen([sys.executable, "-m", "pip", "install", '-U', '-r', 'requirements.txt'],
#                            stdout=subprocess.PIPE,
#                            stderr=subprocess.PIPE)
#
# while s := process.stdout.read(1024):
#     sys.stdout.write(s.decode())
#
# while s := process.stderr.read(1024):
#     sys.stderr.write(s.decode())

import numpy
import cysignals
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
    setup_requires=["cysignals"],
    install_requires=requirements(),
)
