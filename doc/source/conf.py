# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime
sys.path.insert(0, os.path.abspath('../../src'))
import jpt

# -- Project information -----------------------------------------------------

project = 'pyJPT - Joint Probability Trees in Python'
copyright = '{year}, Daniel Nyga, Mareike Picklum, Tom Schierenbeck'.format(year=datetime.date.today().year)
author = 'Daniel Nyga, Mareike Picklum, Tom Schierenbeck'

# The full version, including alpha/beta/rc tags
release = jpt.__version__

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'autoapi.extension',
    'sphinxcontrib.bibtex',
    'nbsphinx'
]

# auto api setup
autoapi_dirs = ['../../src/jpt']
# autoapi_file_patterns = ["*.py", "*.pyx"]
autoapi_python_class_content = "both"
# autoapi_options = ["show-inheritance-diagram"]

# bibtex setup
bibtex_bibfiles = ['./refs.bib']



# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
