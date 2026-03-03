Getting Started
===============

Installation
************

Via PyPI
~~~~~~~~

The ``pyjpt`` package is available on the standard Python package
index. Install the core package with

.. code:: bash

    $ pip install pyjpt

Optional dependency groups can be installed depending on your use
case:

.. code:: bash

    $ pip install pyjpt[matplotlib]   # matplotlib and graphviz plotting
    $ pip install pyjpt[plotly]       # interactive plotly plotting
    $ pip install pyjpt[seq]          # sequential/temporal models
    $ pip install pyjpt[mlflow]       # MLflow experiment tracking

Multiple groups can be combined:

.. code:: bash

    $ pip install pyjpt[matplotlib,mlflow]

Via GitHub Repository
~~~~~~~~~~~~~~~~~~~~~

Alternatively, clone the repository and install from source:

.. code:: bash

    $ git clone https://github.com/joint-probability-trees/jpt-dev
    $ cd jpt-dev
    $ pip install .

For an editable development install, see the `Testing`_ section
below.

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation, clone the repository, switch to the
``doc`` folder and install the documentation requirements:

.. code:: bash

    $ cd doc
    $ pip install -r requirements.txt

After everything is successfully installed, the documentation can be
built using Sphinx:

.. code:: bash

    $ make html

After the build process has finished, you can view the documentation
in your browser under ``build/html/index.html``.

Supported Platforms
*******************

The package is tested on Ubuntu 22.04, but should work on other
Linux distributions and macOS as well.

Testing
*******

Running the test suite requires the ``dev`` dependency group, which
includes all optional dependencies and test utilities:

.. code:: bash

    $ pip install -e ".[dev]"

Run the full test suite using Python's ``unittest`` discovery:

.. code:: bash

    $ cd test
    $ python -m unittest discover

To run a specific test file:

.. code:: bash

    $ cd test
    $ python -m unittest test_jpt
