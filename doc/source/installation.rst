Getting Started
===============

Installation
************

Via PyPI
~~~~~~~~
The ``pyjpt`` package is available on the standard Python package index. Install the package with

.. code:: bash

    $ pip install pyjpt

Via GitHub Repository
~~~~~~~~~~~~~~~~~~~~~
or clone the repository and install with

.. code:: bash

    $ git clone https://github.com/joint-probability-trees/jpt-dev
    $ cd jpt
    $ python setup.py install

When you are working on Debian-based systems, you will have to install a additional requirements

.. code:: bash

    sudo apt install `(cat requirements-deb.txt)`

Build the Documentation
~~~~~~~~~~~~~~~~~~~~~~~

To build the documentation clone the repository, switch to the ``doc`` folder and
install the documentation requirements:

.. code:: bash

    cd doc
    pip install -r requirements.txt

After everything is successfully installed, the documentation can be built using sphinx, for example

.. code:: bash

    make html

After the build process has finished, you can view the documentation in your browser under ``build/html/index.html``.

Supported Platforms
*******************
Currently the package is only supported and tested for Ubuntu 18+, but *theoretically* it should be working
for other operating systems too.

Testing
*******

How to run the tests.

Running the Jupyter Notebooks
*****************************

In order run the included Jupyter notebooks with your development version of the repository checkout, you can
include the path to the code location into the ``PYTHONPATH`` that is passed to the Jupyter kernel as described
in `this forum`_:

1. Find the location of your ``kernel.json`` file:

    .. code:: bash

        jupyter kernelspec list

2. Add and ``env`` argument to the kernel specification:

    .. code:: json

        {
            "argv": [
                "python",
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}"
            ],
            "display_name": "Python 3",
            "language": "python",
            "env": {
               "PYTHONPATH": "/path/to/repo/src"
            }
        }

.. _this forum: https://discourse.jupyter.org/t/how-can-i-pass-environment-variabel-pythonpath-to-jupyter-notebook/7351/2
