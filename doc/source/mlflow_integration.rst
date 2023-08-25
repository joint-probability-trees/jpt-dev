MLFlow Integration
==================

JPTs can be integrated with MLFlow to manage the full model lifecycle.
Among other things Hyperparameters and the model itself can be logged using a python wrapper.

In order to integrate JPTs with MLFlow, MLFlow needs to be installed. This can be done by either installing the
``pyjpt`` package with MLFlow being added as an extra requirement

.. code:: bash

    $ pip install pyjpt[mlflow]

or by installing the MLFlow package manually.


A full tutorial on the MLFlow integration can be found here

.. toctree::

    notebooks/tutorial_mlflow