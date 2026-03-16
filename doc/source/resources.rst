Resources
=========

This page collects references and further reading to help understand
the library, the theory behind JPTs, and related work.

Papers
------

* **Joint Probability Trees** — the original ``pyjpt`` paper:
  Nyga, Picklum, Schierenbeck, Beetz (2023).
  `arXiv:2302.07167 <http://arxiv.org/abs/2302.07167>`_

* **Probabilistic Circuits: A Unifying Framework** —
  :cite:`ProbCirc20`.  Introduces the probabilistic circuit
  framework that JPTs build upon.

* **CART: Classification and Regression Trees** —
  Breiman et al. (1984).  The basis for the tree-construction
  algorithm used in JPTs.

External Tools
--------------

* `scikit-learn <https://scikit-learn.org>`_ — used for dataset
  utilities and model evaluation in the tutorial notebooks.
* `pandas <https://pandas.pydata.org>`_ — DataFrame-based data
  handling used throughout the API.
* `MLflow <https://mlflow.org>`_ — optional experiment tracking
  integration; see :doc:`mlflow_integration`.

.. bibliography::
    :cited:
