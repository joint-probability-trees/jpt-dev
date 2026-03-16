.. image:: _static/img/jpt-logo-horizontal.svg
   :width: 420px
   :align: center
   :alt: pyjpt — Joint Probability Trees

.. rst-class:: center

   *Learn joint distributions from data. Query, predict, explain.*

|

``pyjpt`` is a Python library for learning and querying joint probability
distributions directly from data — no structural assumptions, no manual
feature engineering.  Feed it a DataFrame, get a model that can answer
marginal and conditional probability queries, compute posteriors, find
most-probable explanations, and generate samples — all in a single,
interpretable tree structure that handles mixed symbolic and numeric data
out of the box.

.. code-block:: bash

   pip install pyjpt[matplotlib]

|

.. code-block:: python

   import pandas as pd
   from jpt.variables import infer_from_dataframe
   from jpt.trees import JPT

   model = JPT(infer_from_dataframe(df), min_samples_leaf=0.1)
   model.fit(df)

   p    = model.infer(query={'species': 'setosa'})
   post = model.posterior(['species'], evidence={'petal length (cm)': [1, 2]})
   mpe  = model.mpe(evidence={'species': 'virginica'})

|

Why pyjpt?
----------

* **Hybrid** — symbolic and numeric variables in a single model, no encoding needed
* **No assumptions** — tree partition and distributions are learned directly from data
* **Tractable** — marginals, conditionals, posteriors, MPE and k-MPE in one tree pass
* **White-box** — every inference result traces back to interpretable leaves
* **Linear scaling** — training and inference scale linearly in the number of leaves


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quickstart
   introduction


.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   guide


.. toctree::
   :maxdepth: 1
   :caption: How-to Guides

   howto


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples


.. toctree::
   :maxdepth: 1
   :caption: Reference

   autoapi/index
   changelog


.. toctree::
   :maxdepth: 1
   :caption: Further Information

   faq
   resources
   integrations


----

Lead Developers
---------------

* Daniel Nyga (`📧 <mailto:nyga@uni-bremen.de>`__, `LinkedIn <https://www.linkedin.com/in/daniel-nyga-66029858/>`__)
* Mareike Picklum (`📧 <mailto:mareikep@uni-bremen.de>`__, `LinkedIn <https://www.linkedin.com/in/mareike-picklum-265ba4121/>`__)
* Tom Schierenbeck (`📧 <mailto:tom_sch@uni-bremen.de>`__)

.. note:: If you use ``pyjpt`` in scientific publications, any
   acknowledgement is highly appreciated.  The original paper can be
   found `here <http://arxiv.org/abs/2302.07167>`_::

       @inproceedings{nyga23jpts,
           title={{Joint Probability Trees}},
           author={Daniel Nyga and Mareike Picklum and Tom Schierenbeck
                   and Michael Beetz},
           year={2023},
           booktitle = {arxiv.org},
           note = {Preprint},
           url = {http://arxiv.org/abs/2302.07167}
       }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
