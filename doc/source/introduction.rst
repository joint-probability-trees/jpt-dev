Introduction
============

What are JPTs?
--------------

Joint Probability Trees (JPTs) are a non-parametric probabilistic model
that learns and represents the joint distribution :math:`P(\mathcal{X})`
over a set of variables :math:`\mathcal{X}` directly from data.

A JPT partitions the data space into a set of regions using a decision
tree.  In each leaf region the distribution over all variables is
approximated by a fully factorised product of univariate distributions.
The overall joint distribution is a mixture across all leaves:

.. math::

    P(X=x) = \sum_{\lambda \in \Lambda} P(L=\lambda)
              \prod_i P(X_i = x_i \mid L = \lambda)

where :math:`\Lambda` is the set of leaves and :math:`P(L=\lambda)` is
the prior probability (mixing weight) of leaf :math:`\lambda`.

.. image:: _static/img/gaussian-jpt.png

Variable Types
--------------

``pyjpt`` natively handles three types of variables in a single model:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable type
     - Data type
     - Leaf distribution
   * - ``NumericVariable``
     - ``float`` / ``int``
     - Quantile-based (piecewise linear CDF)
   * - ``SymbolicVariable``
     - ``str`` / category
     - Multinomial
   * - ``IntegerVariable``
     - integer domain
     - Discrete uniform / histogram

Use :py:func:`~jpt.variables.infer_from_dataframe` to create the right
variable type automatically from a DataFrame's column dtypes.

Why JPTs?
---------

* **No structural assumptions** — the tree partition is learned from
  data; no prior knowledge about dependencies or distribution families
  is required.
* **Hybrid support** — symbolic and continuous variables coexist in a
  single model without manual encoding.
* **Tractable inference** — all query types (marginal, conditional,
  posterior, MPE) are computed in a single pass over the tree.
* **White-box** — every inference result traces back to specific leaves,
  enabling interpretable explanations.
* **Linear complexity** — training and inference both scale linearly in
  the number of leaves.

Supported Inference Types
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Query
     - API method
   * - Marginal :math:`P(Q)`
     - :py:meth:`~jpt.trees.JPT.infer`
   * - Conditional :math:`P(Q \mid E)`
     - :py:meth:`~jpt.trees.JPT.infer`
   * - Posterior distribution
     - :py:meth:`~jpt.trees.JPT.posterior`
   * - Expectation
     - :py:meth:`~jpt.trees.JPT.expectation`
   * - Most Probable Explanation (MPE)
     - :py:meth:`~jpt.trees.JPT.mpe`
   * - k-MPE
     - :py:meth:`~jpt.trees.JPT.kmpe`
   * - Conditional JPT
     - :py:meth:`~jpt.trees.JPT.bind`

Theory
------

Probabilistic Circuits
**********************

JPTs are a shallow, deterministic probabilistic circuit (PC).
A JPT defines a tree-like computational graph: deterministic sum nodes
all the way down to the leaves, where fully factorising product nodes
are used.  For more background on probabilistic circuits see
:cite:`ProbCirc20`.

The sum nodes are decision nodes like in decision trees.  They contain
one variable and a split value that partitions the data into two
subsets.  The product nodes fully factorise the problem into independent
distributions represented by quantile distributions for numeric
variables and multinomials for symbolic variables.

Marginal and Conditional Queries
*********************************

A marginal query (MAR) is a partial assignment:

.. math::

    P(\mathcal{E} = e, Z) = \int_{\mathcal{I}_1} \cdots
    \int_{\mathcal{I}_k} P(e, z_1, \dots, z_k)\, dZ_k \cdots dZ_1

where :math:`Z = \mathcal{X} \setminus \mathcal{E}` are the unassigned
variables.

A conditional query follows from two marginal queries:

.. math::

    P(Q \mid E) = \frac{P(Q, E)}{P(E)}

Most Probable Explanation
*************************

MPE (a.k.a. MAP) solves:

.. math::

    \operatorname{argmax}_{Q \cup E}\, P(Q \mid E)

JPTs return a set of results since the piecewise structure allows
multiple maxima to exist and maxima can be intervals rather than
single points.

Probabilistic Learning
----------------------

Generative Learning
*******************

In generative mode (the default), the tree is built by a modified C4.5
algorithm that maximises information gain across *all* variables.  Each
leaf stores a fully factorised product distribution.  This mode models
the full joint :math:`P(\mathcal{X})`.

Discriminative Learning
***********************

In discriminative mode, the impurity computation is restricted to a
designated set of *target* variables :math:`Y`.  Splits are chosen to
maximise information gain with respect to :math:`Y`, while features
:math:`X = \mathcal{X} \setminus Y` serve solely as split candidates.
This concentrates the model's capacity on predicting :math:`Y` and is
well-suited for classification and regression.

Activate discriminative mode via the ``targets`` argument:

.. code-block:: python

    model = JPT(variables, targets=[varnames['species']])
    model.fit(df)
