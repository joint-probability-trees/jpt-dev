Basic Concepts
===============

Overview
********
The JPT package brings reliable, expressive, efficient and interpretable joint probability distributions to everyone.

Supported types of inference are:
    - Full evidence queries
    - Marginal queries
    - Conditional queries
    - Expectations
    - Most probable explanations
    - Confidence intervals


Probabilistic Reasoning
***********************

Understanding basic concepts of probabilistic inference is the key to smart probabilistic modeling.
This chapter will go through all types of probabilistic inference that are possible in JPTs.
JPTs are a special form of probabilistic circuits (PCs) therefore we will refer to the definitions
from :cite:`ProbCirc20`.
Mathematically speaking, JPTs are a shallow, deterministic probabilistic circuit.
A JPT defines a tree like computational graph. The tree consists of deterministic sum nodes all the way down to the
leaves. In the leaves fully factorizing product nodes are used.
The sum nodes are decision nodes like in decision trees. They contain one variable and a split value, where all
datapoints are divided into two subsets, one subset that satisfies the condition and one subset that satisfies
its negation.
The product nodes fully factorize the problem into independent distributions. They consist of a set of distributions
and a prior (mixing coefficient). The factors are represented by quantile distributions.

The joint probability distribution of a JPT is given by

.. math::

    P(X=x)=\sum_{\lambda\in\Lambda}P(L=\lambda)\prod_iP(X_i=x_i | L=\lambda).

where :math:`P(\Lambda)` is the set of all leaves.

.. image:: _static/img/gaussian-jpt.png

Full evidence query
^^^^^^^^^^^^^^^^^^^

A complete evidence query (EVI) computes :math:`P(\mathcal{X} = x)`, i. e. an assignment of one value to all variables.
In complete discrete universes this is equivalent to the probability of one specific world.
In complete continuous universes the EVI query corresponds to the likelihood. In mixed universes it is a mixture of
likelihoods and probabilities.

In everyday situations it is unlikely that everything about the world is known and observable, therefore the EVI query
is not particular useful as an end user.

However, the EVI query is very useful when evaluating the models performance. This is typically done with the likelihood
function

.. math::

    \mathcal{L}(\mathcal{D}|\theta) = \prod_{d \in \mathcal{D}} P(\mathcal{X}=d)

or its logarithm

.. math::

    log(\mathcal{L}(\mathcal{D}|\theta)) = \sum_{d \in \mathcal{D}} log(P(\mathcal{X}=d)).

The EVI query of a JPT can be used with the likelihood function :py:mod:`jpt.trees.JPT.likelihood`.
Details on how to use the likelihood function are found in :doc:`notebooks/tutorial_iris`.

Marginal Query
^^^^^^^^^^^^^^

A marginal query (MAR) is a partial assignment of the world.
Mathematically it can be expressed as

.. math::
    P(\mathcal{E} = e, Z) = \int_{\mathcal{I_1}} \cdots \int_{\mathcal{I_k}} P(e, z_1, \dots, z_k) dZ_k \cdots dZ_1

where e is a partial state and :math:`Z = \mathcal{X} \setminus \mathcal{E}` is the set of all unassigned variables.

This query is very useful when asking for the probability of a broad range of scenarios. In robotics a common example is
the probability of a two problems occurring together.

Conditional Query
^^^^^^^^^^^^^^^^^

The conditional query is a very common query in machine learning.
It is written as

.. math::
    P(Q|E) = \frac{P(Q,E)}{P(E)}

where Q and E are sets of assignments to a partial set of all variables. Complexity wise this query can be answered by
answering two marginal queries and dividing their results. P(Q|E) can be interpreted as the question:
How likely is Q given that E happened.

In classification this is the standard query that is posed to every model. For example: What is the probability of
a leaf being a Setosa given the sepal length is 5cm, the sepal width is 2cm, the petal length is 3cm and the petal
width is 0.5cm.

Conditional queries are implemented in :py:mod:`jpt.trees.JPT.infer`.

Posterior Distribution
^^^^^^^^^^^^^^^^^^^^^^

The posterior query is very similar to the conditional query. Again, a question of the form P(Q|E) is posed to the
model, but this time the answer is returned as an set of independent distributions over all variables in Q given E.
Be aware that even if the variables are returned independently, they may not be independent.
To return the full conditional distribution with all its dependencies see `Conditional Distribution`_.
Posterior distributions are especially useful for calculating moments of random variables and to visualize the
uncertainty within the answer.
This is implemented in :py:mod:`jpt.trees.JPT.posterior`.
Detailed examples for posteriors can be found in :doc:`notebooks/tutorial_regression`.

Moments of Random Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculating the moments of variables can be done via :py:mod:`jpt.trees.JPT.expectation` for the first order moment
and for any other moment the resulting distributions from :py:mod:`jpt.trees.JPT.posterior` have to be used.
Detailed examples for expectations can be found in :doc:`notebooks/tutorial_regression`.


Most Probable Explanation
^^^^^^^^^^^^^^^^^^^^^^^^^

The most probable explanation (MPE, a. k. a. maximum a posteriori (MAP)) refers to the query that maximizes the
likelihood of the probability distribution

.. math::
    argmax_{Q \cup E} P(Q|E)

In the literature one will find the common misconception that :math:`Q \cup E = \mathcal{X}` and
:math:`Q \cap E = \emptyset`. However for MPE inference only :math:`Q \cup E = \mathcal{X}` is necessary. We will see
see reason for that in section `Variable Maps`_.
The MPE query returns the assignment of variables that is most likely given E. A good example is given by a scenario
where a robot wants to find the parameters for his plan that maximize the success probability, i. e.
P(Parameters|success=True).
In classical machine learning applications the MPE inference returns one vector with a single value for every variable
in Q. In turn, JPTs return a set of sets describing all maxima of the conditional distributions over all variables.
The result over all variables is returned since it is not necessary that evidence is hard.
Multiple results are returned since the functional form of JPTs allows multiple maxima to exist and allows maxima to be
intervals. In the resulting list of MPEResults of the :py:mod:`jpt.trees.JPT.mpe` the dimensions
in the MPEResults are independent of each other. Therefore any combination of maxima within one MPEResult is a correct
maximum. However the maxima in different MPEResults cannot be mixed since they are structurally different.
Detailed examples for MPE inference can be found in :doc:`notebooks/tutorial_mnist`.

Conditional Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes it is necessary to apply evidence to a distribution and alter the distributions parameters and structure
according to that evidence. This way the evidence is forever engraved in the resulting distribution and the distribution
gets smaller. This saves parameters and memory. It is implemented in :py:mod:`jpt.trees.JPT.conditional_jpt`.

Probabilistic Learning
**********************

Generative Learning
^^^^^^^^^^^^^^^^^^^

Discriminative Learning
^^^^^^^^^^^^^^^^^^^^^^^


