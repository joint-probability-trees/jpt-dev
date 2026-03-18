How to Work with Variables
=========================

``pyjpt`` uses typed variable objects to describe the columns of your
data.  This guide covers creation, configuration, and the advanced
``invert_impurity`` setting for symbolic variables.

Variable Types
--------------

There are three variable types:

- :py:class:`~jpt.variables.NumericVariable` — continuous real-valued
  columns (e.g. temperature, price).
- :py:class:`~jpt.variables.IntegerVariable` — discrete integer-valued
  columns (e.g. counts, dice rolls).
- :py:class:`~jpt.variables.SymbolicVariable` — categorical columns
  (e.g. color, species, gender).

Variables can be created manually or inferred from a DataFrame:

.. code-block:: python

    import pandas as pd
    from jpt.variables import (
        NumericVariable,
        SymbolicVariable,
        infer_from_dataframe,
    )
    from jpt.distributions import SymbolicType, Bool

    # Manual creation
    temperature = NumericVariable('temperature')
    color = SymbolicVariable(
        'color',
        SymbolicType('Color', ['red', 'green', 'blue']),
    )
    raining = SymbolicVariable('raining', Bool)

    # Automatic inference
    df = pd.read_csv('data.csv')
    variables = infer_from_dataframe(df)

Impurity Inversion for Symbolic Variables
-----------------------------------------

By default the JPT learning algorithm minimizes the Gini impurity of
every target variable, producing leaves in which each symbolic variable
is as *pure* as possible — ideally a single dominant category.

Setting ``invert_impurity=True`` on a
:py:class:`~jpt.variables.SymbolicVariable` reverses this objective
for that variable: the learner now favors splits that keep the
variable's distribution *mixed* within each leaf instead of
separating it.

.. code-block:: python

    from jpt.distributions import SymbolicType
    from jpt.variables import SymbolicVariable
    from jpt.trees import JPT

    Gender = SymbolicType('Gender', ['female', 'male', 'other'])
    gender = SymbolicVariable(
        'gender',
        Gender,
        invert_impurity=True,
    )

    model = JPT(variables=[gender, ...])
    model.fit(df)

When to Use Impurity Inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fairness-aware learning.**
Mark a protected attribute (e.g. gender, ethnicity) with
``invert_impurity=True``.  The tree will avoid creating splits that
segregate by that attribute, producing leaves where the protected
attribute remains representative of the overall population.  This is
a lightweight structural fairness constraint — the model can still
predict on other variables without building discriminatory
partitions.

**Confound suppression.**
In observational data a confounding variable (e.g. hospital site in a
multi-site medical study) may dominate splits even though the goal is
to learn patient-level patterns.  Inverting impurity on the confounder
forces the tree to find splits driven by other variables while
keeping the confounder mixed in every leaf.

**Balanced stratification for downstream tasks.**
If you need per-leaf statistics that should be computed over a
representative distribution of some grouping variable (e.g. product
category), inversion ensures each leaf retains a mix of all groups
rather than splitting them apart.

Example
~~~~~~~

.. code-block:: python

    import pandas as pd
    from jpt.distributions import SymbolicType
    from jpt.variables import SymbolicVariable
    from jpt.trees import JPT

    df = pd.DataFrame({
        'fst': ['a', 'a', 'a', 'b', 'b', 'b'],
        'snd': ['c', 'd', 'c', 'd', 'c', 'd'],
    })

    AT = SymbolicType('AType', labels=['a', 'b'])
    BT = SymbolicType('BType', labels=['c', 'd'])

    A = SymbolicVariable('fst', AT, invert_impurity=True)
    B = SymbolicVariable('snd', BT)

    model = JPT([A, B])
    model.fit(df)

    # Each leaf retains a mix of 'a' and 'b' for variable
    # ``fst`` instead of splitting them into pure nodes.
    for leaf in model.leaves.values():
        print(leaf.distributions['fst'])

.. seealso::

    :py:class:`~jpt.variables.SymbolicVariable` — full API reference.

    :doc:`howto_classification` — using symbolic targets for
    classification.
