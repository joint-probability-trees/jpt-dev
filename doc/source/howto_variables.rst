How to Work with Variables
==========================

``pyjpt`` uses typed variable objects to describe the columns of your
data.  Every variable has a **name** and a **domain** — a distribution
class that defines the set of values the variable can take and how
they are represented internally.  This guide covers:

- The relationship between variables and their domains
- The three variable types and their settings
- How to create variables manually and infer them from data
- Labels (user-facing) vs. values (internal representation)
- Variable maps and assignments for queries and results
- Impurity inversion for symbolic variables

.. contents:: On this page
    :local:
    :depth: 2


Variables and Domains
---------------------

A variable is a pairing of a **name** (a string identifying a column)
with a **domain** (a distribution *class* that defines the legal
values).  The domain is always a class — not an instance — because
the variable describes a *type* of data, not a fitted distribution.
Fitted distributions are created later, inside each leaf of the tree.

.. code-block:: python

    from jpt.distributions import (
        SymbolicType,
        NumericType,
        Bool,
        Numeric,
    )
    from jpt.distributions.univariate import IntegerType
    from jpt.variables import (
        SymbolicVariable,
        NumericVariable,
        IntegerVariable,
    )

    # The domain is a CLASS (not an instance).
    Color = SymbolicType('Color', ['red', 'green', 'blue'])
    color = SymbolicVariable('color', Color)

    # Numeric domains can be plain or scaled.
    temperature = NumericVariable('temperature')  # Numeric
    height = NumericVariable(                     # ScaledNumeric
        'height',
        domain=NumericType(
            'Height',
            values=[165.0, 170.0, 180.0, 190.0],
        ),
    )

    # Integer domains specify a range.
    die = IntegerVariable(
        'die',
        IntegerType('Die', lmin=1, lmax=6),
    )

    # Bool is a predefined symbolic domain with
    # labels {True, False}.
    raining = SymbolicVariable('raining', Bool)

When the tree calls ``variable.distribution()`` internally, it
instantiates the domain class and passes relevant settings from the
variable to the new distribution instance.  The resulting object
holds fitted parameters (probability vectors, quantile functions,
etc.) for a specific leaf.


Domain Factory Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Domains are created with factory functions that return new *classes*
(not instances):

**SymbolicType(name, labels)**
    Creates a :py:class:`~jpt.distributions.univariate.Multinomial`
    subclass.  ``labels`` is a list of user-facing category names.
    Internally, each label is mapped to a zero-based integer index.

    .. code-block:: python

        Fruit = SymbolicType('Fruit', ['apple', 'banana', 'cherry'])
        Fruit.labels   # {0: 'apple', 1: 'banana', 2: 'cherry'}
        Fruit.values   # {'apple': 0, 'banana': 1, 'cherry': 2}
        Fruit.n_values  # 3

**NumericType(name, values=None)**
    Creates a
    :py:class:`~jpt.distributions.univariate.numeric.ScaledNumeric`
    subclass.  If ``values`` is provided (a sample of the expected
    data), the domain stores mean/scale normalization factors so that
    the tree works in standardized space internally while preserving
    the original scale externally.  If no values are given, the
    plain ``Numeric`` class is used (no scaling).

**IntegerType(name, lmin=None, lmax=None)**
    Creates an :py:class:`~jpt.distributions.univariate.Integer`
    subclass.  ``lmin`` and ``lmax`` define the label-space bounds
    (inclusive).  Omitting a bound creates an open-ended domain.
    Internally, values are mapped to zero-based indices:

    .. code-block:: python

        Score = IntegerType('Score', lmin=-2, lmax=2)
        # labels (external): -2, -1, 0, 1, 2
        # values (internal):  0,  1, 2, 3, 4

**Bool**
    A predefined ``Multinomial`` subclass with two labels, ``False``
    (index 0) and ``True`` (index 1).  It is a ready-made class, not
    a factory.


Variable Types
--------------

All variables inherit from the abstract base class
:py:class:`~jpt.variables.Variable`, which provides the settings
mechanism, serialization, and hashing.  ``Variable`` cannot be
instantiated directly; use one of the three concrete subclasses.


NumericVariable
~~~~~~~~~~~~~~~

For continuous, real-valued columns.

.. code-block:: python

    temp = NumericVariable(
        'temperature',
        domain=Numeric,               # default
        blur=0.05,                    # widen point evidence
        max_std=2.0,                  # stop splitting below
                                      # this standard deviation
        precision=0.01,               # quantile granularity
        min_impurity_improvement=0.0, # minimum split gain
    )

============================================ ================================
Setting                                      Description
============================================ ================================
``blur``                                     Widens a single-value evidence
                                             point into an interval via the
                                             prior quantile function.
                                             Default: ``0``.
``max_std``                                  If the standard deviation
                                             in a node drops below this
                                             limit, no further splits are
                                             attempted for this variable.
                                             Default: ``0`` (disabled).
``precision``                                Controls the granularity of
                                             the quantile-based density
                                             approximation.
                                             Default: ``0.01``.
``min_impurity_improvement``                 Minimum impurity reduction
                                             required for a split on this
                                             variable.
                                             Default: ``0``.
============================================ ================================


IntegerVariable
~~~~~~~~~~~~~~~

For discrete, integer-valued columns with a finite or open-ended
range.

.. code-block:: python

    rolls = IntegerVariable(
        'rolls',
        IntegerType('Rolls', lmin=1, lmax=20),
        min_impurity_improvement=0.0,
    )

============================================ ================================
Setting                                      Description
============================================ ================================
``min_impurity_improvement``                 Minimum impurity reduction
                                             required for a split on this
                                             variable.
                                             Default: ``0``.
============================================ ================================


SymbolicVariable
~~~~~~~~~~~~~~~~

For categorical columns.

.. code-block:: python

    species = SymbolicVariable(
        'species',
        SymbolicType('Species', ['setosa', 'versicolor', 'virginica']),
        invert_impurity=False,
        min_impurity_improvement=0.0,
    )

============================================ ================================
Setting                                      Description
============================================ ================================
``invert_impurity``                          Invert the Gini impurity
                                             for this variable, favoring
                                             *mixed* leaves.
                                             Default: ``False``.
                                             See
                                             :ref:`impurity-inversion`.
``min_impurity_improvement``                 Minimum impurity reduction
                                             required for a split on this
                                             variable.
                                             Default: ``0``.
============================================ ================================


Inferring Variables from a DataFrame
------------------------------------

For quick setup, :py:func:`~jpt.variables.infer_from_dataframe`
inspects column dtypes and creates one variable per column:

.. code-block:: python

    import pandas as pd
    from jpt.variables import infer_from_dataframe

    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'age':  [30.0, 25.0, 35.0],
        'score': [3, 5, 4],
    })

    variables = infer_from_dataframe(df)
    # [SymbolicVariable('name', ...),
    #  NumericVariable('age', ...),
    #  IntegerVariable('score', ...)]

The mapping from dtype to variable type is:

=========================== ==================================
Column dtype                Variable type
=========================== ==================================
``bool``, ``object``,       ``SymbolicVariable``
``string``
``float16/32/64``           ``NumericVariable``
``int8/16/32/64``           ``IntegerVariable``
=========================== ==================================

Useful keyword arguments:

- **scale_numeric_types** (default ``True``): use
  ``ScaledNumeric`` domains for float columns, storing mean/scale
  normalization from the column data.
- **unique_domain_names** (default ``False``): append a UUID to
  every domain name, preventing name collisions when calling the
  function multiple times.
- **excluded_columns**: a dict mapping column names to
  user-provided domain classes, overriding automatic inference for
  those columns.
- **remove_nan** (default ``False``): exclude ``NaN`` / ``±inf``
  values when constructing numeric domains.


Labels vs. Values
-----------------

``pyjpt`` distinguishes two representations of data:

**Labels** (user-facing / exterior)
    The human-readable representation: category strings
    (``'red'``, ``'blue'``), raw floats (``23.5``), or integers
    (``-2``).  Labels are what you see in DataFrames, queries, and
    printed results.

**Values** (internal / interior)
    The representation used during tree learning and inference.  For
    symbolic variables, labels are mapped to zero-based integer
    indices.  For scaled numeric variables, labels are
    z-normalized.  For plain numeric and integer variables, labels
    and values are identical (identity mapping).

Every domain class provides bidirectional conversion:

.. code-block:: python

    Color = SymbolicType('Color', ['red', 'green', 'blue'])

    # Label → Value
    Color.values['red']         # 0
    Color.label2value('green')  # 1
    Color.label2value({'red', 'blue'})  # {0, 2}

    # Value → Label
    Color.labels[0]             # 'red'
    Color.value2label(1)        # 'green'
    Color.value2label({0, 2})   # {'red', 'blue'}

For numeric variables with scaling:

.. code-block:: python

    import numpy as np

    Height = NumericType(
        'Height',
        values=np.array([165., 170., 180., 190.]),
    )

    # label2value normalizes (subtracts mean, divides by std)
    internal = Height.label2value(180.0)

    # value2label denormalizes
    external = Height.value2label(internal)  # ≈ 180.0

The tree's ``bind()`` method works in **label space** by default,
so you pass human-readable values and the conversion happens
automatically:

.. code-block:: python

    evidence = model.bind(color='red', temperature=22.5)


Variable Maps and Assignments
-----------------------------

VariableMap
~~~~~~~~~~~

:py:class:`~jpt.variables.VariableMap` is a dictionary-like
container that maps ``Variable`` objects to arbitrary values.  It
supports lookup by both the ``Variable`` object and its name
string:

.. code-block:: python

    from jpt.variables import VariableMap

    vm = VariableMap()
    vm[color] = 'red'       # set by Variable object
    vm['temperature'] = 22.5  # set by name string (if
                              # registered)

    print(vm[color])         # 'red'
    print(vm['color'])       # 'red'

``VariableMap`` is used throughout the library: leaf distributions,
prior distributions, query results, and moment computations all
return ``VariableMap`` instances.

It supports standard dict operations — iteration (yields
``Variable`` objects), ``in``, ``del``, ``len``, ``keys()``,
``values()``, ``items()`` — as well as ``copy()``, in-place
``+=`` / ``-=``, and JSON serialization.


VariableAssignment
~~~~~~~~~~~~~~~~~~

:py:class:`~jpt.variables.VariableAssignment` extends
``VariableMap`` with **type validation** for the values.  It is
an abstract class with two concrete subclasses that correspond to
the two data representations:

**LabelAssignment** — stores values in label space.

.. code-block:: python

    from jpt.variables import LabelAssignment

    la = LabelAssignment(variables=[color, temperature])
    la[color] = {'red', 'green'}  # set of labels
    la[temperature] = 22.5        # scalar → ContinuousSet

    # Convert to internal representation:
    va = la.value_assignment()

**ValueAssignment** — stores values in value space (integer
indices for symbolic variables, normalized floats for scaled
numerics).

.. code-block:: python

    # Convert back to labels:
    la2 = va.label_assignment()

Assignments validate their inputs: setting a symbolic variable to
a label that is not in its domain raises an error, as does setting
a numeric variable to a non-numeric value.

In practice, you rarely construct assignments directly.  The
tree's ``bind()`` method builds a ``LabelAssignment`` from
user-friendly keyword arguments:

.. code-block:: python

    # These are equivalent:
    evidence = model.bind(color='red', temperature=[20, 25])

    # The list [20, 25] is converted to ContinuousSet(20, 25)
    # automatically.


.. _impurity-inversion:

Impurity Inversion for Symbolic Variables
-----------------------------------------

By default the JPT learning algorithm minimizes the Gini impurity
of every target variable, producing leaves in which each symbolic
variable is as *pure* as possible — ideally a single dominant
category.

Setting ``invert_impurity=True`` on a
:py:class:`~jpt.variables.SymbolicVariable` reverses this
objective for that variable: the learner now favors splits that
keep the variable's distribution *mixed* within each leaf instead
of separating it.

.. code-block:: python

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
``invert_impurity=True``.  The tree will avoid creating splits
that segregate by that attribute, producing leaves where the
protected attribute remains representative of the overall
population.  This is a lightweight structural fairness
constraint — the model can still predict on other variables
without building discriminatory partitions.

**Confound suppression.**
In observational data a confounding variable (e.g. hospital site
in a multi-site medical study) may dominate splits even though
the goal is to learn patient-level patterns.  Inverting impurity
on the confounder forces the tree to find splits driven by other
variables while keeping the confounder mixed in every leaf.

**Balanced stratification for downstream tasks.**
If you need per-leaf statistics that should be computed over a
representative distribution of some grouping variable (e.g.
product category), inversion ensures each leaf retains a mix of
all groups rather than splitting them apart.


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


Variable Settings
-----------------

Every variable carries a ``settings`` dictionary populated from
class-level defaults and overridden by constructor arguments.
Settings are accessible as regular attributes:

.. code-block:: python

    v = NumericVariable('x', blur=0.1, precision=0.05)
    v.blur        # 0.1
    v.precision   # 0.05
    v.settings    # {'min_impurity_improvement': 0,
                  #  'blur': 0.1, 'max_std_lbl': 0.0,
                  #  'precision': 0.05}

Settings are included in equality checks and hashing, so two
variables with the same name and domain but different settings
are considered distinct.  Settings are also preserved through
JSON and pickle serialization.


Serialization
-------------

All variables and variable maps support JSON and pickle
round-trips:

.. code-block:: python

    import json
    import pickle
    from jpt.variables import Variable

    # JSON
    data = json.dumps(color.to_json())
    restored = Variable.from_json(json.loads(data))
    assert color == restored

    # Pickle
    restored = pickle.loads(pickle.dumps(color))
    assert color == restored

The JSON representation includes the variable type
(``'numeric'``, ``'symbolic'``, ``'integer'``), name, serialized
domain, and settings.  ``Variable.from_json()`` dispatches to
the correct subclass based on the ``type`` field.


.. seealso::

    :py:mod:`jpt.variables` — full API reference.

    :doc:`howto_classification` — using symbolic targets for
    classification.

    :doc:`howto_regression` — working with numeric targets.
