How to Use Split Validation
===========================

Split validation is a learning-time regulariser that lets you
withhold a subset of training rows from being used as *candidate
split points* while still letting them influence impurity scoring.
It is useful when you suspect that pathological x-values in the
training data (outliers, near-duplicates, or measurement noise)
would otherwise drive the tree to make splits that do not
generalise.

Two complementary mechanisms are available:

* :doc:`split_validation_mask <howto_split_validation>` — marks
  each row as either a training or an evaluation sample;
* :doc:`split_validation_mode <howto_split_validation>` — controls
  which rows contribute to the impurity score at each candidate
  split.


Basic Usage
-----------

Pass a boolean or ``uint8`` mask to :py:meth:`jpt.trees.JPT.fit`.
``True`` (or ``1``) marks a row whose feature value may serve as a
split candidate; ``False`` (``0``) marks a row that is excluded
from the candidate set but still contributes to target statistics.

.. code-block:: python

    import numpy as np
    import pandas as pd
    from jpt.trees import JPT
    from jpt.variables import NumericVariable, SymbolicVariable
    from jpt.distributions import Bool

    rng = np.random.RandomState(0)
    n = 1000
    x = rng.uniform(0, 1, n)
    y = x > 0.5
    df = pd.DataFrame({'x': x, 'y': y})

    # Hold 30 % out as evaluation rows.
    mask = rng.rand(n) < 0.7   # True = training

    xvar = NumericVariable('x')
    yvar = SymbolicVariable('y', Bool)
    jpt = JPT(
        variables=[xvar, yvar],
        targets=[yvar],
        min_samples_leaf=20,
    )
    jpt.fit(df, split_validation_mask=mask)

With no other arguments, the evaluation rows are treated under
``split_validation_mode='both'`` (the default): their target
values are included in the impurity score at every candidate
split, but their x-coordinates are *not* tried as split points.


Choosing a Mode
---------------

``split_validation_mode`` determines which rows contribute to the
target impurity calculation at each split:

``'both'`` (default)
    All rows contribute to impurity.  Equivalent to a classic
    validation-hold-out: the training rows define the candidate
    splits, but every row tells the optimiser how good each
    candidate is.

``'training'``
    Only training rows contribute to impurity.  The evaluation
    rows act purely as a *don't split on these x-values* signal.

``'evaluation'``
    Only evaluation rows contribute to impurity.  The tree is
    scored *exclusively* on held-out rows; training rows propose
    splits and nothing else.  Works well when the training set has
    many near-duplicate or extreme x-values that you want to
    suggest candidate boundaries but not vote on quality.


``min_eval_samples`` — Require a Minimum of Held-out Rows per Child
-------------------------------------------------------------------

When ``split_validation_mode='evaluation'`` is active the
impurity is scored on a smaller set than the training set.
Splits that leave very few evaluation rows on one side yield
unreliable impurity estimates.  Setting ``min_eval_samples`` in
the :py:class:`jpt.trees.JPT` constructor rejects any candidate
split where either child partition contains fewer than
``min_eval_samples`` evaluation rows:

.. code-block:: python

    jpt = JPT(
        variables=[xvar, yvar],
        targets=[yvar],
        min_samples_leaf=20,
        min_eval_samples=10,      # int: absolute count
    )
    jpt.fit(df, split_validation_mask=mask,
            split_validation_mode='evaluation')

As with ``min_samples_leaf``, a ``float`` in :math:`(0, 1)` is
interpreted as a fraction of the *total* training rows:

.. code-block:: python

    JPT(..., min_eval_samples=0.05)   # 5 % of all rows

``min_eval_samples=0`` (the default) disables the check.
``min_eval_samples`` is ignored for modes other than
``'evaluation'``.


Serialisation
-------------

Both ``min_eval_samples`` and the resulting tree structure are
preserved by :py:meth:`jpt.trees.JPT.to_json` /
:py:meth:`jpt.trees.JPT.from_json`.  The split-validation mask and
mode are *learning-time* parameters only — they are not stored in
the fitted model and do not affect inference.


Troubleshooting
---------------

**All splits rejected, tree ends up with a single leaf.**
    ``min_eval_samples`` is too large for your evaluation set size.
    If you have 200 evaluation rows and set ``min_eval_samples=60``,
    no split can leave both sides with 60+ evaluation rows unless
    the tree is nearly balanced.  Reduce the value.

**Training with a mask is much slower than without.**
    The evaluation-only path requires a second pass over the
    target statistics per candidate split.  For large datasets,
    ``split_validation_mode='both'`` (the default) is the fastest
    option.

**"Mask length must equal number of samples" error.**
    The mask is row-aligned with the ``data`` argument to
    ``fit()`` after any preprocessing (dropping NaN rows,
    etc.).  Build the mask from the cleaned DataFrame, not from
    the raw input.

**"Mask must contain at least one training sample" error.**
    At least one row needs ``mask[i] == True`` so the tree has
    candidate split points to choose from.
