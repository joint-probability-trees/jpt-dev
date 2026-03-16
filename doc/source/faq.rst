Frequently Asked Questions
==========================

General
-------

**What is the difference between Sum-Product Networks (SPNs) and JPTs?**
    SPNs define *dependency* trees where edges between variables indicate
    a direct probabilistic influence.  JPTs define *computational* trees
    that are a mixture of local factorised distributions.  In an SPN the
    structural assumptions are fixed before learning; in a JPT the
    partitioning of the data space is inferred from data, making JPTs
    non-parametric and free of prior structural assumptions.

**Do I need to specify variable types manually?**
    No. :py:func:`~jpt.variables.infer_from_dataframe` inspects the
    DataFrame column dtypes and creates the appropriate variable type
    automatically (``NumericVariable`` for float/int columns,
    ``SymbolicVariable`` for object/category columns).  You only need to
    construct variables manually if you want fine-grained control over
    the domain or resolution.

**Can JPTs handle missing values?**
    Not directly during training.  Drop or impute missing values before
    calling ``fit()``.  During inference, simply omit the variable from
    the ``evidence`` dict — marginalisation is exact and handles
    unobserved variables correctly.

Training
--------

**What does ``min_samples_leaf`` control?**
    It sets the minimum number of training samples required to create a
    leaf.  Values between 0 and 1 are treated as fractions of the
    training set size.  Smaller values allow deeper, more expressive
    trees; larger values produce simpler, smoother models.  Start with
    ``0.01``–``0.05`` and tune using cross-validation or held-out
    likelihood.

**What is the difference between generative and discriminative mode?**
    In *generative* mode (default) the tree is split to maximise
    information gain over all variables simultaneously.  The resulting
    model represents the full joint distribution :math:`P(\mathcal{X})`.
    In *discriminative* mode (``targets=[...]``) splits are scored only
    on the target variables, which gives better predictive accuracy for
    classification and regression at the cost of a less faithful joint
    model.

**How do I avoid overfitting?**
    Increase ``min_samples_leaf`` or set ``min_impurity_improvement``
    to a small positive value (e.g. ``1e-4``).  You can also use
    ``max_leaves`` to hard-cap the number of leaves.

Inference
---------

**What does ``model.infer()`` return?**
    A scalar float: the (conditional) probability of the query given the
    evidence.  For a marginal query (no evidence) this is
    :math:`P(Q)`.  For a conditional query it is :math:`P(Q \mid E)`.

**What does ``model.posterior()`` return?**
    A dict mapping each queried variable to a marginal distribution
    object (:py:class:`~jpt.distributions.univariate.Multinomial` for
    symbolic variables, a quantile-based distribution for numeric
    variables).  The distributions are independent conditional on the
    evidence, although the variables may be correlated.

**What happens when evidence is unsatisfiable?**
    ``infer()`` returns ``0.0``.  ``posterior()`` raises a
    ``ValueError``.  Check your evidence ranges before calling
    ``posterior()`` if you are not sure whether the evidence is
    reachable.

**How does MPE differ from posterior expectation?**
    :py:meth:`~jpt.trees.JPT.mpe` returns the *most likely assignment*
    (mode) of all query variables jointly.  The posterior expectation
    (:py:meth:`~jpt.trees.JPT.expectation`) returns the *mean* of each
    variable's marginal distribution independently.  For multimodal
    distributions they can differ substantially.

Performance
-----------

**My model is slow to query.  What should I do?**
    Use ``min_samples_leaf`` to limit the number of leaves.  For batch
    queries consider wrapping evidence rows in a loop over a pre-built
    ``varnames`` lookup dict to avoid repeated string lookups.  The
    ``bind()`` method also pre-computes an evidence-conditioned subtree
    that can be reused for multiple downstream queries.

**Can I train on very large datasets?**
    Training is O(n log n) per variable per split level.  For datasets
    above a few million rows consider sub-sampling for tree construction
    while keeping the full data for leaf distribution fitting, or use
    ``min_samples_leaf`` with a higher fraction to limit tree depth.
