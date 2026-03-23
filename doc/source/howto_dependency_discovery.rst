How to Use Dependency Discovery and Xi Pruning
===============================================

By default, JPTs assume that every feature can influence every target
variable.  In high-dimensional datasets this leads to unnecessarily
large trees, because the learning algorithm may chase spurious
relationships between unrelated variables.  ``pyjpt`` provides two
mechanisms to address this: **dependency discovery**, which identifies
which feature–target pairs are genuinely related before tree
construction, and **xi-based pruning**, which stops splitting when
no statistically significant dependence remains in a partition.

Both mechanisms are built on **Chatterjee's** :math:`\xi` **correlation
coefficient** :cite:`chatterjee2021new`, a rank-based measure of
functional dependence with several unique properties.


Mathematical Background
-----------------------

Chatterjee's :math:`\xi` coefficient
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given :math:`n` paired observations :math:`(X_1, Y_1), \ldots,
(X_n, Y_n)`, sort the pairs by :math:`X` so that
:math:`X_{(1)} \leq \cdots \leq X_{(n)}` and let :math:`r_{(i)}`
denote the rank of :math:`Y_{(i)}` among all :math:`Y` values.  Then

.. math::

    \xi_n(X, Y) = 1 - \frac{3 \sum_{i=1}^{n-1}
    |r_{(i+1)} - r_{(i)}|}{n^2 - 1}.

**Intuition:**  If :math:`Y` is a function of :math:`X`, sorting by
:math:`X` also sorts :math:`Y`, so consecutive ranks differ by 1 and
:math:`\xi_n \to 1`.  If :math:`X` and :math:`Y` are independent,
the ranks form a random permutation and :math:`\xi_n \to 0`.

**Key properties:**

- :math:`\xi = 0` if and only if :math:`X` and :math:`Y` are
  independent.
- :math:`\xi = 1` if and only if :math:`Y` is a measurable function
  of :math:`X`.
- **Asymmetric:** :math:`\xi(X, Y) \neq \xi(Y, X)` in general.
  This is a feature: it measures how well :math:`Y` can be predicted
  from :math:`X`, which is exactly the question a split on :math:`X`
  answers.
- **Distribution-free:** no assumptions about the distributions of
  :math:`X` or :math:`Y`.
- **Detects any dependence:** unlike Pearson (linear) or Spearman
  (monotonic), :math:`\xi` detects arbitrary functional relationships,
  including periodic or many-to-one mappings.
- :math:`\mathcal{O}(n \log n)` computational complexity.

Significance test
^^^^^^^^^^^^^^^^^

Under the null hypothesis that :math:`X` and :math:`Y` are independent
(with :math:`Y` continuous), the asymptotic distribution of :math:`\xi`
is known :cite:`chatterjee2021new`:

.. math::

    \sqrt{n} \, \xi_n \xrightarrow{d}
    \mathcal{N}\!\left(0, \tfrac{2}{5}\right).

This means that a z-test can be used to decide whether an observed
:math:`\xi` value is statistically significant.  Given a significance
level :math:`\alpha`, the null hypothesis of independence is rejected
when

.. math::

    \frac{\sqrt{n} \, \xi_n}{\sqrt{2/5}} > z_{1-\alpha},

where :math:`z_{1-\alpha}` is the :math:`(1-\alpha)`-quantile of the
standard normal distribution.  Setting :math:`\alpha = 0.05` means:
"only accept a dependence if there is less than a 5% chance it arose
by coincidence."


Dependency Discovery
--------------------

Dependency discovery computes :math:`\xi` for all feature–target
pairs before tree construction and retains only those pairs where
the relationship is statistically significant.  This restricts the
impurity computation during learning to genuine dependencies,
preventing the tree from wasting splits on unrelated variables.

Basic usage
^^^^^^^^^^^

Pass an :py:class:`~jpt.learning.dependency.xi.XiDependencyDiscovery`
instance as the ``dependencies`` parameter:

.. code-block:: python

    from jpt.trees import JPT
    from jpt.learning.dependency import (
        XiDependencyDiscovery,
    )

    model = JPT(
        variables,
        targets=[target_var],
        dependencies=XiDependencyDiscovery(
            alpha=0.05
        ),
        min_samples_leaf=0.01,
    )
    model.fit(data)

After learning, ``model.dependencies`` contains the discovered
dependency map.  Inspect it to see which features were identified
as relevant:

.. code-block:: python

    for feat, targets in model.dependencies.items():
        names = [t.name for t in targets]
        print(f'{feat.name} -> {names}')

The ``alpha`` parameter controls the significance level:

- **Smaller** :math:`\alpha` (e.g. 0.01): stricter, fewer
  dependencies retained, more compact trees.
- **Larger** :math:`\alpha` (e.g. 0.10): more permissive, retains
  weaker relationships.

Persistence
^^^^^^^^^^^

The discovery strategy is preserved during serialization.  When a
model is saved and loaded, calling ``fit()`` again will re-discover
dependencies from the new data:

.. code-block:: python

    model.save('model.json')
    restored = JPT.load('model.json')

    # Re-learning uses the same discovery strategy
    restored.fit(new_data)

Backward compatibility
^^^^^^^^^^^^^^^^^^^^^^

The ``dependencies`` parameter continues to accept ``None`` (fully
connected, the default) and explicit dictionaries:

.. code-block:: python

    # Default: every feature depends on every target
    model = JPT(variables, dependencies=None)

    # Manual: only X1 influences Y
    model = JPT(
        variables,
        dependencies={x1_var: [y_var]}
    )

    # Automatic: discover from data
    model = JPT(
        variables,
        dependencies=XiDependencyDiscovery(alpha=0.05)
    )


Xi-Based Pruning
----------------

Even when the global dependency structure is known, the strength of
a relationship may vary across subregions of the data.  Xi-based
pruning tests at each candidate split whether there is still
significant functional dependence in the current partition.  If not,
the node becomes a leaf.

Using the pruning criterion
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass an :py:class:`~jpt.learning.pruning.xi.XiPruningCriterion`
instance as the ``prune_or_split`` parameter of ``fit()``:

.. code-block:: python

    from jpt.learning.pruning import XiPruningCriterion

    model = JPT(
        variables,
        targets=[target_var],
        min_samples_leaf=0.01,
    )
    model.fit(
        data,
        prune_or_split=XiPruningCriterion(alpha=0.05),
    )

The ``alpha`` parameter has the same interpretation as above: the
probability of a false split (splitting when there is no genuine
dependence).

The ``min_n`` parameter (default 30) sets the minimum partition size
for the test.  Below this threshold, the xi test is not applied and
conventional stopping rules take effect.  This handles the fact that
:math:`\xi` requires a minimum sample size for reliable inference
(typically :math:`n \gtrsim 250` for the asymptotic theory
:cite:`chatterjee2021new`, but the test is still informative for
smaller :math:`n` :cite:`dalitz2024bias`).


Combining Both
--------------

Dependency discovery and xi pruning are complementary: the former
reduces the set of feature–target pairs globally, the latter adapts
the stopping criterion locally.  For maximum effect, use both:

.. code-block:: python

    from jpt.trees import JPT
    from jpt.learning.dependency import (
        XiDependencyDiscovery,
    )
    from jpt.learning.pruning import XiPruningCriterion

    model = JPT(
        variables,
        targets=[target_var],
        dependencies=XiDependencyDiscovery(
            alpha=0.05
        ),
        min_samples_leaf=0.01,
    )
    model.fit(
        data,
        prune_or_split=XiPruningCriterion(alpha=0.05),
    )

    print(f'Leaves: {len(model.leaves)}')


Worked Example
--------------

Consider a dataset with three variables: :math:`X_1`, :math:`X_2`
(both uniform noise), and :math:`Y = X_1^2 + \varepsilon` where
:math:`\varepsilon \sim \mathcal{N}(0, 1.5)`.  By construction,
:math:`Y` depends on :math:`X_1` but not on :math:`X_2`:

.. code-block:: python

    import numpy as np
    from pandas import DataFrame
    from jpt.distributions import Numeric
    from jpt.variables import NumericVariable
    from jpt.trees import JPT
    from jpt.base.correlation import xi_correlation
    from jpt.learning.dependency import (
        XiDependencyDiscovery,
    )
    from jpt.learning.pruning import XiPruningCriterion

    np.random.seed(42)
    n = 2000
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    y = x1 ** 2 + np.random.normal(0, 1.5, n)

    df = DataFrame({'X1': x1, 'X2': x2, 'Y': y})
    vx1 = NumericVariable('X1', Numeric, precision=0.1)
    vx2 = NumericVariable('X2', Numeric, precision=0.1)
    vy = NumericVariable('Y', Numeric, precision=0.1)

    # Check xi values
    print(f'xi(X1, Y) = {xi_correlation(x1, y):.3f}')
    print(f'xi(X2, Y) = {xi_correlation(x2, y):.3f}')

    # Standard JPT
    tree_std = JPT(
        [vx1, vx2, vy],
        targets=[vy],
        min_samples_leaf=0.01
    )
    tree_std.fit(df)

    # JPT with dependency discovery + xi pruning
    tree_xi = JPT(
        [vx1, vx2, vy],
        targets=[vy],
        dependencies=XiDependencyDiscovery(alpha=0.05),
        min_samples_leaf=0.01
    )
    tree_xi.fit(
        df,
        prune_or_split=XiPruningCriterion(alpha=0.05)
    )

    print(f'Standard:  {len(tree_std.leaves)} leaves')
    print(f'Xi-aware:  {len(tree_xi.leaves)} leaves')

Expected output::

    xi(X1, Y) = 0.232
    xi(X2, Y) = -0.009
    Standard:  77 leaves
    Xi-aware:  5 leaves

The standard tree produces 77 leaves, many from splits on the
irrelevant variable :math:`X_2`.  The xi-aware tree correctly
identifies :math:`X_1` as the only relevant feature and stops
splitting once the signal in each partition is exhausted, yielding
only 5 leaves.


Extending with Custom Discovery Strategies
-------------------------------------------

The dependency discovery mechanism is extensible.  To implement a
custom strategy, subclass
:py:class:`~jpt.learning.dependency.base.DependencyDiscovery`
and implement three methods:

.. code-block:: python

    from jpt.learning.dependency.base import (
        DependencyDiscovery,
    )

    class MyDiscovery(DependencyDiscovery):

        def __init__(self, threshold=0.1):
            self.threshold = threshold

        def __call__(
                self, data, features,
                targets, variables
        ):
            # Return a dict mapping each feature
            # to its dependent targets
            ...

        def to_json(self):
            return {
                'type': self.__class__.__name__,
                'threshold': self.threshold,
            }

        @classmethod
        def from_json(cls, data):
            return cls(threshold=data['threshold'])

Subclasses are automatically registered for deserialization, so
``DependencyDiscovery.from_json()`` will dispatch to the correct
class based on the ``'type'`` key.


References
----------

.. bibliography:: refs.bib
   :filter: key in {"chatterjee2021new", "dalitz2024bias", "shi2022power"}
   :style: unsrt
