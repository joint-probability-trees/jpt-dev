Changelog
=========

1.1.0
-----

This release adds dependency discovery and xi-based pruning to the JPT
learning pipeline.

New Features
~~~~~~~~~~~~

*Dependency discovery*

- Added ``jpt.base.correlation`` package with a standalone
  implementation of Chatterjee's xi correlation coefficient
  (``xi_correlation``, ``xi_correlation_matrix``).
- Added ``jpt.learning.dependency`` package with the
  ``DependencyDiscovery`` abstract base class and
  ``XiDependencyDiscovery``, which computes xi for all feature-target
  pairs and retains only statistically significant dependencies.
- The ``dependencies`` parameter of ``JPT.__init__`` now accepts
  ``DependencyDiscovery`` instances in addition to ``None`` and
  explicit dictionaries. Discovery strategies are re-invoked on each
  ``learn()`` call and preserved during JSON serialization.

*Pruning*

- Added ``jpt.learning.pruning`` package with
  ``XiPruningCriterion``, a ``prune_or_split`` callback that stops
  splitting when no feature-target pair shows significant functional
  dependence in the current partition.
- The ``prune_or_split`` callback signature is extended from
  ``(jpt, partition, indices)`` to ``(jpt, partition, indices, data)``,
  eliminating the need to access process-local state.

*Documentation*

- Added how-to guide for dependency discovery and xi pruning with
  mathematical background, worked examples, and extensibility guide.
- Added Chatterjee (2021), Dalitz et al. (2024), and Shi et al. (2022)
  to the bibliography.
- Added ``xi_pruning.py`` example demonstrating both features.

Bug Fixes
~~~~~~~~~

- Fixed outdated ``important_datastructures.ipynb`` tutorial notebook:
  replaced removed ``list2interval`` and ``RealSet`` with current
  ``ContinuousSet`` and ``UnionSet`` API; corrected
  ``infer_from_dataframe`` import path.

Test Suite
~~~~~~~~~~

- Added 24 test cases covering xi correlation properties, dependency
  discovery (structure recovery, serialization, JPT integration), and
  pruning behavior (noise sensitivity, alpha monotonicity).


1.0.0 (unreleased)
-------------------

This release contains substantial new features, bug fixes, and infrastructure
improvements relative to the last ``0.1.x`` series (``0.1.41``).

New Features
~~~~~~~~~~~~

*Inference*

- Corrected k-MPE implementation: max-heap extraction, proper
  ``leaf.prior`` scaling, and quadratic pruned-node requeue for
  multi-leaf correctness.
- Added support for specifying numeric query intervals as Python lists
  in :py:meth:`jpt.distributions.univariate.Numeric.p`.

*Learning*

- Added optional progress bar for monitoring the learning progress.
- Added support for custom pruning criteria during JPT construction.
- Runaway tree growth is now prevented when no ``max_std`` constraints
  are set.

*Parallel processing*

- Added multicore module with a customised process-pool class that
  inherits thread-local state in child processes.
- Added support for parallel likelihood computation over multiple cores.
- Added support for parallel data preprocessing.
- Added support for parallel learning of prior distributions per leaf.
- Added support for parallel rendering of JPT leaves.

*Serialization*

- Added ``JPT.dump()`` / ``JPT.load()`` and ``JPT.dumps()`` /
  ``JPT.loads()`` for JSON-based model persistence.
- Added ``__getstate__()`` and ``__setstate__()`` to ``IntSet`` and
  ``NumberSet`` for pickle support.

*Data structures*

- Added ``IntSet``: a new Cython interval type for integer domains,
  replacing ad-hoc set arithmetic in integer distributions.
- Added ``RealSet.min`` property.
- Added ``QuadraticFunction`` support in ``PiecewiseFunction``,
  including vertex-form construction and maximisation.
- Added ``PiecewiseFunction.__neg__()`` and corresponding
  ``Function.__neg__()`` interface method.
- Added ``PLFApproximator`` test coverage.

*Visualisation*

- Added plotting support for Gaussian distributions in the Matplotlib
  and Plotly rendering engines.
- Added support for fancy tree printing with Unicode box-drawing
  characters via ``anytree``.

Bug Fixes
~~~~~~~~~

*Inference and distributions*

- Fixed numeric imprecision in :py:meth:`jpt.trees.JPT.posterior`.
- Fixed :py:meth:`jpt.trees.JPT.expectation` method signature.
- Fixed :py:meth:`jpt.trees.JPT.encode` function.
- Fixed error tolerance in ``Numeric.pdf_to_cdf()`` and corrected
  handling of Dirac impulse contributions.
- Fixed sampling bug in ``UnionSet._sample`` (memoryview assignment).
- Fixed ``ContinuousSet._sample`` memoryview assignment causing
  ``TypeError``.
- Fixed plotting of unbounded integer distributions.

*Learning*

- Fixed memory leak and logging errors in the C4.5 learning algorithm.
- Fixed ``infer_from_dataframe()`` variable type checks and suppressed
  erroneous deduplication of domain values.
- Fixed ``IntegerVariable.assignment2set()`` value conversion and
  compatibility with the new ``IntSet`` class.
- Fixed ``JPT._preprocess_data()`` DataFrame value transformation.
- Fixed ``df.copy()`` before in-place manipulation to prevent
  side-effects on the caller's DataFrame.

*Parallelism and concurrency*

- Fixed multicore likelihood computation and ``multicore=None``
  handling.
- Fixed signal handling to main thread only.
- Fixed repetitive pickling of the JPT instance in parallel leaf
  plotting.
- Fixed ``ImportError`` when importing the custom Pool class.
- Fixed thread-local JPT storage for worker processes.

*Build and packaging*

- Fixed Cython version detection in ``pyximporter.py``.
- Fixed import issue with Cython >= 3.0.11.
- Fixed concurrent Cython compilation.
- Fixed relative data paths in tests and examples.
- Pinned ``kaleido < 1.0`` to prevent incompatible API changes.

Infrastructure
~~~~~~~~~~~~~~

- Migrated version management to ``setuptools-scm``; version is now
  derived automatically from git tags.
- Consolidated all build and dependency configuration into
  ``pyproject.toml``.
- Made ``graphviz``, ``fglib``, ``factorgraph``, and ``mlflow``
  optional dependencies with lazy imports.
- Added ``typing-extensions`` as an explicit dependency.
- Set Python 3.11 as the default build target.
- Modernised type-hint syntax throughout ``trees.py`` and
  ``variables.py``.
- Migrated all ``print``-based logging to Python's standard
  ``logging`` module.
- Lazy-import plotting engines in distribution modules to avoid
  importing heavy optional dependencies at module load time.
- Updated GitHub Actions CI workflows.

Test Suite
~~~~~~~~~~

- Restructured the test suite into per-module files under
  ``test/distributions/``, ``test/base/``, ``test/variables/``, and
  ``test/learning/``.
- Added docstrings to all test methods.
- Added placeholder test cases for the plotting engine.

Previous Releases
-----------------

For changes in the ``0.1.x`` series please refer to the git history::

    git log 0.1.41 --oneline
