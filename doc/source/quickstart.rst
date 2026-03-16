Quick Start
===========

Install ``pyjpt`` from PyPI:

.. code-block:: bash

    pip install pyjpt[matplotlib]

Complete Workflow
-----------------

The example below loads the Iris dataset, fits a JPT, and asks three
kinds of probabilistic questions — all in under 25 lines.

.. code-block:: python

    import pandas as pd
    import sklearn.datasets
    from jpt.variables import infer_from_dataframe
    from jpt.trees import JPT

    # 1 ── Load data
    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]

    # 2 ── Infer variable types and fit
    variables = infer_from_dataframe(df)
    model = JPT(variables, min_samples_leaf=0.1)
    model.fit(df)

    # 3 ── Marginal probability
    p = model.infer(query={'species': 'setosa'})
    print(f'P(setosa) = {p:.3f}')

    # 4 ── Conditional probability  P(setosa | petal length ∈ [1, 2])
    p_cond = model.infer(
        query={'species': 'setosa'},
        evidence={'petal length (cm)': [1.0, 2.0]}
    )
    print(f'P(setosa | petal length ∈ [1,2]) = {p_cond:.3f}')

    # 5 ── Full posterior over species given petal width ≤ 0.5
    post = model.posterior(
        variables=['species'],
        evidence={'petal width (cm)': [0.0, 0.5]}
    )
    for label in model.varnames['species'].domain:
        print(f'  P({label} | narrow petal) = {post[model.varnames["species"]].p(label):.3f}')

    # 6 ── Most probable explanation
    assignment, likelihood = model.mpe(evidence={'species': 'virginica'})
    print(f'MPE (virginica): {assignment[0]}  likelihood={likelihood:.4f}')

How It Works
------------

:py:func:`jpt.variables.infer_from_dataframe` inspects the DataFrame's
column dtypes and creates one variable per column:

* ``float`` / ``int`` columns → :py:class:`~jpt.variables.NumericVariable`
* ``object`` / ``category`` columns → :py:class:`~jpt.variables.SymbolicVariable`

:py:class:`~jpt.trees.JPT` builds a decision-tree partition of the data
space.  Each leaf stores an independent factorised distribution over all
variables.  The ``min_samples_leaf`` parameter controls how fine-grained
that partition becomes — smaller values produce more leaves and a more
expressive model at the cost of higher variance.

Next Steps
----------

* :doc:`introduction` — understand what JPTs are and how they work
* :doc:`guide` — step-by-step tutorials
* :doc:`howto` — task-oriented recipes for classification, regression,
  visualisation, and model persistence
* :doc:`autoapi/index` — full API reference
