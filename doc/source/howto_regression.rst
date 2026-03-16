How to Predict Continuous Values with JPTs
==========================================

JPTs can be used for regression by querying the posterior distribution of
a continuous target variable given observed feature values. Unlike
point-estimate regressors, a JPT returns a full probability distribution
over the target, which allows uncertainty quantification out of the box.

Problem Setup
-------------

Prepare a DataFrame with feature columns and one or more numeric target
columns:

.. code-block:: python

    import pandas as pd
    import sklearn.datasets
    from jpt.variables import infer_from_dataframe
    from jpt.trees import JPT

    boston = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(boston.data, columns=boston.feature_names)
    df['MedHouseVal'] = boston.target

    variables = infer_from_dataframe(df)
    varnames = {v.name: v for v in variables}

Training a Discriminative JPT
------------------------------

Pass a ``targets`` list to concentrate splits on the target variable:

.. code-block:: python

    model = JPT(
        variables,
        targets=[varnames['MedHouseVal']],
        min_samples_leaf=0.05
    )
    model.fit(df)

Point Predictions via Expectation
----------------------------------

:py:meth:`~jpt.trees.JPT.expectation` returns the conditional mean of
the target given feature evidence:

.. code-block:: python

    evidence = {
        'MedInc':   [5.0, 6.0],
        'HouseAge': [20.0, 30.0],
    }

    result = model.expectation(
        variables=[varnames['MedHouseVal']],
        evidence=evidence
    )
    print(f"E[MedHouseVal | evidence] = {result[varnames['MedHouseVal']]:.3f}")

Full Posterior Distribution
----------------------------

:py:meth:`~jpt.trees.JPT.posterior` returns the conditional
distribution as a quantile-based PDF object.  Use it when you need
more than a point estimate:

.. code-block:: python

    import matplotlib.pyplot as plt
    import numpy as np

    post = model.posterior(
        variables=[varnames['MedHouseVal']],
        evidence=evidence
    )
    dist = post[varnames['MedHouseVal']]

    xs = np.linspace(dist.ppf(.01), dist.ppf(.99), 300)
    plt.plot(xs, [dist.pdf(x) for x in xs])
    plt.xlabel('MedHouseVal')
    plt.ylabel('Density')
    plt.title('Posterior distribution')
    plt.show()

Evaluating RMSE
---------------

Iterate over a held-out test set and compare the predicted mean to the
ground-truth target value:

.. code-block:: python

    import sklearn.model_selection
    import math

    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, random_state=0
    )
    model.fit(train_df)

    squared_errors = []
    for _, row in test_df.iterrows():
        evidence = {col: float(row[col]) for col in boston.feature_names}
        result = model.expectation(
            [varnames['MedHouseVal']],
            evidence=evidence
        )
        pred = result[varnames['MedHouseVal']]
        squared_errors.append((pred - row['MedHouseVal']) ** 2)

    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    print(f'RMSE: {rmse:.4f}')

.. seealso::

    :doc:`notebooks/tutorial_regression` — a worked regression
    analysis with visualisations.

    :doc:`notebooks/tutorial_reasoning` — full walk-through of all
    query types including ``posterior`` and ``expectation``.
