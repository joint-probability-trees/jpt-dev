How to Classify with JPTs
=========================

JPTs can be used for classification by treating the class label as a
symbolic variable and querying the posterior probability of each class
given the observed features.

Problem Setup
-------------

Prepare a DataFrame with feature columns and a string class-label
column, then let ``pyjpt`` infer the variable types:

.. code-block:: python

    import pandas as pd
    import sklearn.datasets
    from jpt.variables import infer_from_dataframe
    from jpt.trees import JPT

    iris = sklearn.datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = [iris.target_names[t] for t in iris.target]

    variables = infer_from_dataframe(df)
    varnames = {v.name: v for v in variables}

Training a Discriminative JPT
------------------------------

Pass a ``targets`` list to concentrate the tree splits on the class
label.  This gives better classification accuracy than the default
generative mode when prediction is the sole goal:

.. code-block:: python

    model = JPT(
        variables,
        targets=[varnames['species']],
        min_samples_leaf=0.05
    )
    model.fit(df)

Making Predictions
------------------

Query the posterior over the label given feature evidence.  The model
returns a :py:class:`~jpt.distributions.univariate.Multinomial`
distribution; call ``.p(label)`` for a specific class probability or
read ``.probabilities`` for all classes at once:

.. code-block:: python

    evidence = {
        'sepal length (cm)': [5.8, 6.2],
        'petal length (cm)': [4.0, 4.5],
    }

    post = model.posterior(
        variables=[varnames['species']],
        evidence=evidence
    )
    dist = post[varnames['species']]

    for label in varnames['species'].domain:
        print(f'P({label} | evidence) = {dist.p(label):.3f}')

    # Hard prediction
    predicted = max(varnames['species'].domain, key=dist.p)
    print(f'Predicted class: {predicted}')

For a scalar conditional probability of a single class use
:py:meth:`~jpt.trees.JPT.infer` directly:

.. code-block:: python

    p = model.infer(
        query={'species': 'virginica'},
        evidence={'petal width (cm)': [1.8, 2.5]}
    )
    print(f'P(virginica | wide petal) = {p:.3f}')

Evaluating Accuracy
--------------------

Iterate over a held-out test set and compare the predicted class to the
ground truth:

.. code-block:: python

    import sklearn.model_selection

    train_df, test_df = sklearn.model_selection.train_test_split(
        df, test_size=0.2, random_state=0
    )
    model.fit(train_df)

    correct = 0
    for _, row in test_df.iterrows():
        evidence = {col: float(row[col]) for col in iris.feature_names}
        post = model.posterior([varnames['species']], evidence=evidence)
        dist = post[varnames['species']]
        pred = max(varnames['species'].domain, key=dist.p)
        correct += (pred == row['species'])

    print(f'Accuracy: {correct / len(test_df):.1%}')

.. seealso::

    :doc:`notebooks/tutorial_iris` — a worked likelihood analysis
    on the Iris dataset.

    :doc:`notebooks/tutorial_reasoning` — a full walk-through of all
    query types including ``posterior`` and ``infer``.
