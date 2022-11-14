Examples
========

This section provides example usages of JPTs to get you started easily.

MNIST Handwritten Digit Recognition
***********************************

Note: This example can be found in examples/tutorial.ipnyb.

The MNIST Handwritten Digit Recognition problem consists of 8x8 pictures and labels from 0-9 describing the number
that is depicted on a picture.

To get started we first have to load the dataset from sklearn.

.. code-block:: python

    from sklearn.datasets import load_digits

In "regular" machine learning a conditional distribution P(Q|E) is approximated.
However, as the name JPT suggests, we are interested in the **joint** distribution P(Q,E).
Therefore, we have to load all the data (images and labels) in one dataframe.

.. code-block:: python
    dataset = load_digits(as_frame=True)
    df = dataset.data
    df["digit"] = dataset.target

Next we have to create variables that can be used in the JPT package.
Firstly we have to import the necessary functionality. We will try to infer the variables from the dataframe.

.. code-block:: python

    from jpt.variables import infer_from_dataframe
    variables = infer_from_dataframe(df)

The "digit" variable gets recognized as a numeric variable, which is technically the truth. However, the numeric
representation of it is not useful for the representation problem. Therefore, we have to change it to a symbolic
variable. To create a variable we need a type and a name.

.. code-block:: python

    from jpt.variables import SymbolicVariable, SymbolicType

    digit_type = SymbolicType("digit", [str(i) for i in range(10)])
    digit = SymbolicVariable("digit", digit_type)
    variables[-1] = digit

Next we have to create the model. We want the model to only acquire new parameters if it yields to an improvement
in information gain of 0.05.

.. code-block:: python

    import jpt.trees
    model = jpt.trees.JPT(variables, min_impurity_improvement=0.05)

To finish the knowledge acquisition part we have to fit the model. This is done sklearn style.