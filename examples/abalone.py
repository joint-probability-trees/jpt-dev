"""Learning from tabular data: UCI Abalone dataset.

Loads the UCI Abalone dataset and learns a Joint
Probability Tree using automatic variable inference from
the DataFrame. The model's fit is evaluated using
log-likelihood on the training data.

Demonstrates:
    - ``infer_from_dataframe()`` for automatic variable
      type detection
    - ``fit()`` for learning from raw numpy arrays
    - ``likelihood()`` for model evaluation
    - Tree visualization with ``plot()``
"""
import os
import tempfile

import numpy as np
import pandas as pd

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT on the Abalone dataset and evaluate
    its log-likelihood.

    :param visualize: whether to show interactive plots
    """
    # Load the Abalone dataset
    path = os.path.join(_DATA_DIR, 'abalone.data')
    df = pd.read_csv(
        path,
        names=[
            "Sex", "Length", "Diameter", "Height",
            "Whole weight", "Shucked weight",
            "Viscera weight", "Shell weight", "Rings",
        ]
    )

    # Infer variable types from the data
    variables = infer_from_dataframe(
        df, scale_numeric_types=False
    )
    print(f'Inferred {len(variables)} variables:')
    for v in variables:
        print(f'  {v}')

    # Learn the JPT
    tree = JPT(variables, min_samples_leaf=0.02)
    data = df.to_numpy()
    tree.fit(data.copy())
    print('Finished learning.')

    # Evaluate log-likelihood on the training data
    l = tree.likelihood(data)
    l[l == 0] = pow(1 / len(data), len(variables))
    log_likelihood = sum(np.log(l))
    print(f'Log-likelihood: {log_likelihood:.4f}')

    # Plot the learned tree
    out_dir = tempfile.mkdtemp(prefix='jpt-abalone-')
    tree.plot(
        plotvars=variables,
        directory=out_dir,
        view=visualize
    )


if __name__ == '__main__':
    main(visualize=True)
