"""Binary classification: banana-shaped clusters.

Loads the banana dataset from Kaggle and learns a JPT
for binary classification of banana-shaped clusters.
The example demonstrates handling of large datasets with
automatic variable type inference.

Demonstrates:
    - ``infer_from_dataframe()`` with scale_numeric_types
    - ``save()`` for model serialization
    - Large dataset handling
"""
import logging
import os
import tempfile

import pandas as pd

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


logger = logging.getLogger(__name__)


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT from the banana dataset.

    :param visualize: whether to show interactive plots
    """
    # Load the banana dataset
    f_csv = os.path.join(_DATA_DIR, 'banana.csv')
    src = (
        'https://www.kaggle.com/saranchandar/'
        'standard-classification-banana-dataset'
    )

    if not os.path.exists(f_csv):
        logger.warning(
            'The banana dataset is not in the '
            'repository (too large). Please download '
            'it from %s (login required).', src
        )
        return

    try:
        data = pd.read_csv(
            f_csv, sep=','
        ).fillna(value='???')
        logger.info(
            'Loaded dataset: %d instances, %d features',
            data.shape[0], data.shape[1]
        )
    except pd.errors.ParserError:
        logger.error(
            'Could not parse file. Please check the '
            'CSV and try again.'
        )
        return

    # Infer variables and learn the JPT
    variables = infer_from_dataframe(
        data, scale_numeric_types=True
    )
    out_dir = tempfile.mkdtemp(prefix='jpt-banana-')

    tree = JPT(
        variables=variables,
        min_samples_leaf=data.shape[0] * .01
    )
    tree.learn(data)
    tree.save(
        os.path.join(out_dir, 'banana.json')
    )
    tree.plot(
        title='Banana',
        directory=out_dir,
        view=visualize
    )
    logger.info(tree)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(visualize=True)
