"""Large-scale data: airline departure delay patterns.

Loads a large airline dataset and learns a JPT over
flight attributes (day of week, departure time,
distance, carrier, origin, destination). The model is
then evaluated using likelihood on a held-out sample.

Demonstrates:
    - Large dataset handling with subsampling
    - Category dtype for efficient symbolic encoding
    - ``likelihood()`` for model evaluation
    - ``save()`` and ``plot()`` for model persistence
"""
import logging
import os
import sys
import tempfile

import pandas as pd

from jpt.trees import JPT
from jpt.variables import infer_from_dataframe


logger = logging.getLogger(__name__)


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def preprocess_airline():
    """Load the airlines dataset from a local file or
    download it from OpenML.

    :returns: the loaded DataFrame
    """
    local_src = os.path.join(
        _DATA_DIR,
        'airlines_train_regression_10000000.csv'
    )
    try:
        logger.info(
            'Trying to load dataset from local file...'
        )
        data = pd.read_csv(
            local_src,
            delimiter=',',
            skip_blank_lines=True,
            header=0,
            quotechar="'"
        )
    except FileNotFoundError:
        remote_src = (
            'https://www.openml.org/data/get_csv/'
            '22044760/'
            'airlines_train_regression_10000000.csv'
        )
        logger.warning(
            'Local file not found. Downloading from '
            '%s ...', remote_src
        )
        try:
            data = pd.read_csv(
                remote_src,
                delimiter=',',
                skip_blank_lines=True,
                quoting=0,
                header=0
            )
            data.to_csv(local_src, index=False)
            logger.info(
                'Downloaded dataset: %d instances, '
                '%d features',
                data.shape[0], data.shape[1]
            )
        except pd.errors.ParserError:
            logger.error(
                'Could not download and/or parse file. '
                'Please download it manually.'
            )
            return None

    return data


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT from the airline dataset and evaluate
    it.

    :param visualize: whether to show interactive plots
    """
    all_data = preprocess_airline()
    if all_data is None:
        return

    # Select relevant features
    all_data = all_data[[
        'DayOfWeek', 'CRSDepTime', 'Distance',
        'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest',
    ]]

    # Subsample 10% for training
    data = all_data.sample(frac=0.1)

    # Infer variable types from the full dataset
    variables = infer_from_dataframe(
        all_data, scale_numeric_types=True
    )
    out_dir = tempfile.mkdtemp(prefix='jpt-airline-')

    # Set up category dtypes for symbolic columns
    catcols = data.select_dtypes(['object']).columns
    data[catcols] = data[catcols].astype('category')
    for col, var in zip(
        catcols,
        [v for v in variables if v.symbolic]
    ):
        data[col] = data[col].cat.set_categories(
            var.domain.labels.values()
        )

    # Learn the JPT
    tree = JPT(
        variables=variables,
        min_samples_leaf=int(
            data.shape[0] * 0.1 / len(variables)
        )
    )
    tree.learn(data, verbose=True)
    tree.save(
        os.path.join(out_dir, 'airline.json')
    )
    tree.plot(
        title='Airline Departure Delays',
        directory=out_dir,
        view=visualize,
        verbose=True,
        plotvars=tree.variables
    )
    logger.info(tree)

    # Evaluate likelihood on a test sample
    logger.info('Computing likelihood...')
    test_data = all_data.sample(frac=.1)
    print(
        tree.likelihood(
            test_data,
            verbose=True,
            single_likelihoods=False
        )
    )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(visualize=True)
