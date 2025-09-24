import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import dnutils
from jpt.trees import JPT
from jpt.variables import infer_from_dataframe

logger = logging.getLogger('/banana')
start = datetime.now()
d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-banana')
Path(d).mkdir(parents=True, exist_ok=True)
prefix = f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-banana-FOLD-'
data = variables = kf = None
data_train = data_test = []


def preprocess_banana():
    logger.info('Trying to load dataset from local file...')
    f_csv = '../examples/data/banana.csv'
    src = 'https://www.kaggle.com/saranchandar/standard-classification-banana-dataset'

    if not os.path.exists(f_csv):
        logger.error(f'The file containing this dataset is not in the repository, as it is very large.\nPlease download it from {src} (login required).')
        sys.exit(-1)

    try:
        data = pd.read_csv(f_csv, sep=',').fillna(value='???')
        logger.info(f'Success! Loaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
    except pd.errors.ParserError:
        logger.error('Could not parse file. Please check csv and try again.')
        sys.exit(-1)

    return data


def main(*args):
    data = preprocess_banana()
    variables = infer_from_dataframe(data, scale_numeric_types=True)
    d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-banana')
    Path(d).mkdir(parents=True, exist_ok=True)

    tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    tree.learn(columns=data.values.T)
    tree.save(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-banana.json'))
    tree.plot(title='airline', directory=d, view=False)
    logger.info(tree)


if __name__ == '__main__':
    main()
