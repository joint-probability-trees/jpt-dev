import os
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

import dnutils
from jpt.trees import JPT
from jpt.variables import infer_from_dataframe

# globals
start = datetime.now()
d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-airline')
Path(d).mkdir(parents=True, exist_ok=True)
prefix = f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-airline-FOLD-'
data = variables = kf = None
data_train = data_test = []

dnutils.loggers({'/airline': dnutils.newlogger(dnutils.logs.console,
                                               dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-airline.log')),
                                               level=dnutils.DEBUG)
                 })

logger = dnutils.getlogger('/airline', level=dnutils.DEBUG)


def preprocess_airline():
    try:
        local_src = '../examples/data/airlines_train_regression_10000000.csv'
        logger.info('Trying to load dataset from local file...')
        data = pd.read_csv(local_src, delimiter=',', skip_blank_lines=True, header=0, quotechar="'")
    except FileNotFoundError:
        remote_src = 'https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.csv'
        logger.warning(f'The file containing this dataset is not in the repository, as it is very large.\nI will try downloading file {remote_src} now...')
        try:
            data = pd.read_csv(remote_src, delimiter=',', skip_blank_lines=True, quoting=0, header=0)
            data.to_csv(local_src, index=False)
            logger.info(f'Success! Downloaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
        except pd.errors.ParserError:
            logger.error('Could not download and/or parse file. Please download it manually and try again.')
            sys.exit(-1)

    return data


def main():
    data = preprocess_airline()
    variables = infer_from_dataframe(data, scale_numeric_types=True)
    d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-airline')
    Path(d).mkdir(parents=True, exist_ok=True)

    # tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    tree = JPT(variables=variables, max_depth=8)
    tree.learn(columns=data.values.T)
    tree.save(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-airline.json'))
    tree.plot(title='airline', directory=d, view=False)
    logger.info(tree)


if __name__ == '__main__':
    main()
