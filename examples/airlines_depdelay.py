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
    data = data[['DayOfWeek', 'CRSDepTime', 'Distance', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest']]  #
    data = data.sample(frac=0.1)
    variables = infer_from_dataframe(data, scale_numeric_types=True)
    d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-airline')
    Path(d).mkdir(parents=True, exist_ok=True)

    catcols = data.select_dtypes(['object']).columns
    data[catcols] = data[catcols].astype('category')
    for col, var in zip(catcols, [v for v in variables if v.symbolic]):
        data[col] = data[col].cat.set_categories(var.domain.labels.values())

    # tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    # tree = JPT(variables=variables, max_depth=8)
    tree = JPT(variables=variables, min_samples_leaf=int(data.shape[0] * 0.1 / len(variables)))
    tree.learn(columns=data.values.T)
    tree.save(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-airline.json'))
    tree.plot(title='airline', directory=d, view=True)
    logger.info(tree)


if __name__ == '__main__':
    main()
