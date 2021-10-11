import multiprocessing
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import dnutils
from jpt.learning.distributions import Numeric, SymbolicType, NumericType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

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
        src = '../examples/data/airlines_train_regression_10000000.csv'
        logger.info('Trying to load dataset from local file...')
        data = pd.read_csv(src, delimiter=',', sep=',', skip_blank_lines=True, header=0, quotechar="'")
    except FileNotFoundError:
        src = 'https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.csv'
        logger.warning(f'The file containing this dataset is not in the repository, as it is very large.\nI will try downloading file {src} now...')
        try:
            data = pd.read_csv(src, delimiter=',', sep=',', skip_blank_lines=True, quoting=0, header=0)
            logger.info(f'Success! Downloaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
        except pd.errors.ParserError:
            logger.error('Could not download and/or parse file. Please download it manually and try again.')
            sys.exit(-1)

    data = data.sample(frac=0.0001)
    logger.info('creating types and variables...')
    DepDelay_type = NumericType('DepDelay_type', np.array(data['DepDelay']))
    Month_type = NumericType('Month_type', np.array(data['Month']))
    DayofMonth_type = NumericType('DayofMonth_type', np.array(data['DayofMonth']))
    DayOfWeek_type = NumericType('DayOfWeek_type', np.array(data['DayOfWeek']))
    CRSDepTime_type = NumericType('CRSDepTime_type', np.array(data['CRSDepTime']))
    CRSArrTime_type = NumericType('CRSArrTime_type', np.array(data['CRSArrTime']))
    UniqueCarrier_type = SymbolicType('UniqueCarrier_type', data['UniqueCarrier'].unique())
    Origin_type = SymbolicType('Origin_type', data['Origin'].unique())
    Dest_type = SymbolicType('Dest_type', data['Dest'].unique())
    Distance_type = NumericType('Distance_type', np.array(data['Distance']))

    DepDelay = NumericVariable('DepDelay', DepDelay_type)
    Month = NumericVariable('Month', Month_type)
    DayofMonth = NumericVariable('DayofMonth', DayofMonth_type)
    DayOfWeek = NumericVariable('DayOfWeek', DayOfWeek_type)
    CRSDepTime = NumericVariable('CRSDepTime', CRSDepTime_type)
    CRSArrTime = NumericVariable('CRSArrTime', CRSArrTime_type)
    UniqueCarrier = SymbolicVariable('UniqueCarrier', UniqueCarrier_type)
    Origin = SymbolicVariable('Origin', Origin_type)
    Dest = SymbolicVariable('Dest', Dest_type)
    Distance = NumericVariable('Distance', Distance_type)

    variables = [DepDelay, Month, DayofMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, Origin, Dest, Distance]
    return data, variables


def main():
    data, variables = preprocess_airline()
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
