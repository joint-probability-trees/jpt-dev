import multiprocessing
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import dnutils
from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

# globals
start = datetime.now()
d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-Airline')
Path(d).mkdir(parents=True, exist_ok=True)
prefix = f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-Airline-FOLD-'
data = variables = kf = None
data_train = data_test = []

dnutils.loggers({'/airline': dnutils.newlogger(dnutils.logs.console,
                                               dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-Airline.log')),
                                               level=dnutils.DEBUG)
                 })

logger = dnutils.getlogger('/airline', level=dnutils.DEBUG)


def preprocess_airline():
    try:
        logger.info('Trying to load dataset from local file...')
        data = pd.read_csv('../examples/data/airlines_train_regression_10000000.csv')
    except FileNotFoundError:
        src = 'https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.csv'
        logger.warning(f'The file containing this dataset is not in the repository, as it is very large.\nI will try downloading file {src} now...')
        try:
            data = pd.read_csv(src, delimiter=',', sep=',', skip_blank_lines=True, header=0)
            logger.info(f'Success! Downloaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
        except pd.errors.ParserError:
            logger.error('Could not download and/or parse file. Please download it manually and try again.')
            sys.exit(-1)

    data = data.sample(frac=1)

    logger.info('creating types and variables...')
    UniqueCarrier_type = SymbolicType('UniqueCarrier_type', data['UniqueCarrier'].unique())
    Origin_type = SymbolicType('Origin_type', data['Origin'].unique())
    Dest_type = SymbolicType('Dest_type', data['Dest'].unique())

    DepDelay = NumericVariable('DepDelay', Numeric)
    Month = NumericVariable('Month', Numeric)
    DayofMonth = NumericVariable('DayofMonth', Numeric)
    DayOfWeek = NumericVariable('DayOfWeek', Numeric)
    CRSDepTime = NumericVariable('CRSDepTime', Numeric)
    CRSArrTime = NumericVariable('CRSArrTime', Numeric)
    UniqueCarrier = SymbolicVariable('UniqueCarrier', UniqueCarrier_type)
    Origin = SymbolicVariable('Origin', Origin_type)
    Dest = SymbolicVariable('Dest', Dest_type)
    Distance = NumericVariable('Distance', Numeric)

    variables = [DepDelay, Month, DayofMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, Origin, Dest, Distance]
    return data, variables


def main():
    data, variables = preprocess_airline()
    d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-Airline')
    Path(d).mkdir(parents=True, exist_ok=True)

    # tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    tree = JPT(variables=variables, max_depth=8)
    tree.learn(columns=data.values.T)
    tree.save(os.path.join(d, f'{start.strftime("%d.%m.%Y-%H:%M:%S")}-Airline.json'))
    tree.plot(title='Airline', directory=d, view=False)
    logger.info(tree)


def discrtree(args, max_depth=8):
    i, idx = args
    var = variables[i]
    logger.debug(f'Learning {"Decision" if var.symbolic else "Regression"} tree #{i} with target variable {var.name} for FOLD {idx}')
    tgt = data_train[[var.name]]
    X = data_train[[v.name for v in variables if v != var]]

    # transform categorical features
    catcols = X.select_dtypes(['object']).columns
    X[catcols] = X[catcols].astype('category')
    X[catcols] = X[catcols].apply(lambda x: x.cat.codes)

    if var.numeric:
        t = DecisionTreeRegressor(max_depth=max_depth)

    else:
        t = DecisionTreeClassifier(max_depth=max_depth)

    logger.debug(f'Pickling tree {var.name} for FOLD {idx}...')
    t.fit(X, tgt)
    with open(os.path.abspath(os.path.join(d, f'{prefix}{idx}-{var.name}.pkl')), 'wb') as f:
        pickle.dump(t, f)


def fold(idx, train_index, test_index, max_depth=8):
    # for each split, learn separate regression/decision trees for each variable of training set and JPT over
    # entire training set and then compare the results for queries using test set
    # for fld_idx, (train_index, test_index) in enumerate(kf.split(data)):
    logger.info(f'{"":=<100}\nFOLD {idx}: Learning separate regression/decision trees for each variable...')
    global data_train, data_test
    data_train = data.iloc[train_index]
    data_test = data.iloc[test_index]
    data_test.to_pickle(os.path.join(d, f'{prefix}{idx}-testdata.data'))

    # learn separate regression/decision trees for each variable simultaneously
    pool = multiprocessing.Pool()
    pool.map(discrtree, zip(range(len(variables)), [idx]*len(variables)))
    pool.close()
    pool.join()

    # learn full JPT
    logger.debug(f'Learning full JPT over all variables for FOLD {idx}...')
    tree = JPT(variables=variables, min_samples_leaf=data_train.shape[0]*.01, max_depth=max_depth)
    tree.learn(columns=data_train.values.T)
    tree.save(os.path.join(d, f'{prefix}{idx}-JPT.json'))
    tree.plot(title=f'{prefix}{idx}', directory=d, view=False)
    logger.debug(tree)

    # compare
    logger.debug(f'Comparing full JPT over all variables to separately learnt trees for FOLD {idx}...')
    # for d in data_test:
    #     for t in trees:
    #         # TODO: check representation of d as input for prediction/inference, only use respective variable for t
    #         out(t.predict(d), tree.infer(d))

    logger.info(f'FOLD {idx}: done!\n{"":=<100}\n')


def eval(max_depth=8):
    logger.info(f'Start 10-fold cross validation at {start}')

    # read data
    global data, variables
    data, variables = preprocess_airline()

    # create KFold splits over dataset
    global kf
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(data)

    # run folds without pool
    for idx, (train_index, test_index) in enumerate(kf.split(data)):
        fold(idx, train_index, test_index, max_depth=max_depth)

    logger.info(f'10-fold cross validation took {datetime.now() - start}')


if __name__ == '__main__':
    eval()
