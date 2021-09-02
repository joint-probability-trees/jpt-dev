import glob
import multiprocessing
import os
import pickle
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import dnutils
from airlines_depdelay import preprocess_airline
from dnutils import stop, out
from jpt.trees import JPT

# globals
start = datetime.now()
timeformat = "%d.%m.%Y-%H:%M:%S"
d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}')
prefix = f'{start.strftime(timeformat)}'
data = variables = kf = None
data_train = data_test = []
dataset = 'airline'

clogger = dnutils.getlogger('/crossval-learning', level=dnutils.DEBUG)
rlogger = dnutils.getlogger('/crossval-results', level=dnutils.DEBUG)


def init_globals():
    global d, prefix, clogger, rlogger, dataset
    d = os.path.join('/tmp', f'{start.strftime("%Y-%m-%d")}-{dataset}')
    Path(d).mkdir(parents=True, exist_ok=True)
    prefix = f'{start.strftime(timeformat)}-{dataset}-FOLD-'
    dnutils.loggers({'/crossval-learning': dnutils.newlogger(dnutils.logs.console,
                                                    dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime(timeformat)}-{dataset}-learning.log')),
                                                    level=dnutils.DEBUG),
                     '/crossval-results': dnutils.newlogger(dnutils.logs.console,
                                                    dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime(timeformat)}-{dataset}-results.log')),
                                                    level=dnutils.DEBUG),
                     })

    clogger = dnutils.getlogger('/crossval-learning', level=dnutils.DEBUG)
    rlogger = dnutils.getlogger('/crossval-results', level=dnutils.DEBUG)


def preprocess():
    global data, variables, dataset

    if dataset == 'airline':
        data, variables = preprocess_airline()
    else:
        data, variables = None, None

    # set variable value/code mappings for each symbolic variable
    catcols = data.select_dtypes(['object']).columns
    data[catcols] = data[catcols].astype('category')
    for col, var in zip(catcols, [v for v in variables if v.symbolic]):
        data[col] = data[col].cat.set_categories(var.domain.labels.values())


def discrtree(i, idx, max_depth=8):
    var = variables[i]
    clogger.debug(f'Learning {"Decision" if var.symbolic else "Regression"} tree #{i} with target variable {var.name} for FOLD {idx}')
    tgt = data_train[[var.name]]
    X = data_train[[v.name for v in variables if v != var]]

    # transform categorical features
    catcols = X.select_dtypes(['category']).columns
    X[catcols] = X[catcols].apply(lambda x: x.cat.codes)

    # debug
    # u = data['UniqueCarrier'].astype('category')
    # ud = dict(enumerate(u.cat.categories))
    # o = data['Origin'].astype('category')
    # od = dict(enumerate(o.cat.categories))
    # a = data['Dest'].astype('category')
    # dd = dict(enumerate(a.cat.categories))
    # out('UniqueCarrier')
    # out(list(variables[6].domain.labels.values()) == list(ud.values()))
    # out(variables[6].domain.labels.values())
    # out(ud)
    # out('\nOrigin')
    # out(list(variables[7].domain.labels.values()) == list(od.values()))
    # out(variables[7].domain.labels.values())
    # out(od)
    # out('\nDest')
    # out(list(variables[8].domain.labels.values()) == list(dd.values()))
    # out(variables[8].domain.labels.values())
    # out(dd)

    if var.numeric:
        t = DecisionTreeRegressor(max_depth=max_depth)

    else:
        t = DecisionTreeClassifier(max_depth=max_depth)

    clogger.debug(f'Pickling tree {var.name} for FOLD {idx}...')
    t.fit(X, tgt)
    with open(os.path.abspath(os.path.join(d, f'{prefix}{idx}-{var.name}.pkl')), 'wb') as f:
        pickle.dump(t, f)


def fold(fld_idx, train_index, test_index, max_depth=8):
    # for each split, learn separate regression/decision trees for each variable of training set and JPT over
    # entire training set and then compare the results for queries using test set
    # for fld_idx, (train_index, test_index) in enumerate(kf.split(data)):
    clogger.info(f'{"":=<100}\nFOLD {fld_idx}: Learning separate regression/decision trees for each variable...')
    global data_train, data_test
    data_train = data.iloc[train_index]
    data_test = data.iloc[test_index]
    data_test.to_pickle(os.path.join(d, f'{prefix}{fld_idx}-testdata.data'))

    # learn separate regression/decision trees for each variable simultaneously
    pool = multiprocessing.Pool()
    pool.starmap(discrtree, zip(range(len(variables)), [fld_idx] * len(variables)))
    pool.close()
    pool.join()

    # learn full JPT
    clogger.debug(f'Learning full JPT over all variables for FOLD {fld_idx}...')
    jpt = JPT(variables=variables, min_samples_leaf=data_train.shape[0]*.01, max_depth=max_depth)
    jpt.learn(columns=data_train.values.T)
    jpt.save(os.path.join(d, f'{prefix}{fld_idx}-JPT.json'))
    jpt.plot(title=f'{prefix}{fld_idx}', directory=d, view=False)
    clogger.debug(jpt)

    clogger.info(f'FOLD {fld_idx}: done!\n{"":=<100}\n')


def crossval(max_depth=8):
    cfstart = datetime.now()
    clogger.info(f'Start 10-fold cross validation at {cfstart}')

    # create KFold splits over dataset
    global kf
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(data)

    # run folds
    for idx, (train_index, test_index) in enumerate(kf.split(data)):
        fold(idx, train_index, test_index, max_depth=max_depth)

    clogger.info(f'10-fold cross validation took {datetime.now() - cfstart}')


def compare(folds):
    # compare

    for fld_idx in range(folds):
        rlogger.debug(f'Comparing full JPT over all variables to separately learnt trees for FOLD {fld_idx}...')
        jpt = JPT.load(os.path.join(d, f'{prefix}{fld_idx}-JPT.json'))
        for i, v in enumerate(variables):
            with open(os.path.join(d, f'{prefix}{fld_idx}-{v.name}.pkl'), 'rb') as f:
                dectree = pickle.load(f)
            for index, row in data_test.iterrows():
                dp_jpt = {ri: val for ri, val in zip(row.index, row.values)}
                dp_dec = [v.domain.values[x] for j, x in enumerate(row.values) if i != j]  # translate variable value to code for dec tree
                res_jpt = jpt.infer(dp_jpt).result
                # res_dt = dectree.predict(dp_dec)
                rlogger.debug(f'res_jpt | res_dt: {res_jpt} | {res_jpt}: Comparing datapoint { dp_jpt } in decision tree loaded from {prefix}{fld_idx}-{v.name}.pkl and JPT from {prefix}{fld_idx}-JPT.json')


if __name__ == '__main__':
    global dataset
    dataset = 'airline'

    init_globals()
    preprocess()
    crossval()
    compare(10)

    # load pickled dataframe and print first data point
    # data = pd.read_pickle('31.08.2021-16:51:47-airline-FOLD-0-testdata.data')
    # data.iloc[0]

    # load pickled decision tree
    # with open('31.08.2021-16:51:47-airline-FOLD-0-Dest.pkl', 'rb') as f:
    #     dectree = pickle.load(f)

    # load pickled jpt
    # jpt = JPT.load('31.08.2021-16:51:47-airline-FOLD-0-JPT.json')
