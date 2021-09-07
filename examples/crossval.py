import glob
import multiprocessing
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
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
folds = 10

clogger = dnutils.getlogger('/crossval-learning', level=dnutils.DEBUG)
rlogger = dnutils.getlogger('/crossval-results', level=dnutils.DEBUG)


def init_globals():
    global d, start, prefix, clogger, rlogger, dataset
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
    global folds
    cfstart = datetime.now()
    clogger.info(f'Start {folds}-fold cross validation at {cfstart}')

    # create KFold splits over dataset
    global kf
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(data)

    # run folds
    for idx, (train_index, test_index) in enumerate(kf.split(data)):
        fold(idx, train_index, test_index, max_depth=max_depth)

    clogger.info(f'{folds}-fold cross validation took {datetime.now() - cfstart}')


def compare():
    global d, variables, folds
    sep = "\n"
    pool = multiprocessing.Pool()
    res_jpt, res_dec = zip(*pool.starmap(compare_, zip(range(len(variables)))))
    out(res_jpt, res_dec)
    rlogger.debug(f'Crossvalidation results: {sep.join(f"{v}: {j} | {d}" for v, j, d in zip(variables, res_jpt, res_dec))}')

    with open(os.path.join(d, f'{prefix}crossvalidation.result'), 'w+') as f:
        f.write(sep.join(f"{v};{j};{d}" for v, j, d in zip(variables, res_jpt, res_dec)))

    pool.close()
    pool.join()


def compare_(i):
    global folds
    v = variables[i]
    em_jpt = EvaluationMatrix(v)
    em_dec = EvaluationMatrix(v)

    rlogger.debug(f'Comparing full JPT over all variables to separately learnt tree for variable {v.name}...')
    for fld_idx in range(folds):
        # load test dataset for fold fld_idx
        testdata = pd.read_pickle(os.path.join(d, f'{prefix}{fld_idx}-testdata.data'))

        # load JPT for fold fld_idx
        jpt = JPT.load(os.path.join(d, f'{prefix}{fld_idx}-JPT.json'))

        # load decision tree for fold fld_idx variable v
        with open(os.path.join(d, f'{prefix}{fld_idx}-{v.name}.pkl'), 'rb') as f:
            dectree = pickle.load(f)

        # for each data point in the test dataset check results and store them
        for index, row in testdata.iterrows():
            # ground truth
            var_gt = row[v.name]

            # transform datapoints into expected formats for jpt.expectation and dectree.predict
            dp_jpt = {ri: val for ri, val in zip(row.index, row.values) if ri != v.name}
            out('vals for', v.name, v.domain.values)
            out('rowvalues', v.name, row.values)
            out('dp_dec', v.name, [v.domain.values[x] for j, x in enumerate(row.values) if i != j])
            dp_dec = [v.domain.values[x] for j, x in enumerate(row.values) if i != j]  # translate variable value to code for dec tree

            em_jpt.update(var_gt, jpt.expectation([v], dp_jpt)[0].result)
            em_dec.update(var_gt, dectree.predict([dp_dec])[0])

    rlogger.debug(f'res_jpt | res_dec: {em_jpt.accuracy()} | {em_dec.accuracy()}: Comparing datapoint { dp_jpt } in decision tree loaded from {prefix}{fld_idx}-{v.name}.pkl and JPT from {prefix}{fld_idx}-JPT.json')
    return em_jpt.accuracy(), em_dec.accuracy()


class EvaluationMatrix:
    def __init__(self, variable):
        self.variable = variable
        self._res = []

    def update(self, ground_truth, prediction):
        self._res.append(tuple([ground_truth, prediction]))

    def accuracy(self):
        if self.variable.symbolic:
            return f1_score(list(zip(*self._res))[0], list(zip(*self._res))[1])
        else:
            return sum([abs(gt - exp)**2 for gt, exp in self._res])/len(self._res)


if __name__ == '__main__':
    # dataset = 'airline'

    # init_globals()
    # preprocess()
    # crossval()
    # compare()

    ###################### ONLY RUN COMPARE WITH ON ALREADY EXISTING DATA ##############################################
    start = datetime.strptime('07.09.2021-12:01:58', '%d.%m.%Y-%H:%M:%S')
    dataset = 'airline'
    d = os.path.join('/tmp', f'2021-09-07-airline')
    prefix = f'07.09.2021-12:01:58-airline-FOLD-'
    dnutils.loggers({'/crossval-learning': dnutils.newlogger(dnutils.logs.console,
                                                             dnutils.logs.FileHandler(os.path.join(d,
                                                                                                   f'07.09.2021-12:01:58-airline-learning.log')),
                                                             level=dnutils.DEBUG),
                     '/crossval-results': dnutils.newlogger(dnutils.logs.console,
                                                            dnutils.logs.FileHandler(os.path.join(d,
                                                                                                  f'07.09.2021-12:01:58-airline-results.log')),
                                                            level=dnutils.DEBUG),
                     })

    clogger = dnutils.getlogger('/crossval-learning', level=dnutils.DEBUG)
    rlogger = dnutils.getlogger('/crossval-results', level=dnutils.DEBUG)

    data, variables = preprocess_airline()

    # set variable value/code mappings for each symbolic variable
    catcols = data.select_dtypes(['object']).columns
    out('setting catcols', catcols)
    data[catcols] = data[catcols].astype('category')
    for col, var in zip(catcols, [v for v in variables if v.symbolic]):
        out('setting for ', col, var)
        data[col] = data[col].cat.set_categories(var.domain.labels.values())

    compare()
    ###################### ONLY RUN COMPARE WITH ON ALREADY EXISTING DATA ##############################################
