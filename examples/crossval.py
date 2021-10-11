import glob
import multiprocessing
import os
import pickle
from collections import defaultdict
from datetime import datetime
from math import sqrt
from multiprocessing import current_process
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

import dnutils
from airlines_depdelay import preprocess_airline
from dnutils import stop, out
from jpt.learning.distributions import ScaledNumeric, NumericType
from jpt.trees import JPT

# globals
from jpt.variables import Variable

start = datetime.now()
timeformat = "%d.%m.%Y-%H:%M:%S"
homedir = '../tests/'
d = os.path.join(homedir, f'{start.strftime("%Y-%m-%d")}')
prefix = f'{start.strftime(timeformat)}'
data = variables = kf = None
data_train = data_test = []
dataset = 'airline'
folds = 10

logger = dnutils.getlogger('/crossvalidation', level=dnutils.DEBUG)


def init_globals():
    global d, start, prefix, logger, logger, dataset
    d = os.path.join(homedir, f'{start.strftime("%Y-%m-%d")}-{dataset}')
    Path(d).mkdir(parents=True, exist_ok=True)
    prefix = f'{start.strftime(timeformat)}-{dataset}-FOLD-'
    dnutils.loggers({'/crossvalidation': dnutils.newlogger(dnutils.logs.console,
                                                    dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime(timeformat)}-{dataset}-learning.log')),
                                                    level=dnutils.DEBUG)
                     })

    logger = dnutils.getlogger('/crossvalidation', level=dnutils.DEBUG)


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


def discrtree(i, idx):
    var = variables[i]
    logger.debug(f'Learning {"Decision" if var.symbolic else "Regression"} tree #{i} with target variable {var.name} for FOLD {idx}')
    tgt = data_train[[var.name]]
    X = data_train[[v.name for v in variables if v != var]]

    # transform categorical features
    catcols = X.select_dtypes(['category']).columns
    X[catcols] = X[catcols].apply(lambda x: x.cat.codes)

    if var.numeric:
        t = DecisionTreeRegressor(min_samples_leaf=int(data_train.shape[0]*.01))

    else:
        t = DecisionTreeClassifier(min_samples_leaf=int(data_train.shape[0]*.01))

    logger.debug(f'Pickling tree {var.name} for FOLD {idx}...')
    t.fit(X, tgt)
    with open(os.path.abspath(os.path.join(d, f'{prefix}{idx}-{var.name}.pkl')), 'wb') as f:
        pickle.dump(t, f)


def fold(fld_idx, train_index, test_index, max_depth=8):
    # for each split, learn separate regression/decision trees for each variable of training set and JPT over
    # entire training set and then compare the results for queries using test set
    # for fld_idx, (train_index, test_index) in enumerate(kf.split(data)):
    logger.info(f'{"":=<100}\nFOLD {fld_idx}: Learning separate regression/decision trees for each variable...')
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
    logger.debug(f'Learning full JPT over all variables for FOLD {fld_idx}...')
    jpt = JPT(variables=variables, min_samples_leaf=data_train.shape[0]*.01)
    jpt.learn(columns=data_train.values.T)
    jpt.save(os.path.join(d, f'{prefix}{fld_idx}-JPT.json'))
    jpt.plot(title=f'{prefix}{fld_idx}', directory=d, view=False)
    logger.debug(jpt)

    logger.info(f'FOLD {fld_idx}: done!\n{"":=<100}\n')


def crossval(max_depth=8):
    cfstart = datetime.now()
    logger.info(f'Start learning for {folds}-fold cross validation at {cfstart}')

    # create KFold splits over dataset
    global kf
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(data)

    # run folds
    for idx, (train_index, test_index) in enumerate(kf.split(data)):
        fold(idx, train_index, test_index, max_depth=max_depth)

    logger.info(f'Learning of {folds}-fold cross validation took {datetime.now() - cfstart}')


def compare():
    global d, variables, folds

    cmpstart = datetime.now()
    logger.info(f'Start comparison of {folds}-fold cross validation at {cmpstart}')

    nsep = "\n"
    pool = multiprocessing.Pool()
    res_jpt, res_dec = zip(*pool.map(compare_, [(i, [v.to_json() for v in variables]) for i, _ in enumerate(variables)]))
    pool.close()
    pool.join()
    logger.debug(f'Crossvalidation results (error JPT | error DEC):\n{nsep.join(f"{v.name:<20}{j:.8f} | {d:.8f}" for v, j, d in zip(variables, res_jpt, res_dec))}')

    with open(os.path.join(d, f'{prefix}crossvalidation.result'), 'w+') as f:
        f.write(nsep.join(f"{v.name};{j};{d}" for v, j, d in zip(variables, res_jpt, res_dec)))

    logger.info(f'Comparison of {folds}-fold cross validation took {datetime.now() - cmpstart}')


def compare_(args):
    p = current_process()
    compvaridx, json_var = args
    allvariables = [jv['name'] for jv in json_var]
    compvariable = allvariables[compvaridx]
    symbolic = json_var[compvaridx]['type'] == 'symbolic'

    em_jpt = EvaluationMatrix(compvariable, symbolic=symbolic)
    em_dec = EvaluationMatrix(compvariable, symbolic=symbolic)
    errors = 0.
    datapoints = 0.

    logger.debug(f'Comparing full JPT over all variables to separately learnt tree for variable {compvariable}...')
    for fld_idx in range(folds):
        # load test dataset for fold fld_idx
        testdata = pd.read_pickle(os.path.join(d, f'{prefix}{fld_idx}-testdata.data'))
        # testdata = testdata.sample(frac=1)
        datapoints += len(testdata)

        # create mappings for categorical variables; later to be used to translate variable values to codes for dec tree
        catcols = testdata.select_dtypes(['category']).columns
        mappings = {var: {c: i for i, c in enumerate(testdata[var].cat.categories)} for var in allvariables if var in catcols}

        # load JPT for fold fld_idx
        jpt = JPT.load(os.path.join(d, f'{prefix}{fld_idx}-JPT.json'))

        # load decision tree for fold fld_idx variable v
        with open(os.path.join(d, f'{prefix}{fld_idx}-{compvariable}.pkl'), 'rb') as f:
            dectree = pickle.load(f)

        # for each data point in the test dataset check results and store them
        for n, (index, row) in enumerate(testdata.iterrows()):
            if not n % 10000:
                logger.debug(f'Pool #{p._identity[0]} ({compvariable}) and FOLD {fld_idx} is now at data point {n}...')

            # ground truth
            var_gt = row[compvariable]

            # transform datapoints into expected formats for jpt.expectation and dectree.predict
            dp_jpt = {ri: val for ri, val in zip(row.index, row.values) if ri != compvariable}
            dp_dec = [(mappings[allvariables[colidx]][colval] if json_var[colidx]['type'] == 'symbolic' else colval) for colidx, (colval, var) in enumerate(zip(row.values, allvariables)) if compvaridx != colidx]  # translate variable value to code for dec tree

            jptexp = jpt.expectation([compvariable], dp_jpt, fail_on_unsatisfiability=False)
            if jptexp is None:
                errors += 1.
                logger.warning(f'Errors in Pool #{p._identity[0]} ({compvariable}, FOLD {fld_idx}): {errors} ({datapoints}) (unsatisfiable query: {dp_jpt})')
            else:
                em_jpt.update(var_gt, jptexp[0].result)
                em_dec.update(var_gt, dectree.predict([dp_dec])[0])

    with open(os.path.join(d, f'{prefix}{compvariable}-Matrix-JPT.pkl'), 'wb') as f:
        pickle.dump(em_jpt, f)

    with open(os.path.join(d, f'{prefix}{compvariable}-Matrix-DEC.pkl'), 'wb') as f:
        pickle.dump(em_dec, f)

    logger.error(f'FINAL NUMBER OF ERRORS FOR VARIABLE {compvariable}: {errors} in {datapoints} data points')
    logger.warning(f'res_jpt | res_dec: {em_jpt.accuracy()} | {em_dec.accuracy()}: Comparing datapoint { dp_jpt } in decision tree loaded from {prefix}{fld_idx}-{compvariable}.pkl and JPT from {prefix}{fld_idx}-JPT.json')
    return em_jpt.accuracy(), em_dec.accuracy()


def plot_confusion_matrix():
    mat = pd.read_csv(os.path.join(d, f'{prefix}crossvalidation.result'), sep=';', names=['Variable', 'JPTerr', 'DECerr'])

    x_pos = np.arange(len(variables))
    varnames = [v.name for v in variables]
    vartypes = [v.domain if v.numeric else None for v in variables]
    decerr = [vtype.values[val] if vtype is not None else val for vtype, val in zip(vartypes, mat['DECerr'])]
    jpterr = [vtype.values[val] if vtype is not None else val for vtype, val in zip(vartypes, mat['JPTerr'])]

    decstd = []
    jptstd = []
    for vn in varnames:
        with open(os.path.join(d, f'{prefix}{vn}-Matrix-DEC.pkl'), 'rb') as f:
            matdec = pickle.load(f)
            decstd.append(matdec.error())

        with open(os.path.join(d, f'{prefix}{vn}-Matrix-JPT.pkl'), 'rb') as f:
            matjpt = pickle.load(f)
            jptstd.append(matjpt.error())

    fig, ax = plt.subplots()
    ax.bar(x_pos-0.1, jpterr, width=0.2, yerr=jptstd, align='center', alpha=0.5, ecolor='black', color='orange', capsize=10, label='JPT')
    ax.bar(x_pos+0.1, decerr, width=0.2, yerr=decstd, align='center', alpha=0.5, ecolor='black', color='cornflowerblue', capsize=10, label='DEC')
    ax.set_ylabel('MSE/f1_score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(varnames)
    ax.set_title('DEC VS JPT')
    ax.yaxis.grid(True)
    ax.legend()

    # Save the figure and show
    plt.tight_layout()
    plt.savefig('bar_plot_with_error_bars.png')
    plt.show()


class EvaluationMatrix:
    def __init__(self, variable, symbolic=False):
        self.variable = variable
        self.symbolic = symbolic
        self.res = []

    def update(self, ground_truth, prediction):
        self.res.append(tuple([ground_truth, prediction]))

    def accuracy(self):
        if self.symbolic:
            return f1_score(list(zip(*self.res))[0], list(zip(*self.res))[1], average='macro')
        else:
            return sum([abs(gt - exp)**2 for gt, exp in self.res])/len(self.res)

    def error(self):
        if self.symbolic:
            return 5  # TODO
        else:
            return 5  # TODO


if __name__ == '__main__':
    dataset = 'airline'
    homedir = '../tests/'
    ovstart = datetime.now()
    logger.info(f'Starting overall cross validation at {ovstart}')

    init_globals()
    preprocess()
    crossval()
    compare()

    logger.info(f'Overall cross validation on {len(data)}-instance dataset took {datetime.now() - ovstart}')

    plot_confusion_matrix()


    ###################### ONLY RUN COMPARE WITH ON ALREADY EXISTING DATA ##############################################
    # start = datetime.strptime('07.09.2021-12:01:58', '%d.%m.%Y-%H:%M:%S')
    # dataset = 'airline'
    # d = os.path.join('/tmp', f'2021-09-07-airline')
    # prefix = f'07.09.2021-12:01:58-airline-FOLD-'
    # dnutils.loggers({'/crossval-learning': dnutils.newlogger(dnutils.logs.console,
    #                                                          dnutils.logs.FileHandler(os.path.join(d,
    #                                                                                                f'07.09.2021-12:01:58-airline-learning.log')),
    #                                                          level=dnutils.DEBUG),
    #                  '/crossval-results': dnutils.newlogger(dnutils.logs.console,
    #                                                         dnutils.logs.FileHandler(os.path.join(d,
    #                                                                                               f'07.09.2021-12:01:58-airline-results.log')),
    #                                                         level=dnutils.DEBUG),
    #                  })
    #
    # logger = dnutils.getlogger('/crossval-learning', level=dnutils.DEBUG)
    # logger = dnutils.getlogger('/crossval-results', level=dnutils.DEBUG)
    #
    # data, variables = preprocess_airline()
    #
    # # set variable value/code mappings for each symbolic variable
    # catcols = data.select_dtypes(['object']).columns
    # data[catcols] = data[catcols].astype('category')
    # for col, var in zip(catcols, [v for v in variables if v.symbolic]):
    #     data[col] = data[col].cat.set_categories(var.domain.labels.values())
    #
    # compare()

    #######333333333333333333333############### RUN WITHOUT POOL #######333333333#######################################
    # sep = "\n"
    # res_jpt, res_dec = compare_no_pool()
    # out(res_jpt, res_dec)
    # logger.debug(f'Crossvalidation results: {sep.join(f"{v}: {j} | {d}" for v, j, d in zip(variables, res_jpt, res_dec))}')
    #
    # with open(os.path.join(d, f'{prefix}crossvalidation.result'), 'w+') as f:
    #     f.write(sep.join(f"{v};{j};{d}" for v, j, d in zip(variables, res_jpt, res_dec)))
    #######333333333333333333333############### /RUN WITHOUT POOL #######333333333######################################

    ###################### /ONLY RUN COMPARE WITH ON ALREADY EXISTING DATA #############################################
