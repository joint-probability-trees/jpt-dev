import logging
import multiprocessing
import os
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import pydot as pydot
import tabulate
from sklearn import datasets

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree, export_graphviz

import dnutils
from airlines_depdelay import preprocess_airline
from banana import preprocess_banana
from regression import preprocess_regression
from restaurant import preprocess_restaurant
from gaussians import preprocess_gaussian
from jpt.trees import JPT

# globals
from jpt.variables import infer_from_dataframe

start = datetime.now()
timeformat = "%Y-%m-%d-%H:%M:%S"
homedir = '../tests/'
d = os.path.join(homedir, f'{start.strftime("%Y-%m-%d")}')
prefix = f'{start.strftime(timeformat)}'
data = variables = kf = None
data_train = data_test = []
dataset = 'airline'
folds = 10

logger = logging.getLogger('/crossvalidation')

MIN_SAMPLES_LEAF = 0.1


def init_globals():
    global d, start, prefix, logger, logger, dataset
    d = os.path.join(homedir, f'{start.strftime("%Y-%m-%d")}-{dataset}')
    Path(d).mkdir(parents=True, exist_ok=True)
    prefix = f'{start.strftime(timeformat)}-{dataset}'
    # dnutils.loggers({'/crossvalidation': dnutils.newlogger(dnutils.logs.console,
    #                                                 dnutils.logs.FileHandler(os.path.join(d, f'{start.strftime(timeformat)}-{dataset}-learning.log')),
    #                                                 level=dnutils.DEBUG)
    #                  })

    logger = logging.getLogger('/crossvalidation')


def preprocess():
    global data, variables, dataset

    if dataset == 'airline':
        data = preprocess_airline()
        data = data[['DayOfWeek', 'CRSDepTime', 'Distance', 'CRSArrTime', 'UniqueCarrier', 'Origin', 'Dest']]  #
        # data = data[['UniqueCarrier', 'Origin', 'Dest']]
    elif dataset == 'regression':
        data = preprocess_regression()
    elif dataset == 'iris':
        iris = datasets.load_iris()
        data = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
        data['species'] = [['setosa', 'versicolor', 'virginica'][x] for x in iris['target']]  # convert target integers to symbolics and add to dataframe
    elif dataset == 'banana':
        data = preprocess_banana()
    elif dataset == 'restaurant':
        data = preprocess_restaurant()
        data = data[['Price', 'Food', 'WaitEstimate']]
        logger.debug('Restaurant data\n', data)
    elif dataset == 'gaussian':
        data = preprocess_gaussian()
    else:
        data = None

    variables = infer_from_dataframe(data, scale_numeric_types=True, precision=.01, haze=.01)
    if dataset == 'airline':
        data = data.sample(frac=0.001)  # TODO remove; only for debugging
    logger.debug(f'Loaded {len(data)} datapoints')

    # set variable value/code mappings for each symbolic variable
    catcols = data.select_dtypes(['object']).columns
    data[catcols] = data[catcols].astype('category')
    for col, var in zip(catcols, [v for v in variables if v.symbolic]):
        data[col] = data[col].cat.set_categories(var.domain.labels.values())


def discrtree(i, fld_idx):
    var = variables[i]
    logger.debug(f'Learning {"Decision" if var.symbolic else "Regression"} tree #{i} with target variable {var.name} for FOLD {fld_idx}')
    tgt = data_train[[var.name]]
    X = data_train[[v.name for v in variables if v != var]]

    # transform categorical features
    catcols = X.select_dtypes(['category']).columns
    X[catcols] = X[catcols].apply(lambda x: x.cat.codes)

    if var.numeric:
        t = DecisionTreeRegressor(min_samples_leaf=1 if dataset == 'restaurant' else int(data_train.shape[0] * MIN_SAMPLES_LEAF))
    else:
        t = DecisionTreeClassifier(min_samples_leaf=1 if dataset == 'restaurant' else int(data_train.shape[0] * MIN_SAMPLES_LEAF))

    t.fit(X, tgt)
    logger.debug(f'Pickling tree {var.name} ({t.get_n_leaves()} leaves) for FOLD {fld_idx + 1}...')
    with open(os.path.abspath(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-{var.name}.pkl')), 'wb') as f:
        pickle.dump(t, f)
        plot_tree(t)
        export_graphviz(t, out_file=os.path.abspath(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-{var.name}.dot')))
        (graph,) = pydot.graph_from_dot_file(os.path.abspath(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-{var.name}.dot')))
        graph.write_png(os.path.abspath(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-{var.name}.png')))


def fold(fld_idx, train_index, test_index, max_depth=8):
    # for each split, learn separate regression/decision trees for each variable of training set and JPT over
    # entire training set and then compare the results for queries using test set
    # for fld_idx, (train_index, test_index) in enumerate(kf.split(data)):
    global data_train, data_test
    data_train = data.iloc[train_index]
    data_test = data.iloc[test_index]
    data_test.to_pickle(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-testdata.data'))
    logger.info(f'{"":=<100}\nFOLD {fld_idx}: Learning separate regression/decision trees for each variable...')

    # learn separate regression/decision trees for each variable simultaneously
    pool = multiprocessing.Pool()
    pool.starmap(discrtree, zip(range(len(variables)), [fld_idx] * len(variables)))
    pool.close()
    pool.join()

    # learn full JPT
    logger.debug(f'Learning full JPT over all variables for FOLD {fld_idx}...')
    jpt = JPT(variables=variables, min_samples_leaf=1 if dataset == 'restaurant' else int(data_train.shape[0] * MIN_SAMPLES_LEAF / len(variables)))
    jpt.learn(columns=data_train.values.T)
    jpt.save(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-JPT.json'))
    if dataset in ['iris', 'banana', 'restaurant', 'gaussian']:
        jpt.plot(title=f'{prefix}-FOLD-{fld_idx}', directory=d, view=False)
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
    logger.debug('Numeric Variable Results:')
    logger.debug(tabulate.tabulate([[v.name,
                                     j.accuracy(),
                                     j.error(),
                                     d.accuracy(),
                                     d.error()] for v, j, d in zip(variables, res_jpt, res_dec) if v.numeric],
                                   headers=['Variable', 'JPT-MAE', '(+/-)', 'DEC-MAE', '(+/-)']))
    logger.debug('Symbolic Variable Results:')
    logger.debug(tabulate.tabulate([[v.name,
                                     j.accuracy(),
                                     j.error(),
                                     d.accuracy(),
                                     d.error()] for v, j, d in zip(variables, res_jpt, res_dec) if v.symbolic],
                                   headers=['Variable', 'JPT-F1', '(+/-)', 'DEC-F1', '(+/-)']))

    # save crossvalidation results to file
    with open(os.path.join(d, f'{prefix}-Matrix-DEC.pkl'), 'wb') as f:
        pickle.dump(res_dec, f)

    with open(os.path.join(d, f'{prefix}-Matrix-JPT.pkl'), 'wb') as f:
        pickle.dump(res_jpt, f)

    # TODO remove (human-readable version of crossvalidation results; not used anywhere)
    with open(os.path.join(d, f'{prefix}-crossvalidation.csv'), 'w+') as f:
        f.write(f'Variable;JPTacc;JPTerr;DECacc;DECerr\n')
        f.write(nsep.join(f"{v.name};{j.accuracy()};{j.error()};{d.accuracy()};{d.error()}" for v, j, d in zip(variables, res_jpt, res_dec)))

    logger.info(f'Comparison of {folds}-fold cross validation took {datetime.now() - cmpstart}')


def compare_(args):
    p = multiprocessing.current_process()
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
        testdata = pd.read_pickle(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-testdata.data'))
        datapoints += len(testdata)

        # create mappings for categorical variables; later to be used to translate variable values to codes for dec tree
        catcols = testdata.select_dtypes(['category']).columns
        mappings = {var: {c: i for i, c in enumerate(testdata[var].cat.categories)} for var in allvariables if var in catcols}

        # load JPT for fold fld_idx
        jpt = JPT.load(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-JPT.json'))

        # load decision tree for fold fld_idx variable v
        with open(os.path.join(d, f'{prefix}-FOLD-{fld_idx}-{compvariable}.pkl'), 'rb') as f:
            dectree = pickle.load(f)

        # for each data point in the test dataset check results and store them
        do_print = np.inf
        for n, (index, row) in enumerate(testdata.iterrows()):
            if not n % min(10000, max(1, int(len(testdata)*.1))):
                logger.debug(f'Worker #{p._identity[0]} ({compvariable}) and FOLD {fld_idx} is now at data point {n}...')

            # ground truth
            var_gt = row[compvariable]

            # transform datapoints into expected formats for jpt.expectation and dectree.predict
            dp_jpt = {ri: val for ri, val in zip(row.index, row.values) if ri != compvariable}
            dp_dec = [(mappings[allvariables[colidx]][colval] if json_var[colidx]['type'] == 'symbolic' else colval) for colidx, (colval, var) in enumerate(zip(row.values, allvariables)) if compvaridx != colidx]  # translate variable value to code for dec tree

            jptexp = jpt.expectation([compvariable], dp_jpt, fail_on_unsatisfiability=False)
            if jptexp is None:
                errors += 1.
                # logger.warning(f'Errors in Worker #{p._identity[0]} ({compvariable}, FOLD {fld_idx}): {errors} ({datapoints}); current data point: {n} (unsatisfiable query: {dp_jpt})')
            else:
                em_jpt.update(fld_idx, var_gt, jptexp[0].result)
                if jptexp[0].result != jptexp[0].result:
                    print(compvariable, dp_jpt)
                    print(jpt)
                em_dec.update(fld_idx, var_gt, dectree.predict([dp_dec])[0])

    with open(os.path.join(d, f'{prefix}-Matrix-JPT-{compvariable}.pkl'), 'wb') as f:
        pickle.dump(em_jpt, f)

    with open(os.path.join(d, f'{prefix}-Matrix-DEC-{compvariable}.pkl'), 'wb') as f:
        pickle.dump(em_dec, f)

    logger.error(f'FINAL NUMBER OF ERRORS FOR VARIABLE {compvariable}: {int(errors)} in {int(datapoints)} data points')
    logger.warning(f'res_jpt | res_dec: {em_jpt.accuracy()} | {em_dec.accuracy()}: Comparing datapoint { dp_jpt } in decision tree loaded from {prefix}-FOLD-{fld_idx}-{compvariable}.pkl and JPT from {prefix}{fld_idx}-JPT.json')

    return em_jpt, em_dec


def plot_confusion_matrix(show=True):
    x_pos = np.arange(len(variables))
    varnames = [v.name for v in variables]
    vartypes = [v.domain if v.numeric else None for v in variables]

    with open(os.path.join(d, f'{prefix}-Matrix-DEC.pkl'), 'rb') as f:
        matdec = pickle.load(f)
    decacc = [vt.values[m.accuracy()] if vt is not None else m.accuracy() for vt, m in zip(vartypes, matdec)]
    decerr = [vt.values[m.error()] if vt is not None else m.accuracy() for vt, m in zip(vartypes, matdec)]

    with open(os.path.join(d, f'{prefix}-Matrix-JPT.pkl'), 'rb') as f:
        matjpt = pickle.load(f)
    jptacc = [vt.values[j.accuracy()] if vt is not None else j.accuracy() for vt, j in zip(vartypes, matjpt)]
    jpterr = [vt.values[j.error()] if vt is not None else j.error() for vt, j in zip(vartypes, matjpt)]

    fig, ax = plt.subplots()
    # ax.bar(x_pos-0.1, jptacc, width=0.2, yerr=[0]*len(variables), align='center', alpha=0.5, ecolor='black', color='orange', capsize=10, label='JPT')
    # ax.bar(x_pos+0.1, decacc, width=0.2, yerr=[0]*len(variables), align='center', alpha=0.5, ecolor='black', color='cornflowerblue', capsize=10, label='DEC')
    ax.bar(x_pos-0.1, jptacc, width=0.2, yerr=jpterr, align='center', alpha=0.5, ecolor='black', color='orange', capsize=10, label='JPT')
    ax.bar(x_pos+0.1, decacc, width=0.2, yerr=decerr, align='center', alpha=0.5, ecolor='black', color='cornflowerblue', capsize=10, label='DEC')
    ax.set_ylabel('MSE/f1_score')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(varnames)
    ax.set_title('DEC VS JPT')
    ax.yaxis.grid(True)
    ax.legend()

    # Save the figure and show
    plt.tight_layout()
    plt.xticks(rotation=-45)
    plt.savefig(os.path.join(d, f'{prefix}-crossvalidation.svg'))
    plt.savefig(os.path.join(d, f'{prefix}-crossvalidation.png'))
    if show: plt.show()


class EvaluationMatrix:
    def __init__(self, varname, symbolic=False):
        self.varname = varname
        self.symbolic = symbolic
        self.res = defaultdict(list)

    def update(self, fld, ground_truth, prediction):
        self.res[fld].append(tuple([ground_truth, prediction]))

    def accuracy(self):
        res = list(chain(*self.res.values()))
        if self.symbolic:
            return f1_score(list(zip(*res))[0], list(zip(*res))[1], average='micro')
        else:
            return mean_absolute_error(list(zip(*res))[0], list(zip(*res))[1])  # identical result

    def error(self):
        if self.symbolic:
            return np.std([f1_score(list(zip(*_res))[0], list(zip(*_res))[1], average='macro') for _res in self.res.values()])
        else:
            return np.std([mean_absolute_error(list(zip(*_res))[0], list(zip(*_res))[1]) for _res in self.res.values()])


if __name__ == '__main__':
    dataset = 'airline'
    # dataset = 'regression'
    # dataset = 'iris'
    # dataset = 'banana'
    # dataset = 'restaurant'
    # dataset = 'gaussian'

    homedir = '../tests/'
    ovstart = datetime.now()
    logger.info(f'Starting overall cross validation at {ovstart}')

    init_globals()
    preprocess()
    crossval()
    compare()

    logger.info(f'Overall cross validation on {len(data)}-instance dataset took {datetime.now() - ovstart}')

    plot_confusion_matrix(show=False)


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
    # plot_confusion_matrix()
    ###################### /ONLY RUN COMPARE WITH ON ALREADY EXISTING DATA #############################################
