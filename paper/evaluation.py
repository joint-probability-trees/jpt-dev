"""Empirical evaluation for the ensemble-JPT paper.

Reproduces every number in the Experiments section from the public
library API. Writes incremental results to ``paper/results.json`` (or
``$EVAL_OUT`` if set, enabling parallel per-dataset runs).

Experiments:
    E1  held-out joint log-likelihood and predictive quality of
        single JPT / JPTForest / JPTLikelihoodBoost / JPTBoost vs.
        sklearn RandomForest / HistGradientBoosting / LightGBM
        (5-fold CV; digits, california, adult: single 75/25 split).
        Datasets already present in the results file are topped up
        incrementally: only models missing from their entry are run
        (the split RNG is seeded, so folds are reproducible).
    E2  forest quality vs. number of members M (wine, diabetes).
    E3  likelihood-boost train/held-out LL per round (iris, diabetes).
    E4  prior_alpha ablation (0 vs. 1) on the classification datasets.

Usage:  python evaluation.py [E1 E2 E3 E4] [dataset ...]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from jpt.ensembles import JPTBoost, JPTForest, JPTLikelihoodBoost
from jpt.learning.preprocessing import preprocess_data
from jpt.trees import JPT
from jpt.variables import infer_from_dataframe

warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get(
    'EVAL_OUT', os.path.join(HERE, 'results.json')
)
EPS = 1e-12

MSL = .05          # min_samples_leaf of all generative JPT models
M_FOREST = 10      # forest size
LB_ROUNDS = 8      # likelihood-boost rounds
SEED = 0
N_BIG = 10000      # subsample size of the large datasets

E1_MODELS = ['jpt', 'forest', 'lboost', 'jptboost', 'rf', 'histgb', 'lgbm']
# single 75/25 split instead of 5-fold CV, for runtime reasons
SINGLE_SPLIT = {'digits', 'california', 'adult'}


# ----------------------------------------------------------------------------------------------------------------------
# Datasets

def _sk_frame(bunch) -> pd.DataFrame:
    df = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    df['target'] = [str(bunch.target_names[t]) for t in bunch.target]
    return df


def load_datasets() -> dict:
    digits = load_digits()
    # drop near-constant pixels (undefined density), keep the rest
    mask = digits.data.std(axis=0) > 1.
    ddf = pd.DataFrame(
        digits.data[:, mask],
        columns=['p%d' % i for i in np.flatnonzero(mask)]
    )
    ddf['target'] = [str(t) for t in digits.target]

    diabetes = load_diabetes()
    bdf = pd.DataFrame(diabetes.data, columns=list(diabetes.feature_names))
    bdf['target'] = diabetes.target.astype(float)

    abalone = pd.read_csv(
        os.path.join(HERE, '..', 'examples', 'data', 'abalone.data'),
        names=['sex', 'length', 'diameter', 'height', 'whole', 'shucked',
               'viscera', 'shell', 'target']
    )
    abalone['target'] = abalone['target'].astype(float)
    abalone = abalone.sample(2000, random_state=SEED).reset_index(drop=True)

    california = fetch_california_housing(as_frame=True).frame \
        .rename(columns={'MedHouseVal': 'target'}) \
        .sample(N_BIG, random_state=SEED).reset_index(drop=True)

    adult = fetch_openml(
        'adult', version=2, as_frame=True, parser='auto'
    ).frame
    adult = adult.drop(columns=['fnlwgt', 'education-num']).dropna()
    for col in adult.columns:
        if not pd.api.types.is_numeric_dtype(adult[col]):
            adult[col] = adult[col].astype(str)
    adult = adult.rename(columns={'class': 'target'}) \
        .sample(N_BIG, random_state=SEED).reset_index(drop=True)
    # lump categories rarer than 1% into 'Other': JPT domains are inferred
    # from the train split, so ultra-rare levels would show up unseen at
    # test time and receive undefined likelihood
    for col in adult.columns:
        if col != 'target' and adult[col].dtype == object:
            counts = adult[col].value_counts()
            rare = counts[counts < .01 * len(adult)].index
            adult.loc[adult[col].isin(rare), col] = 'Other'

    return {
        'iris': ('clf', _sk_frame(load_iris())),
        'wine': ('clf', _sk_frame(load_wine())),
        'breast_cancer': ('clf', _sk_frame(load_breast_cancer())),
        'digits': ('clf', ddf),
        'diabetes': ('reg', bdf),
        'abalone': ('reg', abalone),
        'california': ('reg', california),
        'adult': ('clf', adult),
    }


# ----------------------------------------------------------------------------------------------------------------------
# Generative prediction helpers (density-weighted posterior over leaves).
#
# Leaf numeric densities are piecewise-linear pdfs with BOUNDED support:
# a held-out point outside a leaf's per-variable sample range has density
# exactly zero there, and in higher dimensions almost every held-out
# point escapes every leaf's box in some coordinate. All densities are
# therefore smoothed per variable with a uniform background over the
# training range, p~ = (1-eps)p + eps*u_v -- identically for every
# generative model, so comparisons are unaffected.

EPS_BG = 1e-3


class Uniforms:
    """Per-variable uniform background densities from the training data."""

    def __init__(self, train: pd.DataFrame):
        self.u = {}
        for col in train.columns:
            if pd.api.types.is_numeric_dtype(train[col]):
                lo, hi = float(train[col].min()), float(train[col].max())
                self.u[col] = 1. / max(hi - lo, 1e-6)
            else:
                self.u[col] = 1. / train[col].nunique()

    def __getitem__(self, name: str) -> float:
        return self.u[name]


def first_label(var):
    return var.domain.labels[0]


def _leaf_smoothed_likelihoods(tree: JPT, data: pd.DataFrame,
                               uniforms: Uniforms,
                               exclude: str | None = None):
    """Yield ``(leaf, prod_v p~_leaf,v(x_v))`` for every leaf, vectorized;
    ``exclude`` omits one variable (the prediction target) from the
    product."""
    variables = [v for v in tree.variables if v.name != exclude]
    frame = data[[v.name for v in variables]].copy()
    if exclude is not None:
        tvar = tree.varnames[exclude]
        frame[exclude] = first_label(tvar) if tvar.symbolic else 0.
    pp = preprocess_data(tree, frame)
    md = tree.minimal_distances
    for leaf in tree.leaves.values():
        prod = np.ones(len(data))
        for v in variables:
            lv = leaf.likelihood(pp, 2., md, variables=[v])[:, 0]
            prod *= (1. - EPS_BG) * lv + EPS_BG * uniforms[v.name]
        yield leaf, prod


def smoothed_density(tree: JPT, data: pd.DataFrame,
                     uniforms: Uniforms) -> np.ndarray:
    """Smoothed joint density sum_l P(l) prod_v p~_l(x_v)."""
    result = np.zeros(len(data))
    for leaf, prod in _leaf_smoothed_likelihoods(tree, data, uniforms):
        result += leaf.prior * prod
    return result


def mixture_ll(models_weights, data, uniforms) -> float:
    """Mean log-likelihood of a weighted list of (weight, tree)."""
    density = np.zeros(len(data))
    for w, tree in models_weights:
        density += w * smoothed_density(tree, data, uniforms)
    return mean_ll(density)


def class_scores(tree: JPT, data: pd.DataFrame, target: str,
                 classes: list, uniforms: Uniforms) -> np.ndarray:
    """Per-class joint scores p(x, c) = sum_l P(l) L~_l(x) p~_l(c)."""
    tvar = tree.varnames[target]
    idx = [int(tvar.domain.values[c]) for c in classes]
    u = uniforms[target]
    scores = np.zeros((len(data), len(classes)))
    for leaf, fl in _leaf_smoothed_likelihoods(
            tree, data, uniforms, exclude=target):
        probs = np.asarray(leaf.distributions[target].probabilities)[idx]
        probs = (1. - EPS_BG) * probs + EPS_BG * u
        scores += leaf.prior * np.outer(fl, probs)
    return scores


def posterior_mean(tree: JPT, data: pd.DataFrame, target: str,
                   uniforms: Uniforms) -> np.ndarray:
    """E[y | x] = sum_l p(l | x) E_l[y], density-weighted."""
    num = np.zeros(len(data))
    den = np.zeros(len(data))
    for leaf, fl in _leaf_smoothed_likelihoods(
            tree, data, uniforms, exclude=target):
        e = leaf.distributions[target].expectation()
        num += leaf.prior * fl * e
        den += leaf.prior * fl
    return num / np.clip(den, EPS, None)


def mixture_class_scores(mixture, data, target, classes,
                         uniforms) -> np.ndarray:
    return sum(
        w * class_scores(m, data, target, classes, uniforms)
        for w, m in zip(mixture.weights, mixture.members)
    )


def mixture_posterior_mean(mixture, data, target, uniforms) -> np.ndarray:
    num = np.zeros(len(data))
    den = np.zeros(len(data))
    for w, member in zip(mixture.weights, mixture.members):
        for leaf, fl in _leaf_smoothed_likelihoods(
                member, data, uniforms, exclude=target):
            e = leaf.distributions[target].expectation()
            num += w * leaf.prior * fl * e
            den += w * leaf.prior * fl
    return num / np.clip(den, EPS, None)


def mean_ll(density: np.ndarray) -> float:
    return float(np.mean(np.log(np.clip(density, EPS, None))))


# ----------------------------------------------------------------------------------------------------------------------
# Model runners: each returns {metric: value} for one train/test split

def run_split(task: str, train: pd.DataFrame, test: pd.DataFrame,
              prior_alpha: float | None = None,
              models: set | None = None) -> dict:
    """Run one train/test split; ``models`` restricts which model keys
    are computed (None = all), enabling incremental top-ups."""
    out = {}
    target = 'target'
    classes = sorted(train[target].unique()) if task == 'clf' else None
    y_test = test[target].to_numpy()
    uniforms = Uniforms(train)

    def want(name: str) -> bool:
        return models is None or name in models

    def score_pred(name, pred, elapsed=None):
        key = 'acc' if task == 'clf' else 'r2'
        metric = (
            accuracy_score(y_test, pred) if task == 'clf'
            else r2_score(y_test, pred)
        )
        out.setdefault(name, {})[key] = float(metric)
        if elapsed is not None:
            out[name]['time'] = elapsed

    # --- single JPT
    if want('jpt'):
        t0 = time.time()
        single = JPT(
            variables=infer_from_dataframe(
                train, scale_numeric_types=False, prior_alpha=prior_alpha
            ),
            min_samples_leaf=MSL
        ).learn(train)
        t_single = time.time() - t0
        out['jpt'] = {'ll': mixture_ll([(1., single)], test, uniforms),
                      'time': t_single}
        if task == 'clf':
            pred = np.asarray(classes)[
                class_scores(single, test, target, classes, uniforms)
                .argmax(axis=1)
            ]
        else:
            pred = posterior_mean(single, test, target, uniforms)
        score_pred('jpt', pred)

    # --- forest
    if want('forest'):
        t0 = time.time()
        forest = JPTForest(
            n_estimators=M_FOREST, min_samples_leaf=MSL,
            prior_alpha=prior_alpha, random_state=SEED
        ).learn(train)
        t_forest = time.time() - t0
        out['forest'] = {
            'll': mixture_ll(
                list(zip(forest.weights, forest.members)), test, uniforms
            ),
            'time': t_forest
        }
        if task == 'clf':
            pred = np.asarray(classes)[
                mixture_class_scores(forest, test, target, classes,
                                     uniforms).argmax(axis=1)
            ]
        else:
            pred = mixture_posterior_mean(forest, test, target, uniforms)
        score_pred('forest', pred)

    # --- likelihood boosting
    if want('lboost'):
        t0 = time.time()
        lboost = JPTLikelihoodBoost(
            n_rounds=LB_ROUNDS, min_samples_leaf=MSL,
            prior_alpha=prior_alpha, random_state=SEED
        ).learn(train)
        t_lb = time.time() - t0
        out['lboost'] = {
            'll': mixture_ll(
                list(zip(lboost.weights, lboost.members)), test, uniforms
            ),
            'time': t_lb,
            'rounds': len(lboost.members)
        }
        if task == 'clf':
            pred = np.asarray(classes)[
                mixture_class_scores(lboost, test, target, classes,
                                     uniforms).argmax(axis=1)
            ]
        else:
            pred = mixture_posterior_mean(lboost, test, target, uniforms)
        score_pred('lboost', pred)

    # --- discriminative JPTBoost
    if want('jptboost'):
        t0 = time.time()
        if task == 'clf':
            boost = JPTBoost(
                target=target, n_rounds=25, learning_rate=.3,
                min_samples_leaf=.1
            ).learn(train)
        else:
            boost = JPTBoost(
                target=target, n_rounds=40, learning_rate=.1,
                min_samples_leaf=MSL
            ).learn(train)
        t_boost = time.time() - t0
        score_pred('jptboost', boost.predict(test[boost.feature_names]),
                   t_boost)

    # --- discriminative baselines (one-hot for categoricals)
    if task == 'clf':
        baselines = [
            ('rf', RandomForestClassifier(n_estimators=100,
                                          random_state=SEED)),
            ('histgb', HistGradientBoostingClassifier(random_state=SEED)),
            ('lgbm', LGBMClassifier(random_state=SEED, verbose=-1)),
        ]
    else:
        baselines = [
            ('rf', RandomForestRegressor(n_estimators=100,
                                         random_state=SEED)),
            ('histgb', HistGradientBoostingRegressor(random_state=SEED)),
            ('lgbm', LGBMRegressor(random_state=SEED, verbose=-1)),
        ]
    baselines = [(n, m) for n, m in baselines if want(n)]
    if baselines:
        X_train = pd.get_dummies(train.drop(columns=[target]))
        X_test = pd.get_dummies(test.drop(columns=[target]))
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        for name, model in baselines:
            t0 = time.time()
            model.fit(X_train, train[target])
            score_pred(name, model.predict(X_test), time.time() - t0)
    return out


# ----------------------------------------------------------------------------------------------------------------------

def folds(task: str, df: pd.DataFrame, n_splits: int = 5):
    if task == 'clf':
        splitter = StratifiedKFold(n_splits, shuffle=True, random_state=SEED)
        split = splitter.split(df, df['target'])
    else:
        splitter = KFold(n_splits, shuffle=True, random_state=SEED)
        split = splitter.split(df)
    for tr, te in split:
        yield df.iloc[tr].reset_index(drop=True), \
              df.iloc[te].reset_index(drop=True)


def save(results: dict) -> None:
    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)


def load_results() -> dict:
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def aggregate(fold_results: list) -> dict:
    """[{model: {metric: v}}] -> {model: {metric: {mean, std}}}"""
    agg = {}
    for model in fold_results[0]:
        agg[model] = {}
        for metric in fold_results[0][model]:
            vals = [fr[model][metric] for fr in fold_results]
            agg[model][metric] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals)),
            }
    return agg


# ----------------------------------------------------------------------------------------------------------------------

def e1(results: dict, datasets: dict, only: set | None = None) -> None:
    exp = results.setdefault('E1', {})
    for name, (task, df) in datasets.items():
        if only and name not in only:
            continue
        missing = {m for m in E1_MODELS if m not in exp.get(name, {})}
        if not missing:
            continue
        print('[E1] %s (%s, %d rows; models: %s) ...'
              % (name, task, len(df), ', '.join(sorted(missing))),
              flush=True)
        t0 = time.time()

        def splits():
            if name in SINGLE_SPLIT:
                train, test = train_test_split(
                    df, test_size=.25, random_state=SEED,
                    stratify=df['target'] if task == 'clf' else None
                )
                yield (train.reset_index(drop=True),
                       test.reset_index(drop=True))
            else:
                yield from folds(task, df)

        # one model at a time, saving after each, so an interrupted run
        # never loses a completed model
        for model in [m for m in E1_MODELS if m in missing]:
            t1 = time.time()
            fold_results = [
                run_split(task, tr, te, models={model})
                for tr, te in splits()
            ]
            exp.setdefault(name, {}).update(aggregate(fold_results))
            save(results)
            print('[E1] %s/%s done in %.1fs'
                  % (name, model, time.time() - t1), flush=True)
        print('[E1] %s done in %.1fs' % (name, time.time() - t0),
              flush=True)


def e2(results: dict, datasets: dict) -> None:
    exp = results.setdefault('E2', {})
    for name in ('wine', 'diabetes'):
        if name in exp:
            continue
        task, df = datasets[name]
        print('[E2] %s ...' % name, flush=True)
        sweep = {}
        for m in (1, 2, 5, 10, 15):
            fold_results = []
            for train, test in folds(task, df):
                classes = (
                    sorted(train['target'].unique()) if task == 'clf'
                    else None
                )
                uniforms = Uniforms(train)
                forest = JPTForest(
                    n_estimators=m, min_samples_leaf=MSL, random_state=SEED
                ).learn(train)
                r = {'ll': mixture_ll(
                    list(zip(forest.weights, forest.members)),
                    test, uniforms
                )}
                if task == 'clf':
                    pred = np.asarray(classes)[
                        mixture_class_scores(
                            forest, test, 'target', classes, uniforms
                        ).argmax(axis=1)
                    ]
                    r['acc'] = float(
                        accuracy_score(test['target'], pred)
                    )
                else:
                    pred = mixture_posterior_mean(
                        forest, test, 'target', uniforms
                    )
                    r['r2'] = float(r2_score(test['target'], pred))
                fold_results.append({'forest': r})
            sweep[str(m)] = aggregate(fold_results)['forest']
        exp[name] = sweep
        save(results)
        print('[E2] %s done' % name, flush=True)


def e3(results: dict, datasets: dict) -> None:
    exp = results.setdefault('E3', {})
    for name in ('iris', 'diabetes'):
        if name in exp:
            continue
        task, df = datasets[name]
        print('[E3] %s ...' % name, flush=True)
        train, test = train_test_split(
            df, test_size=.25, random_state=SEED,
            stratify=df['target'] if task == 'clf' else None
        )
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)
        uniforms = Uniforms(train)
        lboost = JPTLikelihoodBoost(
            n_rounds=LB_ROUNDS, min_samples_leaf=MSL, random_state=SEED
        ).learn(train)
        # FW prefix mixtures: renormalizing the first k final weights
        # exactly recovers the round-k mixture (uniform (1-alpha) scaling).
        densities = np.array([
            np.clip(smoothed_density(m, test, uniforms), EPS, None)
            for m in lboost.members
        ])
        train_densities = np.array([
            np.clip(smoothed_density(m, train, uniforms), EPS, None)
            for m in lboost.members
        ])
        curve = []
        for k in range(1, len(lboost.members) + 1):
            w = np.asarray(lboost.weights[:k])
            w = w / w.sum()
            curve.append({
                'round': k,
                'train_ll': mean_ll(w @ train_densities[:k]),
                'test_ll': mean_ll(w @ densities[:k]),
            })
        exp[name] = curve
        save(results)
        print('[E3] %s done' % name, flush=True)


def e4(results: dict, datasets: dict) -> None:
    exp = results.setdefault('E4', {})
    for name in ('iris', 'wine', 'breast_cancer'):
        if name in exp:
            continue
        task, df = datasets[name]
        print('[E4] %s ...' % name, flush=True)
        by_alpha = {}
        for alpha in (0., 1.):
            fold_results = []
            for train, test in folds(task, df):
                classes = sorted(train['target'].unique())
                uniforms = Uniforms(train)
                single = JPT(
                    variables=infer_from_dataframe(
                        train, scale_numeric_types=False, prior_alpha=alpha
                    ),
                    min_samples_leaf=MSL
                ).learn(train)
                forest = JPTForest(
                    n_estimators=M_FOREST, min_samples_leaf=MSL,
                    prior_alpha=alpha, random_state=SEED
                ).learn(train)
                fr = {
                    'jpt': {'ll': mixture_ll(
                        [(1., single)], test, uniforms)},
                    'forest': {'ll': mixture_ll(
                        list(zip(forest.weights, forest.members)),
                        test, uniforms)},
                }
                for label, model in (('jpt', single), ('forest', forest)):
                    scores = (
                        class_scores(model, test, 'target', classes,
                                     uniforms)
                        if label == 'jpt'
                        else mixture_class_scores(
                            model, test, 'target', classes, uniforms
                        )
                    )
                    fr[label]['acc'] = float(accuracy_score(
                        test['target'],
                        np.asarray(classes)[scores.argmax(axis=1)]
                    ))
                fold_results.append(fr)
            by_alpha[str(alpha)] = aggregate(fold_results)
        exp[name] = by_alpha
        save(results)
        print('[E4] %s done' % name, flush=True)


# ----------------------------------------------------------------------------------------------------------------------

def main() -> None:
    args = set(sys.argv[1:])
    which = {a for a in args if a.startswith('E')} or \
        {'E1', 'E2', 'E3', 'E4'}
    only = {a for a in args if not a.startswith('E')} or None
    datasets = load_datasets()
    results = load_results()
    if 'E4' in which:
        e4(results, datasets)
    if 'E3' in which:
        e3(results, datasets)
    if 'E2' in which:
        e2(results, datasets)
    if 'E1' in which:
        e1(results, datasets, only)
    print('all done.', flush=True)


if __name__ == '__main__':
    main()
