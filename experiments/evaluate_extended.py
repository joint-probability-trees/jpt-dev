"""
Extended evaluation of split validation for JPT learning.

Adds:
- Likelihood-based evaluation metrics (mean log-likelihood)
- Real-world datasets (Iris, Wine, Breast Cancer from sklearn)
- Experiments 6-8 with real data
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '/home/user/jpt-dev/src')

from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.distributions import SymbolicType


# --------------------------------------------------------------------------- #
# Data generation (synthetic)
# --------------------------------------------------------------------------- #

def make_classification_data(n_samples, n_features, noise_rate, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    n_flip = int(noise_rate * n_samples)
    if n_flip > 0:
        flip_idx = rng.choice(n_samples, n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]
    return X, y


def make_regression_data(n_samples, n_features, noise_std, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.where(X[:, 0] < 0, X[:, 0] * 2 + X[:, 1], -X[:, 0] + X[:, 1] + 3)
    y += rng.randn(n_samples) * noise_std
    return X, y


# --------------------------------------------------------------------------- #
# Real-world dataset loaders
# --------------------------------------------------------------------------- #

def load_real_dataset(name):
    """Load a real dataset, return X (float64), y (int), class_labels, feature_names."""
    if name == 'iris':
        d = load_iris()
    elif name == 'wine':
        d = load_wine()
    elif name == 'breast_cancer':
        d = load_breast_cancer()
    else:
        raise ValueError(f'Unknown dataset: {name}')
    return (d.data.astype(np.float64), d.target.astype(int),
            [str(c) for c in d.target_names], list(d.feature_names))


# --------------------------------------------------------------------------- #
# JPT construction
# --------------------------------------------------------------------------- #

def build_cls_jpt(n_features, n_classes, class_labels, feature_names=None, msl=3):
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(n_features)]
    feats = [NumericVariable(feature_names[i]) for i in range(n_features)]
    YT = SymbolicType('YT', class_labels)
    tgt = SymbolicVariable('y', YT)
    return JPT(variables=feats + [tgt], targets=[tgt], min_samples_leaf=msl)


def build_reg_jpt(n_features, msl=5, feature_names=None):
    if feature_names is None:
        feature_names = [f'x{i}' for i in range(n_features)]
    feats = [NumericVariable(feature_names[i]) for i in range(n_features)]
    tgt = NumericVariable('y')
    return JPT(variables=feats + [tgt], targets=[tgt], min_samples_leaf=msl)


def to_df_cls(X, y, feature_names, class_labels):
    d = {feature_names[i]: X[:, i] for i in range(X.shape[1])}
    d['y'] = [class_labels[int(v)] for v in y]
    return pd.DataFrame(d)


def to_df_reg(X, y, feature_names):
    d = {feature_names[i]: X[:, i] for i in range(X.shape[1])}
    d['y'] = y.astype(np.float64)
    return pd.DataFrame(d)


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #

def predict_classification(jpt, X, feature_names):
    tgt = jpt.targets[0]
    preds = np.full(len(X), -1, dtype=int)
    for i in range(len(X)):
        evidence = {feature_names[j]: float(X[i, j]) for j in range(X.shape[1])}
        leaves = list(jpt.apply(evidence))
        if leaves:
            best = max(leaves, key=lambda l: l.prior)
            probs = best.distributions[tgt].probabilities
            preds[i] = int(np.argmax(probs))
    return preds


def predict_regression(jpt, X, feature_names):
    tgt = jpt.targets[0]
    preds = np.zeros(len(X))
    for i in range(len(X)):
        evidence = {feature_names[j]: float(X[i, j]) for j in range(X.shape[1])}
        leaves = list(jpt.apply(evidence))
        if leaves:
            best = max(leaves, key=lambda l: l.prior)
            preds[i] = best.distributions[tgt].expectation()
    return preds


# --------------------------------------------------------------------------- #
# Likelihood computation
# --------------------------------------------------------------------------- #

def compute_log_likelihood(jpt, df):
    """Compute mean log-likelihood. Handles zero likelihoods gracefully."""
    ll = jpt.likelihood(df)
    # Replace zeros with a small value to avoid -inf
    ll = np.where(ll > 0, ll, 1e-300)
    return np.mean(np.log(ll))


# --------------------------------------------------------------------------- #
# Run helpers
# --------------------------------------------------------------------------- #

def make_mask(n, ratio, seed):
    rng = np.random.RandomState(seed)
    m = (rng.rand(n) < ratio).astype(np.uint8)
    if not np.any(m):
        m[0] = 1
    if not np.any(1 - m):
        m[-1] = 0
    return m


def run_cls(X_tr, y_tr, X_te, y_te, feature_names, class_labels,
            mask, mode, msl):
    n_classes = len(class_labels)
    nf = X_tr.shape[1]
    jpt = build_cls_jpt(nf, n_classes, class_labels, feature_names, msl)
    df_tr = to_df_cls(X_tr, y_tr, feature_names, class_labels)
    df_te = to_df_cls(X_te, y_te, feature_names, class_labels)

    t0 = time.time()
    jpt.fit(df_tr, multicore=1, split_validation_mask=mask,
            split_validation_mode=mode)
    t = time.time() - t0

    p_tr = predict_classification(jpt, X_tr, feature_names)
    p_te = predict_classification(jpt, X_te, feature_names)
    acc_tr = np.mean(p_tr == y_tr)
    acc_te = np.mean(p_te == y_te)

    ll_tr = compute_log_likelihood(jpt, df_tr)
    ll_te = compute_log_likelihood(jpt, df_te)

    return dict(train_acc=acc_tr, test_acc=acc_te,
                train_ll=ll_tr, test_ll=ll_te,
                n_leaves=len(jpt.leaves), time=t)


def run_reg(X_tr, y_tr, X_te, y_te, feature_names, mask, mode, msl):
    nf = X_tr.shape[1]
    jpt = build_reg_jpt(nf, msl, feature_names)
    df_tr = to_df_reg(X_tr, y_tr, feature_names)
    df_te = to_df_reg(X_te, y_te, feature_names)

    t0 = time.time()
    jpt.fit(df_tr, multicore=1, split_validation_mask=mask,
            split_validation_mode=mode)
    t = time.time() - t0

    p_tr = predict_regression(jpt, X_tr, feature_names)
    p_te = predict_regression(jpt, X_te, feature_names)
    mse_tr = np.mean((p_tr - y_tr) ** 2)
    mse_te = np.mean((p_te - y_te) ** 2)

    ll_tr = compute_log_likelihood(jpt, df_tr)
    ll_te = compute_log_likelihood(jpt, df_te)

    return dict(train_mse=mse_tr, test_mse=mse_te,
                train_ll=ll_tr, test_ll=ll_te,
                n_leaves=len(jpt.leaves), time=t)


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def experiment_1(R=10):
    """Overfitting reduction: baseline vs split validation (with likelihood)."""
    print("=" * 70)
    print("EXP 1: Overfitting Reduction (Classification, noise=0.15)")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
    fn = [f'x{i}' for i in range(NF)]
    cl = ['0', '1']
    base, sv = [], []
    for s in range(R):
        X, y = make_classification_data(N_TR + N_TE, NF, NOISE, s)
        Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
        mask = make_mask(N_TR, 0.7, s + 1000)
        base.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, None, 'both', MSL))
        sv.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, mask, 'both', MSL))
        print(f"  seed {s}: base acc={base[-1]['test_acc']:.3f} ll={base[-1]['test_ll']:.2f} "
              f"lv={base[-1]['n_leaves']}  |  "
              f"sv acc={sv[-1]['test_acc']:.3f} ll={sv[-1]['test_ll']:.2f} "
              f"lv={sv[-1]['n_leaves']}")
    _summary2(base, sv)
    return dict(baseline=base, split_val=sv)


def experiment_2(R=10):
    """Effect of mask ratio."""
    print("\n" + "=" * 70)
    print("EXP 2: Mask Ratio Effect")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
    fn = [f'x{i}' for i in range(NF)]
    cl = ['0', '1']
    ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    for ratio in ratios:
        runs = []
        for s in range(R):
            X, y = make_classification_data(N_TR + N_TE, NF, NOISE, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, ratio, s + 2000) if ratio < 1.0 else None
            runs.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, mask, 'both', MSL))
        acc = [r['test_acc'] for r in runs]
        ll = [r['test_ll'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  ratio={ratio:.1f}: acc={np.mean(acc):.3f}+-{np.std(acc):.3f} "
              f"ll={np.mean(ll):.2f}+-{np.std(ll):.2f} "
              f"lv={np.mean(lv):.1f}")
        results[f'{ratio:.1f}'] = runs
    return results


def experiment_3(R=10):
    """Validation mode comparison."""
    print("\n" + "=" * 70)
    print("EXP 3: Validation Mode Comparison")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
    fn = [f'x{i}' for i in range(NF)]
    cl = ['0', '1']
    configs = [
        ('no_mask', None, 'both'),
        ('both', 'mask', 'both'),
        ('training', 'mask', 'training'),
        ('evaluation', 'mask', 'evaluation'),
    ]
    results = {}
    for label, mt, mode in configs:
        runs = []
        for s in range(R):
            X, y = make_classification_data(N_TR + N_TE, NF, NOISE, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, 0.7, s + 3000) if mt else None
            runs.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, mask, mode, MSL))
        acc = [r['test_acc'] for r in runs]
        ll = [r['test_ll'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  {label:15s}: acc={np.mean(acc):.3f}+-{np.std(acc):.3f} "
              f"ll={np.mean(ll):.2f}+-{np.std(ll):.2f} "
              f"lv={np.mean(lv):.1f}")
        results[label] = runs
    return results


def experiment_4(R=10):
    """Noise level sweep."""
    print("\n" + "=" * 70)
    print("EXP 4: Noise Level Sensitivity")
    print("=" * 70)
    N_TR, N_TE, NF, MSL = 200, 500, 5, 3
    fn = [f'x{i}' for i in range(NF)]
    cl = ['0', '1']
    noises = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}
    for noise in noises:
        base_r, sv_r = [], []
        for s in range(R):
            X, y = make_classification_data(N_TR + N_TE, NF, noise, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, 0.7, s + 4000)
            base_r.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, None, 'both', MSL))
            sv_r.append(run_cls(Xtr, ytr, Xte, yte, fn, cl, mask, 'both', MSL))
        ba = [r['test_acc'] for r in base_r]
        sa = [r['test_acc'] for r in sv_r]
        bl_ll = [r['test_ll'] for r in base_r]
        sv_ll = [r['test_ll'] for r in sv_r]
        print(f"  noise={noise:.2f}: base acc={np.mean(ba):.3f} ll={np.mean(bl_ll):.2f} | "
              f"sv acc={np.mean(sa):.3f} ll={np.mean(sv_ll):.2f}")
        results[f'{noise:.2f}'] = dict(baseline=base_r, split_val=sv_r)
    return results


def experiment_5(R=10):
    """Regression with likelihood."""
    print("\n" + "=" * 70)
    print("EXP 5: Regression")
    print("=" * 70)
    N_TR, N_TE, NF, MSL = 200, 500, 3, 5
    fn = [f'x{i}' for i in range(NF)]
    noises = [0.5, 1.0, 2.0]
    results = {}
    for noise in noises:
        base_r, sv_r = [], []
        for s in range(R):
            X, y = make_regression_data(N_TR + N_TE, NF, noise, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, 0.7, s + 5000)
            base_r.append(run_reg(Xtr, ytr, Xte, yte, fn, None, 'both', MSL))
            sv_r.append(run_reg(Xtr, ytr, Xte, yte, fn, mask, 'both', MSL))
        bm = [r['test_mse'] for r in base_r]
        sm = [r['test_mse'] for r in sv_r]
        bl_ll = [r['test_ll'] for r in base_r]
        sv_ll = [r['test_ll'] for r in sv_r]
        print(f"  noise={noise:.1f}: base MSE={np.mean(bm):.3f} ll={np.mean(bl_ll):.2f} | "
              f"sv MSE={np.mean(sm):.3f} ll={np.mean(sv_ll):.2f}")
        results[f'{noise:.1f}'] = dict(baseline=base_r, split_val=sv_r)
    return results


def experiment_6_real_data():
    """Real-world datasets with stratified 5-fold CV."""
    print("\n" + "=" * 70)
    print("EXP 6: Real-World Datasets (5-Fold CV)")
    print("=" * 70)
    datasets = ['iris', 'wine', 'breast_cancer']
    results = {}
    for ds_name in datasets:
        X, y, class_labels, feat_names = load_real_dataset(ds_name)
        # Sanitize feature names (remove special chars for JPT variable names)
        feat_names = [f.replace(' ', '_').replace('(', '').replace(')', '')
                      .replace('/', '_') for f in feat_names]
        n_classes = len(class_labels)
        nf = X.shape[1]

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_runs, sv_runs = [], []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            Xtr, ytr = X[train_idx], y[train_idx]
            Xte, yte = X[test_idx], y[test_idx]
            mask = make_mask(len(train_idx), 0.7, fold + 6000)

            msl = max(3, len(train_idx) // 50)
            base_runs.append(run_cls(Xtr, ytr, Xte, yte, feat_names,
                                     class_labels, None, 'both', msl))
            sv_runs.append(run_cls(Xtr, ytr, Xte, yte, feat_names,
                                   class_labels, mask, 'both', msl))

        ba = [r['test_acc'] for r in base_runs]
        sa = [r['test_acc'] for r in sv_runs]
        bl_ll = [r['test_ll'] for r in base_runs]
        sv_ll = [r['test_ll'] for r in sv_runs]
        bl = [r['n_leaves'] for r in base_runs]
        sl = [r['n_leaves'] for r in sv_runs]
        t, p = sp_stats.ttest_rel(ba, sa)
        t_ll, p_ll = sp_stats.ttest_rel(bl_ll, sv_ll)

        print(f"\n  {ds_name} (n={len(X)}, d={nf}, k={n_classes}):")
        print(f"    Baseline : acc={np.mean(ba):.3f}+-{np.std(ba):.3f}  "
              f"ll={np.mean(bl_ll):.2f}+-{np.std(bl_ll):.2f}  "
              f"lv={np.mean(bl):.1f}")
        print(f"    Split Val: acc={np.mean(sa):.3f}+-{np.std(sa):.3f}  "
              f"ll={np.mean(sv_ll):.2f}+-{np.std(sv_ll):.2f}  "
              f"lv={np.mean(sl):.1f}")
        print(f"    p-value (acc): {p:.4f}   p-value (ll): {p_ll:.4f}")

        results[ds_name] = dict(baseline=base_runs, split_val=sv_runs)
    return results


def experiment_7_real_modes():
    """Mode comparison on real data (breast cancer - largest dataset)."""
    print("\n" + "=" * 70)
    print("EXP 7: Mode Comparison on Breast Cancer Dataset")
    print("=" * 70)
    X, y, class_labels, feat_names = load_real_dataset('breast_cancer')
    feat_names = [f.replace(' ', '_').replace('(', '').replace(')', '')
                  .replace('/', '_') for f in feat_names]
    nf = X.shape[1]

    configs = [
        ('no_mask', None, 'both'),
        ('both', 'mask', 'both'),
        ('training', 'mask', 'training'),
        ('evaluation', 'mask', 'evaluation'),
    ]
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for label, mt, mode in configs:
        runs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            Xtr, ytr = X[train_idx], y[train_idx]
            Xte, yte = X[test_idx], y[test_idx]
            mask = make_mask(len(train_idx), 0.7, fold + 7000) if mt else None
            msl = max(3, len(train_idx) // 50)
            runs.append(run_cls(Xtr, ytr, Xte, yte, feat_names,
                                class_labels, mask, mode, msl))
        acc = [r['test_acc'] for r in runs]
        ll = [r['test_ll'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  {label:15s}: acc={np.mean(acc):.3f}+-{np.std(acc):.3f}  "
              f"ll={np.mean(ll):.2f}+-{np.std(ll):.2f}  "
              f"lv={np.mean(lv):.1f}")
        results[label] = runs
    return results


def experiment_8_real_mask_ratio():
    """Mask ratio sweep on Wine dataset."""
    print("\n" + "=" * 70)
    print("EXP 8: Mask Ratio on Wine Dataset")
    print("=" * 70)
    X, y, class_labels, feat_names = load_real_dataset('wine')
    feat_names = [f.replace(' ', '_').replace('(', '').replace(')', '')
                  .replace('/', '_') for f in feat_names]
    nf = X.shape[1]
    ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for ratio in ratios:
        runs = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            Xtr, ytr = X[train_idx], y[train_idx]
            Xte, yte = X[test_idx], y[test_idx]
            mask = make_mask(len(train_idx), ratio, fold + 8000) if ratio < 1.0 else None
            msl = max(3, len(train_idx) // 50)
            runs.append(run_cls(Xtr, ytr, Xte, yte, feat_names,
                                class_labels, mask, 'both', msl))
        acc = [r['test_acc'] for r in runs]
        ll = [r['test_ll'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  ratio={ratio:.1f}: acc={np.mean(acc):.3f}+-{np.std(acc):.3f}  "
              f"ll={np.mean(ll):.2f}+-{np.std(ll):.2f}  "
              f"lv={np.mean(lv):.1f}")
        results[f'{ratio:.1f}'] = runs
    return results


# --------------------------------------------------------------------------- #
# Summary helper
# --------------------------------------------------------------------------- #

def _summary2(a, b):
    for key in ['test_acc', 'train_acc', 'test_ll', 'train_ll', 'n_leaves']:
        if key not in a[0]:
            continue
        va = [r[key] for r in a]
        vb = [r[key] for r in b]
        t, p = sp_stats.ttest_rel(va, vb)
        print(f"  {key:12s}: baseline={np.mean(va):.3f}+-{np.std(va):.3f}  "
              f"sv={np.mean(vb):.3f}+-{np.std(vb):.3f}  (p={p:.4f})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    R = 10
    print(f"Running extended experiments (R={R} repeats for synthetic, 5-fold CV for real)...\n")

    all_results = {}
    all_results['exp1'] = experiment_1(R)
    all_results['exp2'] = experiment_2(R)
    all_results['exp3'] = experiment_3(R)
    all_results['exp4'] = experiment_4(R)
    all_results['exp5'] = experiment_5(R)
    all_results['exp6'] = experiment_6_real_data()
    all_results['exp7'] = experiment_7_real_modes()
    all_results['exp8'] = experiment_8_real_mask_ratio()

    with open('/home/user/jpt-dev/experiments/results_extended.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n\nResults saved to experiments/results_extended.json")
