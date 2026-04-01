"""
Scientific evaluation of the split validation feature for JPT learning.

Experiments:
1. Overfitting reduction on noisy classification data
2. Effect of mask ratio (train/eval split proportion)
3. Comparison of validation modes (both, training, evaluation)
4. Impact across different noise levels
5. Regression performance
"""

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

sys.path.insert(0, '/home/user/jpt-dev/src')

from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable
from jpt.distributions import SymbolicType


# --------------------------------------------------------------------------- #
# Data generation
# --------------------------------------------------------------------------- #

def make_classification_data(n_samples, n_features, noise_rate, seed):
    """Binary classification: y = 1 if x0 + x1 > 0, with label noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    n_flip = int(noise_rate * n_samples)
    if n_flip > 0:
        flip_idx = rng.choice(n_samples, n_flip, replace=False)
        y[flip_idx] = 1 - y[flip_idx]
    return X, y


def make_regression_data(n_samples, n_features, noise_std, seed):
    """Regression: piecewise-linear with Gaussian noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = np.where(X[:, 0] < 0, X[:, 0] * 2 + X[:, 1], -X[:, 0] + X[:, 1] + 3)
    y += rng.randn(n_samples) * noise_std
    return X, y


# --------------------------------------------------------------------------- #
# JPT construction helpers
# --------------------------------------------------------------------------- #

YT = SymbolicType('YT', ['0', '1'])


def build_cls_jpt(n_features, min_samples_leaf):
    feats = [NumericVariable(f'x{i}') for i in range(n_features)]
    tgt = SymbolicVariable('y', YT)
    return JPT(variables=feats + [tgt], targets=[tgt],
               min_samples_leaf=min_samples_leaf)


def build_reg_jpt(n_features, min_samples_leaf):
    feats = [NumericVariable(f'x{i}') for i in range(n_features)]
    tgt = NumericVariable('y')
    return JPT(variables=feats + [tgt], targets=[tgt],
               min_samples_leaf=min_samples_leaf)


def to_df(X, y, n_features, classification):
    d = {f'x{i}': X[:, i] for i in range(n_features)}
    d['y'] = [str(int(v)) for v in y] if classification else y.astype(np.float64)
    return pd.DataFrame(d)


# --------------------------------------------------------------------------- #
# Leaf-based prediction
# --------------------------------------------------------------------------- #

def predict_classification(jpt, X, n_features):
    """Predict class labels using the highest-prior matching leaf."""
    tgt = jpt.targets[0]
    preds = np.full(len(X), -1, dtype=int)
    for i in range(len(X)):
        evidence = {f'x{j}': float(X[i, j]) for j in range(n_features)}
        leaves = list(jpt.apply(evidence))
        if leaves:
            best = max(leaves, key=lambda l: l.prior)
            probs = best.distributions[tgt].probabilities
            preds[i] = int(np.argmax(probs))
    return preds


def predict_regression(jpt, X, n_features):
    """Predict target mean using the highest-prior matching leaf."""
    tgt = jpt.targets[0]
    preds = np.zeros(len(X))
    for i in range(len(X)):
        evidence = {f'x{j}': float(X[i, j]) for j in range(n_features)}
        leaves = list(jpt.apply(evidence))
        if leaves:
            best = max(leaves, key=lambda l: l.prior)
            preds[i] = best.distributions[tgt].expectation()
    return preds


# --------------------------------------------------------------------------- #
# Training + evaluation
# --------------------------------------------------------------------------- #

def run_cls(X_tr, y_tr, X_te, y_te, nf, mask, mode, msl):
    jpt = build_cls_jpt(nf, msl)
    df = to_df(X_tr, y_tr, nf, True)
    t0 = time.time()
    jpt.fit(df, multicore=1, split_validation_mask=mask,
            split_validation_mode=mode)
    t = time.time() - t0

    p_tr = predict_classification(jpt, X_tr, nf)
    p_te = predict_classification(jpt, X_te, nf)
    acc_tr = np.mean(p_tr == y_tr)
    acc_te = np.mean(p_te == y_te)
    return dict(train_acc=acc_tr, test_acc=acc_te,
                n_leaves=len(jpt.leaves), time=t)


def run_reg(X_tr, y_tr, X_te, y_te, nf, mask, mode, msl):
    jpt = build_reg_jpt(nf, msl)
    df = to_df(X_tr, y_tr, nf, False)
    t0 = time.time()
    jpt.fit(df, multicore=1, split_validation_mask=mask,
            split_validation_mode=mode)
    t = time.time() - t0

    p_tr = predict_regression(jpt, X_tr, nf)
    p_te = predict_regression(jpt, X_te, nf)
    mse_tr = np.mean((p_tr - y_tr) ** 2)
    mse_te = np.mean((p_te - y_te) ** 2)
    return dict(train_mse=mse_tr, test_mse=mse_te,
                n_leaves=len(jpt.leaves), time=t)


def make_mask(n, ratio, seed):
    rng = np.random.RandomState(seed)
    m = (rng.rand(n) < ratio).astype(np.uint8)
    if not np.any(m):
        m[0] = 1
    if not np.any(1 - m):
        m[-1] = 0
    return m


# --------------------------------------------------------------------------- #
# Experiments
# --------------------------------------------------------------------------- #

def experiment_1(R=10):
    """Overfitting reduction: baseline vs split validation."""
    print("=" * 70)
    print("EXP 1: Overfitting Reduction (Classification, noise=0.15)")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
    base, sv = [], []
    for s in range(R):
        X, y = make_classification_data(N_TR + N_TE, NF, NOISE, s)
        Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
        mask = make_mask(N_TR, 0.7, s + 1000)
        base.append(run_cls(Xtr, ytr, Xte, yte, NF, None, 'both', MSL))
        sv.append(run_cls(Xtr, ytr, Xte, yte, NF, mask, 'both', MSL))
        print(f"  seed {s}: base test={base[-1]['test_acc']:.3f} "
              f"leaves={base[-1]['n_leaves']:3d}  |  "
              f"sv test={sv[-1]['test_acc']:.3f} "
              f"leaves={sv[-1]['n_leaves']:3d}")
    _summary('test_acc', base, sv, 'baseline', 'split_val')
    _summary('train_acc', base, sv, 'baseline', 'split_val')
    _summary('n_leaves', base, sv, 'baseline', 'split_val')
    return dict(baseline=base, split_val=sv)


def experiment_2(R=10):
    """Effect of mask ratio."""
    print("\n" + "=" * 70)
    print("EXP 2: Mask Ratio Effect")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
    ratios = [0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    for ratio in ratios:
        runs = []
        for s in range(R):
            X, y = make_classification_data(N_TR + N_TE, NF, NOISE, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, ratio, s + 2000) if ratio < 1.0 else None
            runs.append(run_cls(Xtr, ytr, Xte, yte, NF, mask, 'both', MSL))
        acc = [r['test_acc'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  ratio={ratio:.1f}: acc={np.mean(acc):.3f}±{np.std(acc):.3f} "
              f"leaves={np.mean(lv):.1f}±{np.std(lv):.1f}")
        results[f'{ratio:.1f}'] = runs
    return results


def experiment_3(R=10):
    """Validation mode comparison."""
    print("\n" + "=" * 70)
    print("EXP 3: Validation Mode Comparison")
    print("=" * 70)
    N_TR, N_TE, NF, NOISE, MSL = 200, 500, 5, 0.15, 3
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
            runs.append(run_cls(Xtr, ytr, Xte, yte, NF, mask, mode, MSL))
        acc = [r['test_acc'] for r in runs]
        lv = [r['n_leaves'] for r in runs]
        print(f"  {label:15s}: acc={np.mean(acc):.3f}±{np.std(acc):.3f} "
              f"leaves={np.mean(lv):.1f}±{np.std(lv):.1f}")
        results[label] = runs
    return results


def experiment_4(R=10):
    """Noise level sweep."""
    print("\n" + "=" * 70)
    print("EXP 4: Noise Level Sensitivity")
    print("=" * 70)
    N_TR, N_TE, NF, MSL = 200, 500, 5, 3
    noises = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}
    for noise in noises:
        base_runs, sv_runs = [], []
        for s in range(R):
            X, y = make_classification_data(N_TR + N_TE, NF, noise, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, 0.7, s + 4000)
            base_runs.append(run_cls(Xtr, ytr, Xte, yte, NF, None, 'both', MSL))
            sv_runs.append(run_cls(Xtr, ytr, Xte, yte, NF, mask, 'both', MSL))
        ba = [r['test_acc'] for r in base_runs]
        sa = [r['test_acc'] for r in sv_runs]
        bl = [r['n_leaves'] for r in base_runs]
        sl = [r['n_leaves'] for r in sv_runs]
        print(f"  noise={noise:.2f}: base={np.mean(ba):.3f}±{np.std(ba):.3f} "
              f"(lv={np.mean(bl):.0f}) | "
              f"sv={np.mean(sa):.3f}±{np.std(sa):.3f} "
              f"(lv={np.mean(sl):.0f})")
        results[f'{noise:.2f}'] = dict(baseline=base_runs, split_val=sv_runs)
    return results


def experiment_5(R=10):
    """Regression performance."""
    print("\n" + "=" * 70)
    print("EXP 5: Regression")
    print("=" * 70)
    N_TR, N_TE, NF, MSL = 200, 500, 3, 5
    noises = [0.5, 1.0, 2.0]
    results = {}
    for noise in noises:
        base_runs, sv_runs = [], []
        for s in range(R):
            X, y = make_regression_data(N_TR + N_TE, NF, noise, s)
            Xtr, ytr, Xte, yte = X[:N_TR], y[:N_TR], X[N_TR:], y[N_TR:]
            mask = make_mask(N_TR, 0.7, s + 5000)
            base_runs.append(run_reg(Xtr, ytr, Xte, yte, NF, None, 'both', MSL))
            sv_runs.append(run_reg(Xtr, ytr, Xte, yte, NF, mask, 'both', MSL))
        bm = [r['test_mse'] for r in base_runs]
        sm = [r['test_mse'] for r in sv_runs]
        bl = [r['n_leaves'] for r in base_runs]
        sl = [r['n_leaves'] for r in sv_runs]
        print(f"  noise={noise:.1f}: base MSE={np.mean(bm):.3f}±{np.std(bm):.3f} "
              f"(lv={np.mean(bl):.0f}) | "
              f"sv MSE={np.mean(sm):.3f}±{np.std(sm):.3f} "
              f"(lv={np.mean(sl):.0f})")
        results[f'{noise:.1f}'] = dict(baseline=base_runs, split_val=sv_runs)
    return results


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _summary(key, a, b, la, lb):
    va = [r[key] for r in a]
    vb = [r[key] for r in b]
    t, p = sp_stats.ttest_rel(va, vb)
    print(f"  {key}: {la}={np.mean(va):.3f}±{np.std(va):.3f}  "
          f"{lb}={np.mean(vb):.3f}±{np.std(vb):.3f}  "
          f"(paired-t p={p:.4f})")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    R = 10
    print(f"Running experiments (R={R} repeats)...\n")

    all_results = {}
    all_results['exp1'] = experiment_1(R)
    all_results['exp2'] = experiment_2(R)
    all_results['exp3'] = experiment_3(R)
    all_results['exp4'] = experiment_4(R)
    all_results['exp5'] = experiment_5(R)

    with open('/home/user/jpt-dev/experiments/results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n\nResults saved to experiments/results.json")
