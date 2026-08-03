"""Format results*.json into LaTeX table bodies / pgfplots coordinates."""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

CLF = ['iris', 'wine', 'breast_cancer', 'digits']
REG = ['diabetes', 'abalone']
GEN = ['jpt', 'forest', 'lboost']
ALL = ['jpt', 'forest', 'lboost', 'jptboost', 'rf', 'histgb']


def merged() -> dict:
    """Deep-merge all results*.json files (parallel per-dataset runs)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(HERE, 'results*.json'))):
        with open(path) as f:
            for exp, data in json.load(f).items():
                out.setdefault(exp, {}).update(data)
    return out


def fmt(entry, digits=2, pm=True):
    if entry is None:
        return '---'
    m, s = entry['mean'], entry['std']
    if pm and s > 0:
        return '$%.*f \\pm %.*f$' % (digits, m, digits, s)
    return '$%.*f$' % (digits, m)


def main():
    r = merged()
    e1 = r.get('E1', {})

    print('%% ---- Table: held-out joint log-likelihood (E1)')
    for ds in CLF + REG:
        if ds not in e1:
            continue
        row = [ds.replace('_', '\\_')]
        for model in GEN:
            row.append(fmt(e1[ds].get(model, {}).get('ll')))
        print(' & '.join(row) + ' \\\\')

    print()
    print('%% ---- Table: predictive quality (E1), classification (acc)')
    for ds in CLF:
        if ds not in e1:
            continue
        row = [ds.replace('_', '\\_')]
        for model in ALL:
            row.append(fmt(e1[ds].get(model, {}).get('acc'), 3))
        print(' & '.join(row) + ' \\\\')

    print()
    print('%% ---- Table: predictive quality (E1), regression (R2)')
    for ds in REG:
        if ds not in e1:
            continue
        row = [ds.replace('_', '\\_')]
        for model in ALL:
            row.append(fmt(e1[ds].get(model, {}).get('r2'), 3))
        print(' & '.join(row) + ' \\\\')

    print()
    print('%% ---- E1 training times (seconds, mean)')
    for ds in CLF + REG:
        if ds not in e1:
            continue
        row = [ds.replace('_', '\\_')]
        for model in ALL:
            t = e1[ds].get(model, {}).get('time')
            row.append('$%.1f$' % t['mean'] if t else '---')
        print(' & '.join(row) + ' \\\\')

    print()
    print('%% ---- E2 sweep coordinates')
    for ds, metric in (('wine', 'acc'), ('wine', 'll'),
                       ('diabetes', 'r2'), ('diabetes', 'll')):
        sweep = r.get('E2', {}).get(ds, {})
        coords = ' '.join(
            '(%s,%.4f)' % (m, sweep[m][metric]['mean'])
            for m in sorted(sweep, key=int) if metric in sweep[m]
        )
        print('%% %s %s: %s' % (ds, metric, coords))

    print()
    print('%% ---- E3 boosting curves')
    for ds in ('iris', 'diabetes'):
        curve = r.get('E3', {}).get(ds, [])
        print('%% %s train: %s' % (ds, ' '.join(
            '(%d,%.4f)' % (c['round'], c['train_ll']) for c in curve)))
        print('%% %s test:  %s' % (ds, ' '.join(
            '(%d,%.4f)' % (c['round'], c['test_ll']) for c in curve)))

    print()
    print('%% ---- E4 prior_alpha ablation (ll / acc, alpha=0 vs alpha=1)')
    for ds, by_alpha in r.get('E4', {}).items():
        row = [ds.replace('_', '\\_')]
        for alpha in ('0.0', '1.0'):
            for model in ('jpt', 'forest'):
                entry = by_alpha.get(alpha, {}).get(model, {})
                row.append(fmt(entry.get('ll')))
                row.append(fmt(entry.get('acc'), 3))
        print(' & '.join(row) + ' \\\\')


if __name__ == '__main__':
    main()
