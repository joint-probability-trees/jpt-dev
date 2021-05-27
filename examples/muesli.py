import pyximport
pyximport.install()
import itertools
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dnutils import out
from jpt.base.quantiles import Quantiles
from jpt.learning.distributions import Numeric, Bool, SymbolicType
from jpt.trees import JPT
from jpt.variables import SymbolicVariable, NumericVariable
from jpt.base.intervals import ContinuousSet as Interval


def plot_muesli():
    # Used to create the plots for the paper

    # df = pd.read_pickle('data/human_muesli.dat')
    df = pd.read_csv('data/muesli.csv')
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)

    fig, ax = plt.subplots()
    ax.set_title(f"Breakfast object positions")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # verschiedene Farben UND Formen (Farben fuer prettyprinting, Formen fuer Unterscheidung in b/w Paper)
    icons = itertools.cycle(['o', 'x', '1'])
    colors = ['orange', 'cornflowerblue', 'green', 'yellow']

    # whether extra dimension (success) is used:
    succ = False
    if succ:
        for icon, clazz in zip(icons, df['Class'].unique()):
            samples = df[df['Class'] == clazz]
            for color, succ in zip(colors, df['Success'].unique()):
                spls = samples[samples['Success'] == succ]
                ax.scatter(spls['X'], spls['Y'], color=color, marker=icon, label=f'{clazz} $\u2192$ {succ}')
    else:
        for color, icon, clazz in zip(colors, icons, df['Class'].unique()):
            samples = df[df['Class'] == clazz]
            ax.scatter(samples['X'], samples['Y'], color=color, marker=icon, label=clazz)

    ax.legend()
    plt.show()

    with PdfPages(os.path.join('data', f'muesli-{2 if succ else 1}.pdf'), metadata={'Creator': 'misc',
                                                                                    'Author': 'Picklum & Nyga',
                                                                                    'Title': f'Müsli Example{" Success" if succ else ""}'}) as pdf:
        pdf.savefig(fig)


def test_muesli():
    # TODO: check! This does not work anymore -> fix or delete

    data = pd.read_csv('../examples/data/muesli.csv')
    d = np.array(sorted(data['X']), dtype=np.float64)

    quantiles = Quantiles(d, epsilon=.0001)
    cdf_ = quantiles.cdf()
    d = Numeric(quantile=quantiles)

    interval = Interval(-2.05, -2.0)
    p = d.p(interval)
    out('query', interval, p)

    print(d.cdf.pfmt())
    d.plot(title='Piecewise Linear CDF of Breakfast Data', fname='BreakfastPiecewise', xlabel='X', view=True, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))


def muesli_tree():
    # generate Joint Probability Tree from muesli data (use .csv file because it contains the additional Success column)
    data = pd.read_csv('../examples/data/muesli.csv')
    ObjectType = SymbolicType('ObjectType', data['Class'].unique())

    x = NumericVariable('X', Numeric)
    y = NumericVariable('Y', Numeric)
    o = SymbolicVariable('Object', ObjectType)
    s = SymbolicVariable('Success', Bool)

    jpt = JPT([x, y, o, s], name="Breakfasttree", min_samples_leaf=5)
    jpt.learn(columns=data.values.T)
    jpt.plot(plotvars=[x, y, o], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))

    for clazz in data['Class'].unique():
        out(jpt.infer(query={o: clazz}, evidence={x: [.9, None], y: [None, .45]}))
    print()

    for clazz in data['Class'].unique():
        for exp in jpt.expectation([x, y], evidence={o: clazz}, confidence_level=.1):
            out(exp)

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    # jpt.plot(plotvars=[x, y, o, s])

    # q = {o: ("BowlLarge_Bdvg", "JaNougatBits_UE0O"), x: [.812, .827]}
    # r = jpt.reverse(q)
    # out('Query:', q, 'result:', pprint.pformat(r))


def picklemuesli():
    # read in original .pkl file and pickle as proper np.array
    f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = np.array(pickle.load(fi))

    transformed = []
    for c in data.T:
        try:
            transformed.append(np.array(c, dtype=float))
        except:
            transformed.append(np.array(c))

    with open(f, 'wb+') as fi:
        pickle.dump(transformed, fi)


def main(*args):
    # plot_muesli()
    # test_muesli()
    muesli_tree()
    # picklemuesli()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
