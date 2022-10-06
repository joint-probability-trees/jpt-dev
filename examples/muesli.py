
from jpt.base.utils import format_path

import itertools
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dnutils import out, ifnone
from jpt.distributions.quantile.quantiles import QuantileDistribution
from jpt.distributions import Numeric, Bool, SymbolicType, NumericType
from jpt.trees import JPT
from jpt.variables import SymbolicVariable, NumericVariable
from jpt.base.intervals import ContinuousSet


def plot_muesli(visualize=True):
    # Used to create the plots for the paper

    # df = pd.read_pickle('data/human_muesli.dat')
    df = pd.read_csv(os.path.join('..', 'examples', 'data', 'muesli.csv'))
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)

    fig, ax = plt.subplots()
    # ax.set_title(f"Breakfast object positions")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # verschiedene Farben UND Formen (Farben fuer prettyprinting, Formen fuer Unterscheidung in b/w Paper)
    icons = itertools.cycle(['o', 'x', '1'])
    colors = ['orange', 'cornflowerblue', 'green', 'yellow']

    # whether extra dimension (success) is used:
    succ = True
    if succ:
        for icon, clazz in zip(icons, df['Class'].unique()):
            samples = df[df['Class'] == clazz]
            for color, succ in zip(colors, df['Success'].unique()):
                spls = samples[samples['Success'] == succ]
                ax.scatter(spls['X'], spls['Y'], color=color, marker=icon, label=f'{clazz} $\u2192$ {succ}')
            # ax.set_xticks([round(x, 2) for x in np.arange(sorted(df['X'])[0], sorted(df['X'])[-1], .02)])
    else:
        for color, icon, clazz in zip(colors, icons, df['Class'].unique()):
            samples = df[df['Class'] == clazz]
            ax.scatter(samples['X'], samples['Y'], color=color, marker=icon, label=clazz)
            # ax.set_xticks([round(x, 2) for x in np.arange(sorted(df['X'])[0], sorted(df['X'])[-1], .02)])

    ObjectType = SymbolicType('ObjectType', df['Class'].unique())

    x = NumericVariable('X', Numeric, blur=.01)
    y = NumericVariable('Y', Numeric, blur=.01)
    o = SymbolicVariable('Object', ObjectType)
    s = SymbolicVariable('Success', Bool)

    jpt = JPT([x, y, o, s], min_samples_leaf=.25)
    jpt.learn(columns=df.values.T)

    for leaf in jpt.leaves.values():
        xlower = x.domain.labels[leaf.path[x].lower if x in leaf.path else -np.inf]
        xupper = x.domain.labels[leaf.path[x].upper if x in leaf.path else np.inf]
        ylower = y.domain.labels[leaf.path[y].lower if y in leaf.path else -np.inf]
        yupper = y.domain.labels[leaf.path[y].upper if y in leaf.path else np.inf]
        vlines = []
        hlines = []
        if xlower != np.NINF:
            vlines.append(xlower)
        if xupper != np.PINF:
            vlines.append(xupper)
        if ylower != np.NINF:
            hlines.append(ylower)
        if yupper != np.PINF:
            hlines.append(yupper)
        plt.vlines(vlines, max(ylower, df['Y'].min()), min(yupper, df['Y'].max()))
        plt.hlines(hlines, max(xlower, df['X'].min()), min(xupper, df['X'].max()))

    ax.legend()
    if visualize:
        plt.show()
    with PdfPages(os.path.join(os.path.split(__file__)[0], 'data', f'muesli-{2 if succ else 1}.pdf'),
                  metadata={'Creator': 'misc',
                            'Author': 'Picklum & Nyga',
                            'Title': f'Müsli Example{" Success" if succ else ""}'}) as pdf:
        pdf.savefig(fig)


def test_muesli(visualize=True):

    data = pd.read_csv(os.path.join('..', 'examples', 'data', 'muesli.csv'))
    d = np.array(sorted(data['X']), dtype=np.float64)

    quantiles = QuantileDistribution(epsilon=.01)
    quantiles.fit(d.reshape((-1, 1)), None, 0)
    d = Numeric().set(params=quantiles)

    interval = ContinuousSet(-2.05, -2.0)
    p = d.p(interval)
    if visualize:
        out('query', interval, p)

    d.plot(title=' ',
           fname='BreakfastPiecewise',
           xlabel='X',
           view=visualize,
           directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))


def muesli_tree(visualize=True):
    # generate Joint Probability Tree from muesli data (use .csv file because it contains the additional Success column)
    data = pd.read_csv(os.path.join('..', 'examples', 'data', 'muesli.csv'))
    data["Success"] = data["Success"].astype(str)
    ObjectType = SymbolicType('ObjectType', data['Class'].unique())
    SuccessType = SymbolicType("Success", data['Success'].unique())
    XType = NumericType('XType', data['X'].values)

    x = NumericVariable('X', Numeric, blur=.01)
    y = NumericVariable('Y', Numeric, blur=.01)
    o = SymbolicVariable('Object', ObjectType)
    s = SymbolicVariable('Success', SuccessType)

    # pprint.pprint([x.to_json(), y.to_json(), o.to_json(), s.to_json()])

    jpt = JPT([x, y, o, s], min_samples_leaf=.2)
    jpt.learn(columns=data.values.T)
    print(len(jpt.leaves))
    jpt.save("muesli.jpt")

    # json_data = jpt.to_json()
    # pprint.pprint(json_data)
    # jpt.plot(plotvars=[x, y, o], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))
    # jpt = JPTBase.from_json(json_data)

    for clazz in data['Class'].unique():
        out(jpt.infer(query={o.name: clazz}, evidence={x.name: [.9, None], y.name: [None, .45]}))
    print()

    for clazz in data['Class'].unique():
        for exp in jpt.expectation([x.name, y.name], evidence={o.name: clazz}, confidence_level=.95):
            out(exp)

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    print(jpt)

    # jpt.plot(plotvars=[x, y, o, s])

    # q = {o: ("BowlLarge_Bdvg", "JaNougatBits_UE0O"), x: [.812, .827]}
    # r = jpt.reverse(q)
    # out('Query:', q, 'result:', pprint.pformat(r))
    if visualize:
        plot_conditional(jpt, x, y, evidence={o: 'BaerenMarkeFrischeAlpenmilch'})

    for clazz in data['Class'].unique():
        out(jpt.infer(query={o.name: clazz}, evidence={x.name: [.95 - FUZZYNESS, .95 + FUZZYNESS],
                                                       y.name: [.45 - FUZZYNESS, .45 + FUZZYNESS]}))


FUZZYNESS = .01


def plot_conditional(jpt, qvarx, qvary, evidence=None, title=None):
    x = np.linspace(.7, 1.05, 50)
    y = np.linspace(.15, .55, 50)

    X, Y = np.meshgrid(x, y)
    Z = np.array([jpt.infer({qvarx: ContinuousSet(x - FUZZYNESS, x + FUZZYNESS),
                             qvary: ContinuousSet(y - FUZZYNESS, y + FUZZYNESS)},
                            evidence=evidence).result for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    # ax.plot_wireframe(X, Y, Z, color='black')
    # ax.contour(X, Y, Z, 10)
    ax.set_title(ifnone(title, 'P(%s, %s|%s)' % (qvarx.name,
                                                 qvary.name,
                                                 format_path(evidence) if evidence else '$\emptyset$')))
    plt.show()


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


def main(visualize=True):
    # plot_muesli(visualize=visualize)
    # test_muesli(visualize=visualize)
    muesli_tree(visualize=visualize)
    # picklemuesli()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
