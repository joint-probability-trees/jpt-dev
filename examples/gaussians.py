import logging
from functools import reduce

import numpy as np
from matplotlib._color_data import BASE_COLORS

from dnutils import first, ifnone
from pandas import DataFrame

from jpt.base.utils import format_path
from jpt.distributions import Gaussian, Numeric, SymbolicType
from matplotlib import pyplot as plt

from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable, VariableMap


visualize = True


def plot_gaussian(gaussians):
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.flat, Y.flat])
    Z = np.zeros(shape=xy.shape[0])
    for gaussian in gaussians:
        Z += 1 / len(gaussians) * gaussian.pdf(xy)
    Z = Z.reshape(X.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')


def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in list(BASE_COLORS.keys())[:len(gaussians)]]

    all_data = np.vstack(data)
    for d, c in zip(data, colors):
        plt.scatter(d[:, 0], d[:, 1], color=c, marker='x')
    # plt.scatter(gauss2_data[:, 0], gauss2_data[:, 1], color='b', marker='x')
    # all_data = np.hstack([all_data, reduce(list.__add__, colors)])

    df = DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': reduce(list.__add__, colors)})
    return df


def main(verbose=True):
    plt.close()
    global visualize
    visualize = verbose

    gauss1 = Gaussian([-.25, -.25], [[.2, -.07], [-.07, .1]])
    gauss2 = Gaussian([.5, 1], [[.2, .07], [.07, .05]])

    df = generate_gaussian_samples([gauss1, gauss2], 1000)

    varx = NumericVariable('X', Numeric, precision=.05)
    vary = NumericVariable('Y', Numeric, precision=.05)
    varcolor = SymbolicVariable('Color', SymbolicType('ColorType', df.Color.unique()))

    JPT.logger.level = logging.DEBUG
    jpt = JPT([varx, vary, varcolor], min_samples_leaf=.1)
    jpt.learn(df)

    for leaf in jpt.leaves.values():
        xlower = varx.domain.labels[leaf.path[varx].lower if varx in leaf.path else -np.inf]
        xupper = varx.domain.labels[leaf.path[varx].upper if varx in leaf.path else np.inf]
        ylower = vary.domain.labels[leaf.path[vary].lower if vary in leaf.path else -np.inf]
        yupper = vary.domain.labels[leaf.path[vary].upper if vary in leaf.path else np.inf]
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
        plt.vlines(vlines, max(ylower, -2), min(yupper, 2),
                   color={0: 'r', 1: 'b', None: 'gray'}[first(leaf.path[varcolor])
                   if varcolor in leaf.path else None])
        plt.hlines(hlines, max(xlower, -2.5), min(xupper, 2.5),
                   color={0: 'r', 1: 'b', None: 'gray'}[first(leaf.path[varcolor])
                   if varcolor in leaf.path else None])
    if visualize:
        jpt.plot(plotvars=jpt.variables, view=True)
        plt.show()

    print('10-MPE states:')
    for mpe in jpt.kmpe(k=10):
        print('  ', mpe)

    plot_conditional(jpt, varx, vary)
    plot_gaussian([gauss1, gauss2])
    # plot_conditional(jpt, varx, vary, {varcolor: 'R'})
    # plot_conditional(jpt, varx, vary, {varcolor: 'B'})


def plot_conditional(jpt, qvarx, qvary, evidence=None, title=None):
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)

    X, Y = np.meshgrid(x, y)
    Z = np.array([jpt.pdf(VariableMap([(qvarx, x),
                                       (qvary, y)])) for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(ifnone(title, 'P(%s, %s|%s)' % (qvarx.name,
                                                 qvary.name,
                                                 format_path(evidence) if evidence else r'$\emptyset$')))
    if visualize:
        plt.show()
    plt.plot(x, np.array([jpt.pdf(VariableMap([(qvary, x_)])) for x_ in x]))
    if visualize:
        plt.show()


if __name__ == '__main__':
    main(verbose=True)
