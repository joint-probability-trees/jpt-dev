import logging

import numpy as np
from dnutils import out
from pandas import DataFrame

from jpt.learning.distributions import Gaussian, Numeric, SymbolicType
from matplotlib import pyplot as plt

from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def preprocess_gaussian():
    gauss1 = Gaussian([-.25, -.25], [[.2, -.07], [-.07, .1]])
    gauss2 = Gaussian([.5, 1], [[.2, .07], [.07, .05]])

    SAMPLES = 200

    gauss1_data = gauss1.sample(SAMPLES)
    gauss2_data = gauss2.sample(SAMPLES)

    all_data = np.vstack([gauss1_data, gauss2_data])
    plt.scatter(gauss1_data[:, 0], gauss1_data[:, 1], color='r', marker='x')
    plt.scatter(gauss2_data[:, 0], gauss2_data[:, 1], color='b', marker='x')

    df = DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': ['R'] * SAMPLES + ['B'] * SAMPLES})
    return df


def main():
    df = preprocess_gaussian()
    out(df)

    varx = NumericVariable('X', Numeric)
    vary = NumericVariable('Y', Numeric)
    varcolor = SymbolicVariable('Color', SymbolicType('ColorType', ['R', 'B']))

    JPT.logger.level = logging.DEBUG
    jpt = JPT([varx, vary, varcolor], min_samples_leaf=.01)
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
        plt.vlines(vlines, max(ylower, -2), min(yupper, 2), color='gray')
        plt.hlines(hlines, max(xlower, -2.5), min(xupper, 2.5), color='gray')

    plt.show()
    jpt.plot(plotvars=['X', 'Y', 'Color'])


if __name__ == '__main__':
    main()
