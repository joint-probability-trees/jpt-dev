import numpy as np
from pandas import DataFrame

from dnutils import mapstr, edict, err
from matplotlib import pyplot as plt

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main(visualize=False):
    try:
        from sklearn.datasets import load_digits
        import sklearn.metrics
    except ModuleNotFoundError:
        err('Module sklearn not found. In order to run this example, you have to install this package.')
        return
    plt.close()

    mnist = load_digits()

    # Create the names of the numeric variables
    pixels = ['x_%s%s' % (x1 + 1, x2 + 1) for x1 in range(8) for x2 in range(8)]

    # Create the data frame
    df = DataFrame.from_dict(edict({'digit': mapstr(mnist.target)}) +
                             {pixel: list(values) for pixel, values in zip(pixels, mnist.data.T)})

    targets = list(sorted(set(mapstr(mnist.target))))
    DigitType = SymbolicType('DigitType', targets)

    variables = ([SymbolicVariable('digit', domain=DigitType)] +
                 [NumericVariable(pixel, Numeric) for pixel in pixels])

    # create a "fully connected" dependency matrix
    dependencies = {}
    for var in variables:
        dependencies[var] = [v_ for v_ in variables]

    tree = JPT(variables=variables, min_samples_leaf=100, dependencies=dependencies)

    tree.learn(data=df)

    # testing conditional jpts in a complex scenario
    cjpt = tree.conditional_jpt(tree.bind(
        digit={"5", "6"},
        x_28=[0,2]
    ))

    leaves = list(tree.leaves.values())
    
    rows = 2
    cols = 7
    fig, axes = plt.subplots(rows, cols, figsize=(7, 2))

    if len(axes.shape) == 1:
        axes = np.array([axes])

    for i, leaf in enumerate(leaves):
        model = np.array([leaf.distributions[tree.varnames[pixel]].expectation() for pixel in pixels]).reshape(8, 8)
        idx = (i // 7, i % 7)
        axes[idx].imshow(model, cmap='gray')
        axes[idx].set_title(leaf.distributions[tree.varnames['digit']].mode())

    if visualize:
        plt.tight_layout()
        plt.show()
    
    # tree.plot(plotvars=tree.variables, view=visualize)


if __name__ == '__main__':
    main(visualize=True)
