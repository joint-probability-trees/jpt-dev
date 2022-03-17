import numpy as np
from pandas import DataFrame

from dnutils import mapstr, edict
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable
import tqdm

def main():
    from sklearn.datasets import load_digits
    import sklearn.metrics
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

    dependencies = []
    for var in variables:
        dependencies.append([v_ for v_ in variables if v_ != var])
    dependencies = dict(zip(variables, dependencies))

    tree = JPT(variables=variables, min_samples_leaf=10, variable_dependencies=dependencies)

    tree.learn(data=df)
    leaves = list(tree.leaves.values())

    predictions = np.zeros(len(mnist.data))
    for idx, sample in enumerate(tqdm.tqdm(mnist.data)):
        evidence = dict(zip(variables[1:], [(v-0.5, v+0.5) for v in sample]))
        query = [variables[0]]
        result = tree.expectation(query, evidence)
        print(result[variables[0]].result)
        predictions[idx] = result[variables[0]].result

    print(sklearn.metrics.confusion_matrix(mnist.target, predictions))
    exit()

    """
    rows = 2
    cols = 7
    fig, axes = plt.subplots(rows, cols, figsize=(7, 2))

    if len(axes.shape) == 1:
        axes = np.array([axes])

    for i, leaf in enumerate(leaves):
        model = np.array([leaf.distributions[tree.varnames[pixel]].expectation() for pixel in pixels]).reshape(8, 8)
        idx = (i // 7, i % 7)
        axes[idx].imshow(model, cmap='gray')
        axes[idx].set_title(leaf.distributions[tree.varnames['digit']].expectation())

    plt.tight_layout()
    plt.show()
    """
    tree.plot()


if __name__ == '__main__':
    main()
