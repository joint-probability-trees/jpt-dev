import numpy as np
from pandas import DataFrame

from dnutils import mapstr, edict
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable, VariableMap
from datetime import datetime
import os

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

    # create a "fully connected" dependency matrix
    dependencies = {}
    for var in variables:
        dependencies[var] = [v_ for v_ in variables]

    tree = JPT(variables=variables, min_samples_leaf=100, variable_dependencies=dependencies)

    tree.learn(data=df)
    #tree.plot(directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-mnist'))

    cjpt = tree.conditional_jpt(VariableMap({variables[0]:5, variables[29]:2.}.items()))
    cjpt.plot(directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-mnist'))
    exit()
    #calculate log likelihood
    queries = np.append(np.expand_dims(mnist.target, -1), mnist.data, axis=1)
    likelihood = tree.likelihood(queries)
    print("log-likelihood of tree:", np.sum(np.log(likelihood)))

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
        axes[idx].set_title(leaf.distributions[tree.varnames['digit']].expectation())

    plt.tight_layout()
    plt.show()
    
    tree.plot(plotvars=tree.variables)


if __name__ == '__main__':
    main()
