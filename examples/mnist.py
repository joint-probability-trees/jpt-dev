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
    
    tree.plot(plotvars=tree.variables)


if __name__ == '__main__':
    main()
