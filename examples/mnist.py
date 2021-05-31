import numpy as np
from dnutils import out
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main():
    from sklearn.datasets import load_digits
    mnist = load_digits()

    variables = ([NumericVariable('x_%s%s' % (x1+1, x2+2), Numeric) for x1 in range(8) for x2 in range(8)])
    # +
    #              [SymbolicVariable('class', SymbolicType('DigitType', mnist.target))])

    tree = JPT(variables=variables, min_samples_leaf=150)

    # tree.learn(columns=np.vstack([mnist.data.T, mnist.target.T]))
    tree.learn(columns=mnist.data.T)

    leaves = list(tree.leaves.values())

    rows = len(leaves) // 10 + 1
    cols = max(2, len(leaves))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6))

    if len(axes.shape) == 1:
        axes = np.array([axes])
    # for i in range(20):
    # axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    # axes[i // 10, i % 10].axis('off')
    # axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")

    for i, leaf in enumerate(leaves):
        model = np.array([d.expectation() for d in list(leaf.distributions.values())]).reshape(8, 8)
        # print(leaf.distributions[tree.varnames['class']]._p)
        idx = i // 10, i % 10 if len(axes.shape) == 1 else i
        axes[idx].imshow(model, cmap='gray')

    plt.tight_layout()
    plt.show()
    tree.plot()


if __name__ =='__main__':
    main()