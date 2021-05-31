import numpy as np
from dnutils import out
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric
from jpt.trees import JPT
from jpt.variables import NumericVariable


def main():
    from sklearn.datasets import load_digits
    mnist = load_digits()

    variables = [NumericVariable('x_%s%s' % (x1+1, x2+2), Numeric) for x1 in range(8) for x2 in range(8)]

    tree = JPT(variables=variables, min_samples_leaf=150)

    tree.learn(rows=mnist.data)

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
        out(leaf.distributions, leaf.samples)
        model = np.array([d.expectation() for d in leaf.distributions.values()]).reshape(8, 8)
        out(model)
        idx = i // 10, i % 10 if len(axes.shape) == 1 else i
        axes[idx].imshow(model, cmap='gray')

    plt.tight_layout()
    plt.show()


if __name__ =='__main__':
    main()