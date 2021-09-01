import os
from datetime import datetime

import numpy as np

from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main():
    from sklearn.datasets import fetch_olivetti_faces
    olivetti = fetch_olivetti_faces()
    variables = [NumericVariable(f'Pixel({x1},{x2})', Numeric) for x1 in range(64) for x2 in range(64)] + [SymbolicVariable('IdentityClass', SymbolicType('Identity', set(olivetti.target)))]
    tree = JPT(variables=variables, min_samples_leaf=20)
    tree.learn(columns=np.vstack([olivetti.data.T, olivetti.target.T]))
    # tree.learn(columns=olivetti.data.T)

    leaves = list(tree.leaves.values())

    # rows = len(leaves) // 10 + 1
    # cols = max(2, len(leaves))
    # fig, axes = plt.subplots(rows, cols, figsize=(16, 6))
    #
    # if len(axes.shape) == 1:
    #     axes = np.array([axes])
    # for i in range(20):
    # axes[i // 10, i % 10].imshow(mnist.images[i], cmap='gray')
    # axes[i // 10, i % 10].axis('off')
    # axes[i // 10, i % 10].set_title(f"target: {mnist.target[i]}")

    # for i, leaf in enumerate(leaves):
    #     model = np.array([d.expectation() for d in list(leaf.distributions.values())]).reshape(8, 8)
    #     # print(leaf.distributions[tree.varnames['class']]._p)
    #     fld_idx = i // 10, i % 10 if len(axes.shape) == 1 else i
    #     axes[fld_idx].imshow(model, cmap='gray')
    #
    # plt.tight_layout()
    # plt.show()
    tree.plot(title='Olivetti', directory=os.path.join('/tmp', f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-Olivetti'))


if __name__ =='__main__':
    main()
