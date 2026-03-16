"""Face image learning: Olivetti faces dataset.

Learns a generative model of faces from the Olivetti
dataset (400 images, 64x64 pixels, 40 identities).
The learned JPT captures the statistical structure of
face images and identity classes.

Demonstrates:
    - Very high-dimensional data (4096 pixel features)
    - SymbolicType for identity classification
    - Tree visualization with ``plot()``
"""
import tempfile

import numpy as np

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT from the Olivetti faces dataset.

    :param visualize: whether to show interactive plots
    """
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except ModuleNotFoundError:
        print(
            'Module sklearn not found. Install it to '
            'run this example: pip install scikit-learn'
        )
        return

    # Load the Olivetti faces dataset
    olivetti = fetch_olivetti_faces()

    # Define 4096 pixel variables plus identity class
    variables = [
        NumericVariable(
            f'Pixel({x1},{x2})', Numeric
        )
        for x1 in range(64)
        for x2 in range(64)
    ] + [
        SymbolicVariable(
            'IdentityClass',
            SymbolicType(
                'Identity', set(olivetti.target)
            )
        )
    ]

    # Learn the JPT
    tree = JPT(
        variables=variables,
        min_samples_leaf=20
    )
    tree.learn(
        columns=np.vstack([
            olivetti.data.T,
            olivetti.target.T
        ])
    )

    print(f'Learned tree with {len(tree.leaves)} leaves.')

    # Plot the tree structure
    out_dir = tempfile.mkdtemp(prefix='jpt-olivetti-')
    tree.plot(
        title='Olivetti Faces',
        directory=out_dir,
        view=visualize
    )


if __name__ == '__main__':
    main(visualize=True)
