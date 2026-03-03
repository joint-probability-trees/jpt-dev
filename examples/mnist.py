"""Image classification: handwritten digit recognition.

Learns digit models from sklearn's 8x8 MNIST dataset
using a Joint Probability Tree, then visualizes what
each leaf "sees" as a heatmap of expected pixel values.

Demonstrates:
    - High-dimensional data (64 pixel features)
    - ``conditional_jpt()`` for sub-model extraction
    - Dependency specification for variable relationships
    - Plotly heatmap visualization of learned models
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


# -------------------------------------------------------


def main(visualize=False):
    """Learn digit models and visualize leaf expectations.

    :param visualize: whether to show interactive plots
    """
    try:
        from sklearn.datasets import load_digits
    except ModuleNotFoundError:
        print(
            'Module sklearn not found. Install it to '
            'run this example: pip install scikit-learn'
        )
        return

    # Load the 8x8 digits dataset
    mnist = load_digits()

    # Create pixel variable names (8x8 grid)
    pixels = [
        'x_%s%s' % (x1 + 1, x2 + 1)
        for x1 in range(8)
        for x2 in range(8)
    ]

    # Build DataFrame with digit labels and pixel values
    from dnutils import mapstr, edict
    from pandas import DataFrame
    df = DataFrame.from_dict(
        edict({'digit': mapstr(mnist.target)})
        + {
            pixel: list(values)
            for pixel, values
            in zip(pixels, mnist.data.T)
        }
    )

    # Define variables
    targets = list(sorted(set(mapstr(mnist.target))))
    DigitType = SymbolicType('DigitType', targets)
    variables = (
        [SymbolicVariable('digit', domain=DigitType)]
        + [
            NumericVariable(pixel, Numeric)
            for pixel in pixels
        ]
    )

    # Create a fully connected dependency matrix
    dependencies = {
        var: [v_ for v_ in variables]
        for var in variables
    }

    # Learn the JPT
    tree = JPT(
        variables=variables,
        min_samples_leaf=100,
        dependencies=dependencies
    )
    tree.learn(data=df)

    # Test conditional JPT extraction
    tree.conditional_jpt(tree.bind(
        digit={"5", "6"},
        x_28=[0, 2]
    ))

    # Visualize leaf models as digit heatmaps
    leaves = list(tree.leaves.values())
    models = sorted(
        [
            (
                next(iter(
                    leaf.distributions[
                        tree.varnames['digit']
                    ].expectation()
                )),
                np.array([
                    leaf.distributions[
                        tree.varnames[pixel]
                    ].expectation()
                    for pixel in pixels
                ]).reshape(8, 8),
            )
            for leaf in leaves
        ],
        key=lambda x: x[0]
    )

    ncol = math.ceil(math.sqrt(len(leaves)))
    fig = make_subplots(
        rows=ncol,
        cols=ncol,
        start_cell="top-left",
        horizontal_spacing=.1 / ncol,
        vertical_spacing=.2 / ncol,
        subplot_titles=[m[0] for m in models]
    )

    for i, (_, model) in enumerate(models):
        fig.add_trace(
            go.Heatmap(
                z=model,
                colorscale='gray_r',
                showscale=False
            ),
            row=i // ncol + 1,
            col=i % ncol + 1
        )
        fig.update_yaxes(
            row=i // ncol + 1,
            col=i % ncol + 1,
            autorange='reversed'
        )

    fig.update_layout(
        width=1200,
        height=1000,
        yaxis=dict(autorange='reversed'),
        showlegend=False
    )

    if visualize:
        fig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    scale=1
                )
            )
        )

    # Plot the tree structure
    tree.plot(
        nodefill='#768ABE',
        leaffill='#CCDAFF',
        view=visualize,
    )


if __name__ == '__main__':
    main(visualize=True)
