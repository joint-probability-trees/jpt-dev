"""2D density estimation and partitioning.

Learns a Gaussian mixture model using a Joint Probability
Tree, then visualizes the learned density and the tree's
partition structure. The example shows how JPTs
approximate multivariate distributions by partitioning
the input space into axis-aligned regions.

Demonstrates:
    - Gaussian distributions for data generation
    - NumericVariable with precision
    - ``pdf()`` for density evaluation
    - Leaf path analysis for partition visualization
    - Plotly 3D surface plots
"""
import logging

import numpy as np
import plotly.graph_objects as go
from pandas import DataFrame

from jpt.distributions import Gaussian, Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import (
    NumericVariable,
    SymbolicVariable,
    VariableMap,
)


# -------------------------------------------------------


def plot_gaussian(gaussians, visualize=True):
    """Plot the true Gaussian mixture density as a 3D
    surface.

    :param gaussians: list of Gaussian distributions
    :param visualize: whether to show the plot
    """
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    xy = np.column_stack([X.flat, Y.flat])
    Z = np.zeros(shape=xy.shape[0])
    for gaussian in gaussians:
        Z += 1 / len(gaussians) * gaussian.pdf(xy)
    Z = Z.reshape(X.shape)

    mainfig = go.Figure()
    mainfig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='dense',
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            ),
            showscale=False
        ),
    )

    mainfig.update_layout(
        width=1000,
        height=1000
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    scale=1
                )
            )
        )


# -------------------------------------------------------


def plot_conditional(
        jpt,
        qvarx,
        qvary,
        visualize=True
):
    """Plot the learned joint density as a 3D surface.

    :param jpt:       the learned JPT
    :param qvarx:     the X variable to evaluate
    :param qvary:     the Y variable to evaluate
    :param visualize: whether to show the plot
    """
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)
    X, Y = np.meshgrid(x, y)

    Z = np.array([
        jpt.pdf(VariableMap([(qvarx, x), (qvary, y)]))
        for x, y in zip(X.ravel(), Y.ravel())
    ]).reshape(X.shape)

    mainfig = go.Figure()
    mainfig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='dense',
            contours_z=dict(
                show=True,
                usecolormap=True,
                highlightcolor="limegreen",
                project_z=True
            ),
            showscale=False
        ),
    )

    mainfig.update_layout(
        width=1000,
        height=1000
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    scale=1
                )
            )
        )


# -------------------------------------------------------


def main(visualize=True):
    """Learn a 2D Gaussian mixture and visualize the
    density and partition structure.

    :param visualize: whether to show interactive plots
    """
    # Define two Gaussian components
    gauss1 = Gaussian(
        [-.25, -.25], [[.2, -.07], [-.07, .1]]
    )
    gauss2 = Gaussian(
        [.5, 1], [[.2, .07], [.07, .05]]
    )

    # Generate samples from the mixture
    df = generate_gaussian_samples(
        [gauss1, gauss2], 1000
    )

    # Define variables and learn the JPT
    varx = NumericVariable('X', Numeric, precision=.05)
    vary = NumericVariable('Y', Numeric, precision=.05)
    varcolor = SymbolicVariable(
        'Color',
        SymbolicType('ColorType', df.Color.unique())
    )

    jpt = JPT([varx, vary, varcolor], min_samples_leaf=.1)
    jpt.learn(df)

    # Visualize the training data with partition lines
    mainfig = go.Figure()
    mainfig.add_trace(
        go.Scatter(
            x=df['X'].values,
            y=df['Y'].values,
            marker=dict(
                symbol='x',
                color=df['Color'].values,
                size=10,
            ),
            mode='markers',
            showlegend=False
        )
    )

    # Draw partition boundaries from the tree leaves
    color_map = {
        0: 'rgb(134, 129, 177)',
        1: 'rgb(0, 104, 180)',
        None: 'gray',
    }
    for leaf in jpt.leaves.values():
        xlower = varx.domain.labels[
            leaf.path[varx].lower
            if varx in leaf.path else -np.inf
        ]
        xupper = varx.domain.labels[
            leaf.path[varx].upper
            if varx in leaf.path else np.inf
        ]
        ylower = vary.domain.labels[
            leaf.path[vary].lower
            if vary in leaf.path else -np.inf
        ]
        yupper = vary.domain.labels[
            leaf.path[vary].upper
            if vary in leaf.path else np.inf
        ]

        color_idx = (
            next(iter(leaf.path[varcolor]))
            if varcolor in leaf.path else None
        )
        line_color = color_map.get(color_idx, 'gray')

        if xlower != -np.inf:
            mainfig.add_vline(
                x=xlower, line_color=line_color
            )
        if xupper != np.inf:
            mainfig.add_vline(
                x=xupper, line_color=line_color
            )
        if ylower != -np.inf:
            mainfig.add_hline(
                y=ylower, line_color=line_color
            )
        if yupper != np.inf:
            mainfig.add_hline(
                y=yupper, line_color=line_color
            )

    if visualize:
        mainfig.update_layout(width=1000, height=1000)
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    scale=1
                )
            )
        )

    # Plot the tree structure
    jpt.plot(
        nodefill='#768ABE',
        leaffill='#CCDAFF',
        view=visualize,
        plotvars=[varx, vary, varcolor],
    )

    # Compare learned vs true density
    plot_conditional(jpt, varx, vary, visualize=visualize)
    plot_gaussian(
        [gauss1, gauss2], visualize=visualize
    )


# -------------------------------------------------------


def generate_gaussian_samples(gaussians, n):
    """Generate labeled samples from a Gaussian mixture.

    :param gaussians: list of Gaussian distributions
    :param n:         total number of samples
    :returns:         DataFrame with X, Y, Color columns
    """
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [
        [c] * per_gaussian
        for c in [
            'rgb(134, 129, 177)',
            'rgb(0, 104, 180)',
        ][:len(gaussians)]
    ]

    all_data = np.vstack(data)
    from functools import reduce
    df = DataFrame({
        'X': all_data[:, 0],
        'Y': all_data[:, 1],
        'Color': reduce(list.__add__, colors),
    })
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(visualize=True)
