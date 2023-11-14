import logging
from functools import reduce

import numpy as np
import plotly.graph_objects as go
from dnutils import first
from pandas import DataFrame

from jpt.distributions import Gaussian, Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable, VariableMap

visualize = True


def plot_gaussian(gaussians):
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

    mainfig.show(
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format='svg',  # one of png, svg, jpeg, webp
                scale=1
            )
        )
    )


def generate_gaussian_samples(gaussians, n):
    per_gaussian = int(n / len(gaussians))
    data = [g.sample(per_gaussian) for g in gaussians]
    colors = [[c] * per_gaussian for c in ['rgb(134, 129, 177)', 'rgb(0, 104, 180)'][:len(gaussians)]]

    all_data = np.vstack(data)

    df = DataFrame({'X': all_data[:, 0], 'Y': all_data[:, 1], 'Color': reduce(list.__add__, colors)})
    return df


def main(verbose=True):
    global visualize
    visualize = verbose

    gauss1 = Gaussian([-.25, -.25], [[.2, -.07], [-.07, .1]])
    gauss2 = Gaussian([.5, 1], [[.2, .07], [.07, .05]])

    df = generate_gaussian_samples([gauss1, gauss2], 1000)

    varx = NumericVariable('X', Numeric, precision=.05)
    vary = NumericVariable('Y', Numeric, precision=.05)
    varcolor = SymbolicVariable('Color', SymbolicType('ColorType', df.Color.unique()))

    JPT.logger.level = logging.DEBUG
    jpt = JPT([varx, vary, varcolor], min_samples_leaf=.1)
    jpt.learn(df)

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

    for leaf in jpt.leaves.values():
        xlower = varx.domain.labels[leaf.path[varx].lower if varx in leaf.path else -np.inf]
        xupper = varx.domain.labels[leaf.path[varx].upper if varx in leaf.path else np.inf]
        ylower = vary.domain.labels[leaf.path[vary].lower if vary in leaf.path else -np.inf]
        yupper = vary.domain.labels[leaf.path[vary].upper if vary in leaf.path else np.inf]
        vlines = []
        hlines = []
        if xlower != np.NINF:
            vlines.append(xlower)
        if xupper != np.PINF:
            vlines.append(xupper)
        if ylower != np.NINF:
            hlines.append(ylower)
        if yupper != np.PINF:
            hlines.append(yupper)

        print(first(leaf.path[varcolor]))
        for hl in hlines:
            mainfig.add_hline(
                y=hl,
                line_color={0: 'rgb(134, 129, 177)', 1: 'rgb(0, 104, 180)', None: 'gray'}[first(leaf.path[varcolor]) if varcolor in leaf.path else None])
        for vl in vlines:
            mainfig.add_vline(
                x=vl,
                line_color={0: 'rgb(134, 129, 177)', 1: 'rgb(0, 104, 180)', None: 'gray'}[first(leaf.path[varcolor]) if varcolor in leaf.path else None])

    if visualize:
        mainfig.update_layout(
            width=1000,
            height=1000
        )

        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',  # one of png, svg, jpeg, webp
                    scale=1
                )
            )
        )

    jpt.plot(
        nodefill='#768ABE',
        leaffill='#CCDAFF',
        view=True,
        plotvars=[varx, vary, varcolor],
    )

    # _data = jpt._preprocess_data(df)
    # dec = DecisionTreeClassifier(min_samples_leaf=.1)
    # dec.fit(_data[:, :-1], _data[:, -1:])
    # plot_tree(dec)
    plot_conditional(jpt, varx, vary)
    plot_gaussian([gauss1, gauss2])
    # plot_conditional(jpt, varx, vary, {varcolor: 'R'})
    # plot_conditional(jpt, varx, vary, {varcolor: 'B'})


def plot_conditional(jpt, qvarx, qvary, evidence=None, title=None):
    x = np.linspace(-2, 2, 30)
    y = np.linspace(-2, 2, 30)

    X, Y = np.meshgrid(x, y)
    Z = np.array([jpt.pdf(VariableMap([(qvarx, x),
                                       (qvary, y)])) for x, y, in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

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

    mainfig.show(
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format='svg',  # one of png, svg, jpeg, webp
                scale=1
            )
        )
    )


if __name__ == '__main__':
    main(verbose=True)
