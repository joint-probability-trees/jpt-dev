import math

import numpy as np
import plotly.graph_objects as go
from dnutils import mapstr, edict, err, first
from pandas import DataFrame
from plotly.subplots import make_subplots

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main(visualize=False):
    try:
        from sklearn.datasets import load_digits
        import sklearn.metrics
    except ModuleNotFoundError:
        err('Module sklearn not found. In order to run this example, you have to install this package.')
        return

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

    tree = JPT(variables=variables, min_samples_leaf=100, dependencies=dependencies)

    tree.learn(data=df)

    # testing conditional jpts in a complex scenario
    cjpt = tree.conditional_jpt(tree.bind(
        digit={"5", "6"},
        x_28=[0, 2]
    ))

    leaves = list(tree.leaves.values())
    
    grayscale = [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, "rgb(0, 0, 0)"],
        [0.1, "rgb(0, 0, 0)"],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        [0.1, "rgb(20, 20, 20)"],
        [0.2, "rgb(20, 20, 20)"],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        [0.2, "rgb(40, 40, 40)"],
        [0.3, "rgb(40, 40, 40)"],

        [0.3, "rgb(60, 60, 60)"],
        [0.4, "rgb(60, 60, 60)"],

        [0.4, "rgb(80, 80, 80)"],
        [0.5, "rgb(80, 80, 80)"],

        [0.5, "rgb(100, 100, 100)"],
        [0.6, "rgb(100, 100, 100)"],

        [0.6, "rgb(120, 120, 120)"],
        [0.7, "rgb(120, 120, 120)"],

        [0.7, "rgb(140, 140, 140)"],
        [0.8, "rgb(140, 140, 140)"],

        [0.8, "rgb(160, 160, 160)"],
        [0.9, "rgb(160, 160, 160)"],

        [0.9, "rgb(180, 180, 180)"],
        [1.0, "rgb(180, 180, 180)"]
    ]

    models = []
    for i, leaf in enumerate(leaves):
        models.append([first(leaf.distributions[tree.varnames['digit']].expectation()), np.array([leaf.distributions[tree.varnames[pixel]].expectation() for pixel in pixels]).reshape(8, 8)])

    models = sorted(models, key=lambda x: x[0])

    ncol = math.ceil(math.sqrt(len(leaves)))
    rows = ncol
    cols = ncol
    fig = make_subplots(
        rows=rows,
        cols=cols,
        start_cell="top-left",
        horizontal_spacing=.1/ncol,
        vertical_spacing=.2/ncol,
        subplot_titles=[m[0] for m in models]
    )

    for i, (_, model) in enumerate(models):
        fig.add_trace(
            go.Heatmap(
                z=model,
                colorscale=grayscale,
                showscale=False
            ),
            row=i // ncol + 1,
            col=i % ncol + 1
        )

        fig.update_yaxes(row=i // ncol + 1, col=i % ncol + 1, autorange='reversed')

    fig.update_layout(
        width=1200,
        height=1000,
        yaxis=dict(
            autorange='reversed'
        ),
        showlegend=False
    )

    if visualize:
        fig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',  # one of png, svg, jpeg, webp
                    scale=1
                )
            )
        )

    tree.plot(
        # plotvars=tree.variables,
        nodefill='#768ABE',
        leaffill='#CCDAFF',
        view=False,
    )


if __name__ == '__main__':
    main(visualize=True)
