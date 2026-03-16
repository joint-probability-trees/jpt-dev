"""Nonlinear regression with confidence bands.

Learns the function y = x * sin(x) with additive noise
using a Joint Probability Tree, then predicts with
confidence intervals. The example shows how JPTs can
capture nonlinear relationships and provide uncertainty
estimates through quantile-based posterior distributions.

Demonstrates:
    - NumericVariable with blur
    - Discriminative learning with targets
    - ``posterior()`` for conditional distributions
    - ``ppf`` for confidence band extraction
    - Plotly visualization of predictions
"""
import logging
import tempfile
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from jpt.trees import JPT
from jpt.variables import NumericVariable


# -------------------------------------------------------


def f(x):
    """The target function: x * sin(x)."""
    return x * np.sin(x)


# -------------------------------------------------------


def generate_data(func, n):
    """Generate noisy samples from ``func``.

    Samples are drawn non-uniformly: half from [-20, 0)
    and half from [0, 10).

    :param func: the function to sample from
    :param n:    number of samples to generate
    :returns:    a DataFrame with columns 'x' and 'y'
    """
    X = np.atleast_2d(
        np.random.uniform(-20, 0.0, size=int(n / 2))
    ).T
    X = np.vstack((
        np.atleast_2d(
            np.random.uniform(0, 10.0, size=int(n / 2))
        ).T,
        X
    ))
    X = X.astype(np.float32)
    X = np.array(list(sorted(X)))

    # Generate observations with additive noise
    y = func(X).ravel()
    dy = 1.5 + .5 * np.random.random(y.shape)
    y += np.random.normal(0, dy)
    y = y.astype(np.float32)

    return pd.DataFrame(
        data={'x': X.ravel(), 'y': y}
    )


# -------------------------------------------------------


def main(visualize=True):
    """Learn x*sin(x) and plot predictions with
    confidence bands.

    :param visualize: whether to show interactive plots
    """
    # Generate training data
    df = generate_data(f, 1000)

    # Evaluation grid
    xx = np.atleast_2d(
        np.linspace(-20, 15, 500)
    ).astype(np.float32).T

    # Define variables and learn the JPT
    varx = NumericVariable('x', blur=.05)
    vary = NumericVariable('y')

    jpt = JPT(
        variables=[varx, vary],
        targets=[vary],
        min_samples_leaf=0.01
    )
    jpt.learn(df, verbose=True)

    # Compute posterior predictions with confidence bands
    confidence = .95
    my_predictions = [
        jpt.posterior(
            [vary],
            evidence={varx: x_},
            fail_on_unsatisfiability=False
        )
        for x_ in xx.ravel()
    ]
    y_pred_ = [
        (p[vary].expectation() if p is not None
         else None)
        for p in my_predictions
    ]
    y_lower_ = [
        (p[vary].ppf.eval((1 - confidence) / 2)
         if p is not None else None)
        for p in my_predictions
    ]
    y_upper_ = [
        (p[vary].ppf.eval(1 - (1 - confidence) / 2)
         if p is not None else None)
        for p in my_predictions
    ]

    # Build the plotly figure
    fname = "Regression-Example"
    out_dir = tempfile.mkdtemp(prefix='jpt-regression-')
    mainfig = go.Figure()

    # True function
    mainfig.add_trace(
        go.Scatter(
            x=xx.reshape(-1),
            y=f(xx).reshape(-1),
            line=dict(
                color='black',
                width=2,
                dash='dot'
            ),
            mode='lines',
            name=r'$f(x) = x\,\sin(x)$'
        )
    )

    # Training data scatter
    mainfig.add_trace(
        go.Scatter(
            x=df['x'].values,
            y=df['y'].values,
            marker=dict(
                symbol='circle',
                color='gray',
                size=5,
            ),
            mode='markers',
            name="Training data",
        )
    )

    # JPT prediction
    mainfig.add_trace(
        go.Scatter(
            x=xx.reshape(-1),
            y=y_pred_,
            line=dict(
                color='#C800C8',
                width=2,
                dash='solid'
            ),
            mode='lines',
            name=r'JPT Prediction'
        )
    )

    # Lower confidence band
    mainfig.add_trace(
        go.Scatter(
            x=xx.reshape(-1),
            y=y_lower_,
            line=dict(
                color='#D5A33F',
                width=2,
                dash='dash'
            ),
            mode='lines',
            name='%.1f%% Confidence bands'
                 % (confidence * 100)
        )
    )

    # Upper confidence band
    mainfig.add_trace(
        go.Scatter(
            x=xx.reshape(-1),
            y=y_upper_,
            line=dict(
                color='#D5A33F',
                width=2,
                dash='dash'
            ),
            mode='lines',
            showlegend=False
        )
    )

    mainfig.update_layout(
        width=1200,
        height=1000,
        xaxis=dict(
            title='x',
            side='bottom',
            range=[-20, 15]
        ),
        yaxis=dict(
            title='f(x)',
            range=[-20, 20]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.8
        ),
    )

    fpath = os.path.join(out_dir, fname)
    mainfig.write_html(
        fpath,
        include_plotlyjs="cdn"
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    filename=fname,
                    scale=1
                )
            )
        )

    return mainfig


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(visualize=True)
