import logging
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from jpt.trees import JPT
from jpt.variables import NumericVariable

logging.getLogger("/jpt").setLevel(0)

# ----------------------------------------------------------------------------------------------------------------------
# The function to predict


def f(x):
    return x * np.sin(x)

# ----------------------------------------------------------------------------------------------------------------------


def generate_data(func, x_lower, x_upper, n):
    '''
    Generate a ``DataFrame`` of ``n`` data samples with additive noise from the function ``func``.
    '''
    X = np.atleast_2d(np.random.uniform(-20, 0.0, size=int(n / 2))).T
    X = np.vstack((np.atleast_2d(np.random.uniform(0, 10.0, size=int(n / 2))).T, X))
    X = X.astype(np.float32)
    X = np.array(list(sorted(X)))

    # Observations
    y = func(X).ravel()

    # Add some noise
    dy = 1.5 + .5 * np.random.random(y.shape)
    y += np.random.normal(0, dy)
    y = y.astype(np.float32)

    return pd.DataFrame(data={'x': X.ravel(), 'y': y})


def main(visualize=True):
    df = generate_data(f, -20, 10, 1000)

    # Mesh the input space for evaluations of the real function,
    # the prediction and its MSE
    xx = np.atleast_2d(np.linspace(-20, 15, 500)).astype(np.float32).T

    # Construct the predictive model
    varx = NumericVariable('x', blur=.05)
    vary = NumericVariable('y')

    # For discrimintive learning, uncomment the following line:
    jpt = JPT(variables=[varx, vary], targets=[vary], min_samples_leaf=0.01)
    # For generative learning, uncomment the following line:
    # jpt = JPT(variables=[varx, vary], targets=[vary], min_samples_leaf=.01)

    jpt.learn(df)

    # jpt.plot(view=visualize)

    # Apply the JPT model
    confidence = .95
    conf_level = 0.95
    my_predictions = [jpt.posterior([vary], evidence={varx: x_}, fail_on_unsatisfiability=False) for x_ in xx.ravel()]
    y_pred_ = [(p[vary].expectation() if p is not None else None) for p in my_predictions]
    y_lower_ = [(p[vary].ppf.eval((1 - conf_level) / 2) if p is not None else None) for p in my_predictions]
    y_upper_ = [(p[vary].ppf.eval(1 - (1 - conf_level) / 2) if p is not None else None) for p in my_predictions]

    title = r'$\text{2D Regression Example: }(\vartheta=%.2f\%%)$' % (confidence * 100)
    fname = "Regression-Example"
    directory = '/tmp'
    mainfig = go.Figure()

    # scatter x sin(x) function
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

    # scatter training data
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
            name='%.1f%% Confidence bands' % (confidence * 100)
        )
    )

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
        # title=title
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    fpath = os.path.join(directory, fname)

    mainfig.write_html(
        fpath,
        include_plotlyjs="cdn"
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',  # one of png, svg, jpeg, webp
                    filename=fname,
                    scale=1
                )
            )
        )

    return mainfig


if __name__ == '__main__':
    main(visualize=True)
