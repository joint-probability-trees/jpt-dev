import logging

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from jpt.trees import JPT
from jpt.variables import NumericVariable, VariableMap


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
    plt.close()
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

    jpt.learn(df, verbose=True)

    # jpt.plot(view=visualize)

    # Apply the JPT model
    confidence = .95
    conf_level = 0.95
    my_predictions = [jpt.posterior([vary], evidence={varx: x_}, fail_on_unsatisfiability=False) for x_ in xx.ravel()]
    y_pred_ = [(p[vary].expectation() if p is not None else None) for p in my_predictions]
    y_lower_ = [(p[vary].ppf.eval((1 - conf_level) / 2) if p is not None else None) for p in my_predictions]
    y_upper_ = [(p[vary].ppf.eval(1 - (1 - conf_level) / 2) if p is not None else None) for p in my_predictions]

    # posterior = jpt.posterior([varx], {vary: 0})

    # Plot the function, the prediction and the 90% confidence interval based on the MSE
    plt.plot(xx, f(xx), color='black', linestyle=':', linewidth='2', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(df['x'].values, df['y'].values, '.', color='gray', markersize=5, label='Training data')
    plt.plot(xx, y_pred_, 'm-', label='JPT Prediction', linewidth=2)
    plt.plot(xx, y_lower_, 'y--', label='%.1f%% Confidence bands' % (confidence * 100))
    plt.plot(xx, y_upper_, 'y--')
    # plt.plot(xx, np.asarray(posterior.distributions[varx].pdf.multi_eval(xx.ravel().astype(np.float64))),
    #          label='Posterior')
    plt.plot(xx, np.array([jpt.pdf(VariableMap([(varx, x_), (vary, 0)])) for x_ in xx.ravel().astype(np.float64)]),
             label='Posterior')

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.xlim(-25, 15)
    plt.legend(loc='upper left')
    plt.title(r'2D Regression Example ($\vartheta=%.2f\%%$)' % (confidence * 100))
    plt.grid()
    if visualize:
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
    )
    main()
