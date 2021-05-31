import numpy as np
from dnutils import first
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric, NumericType
from jpt.trees import JPT
from jpt.variables import NumericVariable


def main():

    def f(x):
        """The function to predict."""
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    #  First the noiseless case
    POINTS = 1000
    X = np.atleast_2d(np.random.uniform(0, 10.0, size=int(POINTS / 2))).T
    X = np.vstack((np.atleast_2d(np.random.uniform(-20, 0.0, size=int(POINTS / 2))).T, X))
    X = X.astype(np.float32)

    # Observations
    y = f(X).ravel()

    # Add some noise
    dy = 1.5 + .5 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(-25, 20, 500)).T
    # xx = np.atleast_2d(np.linspace(-10, 20, 500)).T
    xx = xx.astype(np.float32)

    # Construct the predictive model
    varx = NumericVariable('x', Numeric)
    vary = NumericVariable('y', NumericType('y', y))

    jpt = JPT(variables=[varx, vary], min_samples_leaf=10)
    jpt.learn(columns=[X.ravel(), y])

    # Apply the JPT model
    confidence = .25
    my_predictions = [first(jpt.expectation([vary], evidence={varx: x_}, confidence_level=confidence)) for x_ in xx.ravel()]
    y_pred_ = [p.result for p in my_predictions]
    y_lower_ = [p.lower for p in my_predictions]
    y_upper_ = [p.upper for p in my_predictions]

    # Plot the function, the prediction and the 90% confidence interval based on the MSE
    fig = plt.figure()
    plt.plot(xx, f(xx), 'g:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, '.', color='gray', markersize=5, label='Training data')
    plt.plot(xx, y_pred_, 'm-', label='JPT Prediction')
    plt.plot(xx, y_lower_, 'y--', label='%.1f%% Confidence bands' % (confidence * 100))
    plt.plot(xx, y_upper_, 'y--')

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
