import numpy as np
from dnutils import first, out
from matplotlib import pyplot as plt

from jpt.learning.distributions import Numeric, NumericType
from jpt.trees import JPT
from jpt.variables import NumericVariable


def main():

    def f(x):
        """The function to predict."""
        # x -= 20
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    #  First the noiseless case
    POINTS = 1000
    X = np.atleast_2d(np.random.uniform(-20, 0.0, size=int(POINTS / 2))).T
    X = np.vstack((np.atleast_2d(np.random.uniform(0, 10.0, size=int(POINTS / 2))).T, X))
    # X = np.atleast_2d(np.random.uniform(-20, 10.0, size=int(POINTS))).T
    X = X.astype(np.float32)
    X = np.array(list(sorted(X)))

    # Observations
    y = f(X).ravel()

    # Add some noise
    dy = 1.5 + .5 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(-30, 30, 500)).T
    # xx = np.atleast_2d(np.linspace(-10, 20, 500)).T
    xx = xx.astype(np.float32)

    # Construct the predictive model
    varx = NumericVariable('x', NumericType('x', X), haze=.05)
    vary = NumericVariable('y', NumericType('y', y), haze=.05)

    jpt = JPT(variables=[varx, vary], min_samples_leaf=15)
    jpt.learn(columns=[X.ravel(), y])
    # jpt.plot(plotvars=[varx, vary])
    # Apply the JPT model
    confidence = .5

    # for x in xx.ravel():
    #     print(jpt.infer({varx: x}).explain())
    # exit(0)
    my_predictions = [first(jpt.expectation([vary],
                                            evidence={varx: x_},
                                            confidence_level=confidence,
                                            fail_on_unsatisfiability=False)) for x_ in xx.ravel()]
    y_pred_ = [p.result if p else None for p in my_predictions]
    y_lower_ = [p.lower if p else None for p in my_predictions]
    y_upper_ = [p.upper if p else None for p in my_predictions]

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
