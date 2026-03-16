"""Quantile distribution internals.

Visualizes how JPT represents continuous distributions
using piecewise linear CDF approximations. Two examples
are shown:

1. **CDF approximation**: How CDFRegressor fits a
   piecewise linear function to empirical CDFs at
   different epsilon tolerances.
2. **Distribution merging**: How multiple quantile
   distributions can be merged into one using weighted
   combination of their CDFs.

Demonstrates:
    - QuantileDistribution fitting and merging
    - CDFRegressor with different epsilon values
    - Piecewise linear CDF approximation
    - Plotly visualization of CDFs and support points
"""
import numpy as np
import plotly.graph_objects as go

from jpt.distributions import Gaussian
from jpt.distributions.qpd import QuantileDistribution
from jpt.distributions.qpd.cdfreg import CDFRegressor


EPS = 1e-5


# -------------------------------------------------------


def qdata(data):
    """Compute empirical CDF from sorted data.

    :param data: sorted 1D array of samples
    :returns:    2D array [values, cumulative probs]
    """
    data = np.sort(data)
    x, counts = np.unique(data, return_counts=True)
    y = np.asarray(counts, dtype=np.float64)
    np.cumsum(y, out=np.asarray(y))
    n_samples = data.shape[0]
    for i in range(x.shape[0]):
        y[i] /= n_samples
    return np.array([x, y])


# -------------------------------------------------------


def quantile_approximation(visualize=True):
    """Visualize piecewise linear CDF approximation at
    different epsilon tolerances.

    Generates data from a mixture of three Gaussians,
    then fits CDFRegressors with two different epsilon
    values and compares the resulting piecewise linear
    approximations to the true mixture CDF.

    :param visualize: whether to show the plot
    """
    # Generate data from three Gaussians
    gauss1 = Gaussian(-1, 1.5)
    g1data = gauss1.sample(100)
    gauss2 = Gaussian(3, .08)
    g2data = gauss2.sample(100)
    gauss3 = Gaussian(7, .5)
    g3data = gauss3.sample(100)
    data = np.hstack([
        sorted(g1data), sorted(g2data), sorted(g3data)
    ])

    nsamples = data.shape[0]

    # Fit CDFRegressors at two epsilon levels
    reg = CDFRegressor(
        eps=.0, delta_min=1 / nsamples
    )
    reg.fit(qdata(data))

    reg2 = CDFRegressor(
        eps=.01, delta_min=1 / nsamples
    )
    reg2.fit(qdata(data))

    points = np.array(reg.support_points)
    points2 = np.array(reg2.support_points)

    # Compute true mixture CDF
    x = np.linspace(-4, 9, 300)
    x.sort()
    cdf = np.array([
        gauss1.cdf(d)[0] / 3
        + gauss2.cdf(d)[0] / 3
        + gauss3.cdf(d)[0] / 3
        for d in x
    ])

    # Build the plotly figure
    mainfig = go.Figure()

    # PLF approximation with eps=0
    mainfig.add_trace(
        go.Scatter(
            x=points[:, 0],
            y=points[:, 1],
            line=dict(
                color='orange',
                width=2,
                dash='solid',
            ),
            marker=dict(
                symbol='circle',
                color='orange',
                size=10,
            ),
            mode="lines+markers",
            name=(
                r'$\text{PLF of CDF with }'
                r'\varepsilon = %s$' % reg.eps
            )
        )
    )

    # PLF approximation with eps=0.01
    mainfig.add_trace(
        go.Scatter(
            x=points2[:, 0],
            y=points2[:, 1],
            line=dict(
                color='#9300B9',
                width=2,
                dash='solid',
            ),
            marker=dict(
                symbol='x',
                color='#9300B9',
                size=10,
            ),
            mode="lines+markers",
            name=(
                r'$\text{PLF of CDF with }'
                r'\varepsilon = %s$' % reg2.eps
            )
        )
    )

    # Raw data points
    mainfig.add_trace(
        go.Scatter(
            x=data,
            y=np.zeros(data.shape[0]),
            marker=dict(
                symbol='x',
                color='cornflowerblue',
                size=10,
            ),
            mode='markers',
            name="Raw data",
        )
    )

    # True mixture CDF
    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=cdf,
            line=dict(
                color='green',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='Mixture of Gaussians'
        )
    )

    mainfig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(side='bottom'),
        yaxis=dict(range=[0, 1]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.8
        ),
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    filename="Quantile-Example",
                    scale=1
                )
            )
        )


# -------------------------------------------------------


def quantile_merge(visualize=True):
    """Visualize merging of quantile distributions.

    Fits three separate QuantileDistributions to data
    from three Gaussians, then merges them using weighted
    combination and compares the result to a distribution
    fitted on all data at once.

    :param visualize: whether to show the plot
    """
    # Generate data from three Gaussians
    gauss1 = Gaussian(-1, 1.5)
    gauss2 = Gaussian(3, .08)
    gauss3 = Gaussian(7, .5)

    g1data = gauss1.sample(100)
    g2data = gauss2.sample(100)
    g3data = gauss3.sample(100)
    data = np.vstack([
        sorted(g1data), sorted(g2data), sorted(g3data)
    ])

    # Fit a distribution on all data combined
    dist_all = QuantileDistribution(epsilon=EPS)
    dist_all.fit(data.reshape(-1, 1), None, 0)

    # Fit individual distributions
    dist1 = QuantileDistribution(epsilon=EPS)
    dist1.fit(g1data.reshape(-1, 1), None, 0)

    dist2 = QuantileDistribution(epsilon=EPS)
    dist2.fit(g2data.reshape(-1, 1), None, 0)

    dist3 = QuantileDistribution(epsilon=EPS)
    dist3.fit(g3data.reshape(-1, 1), None, 0)

    # Merge the three distributions
    dist = QuantileDistribution.merge(
        [dist1, dist2, dist3], [1 / 3, 1 / 3, 1 / 3]
    )

    # Evaluate all CDFs on a common grid
    x = np.linspace(-7, 9, 500)
    cdf1 = dist1.cdf.multi_eval(x)
    cdf2 = dist2.cdf.multi_eval(x)
    cdf3 = dist3.cdf.multi_eval(x)
    cdf = dist_all.cdf.multi_eval(x)
    cdf_merged = dist.cdf.multi_eval(x)

    # Build the plotly figure
    mainfig = go.Figure()

    # Raw data scatter for each component
    for gdata, color in [
        (g1data, 'blue'),
        (g2data, 'green'),
        (g3data, 'red'),
    ]:
        mainfig.add_trace(
            go.Scatter(
                x=gdata.ravel(),
                y=np.zeros(gdata.shape[0]),
                marker=dict(
                    symbol='circle',
                    color=color,
                    size=10,
                ),
                mode="markers",
                name='Raw data'
            )
        )

    # Individual CDFs
    for cdf_i, color, name in [
        (cdf1, 'blue', 'CDF-1'),
        (cdf2, 'green', 'CDF-2'),
        (cdf3, 'red', 'CDF-3'),
    ]:
        mainfig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(cdf_i),
                line=dict(
                    color=color,
                    width=2,
                    dash='solid',
                ),
                mode="lines",
                name=name
            )
        )

    # Combined CDF (fitted on all data)
    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(cdf),
            line=dict(
                color='purple',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='Combined CDF'
        )
    )

    # Merged CDF (from individual merging)
    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(cdf_merged),
            line=dict(
                color='orange',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='Merged CDF'
        )
    )

    mainfig.update_layout(
        width=1000,
        height=1000,
        xaxis=dict(side='bottom'),
        yaxis=dict(range=[0, 1]),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.8
        ),
    )

    if visualize:
        mainfig.show(
            config=dict(
                displaylogo=False,
                toImageButtonOptions=dict(
                    format='svg',
                    filename="quantile_merge.svg",
                    scale=1
                )
            )
        )


# -------------------------------------------------------


def main(visualize=True):
    """Run both quantile distribution examples.

    :param visualize: whether to show interactive plots
    """
    quantile_approximation(visualize=visualize)
    quantile_merge(visualize=visualize)


if __name__ == '__main__':
    main(visualize=True)
