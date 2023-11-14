import pyximport

pyximport.install()

# from jpt.base.quantiles import QuantileDistribution

from jpt.distributions.quantile.cdfreg import CDFRegressor
import plotly.graph_objects as go


# from jpt.base.intervals import ContinuousSet, INC, EXC


from jpt.distributions.quantile.quantiles import QuantileDistribution


import numpy as np

from jpt.distributions import Gaussian


def qdata(data):
    data = np.sort(data)
    x, counts = np.unique(data, return_counts=True)
    y = np.asarray(counts, dtype=np.float64)
    np.cumsum(y, out=np.asarray(y))
    n_samples = data.shape[0]
    for i in range(x.shape[0]):
        y[i] /= n_samples
    return np.array([x, y])


EPS = 1e-5


def test_quantile_merge():
    gauss1 = Gaussian(-1, 1.5)
    gauss2 = Gaussian(3, .08)
    gauss3 = Gaussian(7, .5)

    g1data = gauss1.sample(100)
    g2data = gauss2.sample(100)
    g3data = gauss3.sample(100)
    data = np.vstack([sorted(g1data), sorted(g2data), sorted(g3data)])

    dist_all = QuantileDistribution(epsilon=EPS)
    dist_all.fit(data.reshape(-1, 1), None, 0)

    mainfig = go.Figure()

    # reg = CDFRegressor(eps=.01)
    # reg.fit(qdata(data))

    dist1 = QuantileDistribution(epsilon=EPS)
    dist1.fit(g1data.reshape(-1, 1), None, 0)
    mainfig.add_trace(
        go.Scatter(
            x=g1data.ravel(),
            y=np.zeros(g1data.shape[0]),
            marker=dict(
                symbol='circle',
                color='blue',
                size=10,
            ),
            mode="markers",
            name='Raw data'
        )
    )

    # reg.fit(np.array(qdata(gauss1)).T, presort=1)
    # points = np.array(reg.points)

    dist2 = QuantileDistribution(epsilon=EPS)
    dist2.fit(g2data.reshape(-1, 1), None, 0)
    mainfig.add_trace(
        go.Scatter(
            x=g2data.ravel(),
            y=np.zeros(g2data.shape[0]),
            marker=dict(
                symbol='circle',
                color='green',
                size=10,
            ),
            mode="markers",
            name='Raw data'
        )
    )

    # reg.fit(np.array(qdata(gauss2)).T, presort=1)
    # points = np.array(reg.points)

    dist3 = QuantileDistribution(epsilon=EPS)
    dist3.fit(g3data.reshape(-1, 1), None, 0)
    mainfig.add_trace(
        go.Scatter(
            x=g3data.ravel(),
            y=np.zeros(g3data.shape[0]),
            marker=dict(
                symbol='circle',
                color='red',
                size=10,
            ),
            mode="markers",
            name='Raw data'
        )
    )

    # reg.fit(np.array(qdata(gauss3)).T, presort=1)
    # points = np.array(reg.points)

    dist = QuantileDistribution.merge([dist1, dist2, dist3], [1/3, 1/3, 1/3])

    x = np.linspace(-7, 9, 500)
    cdf1 = dist1.cdf.multi_eval(x)
    cdf2 = dist2.cdf.multi_eval(x)
    cdf3 = dist3.cdf.multi_eval(x)

    cdf = dist_all.cdf.multi_eval(x)
    cdf_merged = dist.cdf.multi_eval(x)

    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(cdf1),
            line=dict(
                color='blue',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='CDF-1'
        )
    )
    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(cdf2),
            line=dict(
                color='green',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='CDF-2'
        )
    )
    mainfig.add_trace(
        go.Scatter(
            x=x,
            y=np.asarray(cdf3),
            line=dict(
                color='red',
                width=2,
                dash='solid',
            ),
            mode="lines",
            name='CDF-3'
        )
    )
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
        xaxis=dict(
            side='bottom',
        ),
        yaxis=dict(
            range=[0, 1]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.8
        ),
    )

    mainfig.show(
        config=dict(
            displaylogo=False,
            toImageButtonOptions=dict(
                format='svg',  # one of png, svg, jpeg, webp
                filename="quantile_merge.svg",
                scale=1
            )
        )
    )


def test_quantiles():
    gauss1 = Gaussian(-1, 1.5)
    g1data = gauss1.sample(100)
    gauss2 = Gaussian(3, .08)
    g2data = gauss2.sample(100)
    gauss3 = Gaussian(7, .5)
    g3data = gauss3.sample(100)
    data = np.hstack([sorted(g1data), sorted(g2data), sorted(g3data)])

    reg = CDFRegressor(eps=.01)
    reg.fit(qdata(data))

    reg2 = CDFRegressor(eps=.05)
    reg2.fit(qdata(data))

    # print(reg.cdf.pfmt())

    points = np.array(reg.support_points)
    points2 = np.array(reg2.support_points)

    dist_all = QuantileDistribution(epsilon=.1)
    dist_all.fit(data.reshape(-1, 1), None, 0)

    # dist1 = QuantileDistribution(epsilon=1e-10)
    # dist1.fit(gauss1)

    # reg.fit(np.array(qdata(gauss1)).T, presort=1)
    # points = np.array(reg.points)

    # dist2 = QuantileDistribution(epsilon=1e-10)
    # dist2.fit(gauss2)

    # reg.fit(np.array(qdata(gauss2)).T, presort=1)
    # points = np.array(reg.points)

    # dist3 = QuantileDistribution(epsilon=1e-10)
    # dist3.fit(gauss3)

    # reg.fit(np.array(qdata(gauss3)).T, presort=1)
    # points = np.array(reg.points)

    # dist = QuantileDistribution.merge([dist1, dist2, dist3], [.333, .333, .333])

    # x = np.linspace(-7, 9, 500)
    # cdf1 = dist_all.cdf.multi_eval(x)
    # cdf2 = dist_all.cdf.multi_eval(x)
    # cdf = dist_all.cdf.multi_eval(x)

    # out('CDF:', dist.cdf.pfmt())

    # pdf = dist_all.pdf.multi_eval(x)
    # ppf = dist_all.ppf.multi_eval(x)

    # out('PPF:', dist.ppf.pfmt())

    # out('PPF(1) =', dist.ppf.eval(1))
    # out('PPF(0) =', dist.ppf.eval(0))

    x = np.linspace(-4, 9, 300)
    x.sort()
    cdf = np.array([gauss1.cdf(d)[0]/3 + gauss2.cdf(d)[0]/3 + gauss3.cdf(d)[0]/3 for d in x])

    fname = "Quantile-Example"
    mainfig = go.Figure()

    # scatter x sin(x) function
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
            name=r'$\text{PLF of CDF with }\varepsilon = 0.01$'
        )
    )

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
            name=r'$\text{PLF of CDF with }\varepsilon = 0.05$'
        )
    )

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
        xaxis=dict(
            side='bottom',
        ),
        yaxis=dict(
            range=[0, 1]
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=.8
        ),
    )

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


def main(*args):
    test_quantiles()
    test_quantile_merge()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
