import itertools

import pandas as pd
from dnutils import out
from matplotlib import pyplot as plt

from jpt.learning.distributions import SymbolicType, Numeric
from jpt.learning.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def test_tourism():
    df = pd.read_csv('data/tourism.csv')
    df['Price'] *= 2000
    df['DoY'] *= 710

    fig, ax = plt.subplots()
    ax.set_title('Flights from FRA')
    ax.set_xlabel('value')
    ax.set_ylabel('%')

    # ax.plot(data_, np.cumsum([1] * len(data_)) / len(data_), color='green', label='CumSum($\mathcal{D}$)', linewidth=2)
    # ax.plot(data_, cdf.multi_eval(data_), color='orange', linewidth=2, label='Piecewise fn from original data')
    markers = {'AYT': '1', 'BKK': 'x', 'FOR': '+'}
    colors = {'FAMILY': 'red', 'COUPLE': 'green', 'GROUP': 'blue', 'SINGLE+CHILD': 'orange'}

    for dest, persona in [(d, p) for d in df['Destination'].unique() for p in df['Persona'].unique()]:
        out(dest, persona)
        samples = df[(df['Destination'] == dest) & (df['Persona'] == persona)]
        ax.scatter(samples['Price'] , samples['DoY'],
                   color=colors[persona],
                   marker=markers[dest],
                   label=r'%s $\rightarrow$ %s' % (persona, dest),
                   s=150,
                   )
    ax.set_xlabel('Price')
    ax.set_ylabel('Day of Year')
    ax.grid()
    ax.legend()
    # ax.plot(bounds, cdf.multi_eval(bounds), color='cornflowerblue', linestyle='dashed', linewidth=2, markersize=12, label='Piecewise fn from bounds')
    # ax.plot(sampled, cdf.multi_eval(sampled), color='red', linestyle='dotted', linewidth=2, label='Piecewise fn from original data')

    # ax2 = ax.twiny()
    # ax2.set_xlim(left=0., right=1.0)
    # ax2.plot(bounds, cdf.multi_eval(bounds), color='cornflowerblue', linestyle='dashed', linewidth=2, markersize=12, label='Piecewise fn from bounds')
    # ax.legend()

    plt.show()



if __name__ == '__main__':
    test_tourism()
