import os
from datetime import datetime

import pandas as pd
from dnutils import out
from matplotlib import pyplot as plt

from jpt.base.utils import format_path
from jpt.learning.distributions import SymbolicType, Numeric
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def tourism():

    # generate JPT from tourism data
    df = pd.read_csv('../examples/data/tourism.csv')
    df['Price'] *= 2000
    df['DoY'] *= 710

    DestinationType = SymbolicType('DestinationType', df['Destination'].unique())
    PersonaType = SymbolicType('PersonaType', df['Persona'].unique())

    price = NumericVariable('Price', Numeric)
    t = NumericVariable('Time', Numeric, haze=.1)
    d = SymbolicVariable('Destination', DestinationType)
    p = SymbolicVariable('Persona', PersonaType)

    jpt = JPT(variables=[price, t, d, p], min_samples_leaf=15)
    # out(df.values.T)
    jpt.learn(columns=df.values.T[1:])

    for clazz in df['Destination'].unique():
        out(jpt.infer(query={d: clazz}, evidence={t: 300}))
        for exp in jpt.expectation([t, price], evidence={d: clazz}, confidence_level=.95):
            out(exp)
    for persona in df['Persona'].unique():
        for exp in jpt.expectation([t, price], evidence={p: persona}, confidence_level=.95):
            out(exp)

    # Test the MPE inference
    out(jpt.mpe({t: 150}))

    jpt.plot(plotvars=[price, t, d, p],
             directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Tourism'))  # plotvars=[price, t]
    # plot_tourism()


def plot_tourism():
    # generate plot for tree data
    df = pd.read_csv('../examples/data/tourism.csv')
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

    plt.show()

    # ax2 = ax.twiny()
    # ax2.set_xlim(left=0., right=1.0)
    # ax2.plot(bounds, cdf.multi_eval(bounds), color='cornflowerblue', linestyle='dashed', linewidth=2, markersize=12, label='Piecewise fn from bounds')
    # ax.legend()


def main(*args):
    # plot_tourism()
    tourism()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
