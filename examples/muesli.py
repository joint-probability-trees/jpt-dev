import itertools
import os

import pandas as pd
import pyximport
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pyximport.install()


def plot_muesli():

    # df = pd.read_pickle('data/human_muesli.dat')
    df = pd.read_csv('data/muesli.csv')
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)

    fig, ax = plt.subplots()
    ax.set_title(f"Breakfast object positions")
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # verschiedene Farben UND Formen (Farben fuer prettyprinting, Formen fuer Unterscheidung in b/w Paper)
    icons = itertools.cycle(['o', 'x', '1'])
    colors = ['orange', 'cornflowerblue', 'green', 'yellow']

    # if only original data is used:
    succ = False
    for color, icon, clazz in zip(colors, icons, df['Class'].unique()):
        samples = df[df['Class'] == clazz]
        ax.scatter(samples['X'], samples['Y'], color=color, marker=icon, label=clazz)

    # if extra dimension (success) is used:
    # succ = True
    # for icon, clazz in zip(icons, df['Class'].unique()):
    #     samples = df[df['Class'] == clazz]
    #     for color, succ in zip(colors, df['Success'].unique()):
    #         spls = samples[samples['Success'] == succ]
    #         ax.scatter(spls['X'], spls['Y'], color=color, marker=icon, label=f'{clazz} $\u2192$ {succ}')

    ax.legend()

    plt.show()

    with PdfPages(os.path.join('data', f'muesli-{2 if succ else 1}.pdf'), metadata={'Creator': 'misc',
                                                                  'Author': 'Picklum & Nyga',
                                                                  'Title': f'Müsli Example{" Success" if succ else ""}'}) as pdf:
        pdf.savefig(fig)


if __name__ == '__main__':
    plot_muesli()
