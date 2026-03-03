"""Classification with symbolic variables.

Demonstrates the classic restaurant wait-or-leave decision
problem using Joint Probability Trees. Two approaches are
shown:

1. **Manual sampling**: Define variable distributions by
   hand and generate training data from them.
2. **CSV-based learning**: Load a dataset from CSV and
   learn a JPT directly from the data.

Demonstrates:
    - SymbolicVariable, Bool, SymbolicType
    - Manual distribution specification and sampling
    - Learning from CSV data
    - Probabilistic inference with ``infer()``
    - Human-readable explanations with ``explain()``
"""
import os
import tempfile

import pandas as pd

from jpt.distributions import Bool, SymbolicType
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def restaurant_manual_sample(visualize=True):
    """Learn a JPT from manually sampled restaurant data.

    Variable distributions are set by hand and used to
    generate synthetic training data. The resulting JPT
    is then queried for the probability of a bar being
    available without a reservation, given no rain.

    :param visualize: whether to show interactive plots
    """
    # Declare symbolic variable types
    PatronsType = SymbolicType(
        'Patrons', ['Some', 'Full', 'None']
    )
    PriceType = SymbolicType(
        'Price', ['$', '$$', '$$$']
    )
    FoodType = SymbolicType(
        'Food',
        ['French', 'Thai', 'Burger', 'Italian']
    )
    WaitEstType = SymbolicType(
        'WaitEstimate',
        ['0--10', '10--30', '30--60', '>60']
    )

    # Create variables
    al = SymbolicVariable('Alternatives', Bool)
    ba = SymbolicVariable('Bar', Bool)
    fr = SymbolicVariable('Friday', Bool)
    hu = SymbolicVariable('Hungry', Bool)
    pa = SymbolicVariable('Patrons', PatronsType)
    pr = SymbolicVariable('Price', PriceType)
    ra = SymbolicVariable('Rain', Bool)
    re = SymbolicVariable('Reservation', Bool)
    fo = SymbolicVariable('Food', FoodType)
    we = SymbolicVariable('WaitEstimate', WaitEstType)
    wa = SymbolicVariable('WillWait', Bool)

    # Generate training data by sampling from the
    # specified distributions
    numsamples = 500
    data = [
        [
            al.distribution().set(6 / 12.).sample_one(),
            ba.distribution().set(6 / 12.).sample_one(),
            fr.distribution().set(5 / 12.).sample_one(),
            hu.distribution().set(7 / 12.).sample_one(),
            pa.distribution().set(
                [4 / 12., 6 / 12., 2 / 12.]
            ).sample_one(),
            pr.distribution().set(
                [7 / 12., 2 / 12., 3 / 12.]
            ).sample_one(),
            ra.distribution().set(4 / 12.).sample_one(),
            re.distribution().set(5 / 12.).sample_one(),
            fo.distribution().set(
                [2 / 12., 4 / 12., 4 / 12., 2 / 12.]
            ).sample_one(),
            we.distribution().set(
                [6 / 12., 2 / 12., 2 / 12., 2 / 12.]
            ).sample_one(),
            wa.distribution().set(.5).sample_one(),
        ]
        for _ in range(numsamples)
    ]

    # Learn the JPT from the generated data
    variables = [
        al, ba, fr, hu, pa, pr, ra, re, fo, we, wa
    ]
    jpt = JPT(
        variables,
        min_samples_leaf=30,
        min_impurity_improvement=0
    )
    jpt.learn(
        pd.DataFrame(
            data,
            columns=list(jpt.varnames)
        )
    )

    # Plot the learned tree
    out_dir = tempfile.mkdtemp(prefix='jpt-restaurant-')
    jpt.plot(
        plotvars=variables,
        view=visualize,
        directory=out_dir
    )

    # Query: P(Bar=True, Reservation=False | Rain=False)
    q = {ba: True, re: False}
    e = {ra: False}
    res = jpt.infer(q, e)

    print(
        f'P({", ".join(f"{k.name}={v}" for k, v in q.items())}'
        f' | '
        f'{", ".join(f"{k.name}={v}" for k, v in e.items())})'
        f' = {res}'
    )


# -------------------------------------------------------


def restaurant_auto_sample(visualize=True):
    """Learn a JPT from the restaurant CSV dataset.

    Loads the restaurant dataset from CSV and learns a
    JPT directly from the tabular data. The model is
    then queried for conditional probabilities.

    :param visualize: whether to show interactive plots
    """
    # Load the restaurant dataset and drop incomplete rows
    df = pd.read_csv(
        os.path.join(_DATA_DIR, 'restaurant.csv')
    ).dropna()

    # Declare symbolic variable types
    PatronsType = SymbolicType(
        'Patrons', ['Some', 'Full', 'None']
    )
    PriceType = SymbolicType(
        'Price', ['$', '$$', '$$$']
    )
    FoodType = SymbolicType(
        'Food',
        ['French', 'Thai', 'Burger', 'Italian']
    )
    WaitEstType = SymbolicType(
        'WaitEstimate',
        ['0--10', '10--30', '30--60', '>60']
    )

    # Create variables
    al = SymbolicVariable('Alternatives', Bool)
    ba = SymbolicVariable('Bar', Bool)
    fr = SymbolicVariable('Friday', Bool)
    hu = SymbolicVariable('Hungry', Bool)
    pa = SymbolicVariable('Patrons', PatronsType)
    pr = SymbolicVariable('Price', PriceType)
    ra = SymbolicVariable('Rain', Bool)
    re = SymbolicVariable('Reservation', Bool)
    fo = SymbolicVariable('Food', FoodType)
    we = SymbolicVariable('WaitEstimate', WaitEstType)
    wa = SymbolicVariable('WillWait', Bool)

    variables = [
        pa, hu, fo, fr, al, ba, pr, ra, re, we, wa
    ]

    # Learn the JPT from the CSV data
    jpt = JPT(
        variables,
        min_samples_leaf=30,
        min_impurity_improvement=0
    )
    jpt.learn(df[list(jpt.varnames)])

    # Plot the learned tree
    out_dir = tempfile.mkdtemp(prefix='jpt-restaurant-')
    jpt.plot(
        plotvars=variables,
        view=visualize,
        directory=out_dir
    )

    # Query: P(Bar=True, Reservation=False | Rain=False)
    q = {ba: True, re: False}
    e = {ra: False}
    res = jpt.infer(q, e)

    print(
        f'P({", ".join(f"{k.name}={v}" for k, v in q.items())}'
        f' | '
        f'{", ".join(f"{k.name}={v}" for k, v in e.items())})'
        f' = {res}'
    )


# -------------------------------------------------------


def main(visualize=True):
    """Run both restaurant examples.

    :param visualize: whether to show interactive plots
    """
    restaurant_manual_sample(visualize=visualize)
    restaurant_auto_sample(visualize=visualize)


if __name__ == '__main__':
    main(visualize=True)
