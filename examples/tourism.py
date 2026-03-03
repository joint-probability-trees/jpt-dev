"""Real-world mixed-type data: flight price prediction.

Loads a tourism dataset with flight prices, destinations,
travel dates, and traveler types. A JPT is learned over
both symbolic and numeric features, then queried for
conditional expectations and most probable explanations.

Demonstrates:
    - Mixed symbolic/numeric variable models
    - ``expectation()`` for conditional means
    - ``mpe()`` for most probable explanation
    - ``infer()`` with symbolic evidence
"""
import os
import tempfile

import pandas as pd

from jpt.distributions import SymbolicType, Numeric
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data'
)


# -------------------------------------------------------


def main(visualize=True):
    """Learn a JPT from tourism data and run inference
    queries.

    :param visualize: whether to show interactive plots
    """
    # Load and scale the tourism dataset
    df = pd.read_csv(
        os.path.join(_DATA_DIR, 'tourism.csv')
    )
    # Scale normalized values to real-world units
    df['Price'] *= 2000
    df['DoY'] *= 710

    # Define variable types from the data
    DestinationType = SymbolicType(
        'DestinationType', df['Destination'].unique()
    )
    PersonaType = SymbolicType(
        'PersonaType', df['Persona'].unique()
    )

    # Create variables
    price = NumericVariable('Price', Numeric)
    t = NumericVariable('Time', Numeric, haze=.1)
    d = SymbolicVariable('Destination', DestinationType)
    p = SymbolicVariable('Persona', PersonaType)

    # Learn the JPT
    jpt = JPT(
        variables=[price, t, d, p],
        min_samples_leaf=15
    )
    jpt.learn(columns=df.values.T[1:])

    # Query conditional probabilities per destination
    for clazz in df['Destination'].unique():
        prob = jpt.infer(
            query={d: clazz},
            evidence={t: 300}
        )
        print(f'P(Destination={clazz} | Time=300)'
              f' = {prob}')
        for exp in jpt.expectation(
                [t, price],
                evidence={d: clazz},
                confidence_level=.95
        ):
            print(f'  E[{exp}]')

    # Query conditional expectations per persona
    for persona in df['Persona'].unique():
        for exp in jpt.expectation(
                [t, price],
                evidence={p: persona},
                confidence_level=.95
        ):
            print(f'  E[{exp} | Persona={persona}]')

    # Most probable explanation for Time=150
    print('\nMPE for Time=150:')
    print(jpt.mpe({t: 150}))

    # Plot the learned tree
    out_dir = tempfile.mkdtemp(prefix='jpt-tourism-')
    jpt.plot(
        plotvars=[price, t, d, p],
        directory=out_dir,
        view=visualize
    )


if __name__ == '__main__':
    main(visualize=True)
