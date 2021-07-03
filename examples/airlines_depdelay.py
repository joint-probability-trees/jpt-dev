import os
from datetime import datetime

import pandas as pd

from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable


def main():
    print('loading data...')
    try:
        data = pd.read_csv('../examples/data/airlines.csv')
    except FileNotFoundError:
        print('The file containing this dataset is not in the repository, as it is very large.\n'
              'Download it from here first: https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.csv')

    data = data.sample(frac=0.5)

    print('creating types and variables...')
    UniqueCarrier_type = SymbolicType('UniqueCarrier_type', data['UniqueCarrier'].unique())
    Origin_type = SymbolicType('Origin_type', data['Origin'].unique())
    Dest_type = SymbolicType('Dest_type', data['Dest'].unique())

    DepDelay = NumericVariable('DepDelay', Numeric)
    Month = NumericVariable('Month', Numeric)
    DayofMonth = NumericVariable('DayofMonth', Numeric)
    DayOfWeek = NumericVariable('DayOfWeek', Numeric)
    CRSDepTime = NumericVariable('CRSDepTime', Numeric)
    CRSArrTime = NumericVariable('CRSArrTime', Numeric)
    UniqueCarrier = SymbolicVariable('UniqueCarrier', UniqueCarrier_type)
    Origin = SymbolicVariable('Origin', Origin_type)
    Dest = SymbolicVariable('Dest', Dest_type)
    Distance = NumericVariable('Distance', Numeric)

    variables = [DepDelay, Month, DayofMonth, DayOfWeek, CRSDepTime, CRSArrTime, UniqueCarrier, Origin, Dest, Distance]

    started = datetime.now()
    print('Started learning of %s x %s at %s' % (data.shape[0], data.shape[1], started))
    tree = JPT(variables=variables, min_samples_leaf=20)
    tree.learn(columns=data.values.T)
    # tree.plot(title='Airlines Departure Delay Prediction', directory=os.path.join('/tmp', f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-Airlines'), view=False)
    print('Learning and plotting took %s. Saving...' % (datetime.now() - started))
    tree.save(os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Airline.json'))


if __name__ == '__main__':
    main()
