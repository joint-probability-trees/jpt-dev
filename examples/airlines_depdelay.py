import os
import sys
from datetime import datetime

import pandas as pd

import dnutils
from jpt.learning.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import NumericVariable, SymbolicVariable

logger = dnutils.getlogger('/airline', level=dnutils.INFO)


def main():
    try:
        logger.info('Trying to load dataset from local file...')
        data = pd.read_csv('../examples/data/airlines_train_regression_10000000.csv')
    except FileNotFoundError:
        src = 'https://www.openml.org/data/get_csv/22044760/airlines_train_regression_10000000.csv'
        logger.warning(f'The file containing this dataset is not in the repository, as it is very large.\nI will try downloading file {src} now...')
        try:
            data = pd.read_csv(src, delimiter=',', sep=',', skip_blank_lines=True, header=0)
            logger.info(f'Success! Downloaded dataset containing {data.shape[0]} instances of {data.shape[1]} features each')
        except pd.errors.ParserError:
            logger.error('Could not download and/or parse file. Please download it manually and try again.')
            sys.exit(-1)

    data = data.sample(frac=1)

    logger.info('creating types and variables...')
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

    tree = JPT(variables=variables, min_samples_leaf=data.shape[0]*.01)
    tree.learn(columns=data.values.T)
    tree.plot(title='Airlines Departure Delay Prediction', directory=os.path.join('/tmp', f'{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-Airlines'), view=False)
    logger.info(tree)
    tree.save(os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Airline.json'))


if __name__ == '__main__':
    main()
