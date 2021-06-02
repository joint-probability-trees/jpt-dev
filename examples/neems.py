import os
from datetime import datetime

import numpy as np
import pandas as pd

from dnutils import out
from jpt.learning.distributions import SymbolicType, Numeric, Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable, NumericVariable


def neemdata():
    # learn from dataset from sebastian (fetching milk from fridge)

    # location of NEEM_SetTable_Breakfast.tar.xz
    ltargz = 'https://seafile.zfn.uni-bremen.de/f/fa5a760d89234cfc83ad/?dl=1'
    df = pd.read_csv(ltargz, compression='xz', delimiter=';', sep=';', skip_blank_lines=True, header=0,
                     index_col=False,
                     names=['id', 'type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'parent', 'next', 'previous', 'object_acted_on', 'object_type', 'bodyPartsUsed', 'arm', 'grasp', 'effort'],
                     usecols=['type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'object_acted_on', 'bodyPartsUsed', 'arm'],
                     na_values=['type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'object_acted_on', 'bodyPartsUsed', 'arm', np.inf])

    # set default values for empty, infinity or nan values and remove nan rows
    df = df[df['endTime'].notna()]  # this not only removes the lines with endTime=NaN, but in particular all lines where each feature is NaN
    df.replace([np.inf, -np.inf], -1, inplace=True)
    df['object_acted_on'] = df['object_acted_on'].fillna('DEFAULTOBJECT')
    df['bodyPartsUsed'] = df['bodyPartsUsed'].fillna('DEFAULTBP')
    df['arm'] = df['arm'].fillna('DEFAULTARM')
    df['failure'] = df['failure'].fillna('DEFAULTFAIL')
    df['startTime'] = df['startTime'].fillna(-1)
    df['endTime'] = df['endTime'].fillna(-1)
    df['duration'] = df['duration'].fillna(-1)

    # type declarations
    tpTYPE = SymbolicType('type', df['type'].unique())
    failTYPE = SymbolicType('failure', df['failure'].unique())
    oaoTYPE = SymbolicType('object_acted_on', df['object_acted_on'].unique())
    bpuTYPE = SymbolicType('bodyPartsUsed', df['bodyPartsUsed'].unique())
    armTYPE = SymbolicType('arm', df['arm'].unique())

    # variable declarations
    tp = SymbolicVariable('type', tpTYPE)
    st = NumericVariable('startTime', Numeric, haze=.1)
    et = NumericVariable('endTime', Numeric, haze=.1)
    dur = NumericVariable('duration', Numeric, haze=.1)
    succ = SymbolicVariable('success', Bool)
    fail = SymbolicVariable('failure', failTYPE)
    oao = SymbolicVariable('object_acted_on', oaoTYPE)
    bpu = SymbolicVariable('bodyPartsUsed', bpuTYPE)
    arm = SymbolicVariable('arm', armTYPE)

    vars = [tp, st, et, dur, succ, fail, oao, bpu, arm]
    jpt = JPT(variables=vars, min_samples_leaf=500)
    out(f'Learning sebadata-Tree...')
    jpt.learn(columns=df.values.T)
    out(f'Done! Plotting...')
    jpt.plot(filename=jpt.name, plotvars=vars, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-NEEMdata'), view=True)


def main(*args):
    neemdata()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
