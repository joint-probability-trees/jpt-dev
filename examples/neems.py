"""Remote dataset loading: robot manipulation data.

Downloads robot manipulation data (NEEM: Narrative
Enabled Episodic Memory) from a remote server, learns
activity models over action types, durations, success/
failure outcomes, and body part usage.

Demonstrates:
    - Remote data loading via URL
    - NaN handling and data cleaning
    - Symbolic variable inference from data columns
    - Tree serialization with ``save()``
"""
import logging
import os
import tempfile

import numpy as np
import pandas as pd

from jpt.distributions import Numeric, SymbolicType
from jpt.trees import JPT
from jpt.variables import (
    NumericVariable,
    SymbolicVariable,
)


logger = logging.getLogger(__name__)


# -------------------------------------------------------


def main(visualize=True):
    """Download NEEM data, learn a JPT, and save it.

    :param visualize: whether to show interactive plots
    """
    # Download the NEEM dataset from the remote server
    url = (
        'https://seafile.zfn.uni-bremen.de'
        '/f/fa5a760d89234cfc83ad/?dl=1'
    )
    try:
        df = pd.read_csv(
            url,
            compression='xz',
            delimiter=';',
            skip_blank_lines=True,
            header=0,
            index_col=False,
            names=[
                'id', 'type', 'startTime', 'endTime',
                'duration', 'success', 'failure',
                'parent', 'next', 'previous',
                'object_acted_on', 'object_type',
                'bodyPartsUsed', 'arm', 'grasp',
                'effort',
            ],
            usecols=[
                'type', 'startTime', 'endTime',
                'duration', 'success', 'failure',
                'object_acted_on', 'bodyPartsUsed',
                'arm',
            ],
            na_values=[
                'type', 'startTime', 'endTime',
                'duration', 'success', 'failure',
                'object_acted_on', 'bodyPartsUsed',
                'arm', np.inf,
            ],
        )
    except Exception as e:
        logger.error(
            'Could not download NEEM dataset: %s', e
        )
        return

    # Clean the data: remove NaN rows and fill defaults
    df = df[df['endTime'].notna()]
    df.replace([np.inf, -np.inf], -1, inplace=True)
    df['object_acted_on'] = (
        df['object_acted_on'].fillna('DEFAULTOBJECT')
    )
    df['bodyPartsUsed'] = (
        df['bodyPartsUsed'].fillna('DEFAULTBP')
    )
    df['arm'] = df['arm'].fillna('DEFAULTARM')
    df['failure'] = df['failure'].fillna('DEFAULTFAIL')
    df['startTime'] = df['startTime'].fillna(-1)
    df['endTime'] = df['endTime'].fillna(-1)
    df['duration'] = df['duration'].fillna(-1)
    df['success'] = df['success'].astype(str)

    # Define variable types from the data
    tpTYPE = SymbolicType(
        'type', df['type'].unique()
    )
    succTYPE = SymbolicType(
        'success', df['success'].unique()
    )
    failTYPE = SymbolicType(
        'failure', df['failure'].unique()
    )
    oaoTYPE = SymbolicType(
        'object_acted_on',
        df['object_acted_on'].unique()
    )
    bpuTYPE = SymbolicType(
        'bodyPartsUsed', df['bodyPartsUsed'].unique()
    )
    armTYPE = SymbolicType(
        'arm', df['arm'].unique()
    )

    # Create variables
    tp = SymbolicVariable('type', tpTYPE)
    dur = NumericVariable('duration', Numeric, blur=.1)
    succ = SymbolicVariable('success', succTYPE)
    fail = SymbolicVariable('failure', failTYPE)
    oao = SymbolicVariable('object_acted_on', oaoTYPE)
    bpu = SymbolicVariable('bodyPartsUsed', bpuTYPE)
    arm = SymbolicVariable('arm', armTYPE)

    variables = [tp, dur, succ, fail, oao, bpu, arm]

    # Learn the JPT
    jpt = JPT(
        variables=variables,
        min_samples_leaf=0.0005
    )

    print('Learning NEEM tree...')
    jpt.learn(
        columns=df[
            [v.name for v in variables]
        ].values.T
    )
    print(f'Done! Tree has {len(jpt.leaves)} leaves.')

    # Save the learned model
    out_dir = tempfile.mkdtemp(prefix='jpt-neems-')
    save_path = os.path.join(out_dir, 'neem.jpt')
    jpt.save(save_path)
    print(f'Model saved to {save_path}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(visualize=True)
