import ctypes
import threading
from itertools import zip_longest
from multiprocessing import shared_memory, Pool
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from dnutils import mapstr, logs, out
from tqdm import tqdm

from ..trees import JPT

_locals = threading.local()


logger = logs.getlogger('/jpt/learning/preprocessing')


def map_col(args) -> None:
    i, col = args
    df = _locals.df

    shm = shared_memory.SharedMemory(
        _locals.shm
    )

    data_ = np.ndarray(
        shape=(len(df), len(df.columns)),
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )

    try:
        if col in _locals.transformations:
            data_[:, i] = df[col].map(
                _locals.transformations[col]
            ).values

        else:
            data_[:, i] = df[col].values
    except ValueError as e:
        raise ValueError(
            f'{e} of {_locals.jpt.varnames[col].domain.__qualname__} of variable {col}'
        )
    shm.close()

    return col


def preprocess_data(
        jpt: JPT,
        data: Union[np.ndarray, pd.DataFrame],
        rows: Optional[Union[np.ndarray, List]] = None,
        columns: Optional[Union[np.ndarray, List]] = None,
        multicore: int = None,
        verbose: bool = False
) -> np.ndarray:
    """
    Transform the input data into an internal representation.

    :param data: The data to transform
    :param rows: The indices of the rows that will be transformed
    :param columns: The indices of the columns that will be transformed
    :return: the preprocessed data
    """
    if sum(d is not None for d in (data, rows, columns)) > 1:
        raise ValueError('Only either of the three is allowed.')
    elif sum(d is not None for d in (data, rows, columns)) < 1:
        raise ValueError('No data passed.')

    logger.info('Preprocessing data...')

    if isinstance(data, np.ndarray) and data.shape[0] or isinstance(data, list):
        rows = data

    # Transpose the rows
    if isinstance(rows, list) and rows:
        columns = [
            [row[i] for row in rows]
            for i in range(len(jpt.variables))
        ]
    elif isinstance(rows, np.ndarray) and rows.shape[0]:
        columns = rows.T

    if isinstance(columns, list) and columns:
        shape = len(columns[0]), len(columns)
    elif isinstance(columns, np.ndarray) and columns.shape:
        shape = columns.T.shape
    elif isinstance(data, pd.DataFrame):
        shape = data.shape
        data = data.copy()
    else:
        raise ValueError('No data given.')

    # Allocate shared memory for the training data
    shm = shared_memory.SharedMemory(
        name=f"preprocessing-{str(threading.get_ident())}",
        create=True,
        size=shape[0] * shape[1] * ctypes.sizeof(ctypes.c_double)
    )
    _locals.shm = shm.name

    data_ = np.ndarray(
        shape=shape,
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )

    # print(data_)
    # Make the original data available to worker processes
    _locals.df = data

    if isinstance(data, pd.DataFrame):
        if set(jpt.varnames).symmetric_difference(set(data.columns)):
            raise ValueError(
                'Unknown variable names: %s'
                % ', '.join(
                    mapstr(
                        set(jpt.varnames)
                        .symmetric_difference(
                            set(data.columns)
                        )
                    )
                )
            )
        # Check if the order of columns in the data frame is the same
        # as the order of the variables.
        if not all(c == v for c, v in zip_longest(data.columns, jpt.varnames)):
            raise ValueError(
                'Columns in DataFrame must coincide with '
                'variable order: %s' % ', '.join(mapstr(jpt.varnames))
            )

        _locals.transformations = {
            v: jpt.varnames[v].domain.values.map
            for v in data.columns
        }
        if verbose:
            progressbar = tqdm(total=len(data.columns))
        with Pool(multicore) as pool:
            for v in pool.imap_unordered(
                    map_col,
                    iterable=[(i, c) for i, c in enumerate(data.columns)]
            ):
                if verbose:
                    progressbar.update(1)
        if verbose:
            progressbar.close()

    else:
        for i, (var, col) in enumerate(zip(jpt.variables, columns)):
            data_[:, i] = [var.domain.values[v] for v in col]
    result = np.copy(data_, order='C')
    shm.close()
    shm.unlink()

    _locals.__dict__.clear()

    return result
