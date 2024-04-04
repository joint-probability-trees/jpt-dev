import ctypes
import os
import threading
from itertools import zip_longest
from multiprocessing import shared_memory
from typing import Union, Optional, List

import numpy as np
import pandas as pd
from dnutils import mapstr, logs, out, ifnone
from joblib._multiprocessing_helpers import mp
from tqdm import tqdm

from ..base.multicore import Pool
from ..trees import JPT

_locals = threading.local()


logger = logs.getlogger('/jpt/learning/preprocessing')


def _terminate_worker():
    logger.debug(
       f'Closing shared memory "{_locals.shm.name}" in PID {os.getpid()}'
    )
    _locals.shm.close()


# noinspection PyUnresolvedReferences
def _initialize_worker(shm_name):
    logger.debug(
        f'Initializing worker {os.getpid()} with shared memory "{shm_name}"'
    )
    shm = mp.shared_memory.SharedMemory(
        shm_name
    )
    _locals.shm = shm
    df = _locals.df

    _locals.data = np.ndarray(
        shape=(len(df), len(df.columns)),
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )


def map_col(args) -> None:
    i, col = args
    df = _locals.df
    data_ = _locals.data

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

    :param jpt:
    :param verbose:
    :param multicore:
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

    memory_required = shape[0] * shape[1] * ctypes.sizeof(ctypes.c_double)

    try:
        import psutil
        memory_available = psutil.virtual_memory().available
        logger.info(
            f'Required RAM: {memory_required / 1e6:,.2f} MB,'
            f'Available RAM: {memory_available / 1e6:,.2f} MB'
        )
        if memory_available < memory_required:
            logger.warning(
                f'Out of memory: {memory_required / 1e6:,.2f} MB required, '
                f'{memory_available / 1e6:,.2f} MB available.'
            )
    except ModuleNotFoundError:
        pass

    # Allocate shared memory for the training data
    shm = shared_memory.SharedMemory(
        name=f"preprocessing-{str(threading.get_ident())}",
        create=True,
        size=memory_required
    )

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
            progressbar = tqdm(
                total=len(data.columns),
                desc='Preprocessing data'
            )

        n_processes = ifnone(multicore, mp.cpu_count())

        with Pool(
            processes=n_processes,
            local=_locals,
            initializer=_initialize_worker,
            initargs=(shm.name,),
            terminator=_terminate_worker
        ) as processes:
            for _ in processes.imap_unordered(
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

    logger.debug(
        f'Copying data ({data_.nbytes / 1e6} MB)...'
    )
    result = np.copy(data_, order='C')

    logger.debug(
        f'Clearing shared data structures...'
    )
    shm.close()
    shm.unlink()
    _locals.__dict__.clear()

    return result
