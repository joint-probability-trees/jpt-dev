import ctypes
import os
import sys
import threading
import multiprocessing as mp
from typing import Tuple, Optional, Dict, Any

import numpy as np

from dnutils import ifnone, getlogger
from tqdm import tqdm

from jpt.trees import JPT
from jpt.base.multicore import Pool
from jpt.variables import ValueAssignment


_locals = threading.local()

logger = getlogger('/jpt/inference/likelihood')


def _terminate_worker():
    logger.debug(
       f'Closing shared memory "{_locals.shm.name}" in PID {os.getpid()}'
    )
    _locals.shm.close()


# noinspection PyUnresolvedReferences
def _initialize_worker(shm_name, single_likelihoods):
    logger.debug(
        f'Initializing worker {os.getpid()} with shared memory "{shm_name}"'
    )
    shm = mp.shared_memory.SharedMemory(
        shm_name
    )
    _locals.shm = shm

    data = _locals.data

    if single_likelihoods:
        shape = data.shape
    else:
        shape = (data.shape[0], 1)

    _locals.results = np.ndarray(
        shape=shape,
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )


def single_likelihood(args) -> Tuple[int, Any]:
    # print('worker', os.getpid())
    idx, single_likelihoods = args
    jpt = _locals.jpt
    data = _locals.data

    values = ValueAssignment(
        [(var, var.assignment2set(val)) for var, val in zip(jpt.variables, data[idx, :])],
        jpt.variables
    )

    results = _locals.results

    found_leaf = False

    for leaf in jpt.apply(values):
        if found_leaf:
            raise ValueError(
                f'Illegal argument for likelihood reasoning:'
                f'{values}. More than one leaf apply.'
            )
        found_leaf = True

        results[idx:idx + 1, :] = leaf.likelihood(
            data[idx:idx + 1, :],
            dirac_scaling=_locals.dirac_scaling,
            min_distances=ifnone(_locals.min_distances, jpt.minimal_distances),
            single_likelihoods=single_likelihoods
        )

    if not found_leaf:
        raise ValueError(
            f'No leaf applies for {values}'
        )

    return idx


# noinspection PyUnresolvedReferences
def parallel_likelihood(
        jpt: JPT,
        data: np.ndarray,
        dirac_scaling: float = 2.,
        min_distances: Dict = None,
        multicore: Optional[int] = None,
        verbose: bool = False,
        single_likelihoods: bool = False
) -> np.ndarray:

    _locals.data = data
    _locals.jpt = jpt
    _locals.dirac_scaling = dirac_scaling
    _locals.min_distances = min_distances
    _locals.single_likelihoods = single_likelihoods

    if single_likelihoods:
        shape = data.shape
    else:
        shape = (data.shape[0], 1)

    shm = mp.shared_memory.SharedMemory(
        name=f"likelihood-{threading.get_ident():d}",
        create=True,
        size=shape[0] * shape[1] * ctypes.sizeof(ctypes.c_double)
    )

    data_ = np.ndarray(
        shape=shape,
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )

    progress = None
    if verbose:
        progress = tqdm(total=data.shape[0], desc='Computing likelihoods')

    n_processes = ifnone(multicore, mp.cpu_count())
    chunksize = max(1, int(data.shape[0] / n_processes / 2))

    with Pool(
        processes=n_processes,
        local=_locals,
        initializer=_initialize_worker,
        initargs=(shm.name, single_likelihoods),
        terminator=_terminate_worker
    ) as processes:
        for _ in processes.imap_unordered(
            single_likelihood,
            iterable=((i, single_likelihoods) for i in range(data.shape[0])),
            chunksize=chunksize
        ):
            if verbose:
                progress.update(1)
                sys.stderr.flush()

    if verbose:
        progress.close()

    results = np.copy(data_, order='C')

    shm.close()
    shm.unlink()

    _locals.__dict__.clear()

    if single_likelihoods:
        return results
    else:
        return results[:, 0].T
