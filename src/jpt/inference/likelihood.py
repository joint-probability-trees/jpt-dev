import ctypes
import threading
from multiprocessing import Pool, cpu_count, shared_memory
from typing import Tuple, Optional, Dict, Any

import numpy as np
from dnutils import ifnone
from tqdm import tqdm

from jpt import JPT
from jpt.variables import ValueAssignment

_locals = threading.local()


def single_likelihood(args) -> Tuple[int, Any]:
    jpt = _locals.jpt
    data = _locals.data
    idx, single_likelihoods = args
    values = ValueAssignment(
        [(var, var.assignment2set(val)) for var, val in zip(jpt.variables, data[idx, :])],
        jpt.variables
    )
    found_leaf = False

    shm = shared_memory.SharedMemory(
        _locals.shm
    )

    if single_likelihoods:
        shape = data.shape
    else:
        shape = (data.shape[0], 1)

    results = np.ndarray(
        shape=shape,
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )

    for leaf in jpt.apply(values):
        if found_leaf:
            raise ValueError(
                f'Illegal argument for likelihood reasoning:'
                f'{values}. More than one leaf apply.'
            )
        found_leaf = True

        likelihood = leaf.likelihood(
            data[idx:idx + 1, :],
            dirac_scaling=_locals.dirac_scaling,
            min_distances=ifnone(_locals.min_distances, jpt.minimal_distances),
            single_likelihoods=single_likelihoods
        )
        results[idx:idx + 1, :] = likelihood

    if not found_leaf:
        raise ValueError(
            f'No leaf applies for {values}'
        )

    shm.close()
    return idx


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

    n_processes = ifnone(multicore, cpu_count())

    if single_likelihoods:
        shape = (data.shape[0], len(jpt.variables))
    else:
        shape = (data.shape[0], 1)

    shm = shared_memory.SharedMemory(
        name=f"likelihood-{str(threading.get_ident())}",
        create=True,
        size=shape[0] * shape[1] * ctypes.sizeof(ctypes.c_double)
    )

    data_ = np.ndarray(
        shape=shape,
        dtype=np.float64,
        order='C',
        buffer=shm.buf
    )

    _locals.shm = shm.name

    if verbose:
        progress = tqdm(total=data.shape[0], desc='Computing likelihoods')

    with Pool(processes=n_processes) as processes:
        for _ in processes.imap_unordered(
            single_likelihood,
            iterable=((i, single_likelihoods) for i in range(data.shape[0])),
            chunksize=max(1, int(data.shape[0] / n_processes / 2))
        ):
            if verbose:
                progress.update(1)

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
