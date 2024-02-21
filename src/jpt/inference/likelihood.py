import threading
from multiprocessing import Pool, cpu_count
from typing import Tuple, Optional, Dict

import numpy as np
from dnutils import ifnone
from tqdm import tqdm

from jpt import JPT
from jpt.variables import ValueAssignment

_locals = threading.local()


def single_likelihood(idx: int) -> Tuple[int, float]:
    jpt = _locals.jpt
    data = _locals.data
    values = ValueAssignment(
        [(var, var.assignment2set(val)) for var, val in zip(jpt.variables, data[idx, :])],
        jpt.variables
    )
    likelihood = 0
    found_leaf = False

    for leaf in jpt.apply(values):
        if likelihood:
            raise ValueError(
                f'Illegal argument for likelihood reasoning:'
                f'{values}. More than one leaf apply.'
            )
        found_leaf = True
        likelihood += leaf.likelihood(
            data[idx:idx+1, :],
            dirac_scaling=_locals.dirac_scaling,
            min_distances=ifnone(_locals.min_distances, jpt.minimal_distances)
        )

    if not found_leaf:
        raise ValueError(
            f'No leaf applies for {values}'
        )

    return idx, likelihood


def parallel_likelihood(
        jpt: JPT,
        data: np.ndarray,
        dirac_scaling: float = 2.,
        min_distances: Dict = None,
        multicore: Optional[int] = None,
        verbose: bool = False
) -> np.ndarray:

    _locals.data = data
    _locals.jpt = jpt
    _locals.dirac_scaling = dirac_scaling
    _locals.min_distances = min_distances

    n_processes = ifnone(multicore, cpu_count())

    results = np.ndarray(
        shape=data.shape[0],
        dtype=np.float64
    )

    if verbose:
        progress = tqdm(total=data.shape[0], desc='Computing likelihoods')

    with Pool(processes=n_processes) as processes:
        for idx, likelihood in processes.imap_unordered(
            single_likelihood,
            iterable=range(data.shape[0]),
            chunksize=max(1, int(data.shape[0] / n_processes / 2))
        ):
            results[idx] = likelihood
            if verbose:
                progress.update(1)

    if verbose:
        progress.close()

    _locals.__dict__.clear()

    return results
