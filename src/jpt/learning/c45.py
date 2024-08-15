import datetime as dt
import gc
import math
import signal
import threading
from multiprocessing import Lock, Event, Pool, Array, RLock
from operator import attrgetter
from typing import Union, Dict, Tuple, Any, Optional, Callable, Set, List

import numpy as np
import pandas as pd
from .preprocessing import preprocess_data
from dnutils import mapstr, ifnone, getlogger
from tqdm import tqdm
import ctypes as c

from ..base.multicore import DummyPool
from ..base.utils import _write_error
from ..distributions import Distribution


from ..trees import JPT, DecisionNode, Leaf, Node
from ..variables import Variable
from .impurity import Impurity
from ..base.functions import PiecewiseFunction
from ..base.intervals import ContinuousSet, INC, EXC, IntSet, Interval
from ..distributions.qpd import QuantileDistribution

logger = getlogger('/jpt/learning/c45')


# ----------------------------------------------------------------------------------------------------------------------
# Global constants

DISCRIMINATIVE = 'discriminative'
GENERATIVE = 'generative'


# ----------------------------------------------------------------------------------------------------------------------
# Thread-local data structure to make the module thread-safe

_locals = threading.local()


def _initialize_worker_process():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# ----------------------------------------------------------------------------------------------------------------------
# C4.5 splitting criterion to generate recursive partitions

class JPTPartition:
    """
    Represents a partition of the input data during JPT learning.
    """

    def __init__(
            self,
            data: Optional[np.ndarray],
            start: int,
            end: int,
            node_idx: int,
            parent_idx: Optional[int],
            child_idx: Optional[int],
            path: List[Set or Interval],
            min_samples_leaf: int,
            depth: int
    ):
        """
        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent_idx:      the parent node of the current iteration, initially ``None``.
        :param child_idx:   the index of the child in the current iteration.
        :param depth:       the depth of the tree in the current recursion level.
        """
        self.data = data
        self.start = start
        self.end = end
        self.node_idx = node_idx
        self.parent_idx = parent_idx
        self.child_idx = child_idx
        self.depth = depth
        self.min_samples_leaf = min_samples_leaf
        self.path = path

    @property
    def n_samples(self):
        return self.end - self.start


# ----------------------------------------------------------------------------------------------------------------------

def learn_prior(
        variable: Variable,
        column: int
):
    d = variable.distribution()._fit(
        data=_locals.data,
        col=column
    )
    return d.type_to_json(), d.to_json()


# ----------------------------------------------------------------------------------------------------------------------

def c45split(
        partition: JPTPartition,
        prune_or_split: Callable = None
) -> Tuple[Dict[str, Any], JPTPartition, Optional[JPTPartition], Optional[JPTPartition]]:
    """
    Creates a node in the decision tree according to the C4.5 algorithm
    """
    jpt = _locals.jpt
    start = partition.start
    end = partition.end
    depth = partition.depth
    node_idx = partition.node_idx
    parent_idx = partition.parent_idx
    child_idx = partition.child_idx
    data = _locals.data if hasattr(_locals, 'data') else partition.data
    path = partition.path
    indices = np.frombuffer(_locals.indices.get_obj(), dtype=np.int64)
    min_impurity_improvement = ifnone(jpt.min_impurity_improvement, 0)
    n_samples = end - start

    impurity = Impurity.from_tree(jpt)
    impurity.setup(data, indices)
    impurity.min_samples_leaf = partition.min_samples_leaf

    max_gain = impurity.compute_best_split(start, end)
    split_pos = impurity.best_split_pos
    split_var_idx = impurity.best_var
    split_var = jpt.variables[split_var_idx]

    jpt.logger.debug(
        'data range: %d-%d,' % (start, end),
        'split var:', split_var,
        ', split_pos:', split_pos,
        ', gain:', max_gain,
        ', path:', path
    )

    prune = (
        prune_or_split is not None
        and prune_or_split(
            jpt,
            data,
            indices,
            start,
            end,
            node_idx,
            parent_idx,
            child_idx,
            depth,
            list(path)
        )
    )

    logger.debug(
        f'prune_or_split() returned {prune}'
    )

    if (
            not prune
            and max_gain >= min_impurity_improvement
            and depth < jpt.max_depth
    ):
        # Create a decision node ---------------------------------------------------------------------------------------

        node = DecisionNode(
            idx=node_idx,
            variable=split_var,
            parent=None
        )
        node.samples = n_samples

        if split_var.symbolic:  # Symbolic domain ----------------------------------------------------------------------
            split_value = int(
                data[indices[start + split_pos], split_var_idx]
            )
            splits = [
                {split_value},
                set(split_var.domain.values) - {split_value}
            ]

        elif split_var.numeric:  # Numeric domain ----------------------------------------------------------------------
            split_value = (
                data[indices[start + split_pos], split_var_idx] +
                data[indices[start + split_pos + 1], split_var_idx]
            ) / 2
            splits = [
                ContinuousSet(np.NINF, split_value, EXC, EXC),
                ContinuousSet(split_value, np.PINF, INC, EXC)
            ]

        elif split_var.integer:  # Integer domain ----------------------------------------------------------------------
            split_value = (
                data[indices[start + split_pos], split_var_idx] +
                data[indices[start + split_pos + 1], split_var_idx]
            ) / 2
            splits = [
                IntSet(np.NINF, int(math.floor(split_value))),
                IntSet(int(math.floor(split_value)) + 1, np.PINF)
            ]

        else:  # -------------------------------------------------------------------------------------------------------
            raise TypeError(
                'Unknown variable type: %s.' % type(split_var).__name__
            )

        node.splits = [s.copy() for s in splits]

        # recurse left and right
        left = JPTPartition(
            data=partition.data,
            start=start,
            end=start + split_pos + 1,
            node_idx=None,
            parent_idx=node.idx,
            child_idx=0,
            path=path + [(split_var, splits[0].copy())],
            min_samples_leaf=partition.min_samples_leaf,
            depth=depth + 1
        )
        right = JPTPartition(
            data=partition.data,
            start=start + split_pos + 1,
            end=end,
            node_idx=None,
            parent_idx=node.idx,
            child_idx=1,
            path=path + [(split_var, splits[1].copy())],
            min_samples_leaf=partition.min_samples_leaf,
            depth=depth + 1
        )

    else:  # Create a leaf node ----------------------------------------------------------------------------------------
        leaf = node = Leaf(
            idx=node_idx,
            parent=None
        )

        for i, v in enumerate(jpt.variables):
            leaf.distributions[v] = v.distribution()._fit(
                data=data,
                rows=indices[start:end],
                col=i
            )
        leaf.prior = n_samples / data.shape[0]
        leaf.samples = n_samples

        if jpt._keep_samples:
            leaf.s_indices = indices[start:end]

        left = right = None

    node._path = list(path)
    logger.debug(
        'Created', str(node),
        'left=', ifnone(left, None, attrgetter('path')),
        'right=', ifnone(right, None, attrgetter('path'))
    )

    return node.to_json(), partition, left, right


# ----------------------------------------------------------------------------------------------------------------------

class C45Algorithm:

    def __init__(
            self,
            jpt: JPT
    ):
        self.jpt = jpt
        self.lock = None
        self.c45queue = None
        self.finish = None
        self._progressbar = None
        self._prune_or_split = None
        self.queue_length = 0
        self.indices = None
        self.min_samples_leaf = None
        self.node_counter = 0

    def _node_created(
            self,
            args: Tuple
    ) -> None:

        node: Node
        partition: JPTPartition
        left: JPTPartition
        right: JPTPartition
        node, partition, left, right = args

        with self.lock:
            # Re-instantiate the node object in the main process to
            # tie it to the original JPT object
            json_node = node  # .to_json()
            json_node['parent'] = partition.parent_idx
            json_node['child_idx'] = partition.child_idx

            if 'children' in json_node:
                node = DecisionNode.from_json(self.jpt, json_node)
            else:
                node = Leaf.from_json(self.jpt, json_node)

            self.node_counter += 1

            if isinstance(node, DecisionNode):
                left.parent_idx = node.idx
                left.node_idx = self.node_counter
                self.node_counter += 1
                right.parent_idx = node.idx
                right.node_idx = self.node_counter

                self.queue_length += 2
                self.c45queue.apply_async(
                    c45split,
                    args=(left, self._prune_or_split),
                    callback=self._node_created,
                    error_callback=_write_error
                )
                self.c45queue.apply_async(
                    c45split,
                    args=(right, self._prune_or_split),
                    callback=self._node_created,
                    error_callback=_write_error
                )
            else:
                if self._progressbar is not None:
                    self._progressbar.update(
                        partition.n_samples
                    )
            if self._progressbar is not None:
                self._progressbar.set_description(
                    'Learning [Leaves: %.4d, Nodes: %.4d, Queue: %.4d]' % (
                        len(self.jpt.leaves),
                        len(self.jpt.innernodes),
                        self.queue_length - 1,
                    )
                )

            if self.jpt.root is None and node.parent is None:
                self.jpt.root = node

            self.queue_length -= 1

            if not self.queue_length:
                self.finish.set()

    def learn(
            self,
            data: pd.DataFrame = None,
            # rows: Optional[Union[np.ndarray, List]] = None,
            # columns: Optional[Union[np.ndarray, List]] = None,
            keep_samples: bool = False,
            close_convex_gaps: bool = True,
            verbose: bool = False,
            prune_or_split: Optional[Callable] = None,
            multicore: Optional[int] = None
    ) -> None:
        """
        Fit the jpt to ``data``
        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        :param rows:    The training examples (assumed in row-shape)
        :type rows:     [[str or float or bool]]; (according to `self.variables`)
        :param columns: The training examples (assumed in column-shape)
        :type columns:  [[str or float or bool]]; (according to `self.variables`)
        :param keep_samples: If true, stores the indices of the original data samples in the leaf nodes. For debugging
                        purposes only. Default is false.
        :param close_convex_gaps:
        :param prune_or_split:
        :param multicore: The number of cores to use for learning. If ``None``, all cores available will be used.
        :param verbose:

        :return: the fitted model
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check and prepare the data
        _data = preprocess_data(
            self.jpt,
            data=data,
            verbose=verbose,
            multicore=multicore
        )

        logger.info('Initializing JPT learning...')
        self.jpt._reset()

        for idx, variable in enumerate(self.jpt.variables):
            if variable.numeric:
                samples = np.unique(_data.values[:, idx])
                distances = np.diff(samples)
                self.jpt.minimal_distances[variable] = min(distances) if len(distances) > 0 else 2.

        if not len(_data):
            raise ValueError('No data for learning.')

        # --------------------------------------------------------------------------------------------------------------
        # Initialize the internal data structures
        indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        indices[0] = 0
        np.cumsum(indices, out=indices)

        _locals.data = _data.values
        _locals.indices = Array(c.c_long, indices.shape[0])
        _locals.indices[:] = indices

        logger.info(
            f'Data transformation... {_data.shape[0]} x {_data.shape[1]}: '
            f'{_data.values.nbytes / 1e6:,.2f} MB'
        )

        # --------------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        started = dt.datetime.now()
        logger.info('Learning prior distributions...')

        if verbose:
            self._progressbar = tqdm(
                total=len(self.jpt.variables),
                desc='Learning prior distributions'
            )

        if not multicore:
            PoolCls = DummyPool
        else:
            PoolCls = Pool

        pool = PoolCls(
            multicore,
            initializer=_initialize_worker_process
        )
        for i, (dtype, dist) in enumerate(
                pool.starmap(
                    learn_prior,
                    iterable=[(v, i) for i, v in enumerate(self.jpt.variables)]
                )
        ):
            self.jpt.priors[self.jpt.variables[i]] = Distribution.from_json(dtype).from_json(dist)
            if verbose:
                self._progressbar.update(1)
        pool.close()
        pool.join()

        if verbose:
            self._progressbar.close()

        logger.info(
            '%d prior distributions learnt in %s.' % (
                len(self.jpt.priors),
                dt.datetime.now() - started
            )
        )

        # --------------------------------------------------------------------------------------------------------------
        # Start the training
        if type(self.jpt.min_samples_leaf) is int:
            self.min_samples_leaf = self.jpt.min_samples_leaf

        elif type(self.jpt.min_samples_leaf) is float and 0 < self.jpt.min_samples_leaf < 1:
            self.min_samples_leaf = max(1, int(self.jpt.min_samples_leaf * len(_data)))

        else:
            self.min_samples_leaf = self.jpt.min_samples_leaf

        self.keep_samples = keep_samples

        started = dt.datetime.now()
        logger.info(
            'Started learning of %s x %s at %s '
            'requiring at least %s samples per leaf' % (
                _data.shape[0],
                _data.shape[1],
                started,
                self.min_samples_leaf
            )
        )
        learning = GENERATIVE if (
                self.jpt.targets == self.jpt.variables or self.jpt.targets is None
        ) else DISCRIMINATIVE
        logger.info('Learning is %s. ' % learning)

        if learning == DISCRIMINATIVE:
            logger.info(
                'Target variables (%d): %s\n'
                'Feature variables (%d): %s' % (
                    len(self.jpt.targets),
                    ', '.join(mapstr(self.jpt.targets)),
                    len(self.jpt.variables) - len(self.jpt.targets),
                    ', '.join(
                        mapstr(set(self.jpt.variables) - set(self.jpt.targets)))
                )
            )

        _locals.jpt = self.jpt
        self._prune_or_split = prune_or_split
        self.c45queue = PoolCls(
            multicore,
            initializer=_initialize_worker_process
        )
        self.lock = RLock()
        self.finish = Event()
        if verbose:
            self._progressbar = tqdm(
                total=_data.shape[0],
                desc='Learning'
            )

        # build up tree
        with self.lock:
            self.queue_length += 1
            self.c45queue.apply_async(
                c45split,
                args=(
                    JPTPartition(
                        data=None,
                        start=0,
                        end=_data.shape[0],
                        node_idx=self.node_counter,
                        parent_idx=None,
                        child_idx=None,
                        path=[],
                        min_samples_leaf=self.min_samples_leaf,
                        depth=0
                    ),
                    prune_or_split
                ),
                callback=self._node_created,
                error_callback=_write_error
            )

        while 1:
            if self.finish.wait(timeout=10):
                break

        self.c45queue.close()
        self.c45queue.join()

        if verbose:
            self._progressbar.close()
            self._progressbar = None

        if close_convex_gaps:
            self.postprocess_leaves()

        # --------------------------------------------------------------------------------------------------------------
        # Print the statistics
        logger.info(
            'Learning took %s' % (dt.datetime.now() - started),
            repr(self.jpt)
        )

        # --------------------------------------------------------------------------------------------------------------
        # Clean up
        _locals.__dict__.clear()
        gc.collect()

    def postprocess_leaves(self) -> None:
        """
        Postprocess leaves such that the convex hull that is
        postulated from this tree has likelihood > 0 for every
        point inside the hull.
        """

        # get total number of samples and use 1/total as default value
        total_samples = self.jpt.total_samples()

        # for every leaf
        for idx, leaf in self.jpt.leaves.items():
            # for numeric every distribution
            for variable, distribution in leaf.distributions.items():
                if variable.numeric and variable in leaf.path.keys():  # and not distribution.is_dirac_impulse():

                    left = None
                    right = None

                    # if the leaf is not the "lowest" in this dimension
                    if np.NINF < leaf.path[variable].lower < distribution.cdf.intervals[0].upper:
                        # create uniform distribution as bridge between the leaves
                        left = ContinuousSet(
                            leaf.path[variable].lower,
                            distribution.cdf.intervals[0].upper,
                        )

                    # if the leaf is not the "highest" in this dimension
                    if np.PINF > leaf.path[variable].upper > distribution.cdf.intervals[-2].upper:
                        # create uniform distribution as bridge between the leaves
                        right = ContinuousSet(
                            distribution.cdf.intervals[-2].upper,
                            leaf.path[variable].upper,
                        )
                    if distribution.is_dirac_impulse():
                        # noinspection PyTypeChecker
                        interval = ContinuousSet(
                            left.lower if left is not None else distribution.cdf.intervals[0].upper,
                            right.upper if right is not None else distribution.cdf.intervals[1].lower,
                            INC,
                            EXC
                        )
                        # noinspection PyTypeChecker
                        distribution.set(
                            QuantileDistribution.from_pdf(
                                PiecewiseFunction.zero().overwrite({
                                    interval: 1 / interval.width
                                })
                            )
                        )
                    else:
                        distribution.insert_convex_fragments(
                            left,
                            right,
                            total_samples
                        )