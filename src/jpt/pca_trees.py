import datetime
from collections import OrderedDict, ChainMap, deque
from typing import List, Dict, Tuple

import dnutils

import jpt.trees
from jpt.variables import VariableMap, Variable
import numpy as np
import pandas as pd


class PCANode(jpt.trees.Node):
    """
    Superclass for Nodes in a PCAJPT
    """

    def __init__(self, idx, parent):
        """
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        """
        super(PCANode, self).__init__(idx, parent)


# ----------------------------------------------------------------------------------------------------------------------

class PCADecisionNode(PCANode):
    """
    Decision Node in a PCAJPT where linear combinations of all variables are considered instead of just one variable
    """

    def __init__(self, idx: int, parent: PCANode, variables: List[Variable], weights: np.array = None):
        super(PCADecisionNode, self).__init__(idx, parent)

        # this is a list of all variables used for the split.
        self.variables: List[Variable] = variables

        # weights are the factors of the linear combination of numeric variables
        self.weights = weights or np.ones((len([variable for variable in self.variables if variable.numeric])))

        self._splits = None

        self.children: List[PCANode] or None = None


# ----------------------------------------------------------------------------------------------------------------------

class PCALeaf(PCANode):
    def __init__(self, idx: int, parent: PCANode or None = None, prior: float or None = None):
        super(PCALeaf, self).__init__(idx, parent)

        # variable map with distributions
        self.distributions = VariableMap()

        # pca matrix of this leaf
        self.rotation_matrix: np.array or None = None

        # prior probability of this leaf
        self.prior = prior


# ----------------------------------------------------------------------------------------------------------------------

class PCAJPT:
    """
    This class represents an extension to JPTs where the PCA is applied before each split in training
    and Leafs therefore obtain an additional rotation matrix.
    """

    logger = dnutils.getlogger('/pcajpt', level=dnutils.INFO)

    def __init__(self,
                 variables: List[Variable],
                 targets: List[Variable] or None = None,
                 min_samples_leaf: float = .01,
                 min_impurity_improvement: float or None = None,
                 max_depth: int or float = float("infinity")) -> None:
        """
        Create a PCAJPT

        :param variables: the List of all variables
        :param targets: The list of targets to measure the impurity on. Must be a subset of ``variables``
        :param min_samples_leaf: The percentage of samples that are needed to form a leaf
        :param min_impurity_improvement: The minimal amount of information gain needed to accept a split
        :param max_depth: The maximum depth of the tree.
        """

        self._variables = tuple(variables)
        self._targets = targets
        self.max_depth = max_depth

        self.varnames: OrderedDict[str, Variable] = OrderedDict((var.name, var) for var in self._variables)
        self._targets = [self.varnames[v] if type(v) is str else v for v in targets] if targets is not None else None
        self.leaves: Dict[int, PCALeaf] = {}
        self.innernodes: Dict[int, PCADecisionNode] = {}
        self.allnodes: ChainMap[int, PCANode] = ChainMap(self.innernodes, self.leaves)
        self.priors = {}

        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement or None

        # a map saving the minimal distances to prevent infinite high likelihoods
        self.minimal_distances: VariableMap = VariableMap({var: 1. for var in self.numeric_variables}.items())

        self._numsamples = 0
        self.root: PCADecisionNode or None = None
        self.c45queue: deque = deque()
        self._node_counter = 0
        self.indices = None
        self.impurity = None

    @property
    def variables(self) -> Tuple[Variable]:
        return self._variables

    @property
    def targets(self) -> List[Variable]:
        return self._targets

    def _reset(self) -> None:
        self.innernodes.clear()
        self.leaves.clear()
        self.priors.clear()
        self.root = None
        self.c45queue.clear()

    def _preprocess_data(self, data: np.ndarray or pd.DataFrame) -> np.ndarray:
        """
        Transform the input data into an internal representation.
        :param data: the raw data
        :return: the preprocessed data as numpy array
        """

        PCAJPT.logger.info('Preprocessing data...')

        data_ = np.ndarray(shape=data.shape, dtype=np.float64, order='C')
        if isinstance(data, pd.DataFrame):
            if set(self.varnames).symmetric_difference(set(data.columns)):
                raise ValueError('Unknown variable names: %s'
                                 % ', '.join(
                                            dnutils.mapstr(set(self.varnames).symmetric_difference(set(data.columns)))))

            # Check if the order of columns in the data frame is the same
            # as the order of the variables.
            if not all(c == v for c, v in dnutils.zip_longest(data.columns, self.varnames)):
                raise ValueError('Columns in DataFrame must coincide with variable order: %s' %
                                 ', '.join(dnutils.mapstr(self.varnames)))
            transformations = {v: self.varnames[v].domain.values.transformer() for v in data.columns}
            try:
                data_[:] = data.transform(transformations).values
            except ValueError:
                dnutils.err(transformations)
                raise
        else:
            for i, (var, col) in enumerate(zip(self.variables, data.T)):
                data_[:, i] = [var.domain.values[v] for v in col]
        return data_

    def learn(self, data: np.array or pd.DataFrame) -> 'PCAJPT':
        """
        Fits the ``data`` into a regression tree.

        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        """
        # ----------------------------------------------------------------------------------------------------------
        # Check and prepare the data
        _data = self._preprocess_data(data=data)

        for idx, variable in enumerate(self.variables):
            if variable.numeric:
                samples = np.unique(_data[:, idx])
                distances = np.diff(samples)
                self.minimal_distances[variable] = min(distances) if len(distances) > 0 else 2.

        if _data.shape[0] < 1:
            raise ValueError('No data for learning.')

        self.indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        self.indices[0] = 0
        np.cumsum(self.indices, out=self.indices)

        PCAJPT.logger.info('Data transformation... %d x %d' % _data.shape)

        # ----------------------------------------------------------------------------------------------------------
        # Initialize the internal data structures
        self._reset()

        # ----------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        started = datetime.datetime.now()
        PCAJPT.logger.info('Learning prior distributions...')
        self.priors = {}
        for i, (vname, var) in enumerate(self.varnames.items()):
            self.priors[vname] = var.distribution().fit(data=_data,
                                                        col=i)
        PCAJPT.logger.info('Prior distributions learnt in %s.' % (datetime.datetime.now() - started))

        # ----------------------------------------------------------------------------------------------------------
        # Start the training

        if type(self._min_samples_leaf) is int:
            min_samples_leaf = self._min_samples_leaf
        elif type(self._min_samples_leaf) is float and 0 < self._min_samples_leaf < 1:
            min_samples_leaf = max(1, int(self._min_samples_leaf * len(_data)))
        else:
            min_samples_leaf = self._min_samples_leaf

        # Initialize the impurity calculation
        self.impurity = Impurity(self)
        self.impurity.setup(_data, self.indices)
        self.impurity.min_samples_leaf = min_samples_leaf

        started = datetime.datetime.now()
        PCAJPT.logger.info('Started learning of %s x %s at %s '
                        'requiring at least %s samples per leaf' % (_data.shape[0],
                                                                    _data.shape[1],
                                                                    started,
                                                                    min_samples_leaf))
        learning = jpt.trees.GENERATIVE if self.targets is None else jpt.trees.DISCRIMINATIVE
        PCAJPT.logger.info('Learning is %s. ' % learning)
        if learning == jpt.trees.DISCRIMINATIVE:
            PCAJPT.logger.info('Target variables (%d): %s\n'
                            'Feature variables (%d): %s' % (len(self.targets),
                                                            ', '.join(dnutils.mapstr(self.targets)),
                                                            len(self.variables) - len(self.targets),
                                                            ', '.join(
                                                                dnutils.mapstr(
                                                                    set(self.variables) - set(self.targets)))))
        # build up tree
        self.c45queue.append((_data, 0, _data.shape[0], None, None, 0))
        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        # ----------------------------------------------------------------------------------------------------------
        # Print the statistics

        PCAJPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
        # if logger.level >= 20:
        PCAJPT.logger.debug(self)
        return self

    fit = learn

    def c45(self, data, start, end, parent, child_idx, depth) -> None:
        """
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent:      the parent node of the current iteration, initially ``None``.
        :param child_idx:   the index of the child in the current iteration.
        :param depth:       the depth of the tree in the current recursion level.
        """
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = self.min_impurity_improvement or 0
        n_samples = end - start
        split_var_idx = split_pos = -1
        split_var = None
        impurity = self.impurity

        max_gain = impurity.compute_best_split(start, end)

        if max_gain < 0:
            raise ValueError('Something went wrong!')

        self.logger.debug('Data range: %d-%d,' % (start, end),
                          'split var:', split_var,
                          ', split_pos:', split_pos,
                          ', gain:', max_gain)

        if max_gain:
            split_pos = impurity.best_split_pos
            split_var_idx = impurity.best_var
            split_var = self.variables[split_var_idx]

        if max_gain <= min_impurity_improvement or depth >= self.max_depth:  # -----------------------------------------
            leaf = node = PCALeaf(idx=len(self.allnodes), parent=parent)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.distribution().fit(data=data,
                                                             rows=self.indices[start:end],
                                                             col=i)
            leaf.prior = n_samples / data.shape[0]
            leaf.samples = n_samples

            self.leaves[leaf.idx] = leaf

        # TODO creating PCADecisionNode is complex now and not as easy
        else:  # -------------------------------------------------------------------------------------------------------
            node = PCADecisionNode(idx=len(self.allnodes),
                                variable=split_var,
                                parent=parent)
            node.samples = n_samples
            self.innernodes[node.idx] = node


            if split_var.symbolic:  # ----------------------------------------------------------------------------------
                split_value = int(data[self.indices[start + split_pos], split_var_idx])
                splits = [{split_value},
                          set(split_var.domain.values.values()) - {split_value}]

            elif split_var.numeric:  # ---------------------------------------------------------------------------------

                split_value = (data[self.indices[start + split_pos], split_var_idx] +
                               data[self.indices[start + split_pos + 1], split_var_idx]) / 2
                splits = [Interval(np.NINF, split_value, EXC, EXC),
                          Interval(split_value, np.PINF, INC, EXC)]

            else:  # ---------------------------------------------------------------------------------------------------
                raise TypeError('Unknown variable type: %s.' % type(split_var).__name__)

            self.c45queue.append((data, start, start + split_pos + 1, node, 0, depth + 1))
            self.c45queue.append((data, start + split_pos + 1, end, node, 1, depth + 1))

            node.splits = splits

        PCAJPT.logger.debug('Created', str(node))

        if parent is not None:
            parent.set_child(child_idx, node)

        if self.root is None:
            self.root = node