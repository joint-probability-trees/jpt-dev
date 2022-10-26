import datetime
from collections import OrderedDict, ChainMap, deque
from typing import List, Dict, Tuple, Any

import dnutils

from itertools import zip_longest

import jpt.trees
from jpt.variables import VariableMap, Variable, NumericVariable, SymbolicVariable
import numpy as np
import pandas as pd

try:
    from .base.quantiles import __module__
    from .base.intervals import __module__
    from .learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet, RealSet
    from .learning.impurity import PCAImpurity


# ----------------------------------------------------------------------------------------------------------------------

class PCADecisionNode(jpt.trees.Node):
    """
    Represents an inner (decision) node of the the :class:`jpt.trees.Tree`.
    """

    def __init__(self, idx: int, variables: List[Variable], parent: 'jpt.trees.DecisionNode' = None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param variable:   the split feature name
        :type variable:    jpt.variables.Variable
        '''
        self._splits = None
        self.variables = variables
        super().__init__(idx, parent=parent)
        self.children: None or List[jpt.trees.Node] = None
        self.weights: None or np.ndarray = None

    def __eq__(self, o) -> bool:
        return (type(self) is type(o) and
                self.idx == o.idx and
                (self.parent.idx
                 if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
                [n.idx for n in self.children] == [n.idx for n in o.children] and
                self.splits == o.splits and
                self.variables == o.variables and
                self.samples == o.samples and
                self.weights == o.weights)

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'parent': self.parent.idx if self.parent else None,
                'splits': [s.to_json() if isinstance(s, ContinuousSet) else list(s) for s in self.splits],
                'variable': [variable.name for variable in self.variables],
                '_path': [(var.name, split.to_json() if var.numeric else list(split)) for var, split in self._path],
                'children': [node.idx for node in self.children],
                'samples': self.samples,
                'child_idx': self.parent.children.index(self) if self.parent is not None else None}

    @property
    def splits(self) -> List:
        return self._splits

    @splits.setter
    def splits(self, splits):
        if self.children is not None:
            raise ValueError('Children already set: %s' % self.children)
        self._splits = splits
        self.children = [None] * len(self._splits)

    def set_child(self, idx: int, node: jpt.trees.Node) -> None:
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.variable, self.splits[idx]))

    def str_edge(self, idx) -> str:
        return str(ContinuousSet(self.variable.domain.labels[self.splits[idx].lower],
                                 self.variable.domain.labels[self.splits[idx].upper],
                                 self.splits[idx].left,
                                 self.splits[idx].right))

    @property
    def str_node(self) -> str:
        return self.variable.name

    def recursive_children(self):
        return self.children + [item for sublist in
                                [child.recursive_children() for child in self.children] for item in sublist]

    def __str__(self) -> str:
        return (f'<DecisionNode #{self.idx} '
                f'{self.variable.name} = [%s]' % '; '.join(self.str_edge(i) for i in range(len(self.splits))) +
                f'; parent-#: {self.parent.idx if self.parent is not None else None}'
                f'; #children: {len(self.children)}>')

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------

class PCALeaf(jpt.trees.Leaf):
    def __init__(self, idx: int, parent: PCANode or None = None, prior: float or None = None):
        super(PCALeaf, self).__init__(idx, parent)

        # variable map with distributions
        self.distributions = VariableMap()

        # pca matrix of this leaf
        self.rotation_matrix: np.array or None = None

        # prior probability of this leaf
        self.prior = prior


# ----------------------------------------------------------------------------------------------------------------------

class PCAJPT(jpt.trees.JPT):
    """
    This class represents an extension to JPTs where the PCA is applied before each split in training
    and Leafs therefore obtain an additional rotation matrix.
    """

    logger = dnutils.getlogger('/pcajpt', level=dnutils.INFO)

    def __init__(self, variables, targets=None, min_samples_leaf=.01, min_impurity_improvement=None,
                 max_leaves=None, max_depth=None, variable_dependencies=None) -> None:
        """
        Create a PCAJPT

        :param variables: the List of all variables
        :param targets: The list of targets to measure the impurity on. Must be a subset of ``variables``
        :param min_samples_leaf: The percentage of samples that are needed to form a leaf
        :param min_impurity_improvement: The minimal amount of information gain needed to accept a split
        :param max_depth: The maximum depth of the tree.
        """
        super(PCAJPT, self).__init__(variables, targets, min_samples_leaf, min_impurity_improvement, max_leaves,
                                     max_depth, variable_dependencies)

        # dont use targets yet, it is unsure what the correct design for that would be
        if self.targets is not None:
            raise ValueError("Targets are not yet allowed for PCA trees.")

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
            if not all(c == v for c, v in zip_longest(data.columns, self.varnames)):
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
        self.impurity = PCAImpurity(self)
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
            raise ValueError('Something went wrong! max_gain should at least be 0 but was %s' % max_gain)

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

            print(split_var)
            node = PCADecisionNode(idx=len(self.allnodes),
                                variable=split_var,
                                parent=parent)

            # copy number of samples
            node.samples = n_samples
            self.innernodes[node.idx] = node


            # create symbolic decision node
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