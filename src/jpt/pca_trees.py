import datetime
import numbers
from collections import OrderedDict, ChainMap, deque, defaultdict
from typing import List, Dict, Tuple, Any, Union

import dnutils
import os
import math
import html
import matplotlib.pyplot as plt
from itertools import zip_longest
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import jpt.trees
from jpt.variables import VariableMap, Variable, NumericVariable, SymbolicVariable
import numpy as np
import pandas as pd
from graphviz import Digraph
from jpt.distributions.univariate import Numeric

try:
    from .distributions.quantile.quantiles import __module__
    from .base.intervals import __module__
    from .learning.impurity import __module__
    from .base.functions import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet, RealSet
    from .learning.impurity import PCAImpurity, Impurity
    from .base.constants import plotstyle, orange, green, SYMBOL
    from .base.functions import LinearFunction, PiecewiseFunction
    from .base.errors import Unsatisfiability
    from .distributions.quantile.quantiles import QuantileDistribution
    from .base.utils import format_path, normalized


# ----------------------------------------------------------------------------------------------------------------------

class PCADecisionNode(jpt.trees.Node):
    """
    Represents an inner (decision) node of the the :class:`jpt.trees.Tree`.
    """

    def __init__(self, idx: int,
                 variables: List[jpt.variables.Variable],
                 weights: np.ndarray,
                 split_value: float,
                 numeric_indices: np.ndarray,
                 parent: jpt.trees.DecisionNode = None,):
        """
        Create a PCA Decision Node

        :param idx:             the identifier of a node
        :param variables:       the numeric variables involved
        """
        super().__init__(idx, parent=parent)
        # set variable coefficient map
        self.variables: VariableMap = jpt.variables.VariableMap(zip(variables, weights))

        # set splits
        self._splits = [ContinuousSet(np.NINF, split_value, 0, 1),
                        ContinuousSet(split_value, np.PINF, 0, 1)]

        # initialize children
        self.children: List[jpt.trees.Node or None] = [None, None]
        self._path = []
        self.numeric_indices = numeric_indices

    def __eq__(self, o) -> bool:
        return (type(self) is type(o) and
                self.idx == o.idx and
                (self.parent.idx
                 if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
                [n.idx for n in self.children] == [n.idx for n in o.children] and
                self.splits == o.splits and
                self.variables == o.variables and
                self.samples == o.samples)

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'parent': self.parent.idx if self.parent else None,
                'splits': [s.to_json() if isinstance(s, ContinuousSet) else list(s) for s in self.splits],
                'variables': self.variables.to_json(),
                '_path': [(var.name, split.to_json() if var.numeric else list(split)) for var, split in self._path],
                'children': [node.idx for node in self.children],
                'samples': self.samples,
                'child_idx': self.parent.children.index(self) if self.parent is not None else None}

    @property
    def splits(self) -> List:
        return self._splits

    def set_child(self, idx: int, node: jpt.trees.Node) -> None:
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.variables, self.splits[idx]))

    @property
    def str_node(self) -> str:
        result = " + ".join(["%s * %s" % (value, variable.name) for variable, value in self.variables.items()])
        return result

    def recursive_children(self):
        return self.children + [item for sublist in
                                [child.recursive_children() for child in self.children] for item in sublist]

    def __str__(self) -> str:
        additional_format = "+\n      |"
        return "<PCADecisionNode #%s> \n" \
               "Criterion:\n      | %s \n" \
               "parent-#%s\n" \
               "#children: %s" % (self.idx, self.string_criterion().replace('+', additional_format),
                               None if self.parent is None else self.parent.idx, len([c for c in self.children if c]))

    def string_criterion(self) -> str:
        result = " + ".join(["%s * %s" % (value, variable.name) for variable, value in self.variables.items()])
        result += " <= %s" % self._splits[0].upper
        return result

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'

    def str_edge(self, idx) -> str:
        return str(self.splits[idx])

    @property
    def path(self) -> VariableMap:
        res = VariableMap()
        for var, vals in self._path:
            if isinstance(var, VariableMap):
                pass
            elif isinstance(var, Variable):
                res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res


# ----------------------------------------------------------------------------------------------------------------------

class PCALeaf(jpt.trees.Node):
    """
    Represent a Leaf in a PCAJPT.
    A leaf in a PCAJPT consists of distributions for every variable. Furthermore, the numeric variables are represented
    as linear dependent. They are represented as independent in "eigen" coordinates. Therefore, for inference, one has
    to transform queries into the eigen space and answer them there. Be aware that the "eigen" coordinate system is
    constructed from the standardized "data" coordinate system, hence a StandardScaler also exists.
    """
    def __init__(self, idx: int,
                 prior: float,
                 scaler: StandardScaler,
                 decomposer: PCA,
                 numeric_indices: List[int],
                 parent: jpt.trees.Node):

        super(PCALeaf, self).__init__(idx, parent)

        # standard scaler of this leaf
        self.scaler = scaler

        # pca of this leaf
        self.decomposer: PCA = decomposer

        # prior probability of this leaf
        self.prior = prior

        # variable map with distributions
        self.distributions = VariableMap()

        self._path = []

        self.numeric_indices = numeric_indices

        self.numeric_domains_: VariableMap or None = None

    @property
    def numeric_domains(self) -> VariableMap:
        """
        Get the minimum and maximum values of the numeric variables of this leaf in the "data" coordinates.

        :return: VariableMap that maps every numeric variable to a ContinuousSet
        """

        # if it already has been calculated return it
        if self.numeric_domains_ is not None:
            return self.numeric_domains_

        # initialize matrix to hold ranges in "eigen" coordinates
        ranges = np.ndarray((2, len(self.numeric_indices)))

        # for every numeric variable and their distribution and their index
        for idx, (variable, distribution) in enumerate([(variable, distribution) for variable, distribution
                                                        in self.distributions.items() if variable.numeric]):
            domain = distribution.domain()
            ranges[:, idx] = [domain.lower, domain.upper]

        ranges = self.inverse_transform(ranges)

        # initialize result
        result = dict()

        # rewrite the transformed ranges to the resulting map
        for idx, range_ in enumerate(ranges.T):

            # get the corresponding numeric variable
            variable = list(self.distributions.keys())[self.numeric_indices[idx]]

            # use min/max here since the transformation can invert axis without semantics
            result[variable] = ContinuousSet(min(range_), max(range_))

        # construct VariableMap
        self.numeric_domains_ = VariableMap(result.items())
        return self.numeric_domains_

    @property
    def str_node(self) -> str:
        return ""

    @property
    def value(self):
        return self.distributions

    @property
    def path(self) -> dict:
        res = VariableMap()
        for var, vals in self._path:
            if isinstance(var, VariableMap):
                # TODO find suitable representation
                pass
            elif isinstance(var, Variable):
                res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res

    def numeric_variables(self):
        return [variable for variable in self.distributions.keys() if variable.numeric]

    def transform_variable_map(self, query: VariableMap) -> VariableMap:
        """
        Transform all numeric values inside ``query`` to the internal representation of this leaf.
        For every numeric variable that is not in query the min/nax domains are entered.
        :param query: the query to transform
        :return: the transformed VariableMap
        """

        # initialize result
        result = dict()

        # initialize ranges where the first row represents the lower bound in eigen space and the second one the upper
        ranges = np.ndarray((2, len(self.numeric_indices)))

        # for every numeric variable and their index
        for idx, variable in enumerate(self.distributions.keys()):
            result[variable] = query.get(variable)

            # if it is numeric
            if variable.numeric:

                # get the value range for the transformation
                restriction = query[variable] if variable in query.keys() else self.numeric_domains[variable]

                # write to transformation matrix
                ranges[:, self.numeric_indices.index(idx)] = np.array([restriction.lower, restriction.upper])

        # transform to "eigen" space
        ranges = self.transform(ranges)

        # rewrite the transformed ranges to the resulting map
        for idx, range_ in enumerate(ranges.T):

            # get the corresponding numeric variable
            variable = list(self.distributions.keys())[self.numeric_indices[idx]]

            # use min/max here since the transformation can invert axis without semantics
            result[variable] = ContinuousSet(min(range_), max(range_))

        return VariableMap(result.items())

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Forward transform the data such that it can be used for querying.

        :param data: The data to transform
        :return: The transformed data
        """
        return self.decomposer.transform(self.scaler.transform(data))

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Backward transform the data such that it can be used for external representation.

        :param data: The data to inverse transform
        :return: The inverse transformed data
        """
        return self.scaler.inverse_transform(self.decomposer.inverse_transform(data))

    def posterior(self, variables: List[Variable] or None = None, evidence: VariableMap = VariableMap()) -> VariableMap:
        """
        Return the independent distributions in "data" space.
        :param variables: the variables to calculate the posterior over
        :param evidence: the preprocessed evidence in "data" space
        :return: A VariableMap assigning each variable to their distribution
        """

        # initialize variables
        variables = variables or list(self.distributions.keys())

        # transform query to eigen space
        evidence_ = self.transform_variable_map(evidence.copy())

        result = dict()

        # for every variable and its distribution
        for idx, (variable, distribution) in enumerate(self.distributions.items()):

            if variable not in variables:
                continue

            # just copy symbolic variables since they are not distorted by the PCA
            if variable.symbolic:
                result[variable] = self.distributions[variable].crop(evidence_[variable])

            # if the variable is numeric (it gets complicated)
            if variable.numeric:

                distribution = distribution.crop(evidence_[variable])

                # get the index for the transformation via pca
                numeric_index = self.numeric_indices.index(idx)

                # get all points that need to be inverse transformed in "eigen" coordinates
                points_on_axis = [interval.upper for interval in distribution.cdf.intervals[:-1]]

                # construct the whole matrix in "eigen" coordinates
                points = np.zeros((len(points_on_axis), len(self.numeric_indices)))
                points[:, numeric_index] = points_on_axis

                # get the axis of the variable in "data" coordinates
                points_on_data_axis = np.concatenate((np.sort(self.inverse_transform(points)[:, numeric_index]),
                                                     [np.PINF]))

                # list that holds the functions for the posterior distribution
                posterior_functions = [LinearFunction(0, 0)]
                posterior_intervals = [ContinuousSet(np.NINF, points_on_data_axis[0], 2, 2)]

                # for every idx, interval and function in eigen coordinates
                for function_idx, (interval, function) in enumerate(list(distribution.cdf.iter())[1:-1]):

                    # construct next interval in "data" coordinates
                    posterior_interval = ContinuousSet(points_on_data_axis[function_idx],
                                                       points_on_data_axis[function_idx + 1])

                    # construct the function in "data" coordinates
                    previous_prob = posterior_functions[-1].eval(posterior_intervals[-1].upper)
                    next_prob = function.eval(interval.upper)

                    # apply slope formula
                    m = (next_prob - previous_prob) / posterior_interval.range()

                    # calculate intersection with the y-axis as c = f(x) - mx
                    c = previous_prob - (m * posterior_interval.lower)

                    # append new function and intervals
                    posterior_functions.append(LinearFunction(m, c))
                    posterior_intervals.append(posterior_interval)

                # construct posterior piecewise function
                posterior_piecewise_function = PiecewiseFunction()
                posterior_piecewise_function.functions = posterior_functions
                posterior_piecewise_function.intervals = posterior_intervals

                # convert to distribution
                resulting_distribution = Numeric()
                resulting_distribution.set(QuantileDistribution.from_cdf(posterior_piecewise_function))
                result[variable] = resulting_distribution

        return VariableMap(result.items())

    def copy(self):
        """
        :return: the copied PCALeaf
        """
        result = PCALeaf(self.idx, self.prior, self.scaler, self.decomposer, self.numeric_indices, self.parent)
        result.distributions = self.distributions.copy()
        result._path = self._path
        result.samples = self.samples
        return result

    def conditional_leaf(self, evidence: VariableMap):
        """
        Compute the conditional probability distribution of the leaf.
        :param evidence: the evidence to apply
        :return: A copy of this leaf that is consistent with the evidence
        """
        result = self.copy()

        evidence_ = self.transform_variable_map(evidence)

        # for every distribution
        for variable, restriction in evidence_.items():

            # apply symbolic evidence
            if variable.symbolic:
                result.distributions[variable] = result.distributions[variable].crop(evidence_[variable])

            # apply numeric evidence
            if variable.numeric:
                result.distributions[variable] = result.distributions[variable].crop(evidence_[variable])

        return result

    def mpe(self, evidence: VariableMap, minimal_distances: VariableMap):
        """
        Calculate the most probable explanation of the linear dependent distributions.

        This has not yet the full functionality. TODO check if the maxima can be inverse transformed independently

        :return: the likelihood of the maximum as a float and the configuration in ``data`` coordinates as a VariableMap
        """

        # apply conditions
        conditional_leaf = self.conditional_leaf(evidence)

        # initialize likelihood and maximum
        result_likelihood = conditional_leaf.prior

        # initialize maximum
        maximum = dict()

        # initialize explanation in eigen coordinates
        eigen_explanation = np.ndarray((2, len(self.numeric_indices)))

        # for every variable and distribution
        for idx, (variable, distribution) in enumerate(conditional_leaf.distributions.items()):

            # calculate mpe of that distribution
            likelihood, explanation = distribution.mpe()

            # apply upper cap for infinities
            likelihood = minimal_distances[variable] if likelihood == float("inf") else likelihood

            # update likelihood
            result_likelihood *= likelihood

            # for symbolic variables
            if variable.symbolic:

                # save result
                maximum[variable] = explanation

            # for numeric variables
            elif variable.numeric:

                # get th index in eigen_explanation
                eigen_index = self.numeric_indices.index(idx)

                # write data to inverse transform
                eigen_explanation[:, eigen_index] = [explanation.intervals[0].lower, explanation.intervals[0].upper]

        # transform eigen_explanation back to data coordinates
        data_explanation = self.inverse_transform(eigen_explanation)

        # write explanation in data coordinates to result
        for variable, data_explanation_ in zip(self.numeric_variables(), data_explanation.T):
            maximum[variable] = ContinuousSet(min(data_explanation_), max(data_explanation_))

        # create mpe result
        return result_likelihood, maximum

    def probability(self, query: VariableMap, dirac_scaling: float = 2., min_distances: VariableMap = None) -> float:
        """
        Calculate the probability of a (partial) query. Exploits the independence assumptions in eigen coordinates
        :param query: A preprocessed VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type query: VariableMap
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a dirac impulse is used to model the variable.
        :type dirac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: A VariableMap from numeric variables to floats or None
        """

        # initialize result
        result = 1.

        # transform query to eigen space
        query_ = self.transform_variable_map(query)

        # for every variable and its value
        for variable, value in query_.items():

            # if it is numeric
            if variable.numeric:

                # if it is a single value
                if value.lower == value.upper:

                    # get the likelihood
                    likelihood = self.distributions[variable].pdf(value.upper)

                    # if it is infinity and no handling is provided replace it with 1.
                    if likelihood == float("inf") and not min_distances:
                        result *= 1

                    # if it is infinite and a handling is provided, replace with dirac_scaling/min_distance
                    elif likelihood == float("inf") and min_distances:
                        min_distance = min_distances[variable]
                        min_distance = 1 if min_distance == 0 else min_distance
                        result *= dirac_scaling / min_distance

                    # if the likelihood is finite just multiply it
                    else:
                        result *= likelihood

                # handle ordinary probability queries
                else:
                    result *= self.distributions[variable]._p(value)

            # handle symbolic variable
            if variable.symbolic:

                # force the evidence to be a set
                if not isinstance(value, set):
                    value = set([value])

                # return false if the evidence is impossible in this leaf
                result *= self.distributions[variable]._p(value)

        return result

    def parallel_likelihood(self, queries: np.ndarray, dirac_scaling: float = 2.,  min_distances: VariableMap = None) \
            -> np.ndarray:
        """
        Calculate the probability of a (partial) query. Exploits the independence assumption
        :param queries: A VariableMap that maps to singular values (numeric or symbolic)
            or ranges (continuous set, set)
        :type queries: VariableMap
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :type dirac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: A VariableMap from numeric variables to floats or None
        """

        # create result vector
        result = np.ones(len(queries))

        queries_ = queries.copy()
        queries_[:, self.numeric_indices] = self.transform(queries_[:, self.numeric_indices])

        # for each idx, variable and distribution
        for idx, (variable, distribution) in enumerate(self.distributions.items()):

            # if the variable is symbolic
            if isinstance(variable, SymbolicVariable):

                # multiply by probability
                probs = distribution._params[queries_[:, idx].astype(int)]

            # if the variable is numeric
            elif isinstance(variable, NumericVariable):

                # get the likelihoods
                probs = np.asarray(distribution.pdf.multi_eval(queries_[:, idx].copy(order='C').astype(float)))

                if min_distances:

                    # check if the minimal distance is 0 and replace it with one if so
                    min_distance = min_distances[variable]
                    min_distance = 1 if min_distance == 0 else min_distance

                    # replace them with dirac scaling if they are infinite
                    probs[(probs == float("inf")).nonzero()] = dirac_scaling / min_distance

                # if no distances are provided replace infinite values with 1.
                else:
                    probs[(probs == float("inf")).nonzero()] = 1.

            # multiply results
            result *= probs

        return result

    def expectation(self, evidence=VariableMap()) -> VariableMap:
        """
        Calculate the expectation of numeric variables and mpe of symbolic variables
        :param evidence: the preprocessed evidence to apply before calculating the expectation
        :return: the expectation as VariableMap
        """
        # initialize result
        result = dict()

        # if no evidence is provided the calculation can be shortcut without transformation
        if len(evidence) == 0:

            # for every variable
            for idx, (variable, distribution) in enumerate(self.distributions.items()):

                # if it is symbolic, write the expectation directly
                if variable.symbolic:
                    result[variable] = distribution.expectation()

                # if it is numeric, get the mean from the scaler (avoids matrix multiplication)
                else:
                    result[variable] = self.scaler.mean_[self.numeric_indices[idx]]

            return VariableMap(result.items())

        transformed_evidence = self.transform_variable_map(evidence)

        for variable, distribution in self.distributions.items():
            if variable in evidence.items():
                distribution = distribution.crop(transformed_evidence[variable])
                result[variable] = distribution.expectation()

        return VariableMap(result)

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'distributions': self.distributions.to_json(),
                'prior': self.prior,
                'samples': self.samples,
                'parent': self.parent.idx if self.parent else None,
                'child_idx': self.parent.children.index(self) if self.parent is not None else -1,
                "mean": self.scaler.mean_,
                "var": self.scaler.var_,
                "eigenvectors": self.decomposer.components_,
                "numeric_indices": json.dumps(self.numeric_indices)}


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

        # get numeric indices for pca transformations
        self.numeric_indices = [idx for idx, variable in enumerate(self.variables) if variable.numeric]

        if len(self.numeric_indices) <= 1:
            raise ValueError("PCAJPT does not work for 1 or less numeric variables.")

        # don't use targets yet, it is unsure what the correct design for that would be
        if self.targets is not None:
            raise ValueError("Targets are not yet allowed for PCA trees.")

        # for syntax highlighting, update the types of the structures
        self.leaves: Dict[int, PCALeaf] = {}
        self.innernodes: Dict[int, PCADecisionNode] = {}
        self.allnodes: ChainMap[int, jpt.trees.Node] = ChainMap(self.innernodes, self.leaves)


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
        # build up tree)
        self.c45queue.append((_data, 0, _data.shape[0], None, None, 0))
        while self.c45queue:
            params = self.c45queue.popleft()
            self.c45(*params)

        # ----------------------------------------------------------------------------------------------------------
        # Print the statistics
        PCAJPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
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
        original_data = data.copy()

        # --------------------------------------------------------------------------------------------------------------

        # get relevant data
        pca_data = data[np.ix_(self.indices[start:end], self.numeric_indices)]

        # create scaler
        scaler: StandardScaler = StandardScaler()

        # transform data
        pca_data = scaler.fit_transform(pca_data)

        # create a full decomposer
        decomposer = PCA(len(self.numeric_indices))

        # calculate transforms and transform the data
        pca_data = decomposer.fit_transform(pca_data)

        # rewrite data
        data[np.ix_(self.indices[start:end], self.numeric_indices)] = pca_data

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

            leaf = node = PCALeaf(idx=len(self.allnodes),
                                  parent=parent,
                                  prior=n_samples / data.shape[0],
                                  scaler=scaler,
                                  decomposer=decomposer,
                                  numeric_indices=self.numeric_indices)

            # fit distributions
            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.distribution().fit(data=data,
                                                             rows=self.indices[start:end],
                                                             col=i)
            leaf.samples = n_samples

            self.leaves[leaf.idx] = leaf

        else:  # -------------------------------------------------------------------------------------------------------
            # create symbolic decision node
            if split_var.symbolic:  # ----------------------------------------------------------------------------------

                node = jpt.trees.DecisionNode(idx=len(self.allnodes),
                                              variable=split_var,
                                              parent=parent)

                # create symbolic decision
                split_value = int(data[self.indices[start + split_pos], split_var_idx])
                node.splits = [{split_value}, set(split_var.domain.values.values()) - {split_value}]

            elif split_var.numeric:  # ---------------------------------------------------------------------------------

                # load number of dimensions
                n = len(self.numeric_variables)

                # get the index of the splitting dimension if reduced to only numeric variables
                split_dimension = [index for index in self.numeric_indices if index == split_var_idx][0]

                # get the rotation matrix the rotates from the eigen axes to the standardized axes
                standardized_rotation_eigen = decomposer.components_.T

                # create transformation matrix from eigen coordinates to standardized coordinates
                standardized_transformation_eigen = np.identity(n + 1)
                standardized_transformation_eigen[:-1, :-1] = standardized_rotation_eigen

                # get the axis aligned split value in eigen coordinates
                split_value = (data[self.indices[start + split_pos], split_var_idx] +
                               data[self.indices[start + split_pos + 1], split_var_idx]) / 2

                # create the transformation matrix that translates from the split origin to the eigen axes
                eigen_transformation_split = np.identity(n+1)
                eigen_transformation_split[split_dimension, -1] = split_value

                # calculate transformation matrix from split space to standardized space
                standardized_transformation_split = np.dot(standardized_transformation_eigen,
                                                           eigen_transformation_split)

                # create the normal vector of the splitting plane in split coordinates
                split_normal_of_split = np.zeros(n+1)
                split_normal_of_split[split_dimension] = 1

                # create origin of the split coordinate system as point
                split_origin_of_split = np.zeros(n+1)
                split_origin_of_split[-1] = 1

                # calculate the normal vector and origin of the splitting plane in standardized coordinates
                standardized_normal_of_split = np.dot(standardized_transformation_split, split_normal_of_split)
                standardized_origin_of_split = np.dot(standardized_transformation_split, split_origin_of_split)

                # transform standardized normal vector to data normal vector of the splitting plane
                data_normal_of_split = np.sqrt(scaler.var_) * standardized_normal_of_split[:-1]

                data_origin_of_split = scaler.inverse_transform(standardized_origin_of_split[:-1].reshape(1, -1))[0]

                # transform the split value back to data coordinates
                data_split_value = float(np.dot(data_normal_of_split.T, data_origin_of_split))

                # create numeric decision node
                node = PCADecisionNode(idx=len(self.allnodes),
                                       variables=self.numeric_variables,
                                       weights=data_normal_of_split,
                                       split_value=data_split_value,
                                       numeric_indices=self.numeric_indices,
                                       parent=parent)

            else:  # ---------------------------------------------------------------------------------------------------
                raise TypeError('Unknown variable type: %s.' % type(split_var).__name__)

            # set number of samples
            node.samples = n_samples

            if parent is not None:
                parent.set_child(child_idx, node)

            # save node to tree
            self.innernodes[node.idx] = node

            # TODO investigate if original_data or data should be used here
            self.c45queue.append((original_data, start, start + split_pos + 1, node, 0, depth + 1))
            self.c45queue.append((original_data, start + split_pos + 1, end, node, 1, depth + 1))

        PCAJPT.logger.debug('Created', str(node))

        if parent is not None:
            parent.set_child(child_idx, node)

        if self.root is None:
            self.root = node

    def likelihood(self, queries: np.ndarray, dirac_scaling=2., min_distances=None) -> np.ndarray:
        """Get the probabilities of a list of worlds. The worlds must be fully assigned with
        single numbers (no intervals).

        :param queries: An array containing the worlds. The shape is (x, len(variables)).
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        Returns: An np.array with shape (x, ) containing the probabilities.

        """

        # initialize probabilities
        probabilities = np.zeros(len(queries))

        # for all leaves
        for leaf in self.leaves.values():

            # calculate probability in "product node"
            leaf_probabilities = leaf.prior * leaf.parallel_likelihood(queries, dirac_scaling, self.minimal_distances)

            # apply "sum node"
            probabilities = probabilities + leaf_probabilities

        return probabilities

    def expectation(self, variables=None, evidence=None, confidence_level=None, fail_on_unsatisfiability=True) \
            -> jpt.trees.ExpectationResult:
        """
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.

        :param variables:
        :param evidence:
        :param confidence_level:
        :param fail_on_unsatisfiability:
        :return:
        """
        raise NotImplementedError()

    def posterior(self,
                  variables: List[Union[Variable, str]] = None,
                  evidence: Union[Dict[Union[Variable, str], Any], VariableMap] = VariableMap(),
                  fail_on_unsatisfiability: bool = True,
                  report_inconsistencies: bool = False) -> jpt.trees.PosteriorResult or None:
        """
        Compute the posterior over all ``variables`` independently.
        Be aware that the distributions are not actually independent.
        :param variables: the variables to calculate the posterior on.
        :param evidence: the evidence
        :param fail_on_unsatisfiability: Rather to raise an Exception if P(Evidence) = 0 or not
        :param report_inconsistencies:
        :return: jpt.trees.PosteriorResult
        """
        variables = variables or self.variables
        evidence = self._preprocess_query(evidence)
        distributions = defaultdict(list)
        weights = []

        for leaf in self.leaves.values():

            # calculate probability of evidence
            probability = leaf.probability(evidence)

            # if this leaf is impossible skip it
            if probability == 0.:
                continue

            # append probability to priors
            weights.append(leaf.probability(evidence) * leaf.prior)

            # calculate posteriors
            posteriors = leaf.posterior(variables, evidence)

            # append posteriors
            for variable in variables:
                distributions[variable].append(posteriors[variable])

        try:
            weights = normalized(weights)
        except ValueError:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence))
            return None

        # initialize result
        result = dict()

        # merge distributions
        for variable in variables:
            result[variable] = Numeric.merge(distributions[variable], weights=weights)

        # construct posterior result
        posterior_result = jpt.trees.PosteriorResult(variables, evidence)
        posterior_result.result = VariableMap(result.items())
        return posterior_result

    def conditional_jpt(self, evidence: VariableMap):
        """
        Apply evidence on a PCAJPT and get a new PCAJPT that represent P(x|evidence).
        The new JPT still contains all variables.

        :param evidence: A preprocessed VariableMap mapping the observed variables to there observed,
            single values (not intervals)
        :type evidence: ``VariableMap``
        """
        result = self.copy()

        for idx, leave in result.leaves.items():
            p_evidence = leave.probability(evidence, self.minimal_distances)
            print(p_evidence)


    def mpe(self, evidence: VariableMap = VariableMap()) -> List[jpt.trees.MPEResult]:
        """
        Calculate the most probable explanation of all variables if the tree given the evidence.
        :param evidence: The raw evidence
        :return: List[MPEResult] that describes all maxima of the tree given the evidence.
        """

        # transform the evidence
        preprocessed_evidence = self._preprocess_query(evidence, allow_singular_values=True)

        # apply the conditions given
        conditional_jpt = self.conditional_jpt(preprocessed_evidence)

        # calculate the maximal probabilities for each leaf
        maxima = [leaf.mpe(self.minimal_distances) for leaf in conditional_jpt.leaves.values()]

        # get the maximum of those maxima
        highest_likelihood = max([m[0] for m in maxima])

        # create a list for all possible maximal occurrences
        results = []

        # for every leaf and its mpe
        for leaf, (likelihood, mpe) in zip(conditional_jpt.leaves.values(), maxima):

            if likelihood == highest_likelihood:
                # append the argmax to the results
                mpe_result = jpt.trees.MPEResult(evidence, highest_likelihood, mpe, leaf.path)
                results.append(mpe_result)

        # return the results
        return results

    def plot(self, title=None, filename=None, directory='/tmp', plotvars=None, view=True, max_symb_values=10):
        '''Generates an SVG representation of the generated regression tree.

        :param title:   (str) title of the plot
        :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
        :type filename: str
        :param directory: the location to save the SVG file to
        :type directory: str
        :param plotvars: the variables to be plotted in the graph
        :type plotvars: <jpt.variables.Variable>
        :param view: whether the generated SVG file will be opened automatically
        :type view: bool
        :param max_symb_values: limit the maximum number of symbolic values to this number
        '''
        if plotvars is None:
            plotvars = []
        plotvars = [self.varnames[v] if type(v) is str else v for v in plotvars]

        title = title or 'unnamed'

        if not os.path.exists(directory):
            os.makedirs(directory)

        dot = Digraph(format='svg', name=title,
                      directory=directory,
                      filename=f'{filename or title}')

        # create nodes
        sep = ",<BR/>"
        for idx, n in self.allnodes.items():
            imgs = ''

            # plot and save distributions for later use in tree plot
            if isinstance(n, PCALeaf):
                rc = math.ceil(math.sqrt(len(plotvars)))
                img = ''
                for i, pvar in enumerate(plotvars):
                    img_name = html.escape(f'{pvar.name}-{n.idx}')

                    params = {} if pvar.numeric else {'horizontal': True,
                                                      'max_values': max_symb_values}

                    n.distributions[pvar].plot(title=html.escape(pvar.name),
                                               fname=img_name,
                                               directory=directory,
                                               view=False,
                                               **params)
                    img += (f'''{"<TR>" if i % rc == 0 else ""}
                                        <TD><IMG SCALE="TRUE" SRC="{os.path.join(directory, f"{img_name}.png")}"/></TD>
                                {"</TR>" if i % rc == rc - 1 or i == len(plotvars) - 1 else ""}
                                ''')

                    # clear current figure to allow for other plots
                    plt.clf()

                if plotvars:
                    imgs = f'''
                                <TR>
                                    <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2">
                                        <TABLE>
                                            {img}
                                        </TABLE>
                                    </TD>
                                </TR>
                                '''

            land = '<BR/>\u2227 '
            element = ' \u2208 '

            # content for node labels
            nodelabel = f'''<TR>
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{"Leaf" if isinstance(n, PCALeaf) else "Node"} #{n.idx}</B><BR/>{html.escape(n.str_node)}</TD>
                            </TR>'''

            if isinstance(n, PCALeaf):
                nodelabel = f'''{nodelabel}{imgs}
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{n.samples} ({n.prior * 100:.3f}%)</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{"<B>" + html.escape(v.name) + "</B>" }=' + (f'{html.escape(str(exp))!s}' if v.symbolic else f'{exp:.2f}') for v, exp in n.expectation().items()])}</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{f"{land}".join([html.escape(var.str(val, fmt='set')) for var, val in n.path.items()])}</TD>
                                </TR>
                                '''

            # stitch together
            lbl = f'''<<TABLE ALIGN="CENTER" VALIGN="MIDDLE" BORDER="0" CELLBORDER="0" CELLSPACING="0">
                            {nodelabel}
                      </TABLE>>'''

            if isinstance(n, PCALeaf):
                dot.node(str(idx),
                         label=lbl,
                         shape='box',
                         style='rounded,filled',
                         fillcolor=green)
            else:
                dot.node(str(idx),
                         label=lbl,
                         shape='ellipse',
                         style='rounded,filled',
                         fillcolor=orange)

        # create edges
        for idx, n in self.innernodes.items():
            for i, c in enumerate(n.children):
                if c is None: continue
                dot.edge(str(n.idx), str(c.idx), label=html.escape(n.str_edge(i)))

        # show graph
        PCAJPT.logger.info(f'Saving rendered image to {os.path.join(directory, filename or title)}.svg')

        # improve aspect ratio of graph having many leaves or disconnected nodes
        dot = dot.unflatten(stagger=3)
        dot.render(view=view, cleanup=False)
