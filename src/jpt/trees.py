'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import html
import json
import math
import numbers
import operator
import os
import pickle
import pprint
from collections import defaultdict, deque, ChainMap, OrderedDict
import datetime

import numpy as np
import pandas as pd
from dnutils.stats import stopwatch
from graphviz import Digraph
from matplotlib import style, pyplot as plt

import dnutils
from dnutils import first, ifnone, mapstr
from sklearn.tree import DecisionTreeRegressor

try:
    from .base.quantiles import QuantileDistribution
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
    from .base.constants import plotstyle, orange, green, SYMBOL
    from .base.utils import list2interval, format_path, normalized
    from .learning.impurity import Impurity
    from .learning.distributions import Multinomial, Numeric, Identity
    from .variables import Variable
except ImportError:
    import pyximport
    pyximport.install()
finally:
    from .base.quantiles import QuantileDistribution
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
    from .base.constants import plotstyle, orange, green, SYMBOL
    from .base.utils import list2interval, format_path, normalized
    from .learning.impurity import Impurity
    from .learning.distributions import Multinomial, Numeric, Identity
    from .variables import Variable


style.use(plotstyle)


# ----------------------------------------------------------------------------------------------------------------------
# Global data store to exploit copy-on-write in multiprocessing

import multiprocessing as mp

_data = None
_data_queue = mp.Queue()
_node_queue = mp.Queue()
_pool = None


# ----------------------------------------------------------------------------------------------------------------------


def _prior(args):
    var_idx, json_var = args
    return Variable.from_json(json_var).dist(data=_data, col=var_idx).to_json()


class Node:
    '''
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx, parent=None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        '''
        self.idx = idx
        self.parent = parent
        self.samples = 0.
        self.children = None
        self._path = []
        self.distributions = {}

    @property
    def path(self):
        res = OrderedDict()
        for var, vals in self._path:
            res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res

    def format_path(self):
        return format_path(self.path)

    def __str__(self):
        return f'Node<{self.idx}>'

    def __repr__(self):
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class DecisionNode(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx, dec_criterion, parent=None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param splits:
        :type splits:
        :param dec_criterion:   the split feature name
        :type dec_criterion:    jpt.variables.Variable
        '''
        self._splits = None
        self.dec_criterion = dec_criterion
        self.dec_criterion_val = None
        super().__init__(idx, parent=parent)
        self.children = None  # [None] * len(self.splits)

    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, splits):
        if self.children is not None:
            raise ValueError('children already set: %s' % self.children)
        self._splits = splits
        self.children = [None] * len(self._splits)

    def set_child(self, idx, node):
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.dec_criterion, self.splits[idx]))

    def str_edge(self, idx):
        return str(ContinuousSet(self.dec_criterion.domain.labels[self.splits[idx].lower],
                                 self.dec_criterion.domain.labels[self.splits[idx].upper],
                                 self.splits[idx].left,
                                 self.splits[idx].right)
                   if self.dec_criterion.numeric else self.dec_criterion.domain.labels[idx])

    @property
    def str_node(self):
        return self.dec_criterion.name

    def __str__(self):
        return (f'DecisionNode<ID:{self.idx}; '
                f'CRITERION: {self.dec_criterion.name}; '
                f'PARENT: {f"DecisionNode<ID: {self.parent.idx}>" if self.parent else None}; '
                f'#CHILDREN: {len(self.children)}>')

    def __repr__(self):
        return f'Node<{self.idx}> object at {hex(id(self))}'
        return str(self.dec_criterion.str(self.splits[idx], fmt='logic'))


# ----------------------------------------------------------------------------------------------------------------------


class Leaf(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''
    def __init__(self, idx, parent=None, prior=None):
        super().__init__(idx, parent=parent)
        self.distributions = OrderedDict()
        self.prior = prior

    @property
    def str_node(self):
        return ""

    def applies(self, query):
        '''Checks whether this leaf is consistent with the given ``query``.'''
        path = self.path
        for var in set(query.keys()).intersection(set(path.keys())):
            if var.symbolic:
                if query.get(var) not in path.get(var):
                    return False
            else:
                if not path.get(var).intersects(query.get(var)):
                    return False
        return True

    @property
    def value(self):
        return self.distributions

    def __str__(self):
        return (f'Leaf<ID: {self.idx}; '
                #f'VALUE: {",".join([f"{var.name}: {str(dist)}" for var, dist in self.distributions.items()])}; '
                # 'PATH: %s' % format_path(self.)
                f'PARENT: {f"DecisionNode<ID: {self.parent.idx}>" if self.parent else None}>')

    def __repr__(self):
        return f'LeafNode<{self.idx}> object at {hex(id(self))}'

    def to_json(self):
        return {'idx': self.idx,
                'path': [[var.name, list(values) if var.symbolic else values.to_json()] for var, values in self.path.items()],
                'distributions': [[var.name, dist.to_json()] for var, dist in self.distributions.items()],
                'leaf_prior': self.prior}

    @staticmethod
    def from_json(tree, data):
        leaf = Leaf(idx=data['idx'], prior=data['leaf_prior'])
        leaf.distributions = OrderedDict([(tree.varnames[v],
                                           tree.varnames[v].domain.from_json(d)) for v, d in data['distributions']])
        leaf._path = OrderedDict()
        for varname, values in data['path']:
            var = tree.varnames[varname]
            leaf.path[var] = set(values) if var.symbolic else ContinuousSet.from_json(values)
        leaf.prior = data['leaf_prior']
        return leaf


# ----------------------------------------------------------------------------------------------------------------------


class JPTBase:

    def __init__(self, variables, targets=None):
        self._variables = tuple(variables)
        self._targets = targets
        self.varnames = OrderedDict((var.name, var) for var in self._variables)
        self.leaves = {}
        self.priors = {}

    @property
    def variables(self):
        return self._variables

    @property
    def targets(self):
        return self._targets

    def to_json(self):
        return {'variables': [v.to_json() for v in self.variables],
                'targets': [v.name for v in self.variables],
                'leaves': [l.to_json() for l in self.leaves.values()],
                'priors': {varname: p.to_json() for varname, p in self.priors.items()}}

    @staticmethod
    def from_json(data):
        jpt = JPTBase(variables=[Variable.from_json(d) for d in data['variables']])
        jpt._targets = tuple(jpt.varnames[varname] for varname in data['targets'])
        jpt.leaves = {d['idx']: Leaf.from_json(jpt, d) for d in data['leaves']}
        jpt.priors = {varname: jpt.varnames[varname].domain.from_json(dist) for varname, dist in data['priors'].items()}
        return jpt

    def infer(self, query, evidence=None, fail_on_unsatisfiability=True):
        r'''For each candidate leaf ``l`` calculate the number of samples in which `query` is true:

        .. math::
            P(query|evidence) = \frac{p_q}{p_e}
            :label: query

        .. math::
            p_q = \frac{c}{N}
            :label: pq

        .. math::
            c = \frac{\prod{F}}{x^{n-1}}
            :label: c

        where ``Q`` is the set of variables in `query`, :math:`P_{l}` is the set of variables that occur in ``l``,
        :math:`F = \{v | v \in Q \wedge~v \notin P_{l}\}` is the set of variables in the `query` that do not occur in ``l``'s path,
        :math:`x = |S_{l}|` is the number of samples in ``l``, :math:`n = |F|` is the number of free variables and
        ``N`` is the number of samples represented by the entire tree.
        reference to :eq:`query`

        :param query:       the event to query for, i.e. the query part of the conditional P(query|evidence) or the prior P(query)
        :type query:        dict of {jpt.variables.Variable : jpt.learning.distributions.Distribution.value}
        :param evidence:    the event conditioned on, i.e. the evidence part of the conditional P(query|evidence)
        :type evidence:     dict of {jpt.variables.Variable : jpt.learning.distributions.Distribution.value}
        '''
        query_ = self._prepropress_query(query)
        evidence_ = ifnone(evidence, {}, self._prepropress_query)

        r = Result(query_, evidence_)

        p_q = 0.
        p_e = 0.

        for leaf in self.apply(evidence_):
            # out(leaf.format_path(), 'applies', ' ^ '.join([var.str_by_idx(val) for var, val in evidence_.items()]))
            p_m = 1
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                p_m *= leaf.distributions[var]._p(evidence_val)

            w = leaf.prior
            p_m *= w
            p_e += p_m

            if leaf.applies(query_):
                for var in set(query_.keys()):
                    query_val = query_[var]
                    if var.numeric and var in leaf.path:
                        query_val = query_val.intersection(leaf.path[var])
                    elif var.symbolic and var in leaf.path:
                        continue
                    p_m *= leaf.distributions[var]._p(query_val)
                p_q += p_m

                r.candidates.append(leaf)
                r.weights.append(p_m)

        if p_e == 0:
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence_))
            else:
                return None

        r.result = p_q / p_e
        r.weights = [w / p_e for w in r.weights]
        return r

    def posterior(self, variables, evidence):
        '''

        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        '''
        evidence_ = ifnone(evidence, {}, self._prepropress_query)
        result = PosteriorResult(variables, evidence_)
        variables = [self.varnames[v] if type(v) is str else v for v in variables]

        distributions = defaultdict(list)

        likelihoods = []
        priors = []

        for leaf in self.apply(evidence_):
            likelihood = 1
            # check if path of candidate leaf is consistent with evidence
            # (i.e. contains evicence variable with *correct* value or does not contain it at all)
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                likelihood *= leaf.distributions[var]._p(evidence_val)
            likelihoods.append(likelihood)
            priors.append(leaf.prior)

            for var in variables:
                evidence_val = evidence_.get(var)
                distribution = leaf.distributions[var]
                if evidence_val is not None:
                    if var.numeric and var in leaf.path:
                        evidence_val = evidence_val.intersection(leaf.path[var])
                    elif var.symbolic and var in leaf.path:
                        continue
                    distribution = distribution.crop(evidence_val)
                distributions[var].append(distribution)

            result.candidates.append(leaf)

        weights = [l * p for l, p in zip(likelihoods, priors)]
        try:
            weights = normalized(weights)
        except ValueError:
            return None

        # initialize all query variables with None, in case dists
        # is empty (i.e. no candidate leaves -> query unsatisfiable)
        result.dists = {v: None for v in variables}

        for var, dists in distributions.items():
            if var.numeric:
                result.dists[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result.dists[var] = Multinomial.merge(dists, weights=weights)

        return result

    def expectation(self, variables=None, evidence=None, confidence_level=None, fail_on_unsatisfiability=True):
        '''
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.
        '''
        posteriors = self.posterior(variables, evidence)
        conf_level = ifnone(confidence_level, .95)

        variables = ifnone([v if isinstance(v, Variable) else self.varnames[v] for v in variables],
                           set(self.variables) - set(evidence))

        if posteriors is None:
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence))
            else:
                return None

        result = {var: ExpectationResult(var,
                                         posteriors._evidence,
                                         conf_level)
                  for var in variables}

        for var, dist in posteriors.dists.items():
            expectation = dist._expectation()
            result[var]._res = expectation
            if var.numeric:
                exp_quantile = dist.cdf.eval(expectation)
                result[var]._lower = dist.ppf.eval(max(0., (exp_quantile - conf_level / 2.)))
                result[var]._upper = dist.ppf.eval(min(1., (exp_quantile + conf_level / 2.)))

        return list(result.values())

    def mpe(self, evidence=None, fail_on_unsatisfiability=True):
        '''
        Compute the (conditional) MPE state of the model.
        '''
        evidence_ = self._prepropress_query(evidence)
        distributions = {var: deque() for var in self.variables}

        r = MPEResult(evidence_)

        for leaf in self.apply(evidence_):
            p_m = 1
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                p_m *= leaf.distributions[var]._p(evidence_val)

            if not p_m: continue

            for var in self.variables:
                distributions[var].append((leaf.distributions[var], p_m))

        if not all([sum([w for _, w in distributions[var]]) for v in self.variables]):
            if fail_on_unsatisfiability:
                raise ValueError('Query is unsatisfiable: P(%s) is 0.' % var.str(evidence_val, fmt='logic'))
            else:
                return None

        posteriors = {var: var.domain.merge([d for d, _ in distributions[var]],
                                            normalized([w for _, w in distributions[var]])) for var in distributions}

        for var, dist in posteriors.items():
            if var in evidence_:
                continue
            r.path.update({var: dist.mpe()})
        return r

    def _prepropress_query(self, query):
        '''
        Transform a query entered by a user into an internal representation
        that can be further processed.
        '''
        # Transform lists into a numeric interval:
        query_ = {var if isinstance(var, Variable)
                  else self.varnames[var]: list2interval(val) if type(val) in (list, tuple) else val for var, val in query.items()}
        # Transform single numeric values in to intervals given by the haze
        # parameter of the respective variable:
        for var, lbl in list(query_.items()):
            if var.numeric:
                if isinstance(lbl, numbers.Number):
                    val = var.domain.values[lbl]
                    prior = self.priors[var.name]
                    quantile = prior.cdf.functions[max(1, min(len(prior.cdf) - 2,
                                                              prior.cdf.idx_at(val)))].eval(val)
                    lower = quantile - var.haze / 2
                    upper = quantile + var.haze / 2
                    query_[var] = ContinuousSet(prior.ppf.functions[max(1,
                                                                        min(len(prior.cdf) - 2,
                                                                        prior.ppf.idx_at(lower)))].eval(lower),
                                                prior.ppf.functions[min(len(prior.ppf) - 2,
                                                                        max(1,
                                                                        prior.ppf.idx_at(upper)))].eval(upper))
                    # query_[var] = ContinuousSet(val, val)
                    # if query_[var].lower >= query_[var].upper or np.isnan(query_[var].upper) or np.isnan(query_[var].lower):
                    #     out(prior.cdf.pfmt())
                    #     out(prior.ppf.pfmt())
                    #     stop(var, lbl, val, quantile, query_[var].lower, query_[var].upper)
                elif isinstance(lbl, ContinuousSet):
                    query_[var] = ContinuousSet(var.domain.values[lbl.lower],
                                                var.domain.values[lbl.upper], lbl.left, lbl.right)
            if var.symbolic:
                # Transform into internal values (symbolic values to their indices):
                query_[var] = var.domain.values[lbl]
        JPT.logger.debug('Original :', pprint.pformat(query), '\nProcessed:', pprint.pformat(query_))
        return query_

    def apply(self, query):
        # if the sample doesn't match the features of the tree, there is no valid prediction possible
        if not set(query.keys()).issubset(set(self._variables)):
            raise TypeError(f'Invalid query. Query contains variables that are not '
                            f'represented by this tree: {[v for v in query.keys() if v not in self._variables]}')

        # find the leaf (or the leaves) that have each variable either
        # - not occur in the path to this node OR
        # - match the boolean/symbolic value in the path OR
        # - lie in the interval of the numeric value in the path
        # -> return leaf that matches query
        yield from (leaf for leaf in self.leaves.values() if leaf.applies(query))


class JPT(JPTBase):
    '''
    Joint Probability Trees.
    '''

    logger = dnutils.getlogger('/jpt', level=dnutils.INFO)

    def __init__(self, variables, targets=None, min_samples_leaf=.01, min_impurity_improvement=None,
                 max_leaves=None, max_depth=None):
        '''Implementation of Joint Probability Tree (JPT) learning. We store multiple distributions
        induced by its training samples in the nodes so we can later make statements
        about the confidence of the prediction.
        has children :class:`~jpt.learning.trees.Node`.

        :param variables:           the variable declarations of the data being processed by this tree
        :type variables:            [jpt.variables.Variable]
        :param min_samples_leaf:    the minimum number of samples required to generate a leaf node
        :type min_samples_leaf:     int
        '''
        super().__init__(variables, targets=targets)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self._numsamples = 0
        self.innernodes = {}
        self.allnodes = ChainMap(self.innernodes, self.leaves)
        self.root = None
        self.c45queue = deque()
        self.max_leaves = max_leaves
        self.max_depth = max_depth or float('inf')
        self._node_counter = 0
        self.indices = None
        self.impurity = None#Impurity(self)

    def c45(self, data, start, end, parent, child_idx, depth):
        '''
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param indices: the indices for the training samples used to calculate the gain
        :type indices:      [[int]]
        :param parent:      the parent node of the current iteration, initially the root node
        :type parent:       jpt.variables.Variable
        :param child_idx:   the index of the child in the current iteration
        :type child_idx:    int
        '''
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        # --------------------------------------------------------------------------------------------------------------
        n_samples = end - start

        if n_samples > self.min_samples_leaf:
            impurity = self.impurity
            impurity.compute_best_split(start, end)
            max_gain = impurity.max_impurity_improvement
            best_split = impurity.best_split_pos

            if impurity.best_var != -1:
                ft_best_idx = impurity.best_var
                ft_best = self.variables[ft_best_idx]  # if ft_best_idx is not None else None
            else:
                ft_best = None

        else:
            max_gain = 0
            ft_best = None
            sp_best = None
            ft_best_idx = None

        # create decision node splitting on ft_best or leaf node if min_samples_leaf criterion is not met
        if max_gain <= min_impurity_improvement or depth >= self.max_depth:
            leaf = Leaf(idx=len(self.allnodes),
                        parent=parent)
            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.dist(data=data, rows=self.indices[start:end], col=i)

            leaf.prior = n_samples / data.shape[0]
            leaf.samples = n_samples

            self.leaves[leaf.idx] = leaf

            JPT.logger.debug('Created leaf', str(leaf))

        else:
            # divide examples into distinct sets for each value of ft_best
            # split_data = None  # {val: [] for val in ft_best.domain.values}
            node = DecisionNode(idx=len(self.allnodes),
                                dec_criterion=ft_best,
                                parent=parent)
            node.samples = n_samples
            # update path
            self.innernodes[node.idx] = node

            if ft_best.symbolic:
                # CASE SPLIT VARIABLE IS SYMBOLIC
                node.splits = [{i_v} for i_v in range(ft_best.domain.n_values)]

                # split examples into distinct sets for each value of the selected feature
                prev = 0
                for i, val in enumerate(node.splits):
                    if best_split and first(val) == data[self.indices[start + best_split[0]], ft_best_idx]:
                        pos = best_split.popleft()
                        self.c45queue.append((data, start + prev, start + pos + 1, node, i, depth + 1))
                        prev = pos + 1

            elif ft_best.numeric:
                # CASE SPLIT VARIABLE IS NUMERIC
                prev = 0
                splits = [Interval(np.NINF, np.PINF, EXC, EXC)]
                for i, pos in enumerate(best_split):
                    splits[-1].upper = (data[self.indices[start + pos],
                                             ft_best_idx] +
                                        data[self.indices[start + pos + 1],
                                             ft_best_idx]) / 2
                    splits.append(Interval(splits[-1].upper, np.PINF, INC, EXC))
                    self.c45queue.append((data, start + prev, start + pos + 1, node, i, depth + 1))
                    prev = pos + 1
                self.c45queue.append((data, start + prev, end, node, len(splits) - 1, depth + 1))
                node.splits = splits

            else:
                raise TypeError('Unknown variable type: %s.' % type(ft_best).__name__)

            if parent is not None:
                parent.set_child(child_idx, node)

            JPT.logger.debug('Created decision node', str(node))

    def __str__(self):
        return (f'{self.__class__.__name__}\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def __repr__(self):
        return (f'{self.__class__.__name__}\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def _p(self, parent, indent):
        if parent is None:
            return "{}None\n".format(" " * indent)
        return "{}{}\n{}".format(" " * indent,
                                 str(parent),
                                 ''.join([self._p(r, indent + 5) for r in ifnone(parent.children, [])]))

    def _preprocess_data(self, data=None, rows=None, columns=None):
        '''
        Transform the input data into an internal representation.
        '''
        if sum(d is not None for d in (data, rows, columns)) != 1:
            raise ValueError('Only either of the three is allowed.')

        JPT.logger.info('Preprocessing data...')

        if isinstance(data, np.ndarray) and data.shape[0] or isinstance(data, list):
            rows = data

        if isinstance(rows, list) and rows:  # Transpose the rows
            columns = [[row[i] for row in rows] for i in range(len(self.variables))]
        elif isinstance(rows, np.ndarray) and rows.shape[0]:
            columns = rows.T

        if isinstance(columns, list) and columns:
            shape = len(columns[0]), len(columns)
        elif isinstance(columns, np.ndarray) and columns.shape:
            shape = columns.T.shape
        elif isinstance(data, pd.DataFrame):
            shape = data.shape
        else:
            raise ValueError('No data given.')

        data_ = np.ndarray(shape=shape, dtype=np.float64, order='C')
        if isinstance(data, pd.DataFrame):
            if set(self.varnames).symmetric_difference(set(data.columns)):
                raise ValueError('Unknown variable names: %s'
                                 % ', '.join(mapstr(set(self.varnames).symmetric_difference(set(data.columns)))))

            data_[:] = data.transform({c: self.varnames[v].domain.values.transformer()
                                       for v, c in zip(self.varnames, data.columns)},
                                      ).values
        else:
            for i, (var, col) in enumerate(zip(self.variables, columns)):
                data_[:, i] = [var.domain.values[v] for v in col]
        return data_

    def learn(self, data=None, rows=None, columns=None):
        '''Fits the ``data`` into a regression tree.

        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        :param rows:    The training examples (assumed in row-shape)
        :type rows:     [[str or float or bool]]; (according to `self.variables`)
        :param columns: The training examples (assumed in row-shape)
        :type columns:  [[str or float or bool]]; (according to `self.variables`)
        '''
        # --------------------------------------------------------------------------------------------------------------
        # Check and prepare the data
        global _data
        _data = self._preprocess_data(data=data, rows=rows, columns=columns)

        self.indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        self.indices[0] = 0
        np.cumsum(self.indices, out=self.indices)
        # Initialize the impurity calculation
        self.impurity = Impurity(self)
        self.impurity.setup(_data, self.indices)
        self.impurity.min_samples_leaf = max(1, self.min_samples_leaf)

        JPT.logger.info('Data transformation... %d x %d' % _data.shape)

        # --------------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        started = datetime.datetime.now()
        JPT.logger.info('Learning prior distributions...')
        self.priors = {}
        pool = mp.Pool()
        for i, prior in enumerate(pool.map(_prior, [(i, var.to_json()) for i, var in enumerate(self.variables)])):# {var: var.dist(data=data[:, i]) }
            self.priors[self.variables[i].name] = self.variables[i].domain.from_json(prior)
        JPT.logger.info('Prior distributions learnt in %s.' % (datetime.datetime.now() - started))
        self.impurity.priors = [self.priors[v.name] for v in self.variables if v.numeric]
        pool.close()
        pool.join()

        # --------------------------------------------------------------------------------------------------------------
        # Start the training

        started = datetime.datetime.now()
        JPT.logger.info('Started learning of %s x %s at %s '
                        'requiring at least %s samples per leaf' % (_data.shape[0],
                                                                    _data.shape[1],
                                                                    started,
                                                                    int(self.impurity.min_samples_leaf)))
        # build up tree
        self.c45queue.append((_data, 0, _data.shape[0], None, None, 0))
        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        if self.innernodes:
            self.root = self.innernodes[0]

        elif self.leaves:
            self.root = self.leaves[0]

        else:
            JPT.logger.error('NO INNER NODES!', self.innernodes, self.leaves)
            self.root = None

        # --------------------------------------------------------------------------------------------------------------
        # Print the statistics

        JPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
        # if logger.level >= 20:
        JPT.logger.debug(self)

    @property
    def min_samples_leaf(self):
        if type(self._min_samples_leaf) is int: return self._min_samples_leaf
        if type(self._min_samples_leaf) is float and 0 < self._min_samples_leaf < 1: return int(self._min_samples_leaf*len(_data))
        return int(self._min_samples_leaf)

    @staticmethod
    def sample(sample, ft):
        # NOTE: This sampling is NOT uniform for intervals that are infinity in any direction! TODO: FIX to sample from CATEGORICAL
        if ft not in sample:
            return Interval(np.NINF, np.inf, EXC, EXC).sample()
        else:
            iv = sample[ft]

        if isinstance(iv, Interval):
            if iv.lower == -np.inf and iv.upper == np.inf:
                return Interval(np.NINF, np.inf, EXC, EXC).sample()
            if iv.lower == -np.inf:
                if any([i.right == EXC for i in iv.intervals]):
                    # workaround to be able to sample from open interval
                    return iv.upper - 0.01 * iv.upper
                else:
                    return iv.upper
            if iv.upper == np.inf:
                # workaround to be able to sample from open interval
                if any([i.left == EXC for i in iv.intervals]):
                    return iv.lower + 0.01 * iv.lower
                else:
                    return iv.lower

            return iv.sample()
        else:
            return iv

    def reverse(self, query, confidence=.5):
        '''Determines the leaf nodes that match query best and returns their respective paths to the root node.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :type query: dict
        :param confidence:  the confidence level for this MPE inference
        :type confidence: float
        :returns: a mapping from probabilities to lists of matcalo.core.algorithms.JPT.Node (path to root)
        :rtype: dict
        '''
        # if none of the target variables is present in the query, there is no match possible
        if set(query.keys()).isdisjoint(set(self.variables)):
            return []

        # Transform into internal values/intervals (symbolic values to their indices) and update to contain all possible variables
        query = {var: list2interval(val) if type(val) in (list, tuple) and var.numeric else val if type(val) in (list, tuple) else [val] for var, val in query.items()}
        query_ = {var: set(var.domain.value[v] for v in val) for var, val in query.items()}
        for i, var in enumerate(self.variables):
            if var in query_: continue
            if var.numeric:
                query_[var] = list2interval([np.NINF, np.PINF])
            else:
                query_[var] = var.domain.values

        # find the leaf (or the leaves) that matches the query best
        confs = {}
        for k, l in self.leaves.items():
            confs_ = defaultdict(float)
            for v, dist in l.distributions.items():
                if v.numeric:
                    confs_[v] = dist.p(query_[v])
                else:
                    conf = 0.
                    for sv in query_[v]:
                        conf += dist.p(sv)
                    confs_[v] = conf
            confs[l] = confs_

        # the candidates are the one leaves that satisfy the confidence requirement (i.e. each free variable of a leaf must satisfy the requirement)
        candidates = sorted([leaf for leaf, confs in confs.items() if all(c >= confidence for c in confs.values())], key=lambda l: sum(confs[l].values()), reverse=True)

        # for the chosen candidate determine the path to the root
        paths = []
        for c in candidates:
            p = []
            curcand = c
            while curcand is not None:
                p.append(curcand)
                curcand = curcand.parent
            paths.append((confs[c], p))

        # elements of path are tuples (a, b) with a being mappings of {var: confidence} and b being an ordered list of
        # nodes representing a path from a leaf to the root
        return paths

    def compute_best_split(self, indices):
        # calculate gains for each feature/target combination and normalize over targets
        gains_tgt = defaultdict(dict)
        for tgt in self.variables:
            maxval = 0.
            for ft in self.variables:
                gains_tgt[tgt][ft] = self.gains(indices, ft, tgt)
                maxval = max(maxval, *gains_tgt[tgt][ft].values())

            # normalize gains for comparability
            gains_tgt[tgt] = {ft: {v: g / maxval if maxval > 0. else 0 for v, g in gains_tgt[tgt][ft].items()} for ft in self._variables}

        # determine (harmonic) mean of target gains
        gains_ft_hm = defaultdict(lambda: defaultdict(dict))
        for tgt, fts in gains_tgt.items():
            for ft, sps in fts.items():
                for sp, spval in sps.items():
                    gains_ft_hm[ft][sp][tgt] = spval

        # determine attribute with highest normalized information gain and its index
        max_gain = -1
        sp_best = None
        ft_best = None
        for ft, sps in gains_ft_hm.items():
            for sp, tgts in sps.items():
                hm = np.mean(list(gains_ft_hm[ft][sp].values()))
                if max_gain < hm:
                    sp_best = sp
                    ft_best = ft
                    max_gain = hm
        ft_best_idx = self.variables.index(ft_best)
        return ft_best_idx, sp_best, max_gain

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

        title = ifnone(title, 'unnamed')

        if not os.path.exists(directory):
            os.makedirs(directory)

        dot = Digraph(format='svg', name=filename or title,
                      directory=directory,
                      filename=f'{filename or title}.dot')

        # create nodes
        sep = ",<BR/>"
        for idx, n in self.allnodes.items():
            imgs = ''

            # plot and save distributions for later use in tree plot
            if isinstance(n, Leaf):
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
                                {"</TR>" if i % rc == rc-1 or i == len(plotvars) - 1 else ""}
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
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{"Leaf" if isinstance(n, Leaf) else "Node"} #{n.idx}</B><BR/>{html.escape(n.str_node)}</TD>
                            </TR>'''

            if isinstance(n, Leaf):
                nodelabel = f'''{nodelabel}{imgs}
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{n.samples} ({n.prior * 100:.3f}%)</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items()])}</TD>
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

            if isinstance(n, Leaf):
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
        JPT.logger.info(f'Saving rendered image to {os.path.join(directory, filename or title)}.svg')

        # improve aspect ratio of graph having many leaves or disconnected nodes
        dot = dot.unflatten(stagger=3)
        dot.render(view=view, cleanup=False)

    def pickle(self, fpath):
        '''Pickles the fitted regression tree to a file at the given location ``fpath``.

        :param fpath: the location for the pickled file
        :type fpath: str
        '''
        with open(os.path.abspath(fpath), 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath):
        '''Loads the pickled regression tree from the file at the given location ``fpath``.

        :param fpath: the location of the pickled file
        :type fpath: str
        '''
        with open(os.path.abspath(fpath), 'rb') as f:
            try:
                JPT.logger.info(f'Loading JPT {os.path.abspath(fpath)}')
                return pickle.load(f)
            except ModuleNotFoundError:
                JPT.logger.error(f'Could not load file {os.path.abspath(fpath)}')
                raise Exception(f'Could not load file {os.path.abspath(fpath)}. Probably deprecated.')

    @staticmethod
    def calcnorm(sigma, mu, intervals):
        '''Computes the CDF for a multivariate normal distribution.

        :param sigma: the standard deviation
        :param mu: the expected value
        :param intervals: the boundaries of the integral
        :type sigma: float
        :type mu: float
        :type intervals: list of matcalo.utils.utils.Interval
        '''
        from scipy.stats import mvn
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], mu, sigma))

    def sklearn_tree(self, data=None, targets=None):
        if data is None:
            data = self.data
        assert data is not None, 'Gimme data!'

        tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                     min_impurity_decrease=self.min_impurity_improvement,
                                     random_state=0)
        with stopwatch('/sklearn/decisiontree'):
            tree.fit(data, data if targets is None else targets)
        return tree

    def save(self, file):
        '''
        Write this JPT persistently to disk.

        ``file`` can be either a string or file-like object.
        '''
        if type(file) is str:
            with open(file, 'w+') as f:
                json.dump(self.to_json(), f)
        else:
            json.dump(self.to_json(), file)

    @staticmethod
    def load(file):
        '''
        Load a JPT from disk.
        '''
        if type(file) is str:
            with open(file, 'r') as f:
                t = json.load(f)
        else:
            t = json.load(file)
        return JPTBase.from_json(t)


class DistributedJPT(JPTBase):

    def __init__(self, path, **kwargs):
        super().__init__(**kwargs)
        self.path = path

    def apply(self, query):
        pass


class Result:

    def __init__(self, query, evidence, res=None, cand=None, w=None):
        self.query = query
        self._evidence = evidence
        self._res = ifnone(res, [])
        self._cand = ifnone(cand, [])
        self._w = ifnone(w, [])

    def __str__(self):
        return self.format_result()

    @property
    def evidence(self):
        return {k: (k.domain.labels[v] if k.symbolic else ContinuousSet(k.domain.labels[v.lower], k.domain.labels[v.upper], v.left, v.right)) for k, v in self._evidence.items()}

    @property
    def result(self):
        return self._res

    @result.setter
    def result(self, res):
        self._res = res

    @property
    def candidates(self):
        return self._cand

    @candidates.setter
    def candidates(self, cand):
        self._cand = cand

    @property
    def weights(self):
        return self._w

    @weights.setter
    def weights(self, w):
        self._w = w

    def format_result(self):
        return ('P(%s%s%s) = %.3f%%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query.items()]),
                                        ' | ' if self.evidence else '',
                                        ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                        self.result * 100))

    def explain(self):
        result = self.format_result()
        result += '\n'
        for weight, leaf in sorted(zip(self.weights, self.candidates), key=operator.itemgetter(0), reverse=True):
            result += '%.3f%%: %s\n' % (weight, format_path({var: val for var, val in leaf.path.items() if var not in self.evidence}))
        return result


class ExpectationResult(Result):

    def __init__(self, query, evidence, theta, lower=None, upper=None, res=None, cand=None, w=None):
        super().__init__(query, evidence, res=res, cand=cand, w=w)
        self.theta = theta
        self._lower = lower
        self._upper = upper

    @property
    def lower(self):
        return self.query.domain.labels[self._lower]

    @property
    def upper(self):
        return self.query.domain.labels[self._upper]

    @property
    def result(self):
        return self.query.domain.labels[self._res]

    def format_result(self):
        left = 'E(%s%s%s; %s = %.3f)' % (self.query,
                                         ' | ' if self.evidence else '',
                                         ', '.join([var.str(val, fmt='logic') for var, val in self._evidence.items()]),
                                         SYMBOL.THETA,
                                         self.theta)
        right = '[%.3f %s %.3f %s %.3f]' % (self.lower,
                                            SYMBOL.ARROW_BAR_LEFT,
                                            self.result,
                                            SYMBOL.ARROW_BAR_RIGHT,
                                            self.upper) if self.query.numeric else self.query.str(self.result)
        return '%s = %s' % (left, right)


class MPEResult(Result):

    def __init__(self, evidence, res=None, cand=None, w=None):
        super().__init__(None, evidence, res=res, cand=cand, w=w)
        self.path = {}

    def format_result(self):
        return f'MPE({self.evidence}) = {format_path(self.path)}'


class PosteriorResult(Result):

    def __init__(self, query, evidence, dists=None, cand=None, w=None):
        super().__init__(query, evidence, res=None, cand=cand)
        self._w = ifnone(w, {})
        self.dists = dists

    def format_result(self):
        return ('P(%s%s%s) = %.3f%%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query.items()]),
                                        ' | ' if self.evidence else '',
                                        ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                        self.result * 100))

    def __getitem__(self, item):
        return self.dists[item]
