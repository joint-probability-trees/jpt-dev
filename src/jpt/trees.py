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
from itertools import zip_longest
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dnutils.stats import stopwatch
from graphviz import Digraph
from matplotlib import style, pyplot as plt

import dnutils
from dnutils import first, ifnone, mapstr, out, err, fst
from sklearn.tree import DecisionTreeRegressor

from .base.utils import list2interval, format_path, normalized, Unsatisfiability

from .variables import Variable, VariableMap

try:
    from .base.quantiles import QuantileDistribution
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
    from .base.constants import plotstyle, orange, green, SYMBOL
    from .base.utils import list2interval, format_path, normalized
    from .learning.impurity import Impurity
    from .learning.distributions import Multinomial, Numeric, Identity, Distribution
    from .variables import Variable, SymbolicVariable, NumericVariable
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
    try:
        return Variable.from_json(json_var).dist(data=_data, col=var_idx).to_json()
    except ValueError as e:
        raise ValueError('%s: %s' % (Variable.from_json(json_var), str(e)))


# ----------------------------------------------------------------------------------------------------------------------
# Global constants

DISCRIMINATIVE = 'discriminative'
GENERATIVE = 'generative'

# ----------------------------------------------------------------------------------------------------------------------


class Node:
    '''
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx:int, parent=None) -> None:
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        '''
        self.idx = idx
        self.parent = parent
        self.samples = 0.
        self._path = []

    @property
    def path(self) -> VariableMap:
        res = VariableMap()
        for var, vals in self._path:
            res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res

    def format_path(self):
        return format_path(self.path)

    def __str__(self) -> str:
        return f'Node<{self.idx}>'

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class DecisionNode(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx:int, variable:Variable, parent:Node=None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param variable:   the split feature name
        :type variable:    jpt.variables.Variable
        '''
        self._splits = None
        self.variable = variable
        super().__init__(idx, parent=parent)
        self.children:None or List[Node] = None  # [None] * len(self.splits)

    def __eq__(self, o) -> bool:
        return (type(self) is type(o) and
                self.idx == o.idx and
                (self.parent.idx
                 if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
                [n.idx for n in self.children] == [n.idx for n in o.children] and
                self.splits == o.splits and
                self.variable == o.variable and
                self.samples == o.samples)

    def to_json(self) -> str:
        return {'idx': self.idx,
                'parent': self.parent.idx if self.parent is not None else None,
                'splits': [s.to_json() if isinstance(s, ContinuousSet) else s for s in self.splits],
                'variable': self.variable.name,
                '_path': [(var.name, split.to_json() if var.numeric else list(split)) for var, split in self._path],
                'children': [node.idx for node in self.children],
                'samples': self.samples,
                'child_idx': self.parent.children.index(self) if self.parent is not None else None}

    @staticmethod
    def from_json(jpt, data):
        node = DecisionNode(idx=data['idx'], variable=jpt.varnames[data['variable']])
        node.splits = [Interval.from_json(s) if node.variable.numeric else s for s in data['splits']]
        node.children = [None] * len(node.splits)
        node.parent = jpt.innernodes.get(data['parent'])
        node.samples = data['samples']
        if node.parent is not None:
            node.parent.set_child(data['child_idx'], node)
        jpt.innernodes[node.idx] = node
        return node

    @property
    def splits(self) -> List:
        return self._splits

    @splits.setter
    def splits(self, splits):
        if self.children is not None:
            raise ValueError('Children already set: %s' % self.children)
        self._splits = splits
        self.children = [None] * len(self._splits)

    def set_child(self, idx, node) -> None:
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.variable, self.splits[idx]))

    def str_edge(self, idx) -> str:
        if self.variable.numeric:
            return str(ContinuousSet(self.variable.domain.labels[self.splits[idx].lower],
                                     self.variable.domain.labels[self.splits[idx].upper],
                                     self.splits[idx].left,
                                     self.splits[idx].right))
        else:
            negate = len(self.splits[1]) > 1
            if negate:
                label = self.variable.domain.labels[fst(self.splits[0])]
                return '%s%s' % ('\u00AC' if idx > 0 else '', label)
            else:
                return str(self.variable.domain.labels[fst(self.splits[idx])])

    @property
    def str_node(self) -> str:
        return self.variable.name

    def __str__(self) -> str:
        return (f'<DecisionNode #{self.idx} '
                f'{self.variable.name} = [%s]' % '; '.join(self.str_edge(i) for i in range(len(self.splits))) +
                f'; parent-#: {self.parent.idx if self.parent is not None else None}'
                f'; #children: {len(self.children)}>')

    def __repr__(self) -> str:
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class Leaf(Node):
    '''
    Represents a leaf node of the the :class:`jpt.learning.trees.Tree`.
    '''
    def __init__(self, idx:int, parent:Node or None=None, prior=None):
        super().__init__(idx, parent=parent)
        self.distributions = VariableMap()
        self.prior = prior

    @property
    def str_node(self) -> str:
        return ""

    def applies(self, query: VariableMap) -> bool:
        '''Checks whether this leaf is consistent with the given ``query``.'''
        path = self.path
        for var in set(query.keys()).intersection(set(path.keys())):
            if path.get(var).isdisjoint(query.get(var)):
                return False
        return True

    @property
    def value(self):
        return self.distributions

    def __str__(self) -> str:
        return (f'Leaf<ID: {self.idx}; '
                f'parent: {f"DecisionNode<ID: {self.parent.idx}>" if self.parent else None}>')

    def __repr__(self) -> str:
        return f'LeafNode<{self.idx}> object at {hex(id(self))}'

    def to_json(self) -> str:
        return {'idx': self.idx,
                'distributions': self.distributions.to_json(),
                'prior': self.prior,
                'samples': self.samples,
                'parent': self.parent.idx if self.parent is not None else None,
                'child_idx': self.parent.children.index(self)}

    @staticmethod
    def from_json(tree, data):
        leaf = Leaf(idx=data['idx'], prior=data['prior'], parent=tree.innernodes.get(data['parent']))
        leaf.distributions = VariableMap.from_json(tree.variables, data['distributions'], Distribution)
        leaf._path = []
        leaf.parent.set_child(data['child_idx'], leaf)
        leaf.prior = data['prior']
        leaf.samples = data['samples']
        tree.leaves[leaf.idx] = leaf
        return leaf

    def __eq__(self, o) -> bool:
        return (type(o) == type(self) and
                self.idx == o.idx and
                self._path == o._path and
                self.samples == o.samples and
                self.distributions == o.distributions and
                self.prior == o.prior)


# ----------------------------------------------------------------------------------------------------------------------

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
        return {k: (k.domain.labels[fst(v)]
                    if k.symbolic else ContinuousSet(k.domain.labels[v.lower],
                                                     k.domain.labels[v.upper], v.left, v.right))
                for k, v in self._evidence.items()}

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
        return ('P(%s%s) = %.3f%%' % (format_path(self.query),
                                      (' | %s' % format_path(self.evidence)) if self.evidence else '',
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
        left = 'E(%s%s%s; %s = %.3f)' % (self.query.name,
                                         ' | ' if self.evidence else '',
                                         # ', '.join([var.str(val, fmt='logic') for var, val in self._evidence.items()]),
                                         format_path(self.evidence),
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
        self.distributions = dists

    def format_result(self):
        return ('P(%s%s%s) = %.3f%%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query.items()]),
                                        ' | ' if self.evidence else '',
                                        ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                        self.result * 100))

    def __getitem__(self, item):
        return self.distributions[item]


class JPTBase:

    def __init__(self, variables, targets=None):
        self._variables = tuple(variables)
        self._targets = targets
        self.varnames:OrderedDict[str, Variable] = OrderedDict(
                        (var.name, var) for var in self._variables)
        self.leaves:Dict[int, Leaf] = {}
        self.innernodes:Dict[int, DecisionNode] = {}
        self.allnodes:Dict[int,Node] = ChainMap(self.innernodes, self.leaves)
        self.priors = {}

    @property
    def variables(self) -> List[Variable]:
        return self._variables

    @property
    def targets(self) -> List[Variable]:
        return self._targets

    @property
    def features(self) -> List[Variable]:
        return [var for var in self.variables if var not in self.targets]

    @property
    def numeric_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, NumericVariable)]

    @property
    def symbolic_variables(self) -> List[Variable]:
        return [var for var in self.variables if isinstance(var, SymbolicVariable)]

    @property
    def numeric_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, NumericVariable)]

    @property
    def symbolic_targets(self) -> List[Variable]:
        return [var for var in self.targets if isinstance(var, SymbolicVariable)]

    @property
    def numeric_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, NumericVariable)]

    @property
    def symbolic_features(self) -> List[Variable]:
        return [var for var in self.features if isinstance(var, SymbolicVariable)]


    def to_json(self) -> str:
        return {'variables': [v.to_json() for v in self.variables],
                'targets': [v.name for v in self.variables],
                'leaves': [l.to_json() for l in self.leaves.values()],
                'innernodes': [n.to_json() for n in self.innernodes.values()],
                'priors': {varname: p.to_json() for varname, p in self.priors.items()}}

    @staticmethod
    def from_json(data) -> str:
        jpt = JPTBase(variables=[Variable.from_json(d) for d in data['variables']])
        jpt._targets = tuple(jpt.varnames[varname] for varname in data['targets'])
        for d in data['innernodes']:
            DecisionNode.from_json(jpt, d)
        for d in data['leaves']:
            Leaf.from_json(jpt, d)
        jpt.priors = {varname: jpt.varnames[varname].domain.from_json(dist)
                      for varname, dist in data['priors'].items()}
        return jpt

    def __eq__(self, o) -> bool:
        return (isinstance(o, JPTBase) and
                self.variables == o.variables and
                self.innernodes == o.innernodes and
                self.leaves == o.leaves and
                self.priors == o.priors)

    def infer(self, query, evidence=None, fail_on_unsatisfiability=True) -> Result:
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
        querymap = VariableMap()
        for key, value in query.items():
            querymap[key if isinstance(key, Variable) else self.varnames[key]] = value
        query_ = self._prepropress_query(querymap)
        evidencemap = VariableMap()
        if evidence:
            for key, value in evidence.items():
                evidencemap[key if isinstance(key, Variable) else self.varnames[key]] = value
        evidence_ = ifnone(evidencemap, {}, self._prepropress_query)

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
                r.result = None
                r.weights = None
        else:
            r.result = p_q / p_e
            r.weights = [w / p_e for w in r.weights]
        return r

    def posterior(self, variables, evidence, fail_on_unsatisfiability=True) -> Result:
        '''

        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :type fail_on_unsatisfiability:  bool
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
                evidence_set = evidence_[var]
                if var in leaf.path:
                    evidence_set = evidence_set.intersection(leaf.path[var])
                likelihood *= leaf.distributions[var]._p(evidence_set)
            likelihoods.append(likelihood)
            priors.append(leaf.prior)

            for var in variables:
                evidence_set = evidence_.get(var)
                distribution = leaf.distributions[var]
                if evidence_set is not None:
                    if var in leaf.path:
                        evidence_set = evidence_set.intersection(leaf.path[var])
                        distribution = distribution.crop(evidence_set)
                distributions[var].append(distribution)

            result.candidates.append(leaf)

        weights = [l * p for l, p in zip(likelihoods, priors)]
        try:
            weights = normalized(weights)
        except ValueError:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence))
            return None

        # initialize all query variables with None, in case dists
        # is empty (i.e. no candidate leaves -> query unsatisfiable)
        result.distributions = VariableMap()

        for var, dists in distributions.items():
            if var.numeric:
                result.distributions[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result.distributions[var] = Multinomial.merge(dists, weights=weights)

        return result

    def expectation(self, variables=None,
                    evidence=None,
                    confidence_level=None,
                    fail_on_unsatisfiability=True) -> ExpectationResult:
        '''
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.
        '''
        variables = ifnone([v if isinstance(v, Variable) else self.varnames[v] for v in variables],
                           set(self.variables) - set(evidence))
        posteriors = self.posterior(variables, evidence, fail_on_unsatisfiability=fail_on_unsatisfiability)
        conf_level = ifnone(confidence_level, .95)

        if posteriors is None:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Query is unsatisfiable: P(%s) is 0.' % format_path(evidence))
            else:
                return None

        final = VariableMap()
        for var, dist in posteriors.distributions.items():
            result = ExpectationResult(var, posteriors._evidence, conf_level)
            result._res = dist._expectation()
            if var.numeric:
                exp_quantile = dist.cdf.eval(result._res)
                result._lower = dist.ppf.eval(max(0., (exp_quantile - conf_level / 2.)))
                result._upper = dist.ppf.eval(min(1., (exp_quantile + conf_level / 2.)))
            final[var] = result
        return final

    def mpe(self, evidence=None, fail_on_unsatisfiability=True) -> MPEResult:
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
                                            normalized([w for _, w in distributions[var]]))
                      for var in distributions}

        for var, dist in posteriors.items():
            if var in evidence_:
                continue
            r.path.update({var: dist.mpe()})
        return r

    def _prepropress_query(self, query) -> VariableMap:
        '''
        Transform a query entered by a user into an internal representation
        that can be further processed.
        '''
        # Transform lists into a numeric interval:
        query_ = VariableMap()
        # Transform single numeric values in to intervals given by the haze
        # parameter of the respective variable:
        for key, arg in query.items():
            var = key if isinstance(key, Variable) else self.varnames[key]
            if var.numeric:
                if type(arg) is list:
                    arg = list2interval(arg)
                if isinstance(arg, numbers.Number):
                    val = var.domain.values[arg]
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
                elif isinstance(arg, ContinuousSet):
                    query_[var] = ContinuousSet(var.domain.values[arg.lower],
                                                var.domain.values[arg.upper], arg.left, arg.right)
            if var.symbolic:
                # Transform into internal values (symbolic values to their indices):
                if not type(arg) is set:
                    arg = {arg}
                query_[var] = {var.domain.values[v] for v in arg}

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
                 max_leaves=None, max_depth=None, variable_dependencies=None) -> None:
        '''Implementation of Joint Probability Tree (JPT) learning. We store multiple distributions
        induced by its training samples in the nodes so we can later make statements
        about the confidence of the prediction.
        has children :class:`~jpt.learning.trees.Node`.

        :param variables:           the variable declarations of the data being processed by this tree
        :type variables:            [jpt.variables.Variable]
        :param min_samples_leaf:    the minimum number of samples required to generate a leaf node
        :type min_samples_leaf:     int
        :param variable_dependencies: A dict that maps every variable to a list of variables that are 
                                        directly dependent to that variable.
        :type variable_dependencies: None or Dict from variable to list of variables 
        '''
        super().__init__(variables, targets=targets)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self._numsamples = 0
        self.root = None
        self.c45queue = deque()
        self.max_leaves = max_leaves
        self.max_depth = max_depth or float('inf')
        self._node_counter = 0
        self.indices = None
        self.impurity = None

        # initialize the dependencies as fully dependent on each other.
        # the interface isn't modified therefore the jpt should work as before if not
        # specified different
        if variable_dependencies is None:
            self.variable_dependencies: Dict[Variable, List[Variable]] = \
                dict(zip(self.variables, [self.variables]*len(self.variables)))
        else:
            self.variable_dependencies: Dict[Variable, List[Variable]] = variable_dependencies

        # also initialize the dependency structure as indices since it will be usefull in the c45 algorithm
        self.dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                         -1, dtype=np.int64)

        # dependencies to numeric varaibles for every variable
        self.numeric_dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                                 -1,
                                                 dtype=np.int64)

        # dependencies to symbolic variables for every variable
        self.symbolic_dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                                  -1,
                                                  dtype=np.int64)

        # convert variable dependency structure to index dependency structure for easy interpretation in cython
        for key, value in self.variable_dependencies.items():
            
            # get the index version of the dependent variables and store them
            key_ = self.variables.index(key)
            value_ = [self.variables.index(var) for var in value]
            self.dependency_matrix[key_, 0:len(value_)] = value_
            
            # create lists to store the index dependencies for only numeric/symbolic variables
            numeric_dependencies = []
            symbolic_dependencies = []

            for dependent_variable in value:
                #skip dependent variables if one is not allowed to purify them
                if self.targets and dependent_variable not in self.targets:
                    continue
                
                # get index of numeric dependent variable
                if isinstance(dependent_variable, NumericVariable):
                    if self.targets:
                        numeric_dependencies.append(
                            self.numeric_targets.index(dependent_variable)
                            )
                    else:
                        numeric_dependencies.append(
                            self.numeric_variables.index(dependent_variable)
                            )

                # get indices of symbolic dependent variable
                elif isinstance(dependent_variable, SymbolicVariable):
                    if self.targets:
                        symbolic_dependencies.append(
                            self.symbolic_targets.index(dependent_variable)
                            )
                    else:
                        symbolic_dependencies.append(
                            self.symbolic_variables.index(dependent_variable)
                            )

            # save the index dependencies to the matrix later used to calculate impurities
            self.numeric_dependency_matrix[key_, 0:len(numeric_dependencies)] = numeric_dependencies
            self.symbolic_dependency_matrix[key_, 0:len(symbolic_dependencies)] = symbolic_dependencies

    def c45(self, data, start, end, parent, child_idx, depth) -> None:
        '''
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent:      the parent node of the current iteration, initially the root node.
        :param child_idx:   the index of the child in the current iteration.
        :param depth:       the depth of the tree in the current recursion level.
        '''
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        n_samples = end - start
        split_var_idx = split_pos = -1
        split_var = None

        impurity = self.impurity
        max_gain = impurity.compute_best_split(start, end)

        if max_gain:
            split_pos = impurity.best_split_pos
            split_var_idx = impurity.best_var
            split_var = self.variables[split_var_idx]

        JPT.logger.debug('Data range: %d-%d,' % (start, end),
                         'split var:', split_var,
                         ', split_pos:', split_pos,
                         ', gain:', max_gain)
        if max_gain <= min_impurity_improvement or depth >= self.max_depth:  # -----------------------------------------
            leaf = node = Leaf(idx=len(self.allnodes), parent=parent)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.dist(data=data,
                                               rows=self.indices[start:end], col=i)

            leaf.prior = n_samples / data.shape[0]
            leaf.samples = n_samples

            self.leaves[leaf.idx] = leaf

        else:  # -------------------------------------------------------------------------------------------------------
            node = DecisionNode(idx=len(self.allnodes),
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

        JPT.logger.debug('Created', str(node))

        if parent is not None:
            parent.set_child(child_idx, node)

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def _p(self, parent, indent) -> str:
        if parent is None:
            return "{}None\n".format(" " * indent)
        return "{}{}\n{}".format(" " * indent,
                                 str(parent),
                                 ''.join([self._p(r, indent + 5) for r in ([] if isinstance(parent, Leaf)
                                                                           else parent.children)]))

    def _preprocess_data(self, data=None, rows=None, columns=None) -> np.ndarray:
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

            # Check if the order of columns in the data frame is the same
            # as the order of the variables.
            if not all(c == v for c, v in zip_longest(data.columns, self.varnames)):
                raise ValueError('Columns in DataFrame must coincide with variable order: %s' %
                                 ', '.join(mapstr(self.varnames)))
            transformations = {v: self.varnames[v].domain.values.transformer() for v in data.columns}
            try:
                data_[:] = data.transform(transformations).values
            except ValueError:
                err(transformations)
                raise
        else:
            for i, (var, col) in enumerate(zip(self.variables, columns)):
                data_[:, i] = [var.domain.values[v] for v in col]
        return data_

    def learn(self, data=None, rows=None, columns=None) -> None:
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
        # self.impurity.priors = [self.priors[v.name] for v in self.variables if v.numeric]
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
        learning = GENERATIVE if self.targets is None else DISCRIMINATIVE
        JPT.logger.info('Learning is %s. ' % learning)
        if learning == DISCRIMINATIVE:
            JPT.logger.info('Target variables (%d): %s\n'
                            'Feature variables (%d): %s' % (len(self.targets),
                                                            ', '.join(mapstr(self.targets)),
                                                            len(self.variables) - len(self.targets),
                                                            ', '.join(mapstr(set(self.variables) - set(self.targets)))))
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
        if type(self._min_samples_leaf) is int:
            return self._min_samples_leaf
        if type(self._min_samples_leaf) is float and 0 < self._min_samples_leaf < 1:
            return int(self._min_samples_leaf * len(_data))
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

    def reverse(self, query, confidence=.5) -> List[Tuple[Dict, List[Node]]]:
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
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items() if self.targets is None or v in self.targets])}</TD>
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

    def pickle(self, fpath) -> None:
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

    def sklearn_tree(self, data=None, targets=None) -> DecisionTreeRegressor:
        if data is None:
            data = self.data
        assert data is not None, 'Gimme data!'

        tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                     min_impurity_decrease=self.min_impurity_improvement,
                                     random_state=0)
        with stopwatch('/sklearn/decisiontree'):
            tree.fit(data, data if targets is None else targets)
        return tree

    def save(self, file) -> None:
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