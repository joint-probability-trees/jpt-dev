'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import html
import json
import queue
from operator import attrgetter

import math
import numbers
import operator
import os
import pickle
import pprint
from collections import defaultdict, deque, ChainMap, OrderedDict
import datetime
from itertools import zip_longest
from typing import Dict, List, Tuple, Any, Union

import numpy as np
import pandas as pd
from graphviz import Digraph
from matplotlib import style, pyplot as plt

import dnutils
from dnutils import first, ifnone, mapstr, err, fst, out, ifnot

from .base.utils import prod
from .base.errors import Unsatisfiability

from .variables import VariableMap, SymbolicVariable, NumericVariable, Variable
from .distributions import Distribution

from .base.utils import list2interval, format_path, normalized
from .distributions import Multinomial, Numeric
from .base.constants import plotstyle, orange, green, SYMBOL

try:
    from .base.intervals import __module__
    from .learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet
    from .learning.impurity import Impurity


style.use(plotstyle)


# ----------------------------------------------------------------------------------------------------------------------
# Global constants

DISCRIMINATIVE = 'discriminative'
GENERATIVE = 'generative'


# ----------------------------------------------------------------------------------------------------------------------

class Node:
    '''
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx: int, parent: Union[None, 'DecisionNode'] = None) -> None:
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        '''
        self.idx = idx
        self.parent: DecisionNode = parent
        self.samples = 0.
        self._path = []

    @property
    def path(self) -> VariableMap:
        res = VariableMap()
        for var, vals in self._path:
            res[var] = res.get(var, set(range(var.domain.n_values)) if var.symbolic else R).intersection(vals)
        return res

    def consistent_with(self, evidence: VariableMap) -> bool:
        '''
        Check if the node is consistend with the variable assignments in evidence.
        
        :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
        :type evidence: VariableMap
        '''
        for variable, value in evidence.items():
            variable: Variable
            if variable in self.path.keys():
                restriction = self.path[variable]
                if variable.numeric:
                    if not restriction.lower < value <= restriction.upper:
                        return False
                elif variable.symbolic:
                    if value not in restriction:
                        return False
        return True

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

    def __init__(self, idx: int, variable: Variable, parent: 'DecisionNode' = None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param variable:   the split feature name
        :type variable:    jpt.variables.Variable
        '''
        self._splits = None
        self.variable = variable
        super().__init__(idx, parent=parent)
        self.children: None or List[Node] = None  # [None] * len(self.splits)

    def __eq__(self, o) -> bool:
        return (type(self) is type(o) and
                self.idx == o.idx and
                (self.parent.idx
                 if self.parent is not None else None) == (o.parent.idx if o.parent is not None else None) and
                [n.idx for n in self.children] == [n.idx for n in o.children] and
                self.splits == o.splits and
                self.variable == o.variable and
                self.samples == o.samples)

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'parent': ifnone(self.parent, None, attrgetter('idx')),
                'splits': [s.to_json() if isinstance(s, ContinuousSet) else list(s) for s in self.splits],
                'variable': self.variable.name,
                '_path': [(var.name, split.to_json() if var.numeric else list(split)) for var, split in self._path],
                'children': [node.idx for node in self.children],
                'samples': self.samples,
                'child_idx': self.parent.children.index(self) if self.parent is not None else None}

    @staticmethod
    def from_json(jpt: 'JPT', data: Dict[str, Any]) -> 'DecisionNode':
        node = DecisionNode(idx=data['idx'], variable=jpt.varnames[data['variable']])
        node.splits = [Interval.from_json(s) if node.variable.numeric else set(s) for s in data['splits']]
        node.children = [None] * len(node.splits)
        node.parent = ifnone(data['parent'], None, jpt.innernodes.get)
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

    def set_child(self, idx: int, node: Node) -> None:
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

class Leaf(Node):
    '''
    Represents a leaf node of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx: int, parent: Node or None = None, prior=None):
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

    def recursive_children(self):
        return []

    def __str__(self) -> str:
        return (f'<Leaf # {self.idx}; '
                f'parent: <%s # %s>>' % (type(self.parent).__qualname__, ifnone(self.parent, None, attrgetter('idx'))))

    def __repr__(self) -> str:
        return f'Leaf<{self.idx}> object at {hex(id(self))}'

    def to_json(self) -> Dict[str, Any]:
        return {'idx': self.idx,
                'distributions': self.distributions.to_json(),
                'prior': self.prior,
                'samples': self.samples,
                'parent': ifnone(self.parent, None, attrgetter('idx')),
                'child_idx': self.parent.children.index(self) if self.parent is not None else -1}

    @staticmethod
    def from_json(tree: 'JPT', data: Dict[str, Any]) -> 'Leaf':
        leaf = Leaf(idx=data['idx'], prior=data['prior'], parent=tree.innernodes.get(data['parent']))
        leaf.distributions = VariableMap.from_json(tree.variables, data['distributions'], Distribution)
        leaf._path = []
        if leaf.parent is not None:
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
        self.candidate_dists = defaultdict(list)

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
                                      (' | %s' % format_path(self._evidence)) if self.evidence else '',
                                      self.result * 100))

    def explain(self):
        result = self.format_result()
        result += '\n'
        out(self.candidates)
        out(self.weights)
        for weight, leaf in sorted(zip(self.weights, self.candidates),
                                   key=operator.itemgetter(0),
                                   reverse=True):
            if not weight:
                continue
            result += '%.3f%%: %s\n' % (weight * 100,
                                        format_path({var: val for var, val in leaf.path.items()
                                                     if var not in self.evidence}))
        return result


# ----------------------------------------------------------------------------------------------------------------------

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
                                         format_path(self._evidence),
                                         SYMBOL.THETA,
                                         self.theta)
        right = '[%.3f %s %.3f %s %.3f]' % (self.lower,
                                            SYMBOL.ARROW_BAR_LEFT,
                                            self.result,
                                            SYMBOL.ARROW_BAR_RIGHT,
                                            self.upper) if self.query.numeric else self.result
        return '%s = %s' % (left, right)

    def explain(self):
        result = self.format_result()
        result += '\n'
        out(self.candidates)
        out(self.weights)
        for weight, leaf, dist in sorted(zip(self.weights, self.candidates, self.candidate_dists),
                                   key=operator.itemgetter(0),
                                   reverse=True):
            if not weight:
                continue
            result += '%.3f%%: %s: %s - %s - %s\n' % (weight * 100,
                                                      format_path({var: val for var, val in leaf.path.items()}),
                                                      dist.quantile(.05),
                                                      dist.expectation(),
                                                      dist.quantile(.95))
        print(self.distribution.ppf.pfmt())
        self.distribution.plot(view=True)
        return result


# ----------------------------------------------------------------------------------------------------------------------

class MPEResult(Result):

    def __init__(self, evidence, res=None, cand=None, w=None):
        super().__init__(None, evidence, res=res, cand=cand, w=w)
        self.path = {}

    def format_result(self):
        return f'MPE({self.evidence}) = {format_path(self.path)}'


# ----------------------------------------------------------------------------------------------------------------------

class PosteriorResult(Result):

    def __init__(self, query, evidence, dists=None, cand=None, w=None):
        super().__init__(query, evidence, res=None, cand=cand, w=w)
        self.distributions = dists

    def format_result(self):
        return ('P(%s%s%s) = %.3f%%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query]),
                                        ' | ' if self.evidence else '',
                                        ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                        self.result * 100))

    def __getitem__(self, item):
        return self.distributions[item]


# ----------------------------------------------------------------------------------------------------------------------

class JPT:
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
        :type min_samples_leaf:     int or float
        :param variable_dependencies: A dict that maps every variable to a list of variables that are 
                                        directly dependent to that variable.
        :type variable_dependencies: None or Dict from variable to list of variables 
        '''

        self._variables = tuple(variables)
        self.varnames: OrderedDict[str, Variable] = OrderedDict((var.name, var) for var in self._variables)
        self._targets = [self.varnames[v] if type(v) is str else v for v in targets] if targets is not None else None
        self.leaves: Dict[int, Leaf] = {}
        self.innernodes: Dict[int, DecisionNode] = {}
        self.allnodes: ChainMap[int, Node] = ChainMap(self.innernodes, self.leaves)
        self.priors = {}

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
            self.variable_dependencies: VariableMap[Variable, List[Variable]] = \
                VariableMap(zip(self.variables, [list(self.variables)] * len(self.variables)))
        else:
            self.variable_dependencies: VariableMap[Variable, List[Variable]] = variable_dependencies

        # also initialize the dependency structure as indices since it will be usefull in the c45 algorithm
        self.dependency_matrix = np.full((len(self.variables), len(self.variables)),
                                         -1, dtype=np.int64)

        # dependencies to numeric variables for every variable
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
                # skip dependent variables if one is not allowed to purify them
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

    def _reset(self) -> None:
        self.innernodes.clear()
        self.leaves.clear()
        self.priors.clear()
        self.root = None
        self.c45queue.clear()

    @property
    def variables(self) -> Tuple[Variable]:
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

    def to_json(self) -> Dict:
        return {'variables': [v.to_json() for v in self.variables],
                'targets': [v.name for v in self.targets] if self.targets else self.targets,
                'min_samples_leaf': self.min_samples_leaf,
                'min_impurity_improvement': self.min_impurity_improvement,
                'max_leaves': self.max_leaves,
                'max_depth': self.max_depth,
                'variable_dependencies': {var.name: [v.name for v in deps]
                                          for var, deps in self.variable_dependencies.items()},
                'leaves': [l.to_json() for l in self.leaves.values()],
                'innernodes': [n.to_json() for n in self.innernodes.values()],
                'priors': {varname: p.to_json() for varname, p in self.priors.items()},
                'root': ifnone(self.root, None, attrgetter('idx'))
                }

    @staticmethod
    def from_json(data: Dict[str, Any]):
        variables = OrderedDict([(d['name'], Variable.from_json(d)) for d in data['variables']])
        jpt = JPT(variables=list(variables.values()),
                  targets=[variables[v] for v in data['targets']] if data.get('targets') else None,
                  min_samples_leaf=data['min_samples_leaf'],
                  min_impurity_improvement=data['min_impurity_improvement'],
                  max_leaves=data['max_leaves'],
                  max_depth=data['max_depth'],
                  variable_dependencies={variables[var]: [variables[v] for v in deps]
                                         for var, deps in data['variable_dependencies'].items()}
                  )
        for d in data['innernodes']:
            DecisionNode.from_json(jpt, d)
        for d in data['leaves']:
            Leaf.from_json(jpt, d)
        jpt.priors = {varname: jpt.varnames[varname].domain.from_json(dist)
                      for varname, dist in data['priors'].items()}
        jpt.root = jpt.allnodes[data.get('root')] if data.get('root') is not None else None
        return jpt

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = JPT.from_json(state).__dict__

    def __eq__(self, o) -> bool:
        return (isinstance(o, JPT) and
                self.innernodes == o.innernodes and
                self.leaves == o.leaves and
                self.priors == o.priors and
                (self.dependency_matrix == o.dependency_matrix).all() and
                self.min_samples_leaf == o.min_samples_leaf and
                self.min_impurity_improvement == o.min_impurity_improvement and
                self.targets == o.targets and
                self.variables == o.variables and
                self.max_depth == o.max_depth and
                self.max_leaves == o.max_leaves)

    def pdf(self, values: VariableMap) -> float:
        values_ = VariableMap([(var, ContinuousSet(val, val)) for var, val in values.items()])
        pdf = 0
        for leaf in self.apply(values_):
            pdf += leaf.prior * (prod(leaf.distributions[var].pdf(value)
                                      for var, value in values.items()) if values else 1)
        return pdf

    def infer(self,
              query: Union[Dict[Union[Variable, str], Any], VariableMap],
              evidence: Union[Dict[Union[Variable, str], Any], VariableMap] = None,
              fail_on_unsatisfiability: bool = True) -> Result:
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
        :param fail_on_unsatisfiability: whether or not an error is raised in case of unsatisifiable evidence.
        '''
        # querymap = VariableMap()
        # for key, value in query.items():
        #     querymap[key if isinstance(key, Variable) else self.varnames[key]] = value
        query_ = self._prepropress_query(query)
        evidence_ = ifnone(evidence, {}, self._prepropress_query)

        r = Result(query_, evidence_)

        p_q = 0.
        p_e = 0.

        for leaf in self.apply(evidence_):
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

    # noinspection PyProtectedMember
    def posterior(self,
                  variables: List[Union[Variable, str]] = None,
                  evidence: Union[Dict[Union[Variable, str], Any], VariableMap] = None,
                  fail_on_unsatisfiability: bool = True,
                  report_inconsistencies: bool = False) -> PosteriorResult:
        '''

        :param variables:        the query variables of the posterior to be computed
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :param report_inconsistencies:   in case of an ``Unsatisfiability`` error, the exception raise
                                         will contain information about the variable assignments that
                                         provoked the inconsistency.
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        '''
        variables = ifnone(variables, self.variables)
        evidence_ = ifnone(evidence, {}, self._prepropress_query)
        result = PosteriorResult(variables, evidence_)
        variables = [self.varnames[v] if type(v) is str else v for v in variables]

        distributions = defaultdict(list)

        likelihoods = []
        priors = []

        inconsistencies = {}
        for leaf in self.apply(evidence_):
            likelihood = 1
            conflicting_assignment = VariableMap()
            # check if path of candidate leaf is consistent with evidence
            # (i.e. contains evicence variable with *correct* value or does not contain it at all)
            for var in set(evidence_.keys()):
                evidence_set = evidence_[var]
                if var in leaf.path:
                    evidence_set = evidence_set.intersection(leaf.path[var])

                if isinstance(evidence_set, ContinuousSet) and evidence_set.size() == 1:
                    l_var = leaf.distributions[var].pdf(evidence_set.lower)
                    l_var = 1 if np.isinf(l_var) else l_var
                else:
                    l_var = leaf.distributions[var]._p(evidence_set)

                if not l_var:
                    conflicting_assignment[var] = var.domain.value2label(evidence_set)
                    if not report_inconsistencies:
                        break

                likelihood *= ifnot(l_var, 1)

            if conflicting_assignment:
                inconsistencies[conflicting_assignment] = inconsistencies.get(conflicting_assignment, 0) + likelihood
                continue

            likelihoods.append(0 if conflicting_assignment else likelihood)
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
            result.candidate_dists[var].append(distribution)

        weights = [l * p for l, p in zip(likelihoods, priors)]
        try:
            weights = normalized(weights)
        except ValueError:
            if fail_on_unsatisfiability:
                raise Unsatisfiability('Evidence %s is unsatisfiable.' % format_path(evidence_),
                                       reasons=inconsistencies)
            return None
        result.weights = weights

        # initialize all query variables with None, in case dists
        # is empty (i.e. no candidate leaves -> query unsatisfiable)
        result.distributions = VariableMap()

        for var, dists in distributions.items():
            if var.numeric:
                result.distributions[var] = Numeric.merge(dists, weights=weights)
            elif var.symbolic:
                result.distributions[var] = Multinomial.merge(dists, weights=weights)

        return result

    def expectation(self,
                    variables=None,
                    evidence=None,
                    confidence_level=None,
                    fail_on_unsatisfiability=True) -> ExpectationResult:
        '''
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.
        '''
        variables = ifnone([v if isinstance(v, Variable) else self.varnames[v] for v in variables],
                           set(self.variables) - set(evidence))

        posteriors = self.posterior(variables,
                                    evidence,
                                    fail_on_unsatisfiability=fail_on_unsatisfiability)
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
            result.candidates.extend(posteriors.candidates)
            result.weights = posteriors.weights
            result.candidate_dists = posteriors.candidate_dists[var]
            result.distribution = dist
            if var.numeric:
                exp_quantile = dist.cdf.eval(result._res)
                result._lower = dist.ppf.eval((1 - conf_level) / 2)  # max(0., (exp_quantile - conf_level / 2.)))
                result._upper = dist.ppf.eval(1 - (1 - conf_level) / 2)  # min(1., (exp_quantile + conf_level / 2.)))
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

    def _prepropress_query(self,
                           query: Union[VariableMap, Dict[Variable, Any]],
                           remove_none: bool = True) -> VariableMap:
        '''
        Transform a query entered by a user into an internal representation
        that can be further processed.
        '''
        # Transform lists into a numeric interval:
        query_ = VariableMap()
        # parameter of the respective variable:
        for key, arg in query.items():
            if arg is None and remove_none:
                continue
            var = key if isinstance(key, Variable) else self.varnames[key]
            if var.numeric:
                if type(arg) is list:
                    arg = list2interval(arg)
                if isinstance(arg, numbers.Number):
                    val = var.domain.values[arg]
                    # Apply a "blur" to single value evidences, if any blur is set
                    if var.blur:
                        prior = self.priors[var.name]
                        quantile = prior.cdf.functions[max(1, min(len(prior.cdf) - 2,
                                                                  prior.cdf.idx_at(val)))].eval(val)
                        lower = quantile - var.blur / 2
                        upper = quantile + var.blur / 2
                        query_[var] = ContinuousSet(prior.ppf.functions[max(1,
                                                                            min(len(prior.cdf) - 2,
                                                                                prior.ppf.idx_at(lower)))].eval(lower),
                                                    prior.ppf.functions[min(len(prior.ppf) - 2,
                                                                            max(1,
                                                                                prior.ppf.idx_at(upper)))].eval(upper))
                    else:
                        query_[var] = ContinuousSet(val, val)
                elif isinstance(arg, ContinuousSet):
                    query_[var] = ContinuousSet(var.domain.values[arg.lower],
                                                var.domain.values[arg.upper], arg.left, arg.right)
                else:
                    query_[var] = arg
            if var.symbolic:
                # Transform into internal values (symbolic values to their indices):
                if type(arg) is not set:
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

    def c45(self, data, start, end, parent, child_idx, depth) -> None:
        '''
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.

        :param data:        the indices for the training samples used to calculate the gain.
        :param start:       the starting index in the data.
        :param end:         the stopping index in the data.
        :param parent:      the parent node of the current iteration, initially ``None``.
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
            leaf = node = Leaf(idx=len(self.allnodes), parent=parent)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.distribution().fit(data=data,
                                                             rows=self.indices[start:end],
                                                             col=i)
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

        if self.root is None:
            self.root = node

    def __str__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self.pfmt()}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}\n'
                f'{self.pfmt()}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def pfmt(self) -> str:
        '''Return a pretty-format string representation of this JPT.'''
        return self._pfmt(self.root, 0)

    def _pfmt(self, node, indent) -> str:
        return "{}{}\n{}".format(" " * indent,
                                 str(node),
                                 ''.join([self._pfmt(c, indent + 4) for c in node.children])
                                 if isinstance(node, DecisionNode) else '')

    def _preprocess_data(self, data=None, rows=None, columns=None) -> np.ndarray:
        '''
        Transform the input data into an internal representation.
        '''
        if sum(d is not None for d in (data, rows, columns)) > 1:
            raise ValueError('Only either of the three is allowed.')
        elif sum(d is not None for d in (data, rows, columns)) < 1:
            raise ValueError('No data pased.')

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

    def learn(self, data=None, rows=None, columns=None) -> 'JPT':
        '''Fits the ``data`` into a regression tree.

        :param data:    The training examples (assumed in row-shape)
        :type data:     [[str or float or bool]]; (according to `self.variables`)
        :param rows:    The training examples (assumed in row-shape)
        :type rows:     [[str or float or bool]]; (according to `self.variables`)
        :param columns: The training examples (assumed in row-shape)
        :type columns:  [[str or float or bool]]; (according to `self.variables`)
        '''
        # ----------------------------------------------------------------------------------------------------------
        # Check and prepare the data
        _data = self._preprocess_data(data=data, rows=rows, columns=columns)
        if _data.shape[0] < 1:
            raise ValueError('No data for learning.')

        self.indices = np.ones(shape=(_data.shape[0],), dtype=np.int64)
        self.indices[0] = 0
        np.cumsum(self.indices, out=self.indices)

        JPT.logger.info('Data transformation... %d x %d' % _data.shape)

        # ----------------------------------------------------------------------------------------------------------
        # Initialize the internal data structures
        self._reset()

        # ----------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        started = datetime.datetime.now()
        JPT.logger.info('Learning prior distributions...')
        self.priors = {}
        for i, (vname, var) in enumerate(self.varnames.items()):
            self.priors[vname] = var.distribution().fit(data=_data,
                                                        col=i)
        JPT.logger.info('Prior distributions learnt in %s.' % (datetime.datetime.now() - started))

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
        JPT.logger.info('Started learning of %s x %s at %s '
                        'requiring at least %s samples per leaf' % (_data.shape[0],
                                                                    _data.shape[1],
                                                                    started,
                                                                    min_samples_leaf))
        learning = GENERATIVE if self.targets is None else DISCRIMINATIVE
        JPT.logger.info('Learning is %s. ' % learning)
        if learning == DISCRIMINATIVE:
            JPT.logger.info('Target variables (%d): %s\n'
                            'Feature variables (%d): %s' % (len(self.targets),
                                                            ', '.join(mapstr(self.targets)),
                                                            len(self.variables) - len(self.targets),
                                                            ', '.join(
                                                                mapstr(set(self.variables) - set(self.targets)))))
        # build up tree
        self.c45queue.append((_data, 0, _data.shape[0], None, None, 0))
        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        # ----------------------------------------------------------------------------------------------------------
        # Print the statistics

        JPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
        # if logger.level >= 20:
        JPT.logger.debug(self)
        return self

    fit = learn

    @property
    def min_samples_leaf(self):
        return self._min_samples_leaf

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

    def likelihood(self, queries: np.ndarray, durac_scaling=2., min_distances=None) -> np.ndarray:
        """Get the probabilities of a list of worlds. The worlds must be fully assigned with
        single numbers (no intervals).

        :param queries: An array containing the worlds. The shape is (x, len(variables)).
        :type queries: np.array
        :param durac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :type durac_scaling: float
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :type min_distances: Dict[Variable, float]
        Returns: An np.array with shape (x, ) containing the probabilities.
        """
        # create minimal distances for each numeric variable such a senseful metric can be computed if not provided
        if min_distances is None:
            min_distances: Dict[Variable, float] = dict()
            for idx, variable in enumerate(self.variables):
                if variable.numeric:
                    samples = np.unique(queries[:, idx])
                    distances = np.diff(samples)
                    min_distances[variable] = min(distances) if len(distances) > 0 else durac_scaling

        #initialize probabilities
        probabilities = np.zeros(len(queries))

        #for all leaves
        for leaf in self.leaves.values():
            #calculate the samples that are in this leaf
            true_query_indices = np.ones(len(queries))

            for variable, restriction in leaf.path.items():
                idx = self.variables.index(variable)
                column = queries[:, idx]

                if isinstance(variable, NumericVariable):
                    true_indices = ((restriction.lower < column) & (column <= restriction.upper))
                elif isinstance(variable, SymbolicVariable):
                    true_indices = np.isin(column, np.asarray(list(restriction)))
                true_query_indices *= true_indices

            in_leaf_probabilities = np.ones(len(queries))
            for variable, distribution in leaf.distributions.items():
                idx = self.variables.index(variable)

                if isinstance(variable, SymbolicVariable):
                    probs = distribution._params[queries[:, idx].astype(int)]

                elif isinstance(variable, NumericVariable):
                    probs = np.asarray(distribution.pdf.multi_eval(queries[:, idx].copy(order='C')))
                    probs[(probs == float("inf")).nonzero()] = durac_scaling / min_distances[variable]

                in_leaf_probabilities *= probs

            probabilities += (in_leaf_probabilities * leaf.prior * true_query_indices)

        return probabilities

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
        query = {var: list2interval(val) if type(val) in (list, tuple) and var.numeric else val if type(val) in (
        list, tuple) else [val] for var, val in query.items()}
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
        candidates = sorted([leaf for leaf, confs in confs.items() if all(c >= confidence for c in confs.values())],
                            key=lambda l: sum(confs[l].values()), reverse=True)

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

    def conditional_jpt(self, evidence: VariableMap, keep_evidence: bool = False):
        '''
        Apply evidence on a JPT and get a new JPT that represent P(x|evidence).
        The new JPT contains all variables that are not in the evidence and is a 
        full joint probability distribution over those variables.

        :param evidence: A variable Map mapping the observed variables to there observed,
            single values (not intervals)
        :type evidence: ``VariableMap``

        :param keep_evidence: Rather to keep the evidence variables in the new
            JPT or not. If kept, their PDFs are replaced with Durac impulses.
        :type keep_evidence: bool
        '''

        # the new jpt that acts as conditional joint probability distribution
        conditional_jpt: JPT = JPT.from_json(self.to_json())

        if len(evidence) == 0:
            return conditional_jpt

        univisited_nodes = queue.Queue()
        univisited_nodes.put_nowait(conditional_jpt.allnodes[self.root.idx])

        while not univisited_nodes.empty():

            # get the next node to inspect
            current_node: Node = univisited_nodes.get_nowait()

            # if it is a leaf skip this iteration
            if isinstance(current_node, Leaf):
                continue

            # syntax highlighting
            current_node: DecisionNode

            # remember the indices of the nodes that need to get removed
            invalid = []

            # check if the children of the node need to be traversed
            for idx, child in enumerate(current_node.children):

                # traverse consistent children
                if child.consistent_with(evidence):
                    univisited_nodes.put_nowait(child)

                # mark invalid children for removal
                else:
                    invalid = [idx] + invalid

            # remove invalid children from the tree and the children list
            for idx in invalid:
                # get all the indices of the subtree members
                removable_indices = [node.idx for node in
                                     current_node.children[idx].recursive_children()] + \
                                    [current_node.children[idx].idx]

                # for all dead nodes 
                for jdx in removable_indices:
                    # if it is a leaf remove it from the leaves
                    if isinstance(self.allnodes[jdx], Leaf):
                        del conditional_jpt.leaves[jdx]
                    # if it is an innernode remove it from the innernodes
                    else:
                        del conditional_jpt.innernodes[jdx]

                # remove it as child
                del current_node.children[idx]

        # calculate remaining probability mass
        probability_mass = sum(leaf.prior for leaf in conditional_jpt.leaves.values())

        # clean up not needed distributions and redistribute probability mass
        for leaf in conditional_jpt.leaves.values():
            leaf.prior /= probability_mass
            for variable in evidence.keys():
                # TODO: replace with durac impulse if wanted
                if keep_evidence:
                    pass
                else:
                    del leaf.distributions[variable]

        # clean up not needed path restrictions
        for node in conditional_jpt.allnodes.values():
            for variable in evidence.keys():
                if variable in node.path.keys():
                    del node.path[variable]

        # discard unused variables
        if not keep_evidence:
            conditional_jpt._variables = [variable for variable in conditional_jpt.variables
                                          if variable not in evidence.keys()]

        return conditional_jpt

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
        return JPT.from_json(t)
