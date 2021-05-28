'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import html
import itertools
import math
import numbers
import operator
import os
import pickle
import pprint
from collections import defaultdict, deque, ChainMap, OrderedDict
import datetime

import numpy as np
from dnutils.stats import stopwatch
from graphviz import Digraph
from matplotlib import style, pyplot as plt

import dnutils
from dnutils import first, out, ifnone
from sklearn.tree import DecisionTreeRegressor

from .learning.distributions import Distribution
from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet

from .learning.impurity import Impurity
from .base.constants import plotstyle, orange, green, SYMBOL
from .base.utils import list2interval, format_path, normalized

style.use(plotstyle)


# ----------------------------------------------------------------------------------------------------------------------
# Global data store to exploit copy-on-write in multiprocessing

import multiprocessing as mp

_manager = mp.Manager()
_data = None
_data_queue = _manager.Queue()
_node_queue = _manager.Queue()
_pool = None


# ----------------------------------------------------------------------------------------------------------------------


class Node:
    '''
    Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx, parent=None, treename=None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param parent:          the parent node
        :type parent:           jpt.learning.trees.Node
        :param treename:        the name of the decision tree
        :type treename:         str
        '''
        self.idx = idx
        self.parent = parent
        self.samples = 0.
        self.treename = treename
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

    def __init__(self, idx, splits, dec_criterion, parent=None, treename=None):
        '''
        :param idx:             the identifier of a node
        :type idx:              int
        :param splits:
        :type splits:
        :param dec_criterion:   the split feature name
        :type dec_criterion:    jpt.variables.Variable
        '''
        self.splits = splits
        self.dec_criterion = dec_criterion
        self.dec_criterion_val = None
        super().__init__(idx, parent=parent, treename=treename)
        self.children = [None] * len(self.splits)

    def set_child(self, idx, node):
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.dec_criterion, self.splits[idx]))

    def str_edge(self, idx):
        return str(self.splits[idx] if self.dec_criterion.numeric else self.dec_criterion.domain.labels[idx])

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
    def __init__(self, idx, parent=None, prior=None, treename=None):
        super().__init__(idx, parent=parent, treename=treename)
        self.distributions = defaultdict(Distribution)
        self.prior = prior

    @property
    def str_node(self):
        return ""

    def applies(self, query):
        '''Checks whether this leaf is consistent with the given ``query``.'''
        path = self.path
        for var in set(query.keys()).intersection(set(path.keys())):
            # out(var, var.str_by_idx(query[var]), var.str_by_idx(path[var]))
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
        return (f'LeafNode<ID: {self.idx}; '
                f'VALUE: {",".join([f"{var.name}: {str(dist)}" for var, dist in self.distributions.items()])}; '
                f'PARENT: {f"DecisionNode<ID: {self.parent.idx}>" if self.parent else None}>')

    def __repr__(self):
        return f'LeafNode<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class JPT:
    '''
    Joint Probability Trees.
    '''

    logger = dnutils.getlogger('/jpt', level=dnutils.DEBUG)

    def __init__(self, variables, name='regtree', min_samples_leaf=1, min_impurity_improvement=None):
        '''Custom wrapper around Joint Probability Tree (JPT) learning. We store multiple distributions
        induced by its training samples in the nodes so we can later make statements
        about the confidence of the prediction.
        has children :class:`~jpt.learning.trees.Node`.

        :param variables:           the variable declarations of the data being processed by this tree
        :type variables:            [jpt.variables.Variable]
        :param name:                the name of the tree
        :type name:                 str
        :param min_samples_leaf:    the minimum number of samples required to generate a leaf node
        :type min_samples_leaf:     int
        '''
        self._variables = tuple(variables)
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.name = name or self.__class__.__name__
        self._numsamples = 0
        self.leaves = {}
        self.innernodes = {}
        self.allnodes = ChainMap(self.innernodes, self.leaves)
        self.root = None
        self.c45queue = _data_queue
        self.priors = {}

    @property
    def variables(self):
        return self._variables

    def c45(self, indices, parent, child_idx):
        '''

        :param indices: the indices for the training samples used to calculate the gain
        :type indices:      [[int]]
        :param parent:      the parent node of the current iteration, initially the root node
        :type parent:       jpt.variables.Variable
        :param child_idx:   the index of the child in the current iteration
        :type child_idx:    int

        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.
        '''
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        # --------------------------------------------------------------------------------------------------------------
        data = _data[indices, :]

        if len(indices) > self.min_samples_leaf:
            impurity = Impurity(self, indices)
            # ft_best_idx, sp_best, max_gain = self.compute_best_split(indices)
            ft_best_idx, sp_best, max_gain = impurity.compute_best_split()

            if ft_best_idx is not None:
                ft_best = self.variables[ft_best_idx]  # if ft_best_idx is not None else None
            else:
                ft_best = None

        else:
            max_gain = 0
            ft_best = None
            sp_best = None
            ft_best_idx = None

        # create decision node splitting on ft_best or leaf node if min_samples_leaf criterion is not met
        if max_gain <= min_impurity_improvement:
            leaf = Leaf(idx=len(self.allnodes),
                        parent=parent,
                        treename=self.name)

            if parent is not None:
                parent.set_child(child_idx, leaf)

            for i, v in enumerate(self.variables):
                leaf.distributions[v] = v.dist(data=data[:, i].T)
                leaf.prior = data.shape[0] / self.data.shape[0]

            leaf.samples = len(indices)

            self.leaves[leaf.idx] = leaf

        else:
            # divide examples into distinct sets for each value of ft_best
            split_data = None  # {val: [] for val in ft_best.domain.values}

            if ft_best.symbolic:
                # CASE SPLIT VARIABLE IS SYMBOLIC
                split_data = [deque() for _ in range(ft_best.domain.n_values)]
                splits = [{i_v} for i_v in range(ft_best.domain.n_values)]

                # split examples into distinct sets for each value of the selected feature
                for i, d in zip(indices, data):
                    split_data[int(d[ft_best_idx])].append(i)

            elif ft_best.numeric:
                # CASE SPLIT VARIABLE IS NUMERIC
                split_data = [deque(), deque()]
                splits = [Interval(np.NINF, sp_best, EXC, INC),
                          Interval(sp_best, np.PINF, EXC, EXC)]

                # split examples into distinct sets for smaller and higher values of the selected feature than the selected split value
                for i, d in zip(indices, data):
                    split_data[d[ft_best_idx] > sp_best].append(i)

            else:
                raise TypeError('Unknown variable type: %s.' % type(ft_best).__name__)

            # ----------------------------------------------------------------------------------------------------------

            node = DecisionNode(idx=len(self.allnodes),
                                splits=splits,
                                dec_criterion=ft_best,
                                parent=parent,
                                treename=self.name)
            if parent is not None:
                parent.set_child(child_idx, node)
            node.samples = len(indices)

            # update path
            self.innernodes[node.idx] = node

            # recurse on sublists
            for i, d_ft in enumerate(split_data):
                if not d_ft:
                    continue
                self.c45queue.append((tuple(d_ft), node, i))

    def __str__(self):
        return (f'{self.__class__.__name__}<{self.name}>:\n'
                f'{"=" * (len(self.name) + 7)}\n\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def __repr__(self):
        return (f'{self.__class__.__name__}<{self.name}>:\n'
                f'{"=" * (len(self.name) + 7)}\n\n'
                f'{self._p(self.root, 0)}\n'
                f'JPT stats: #innernodes = {len(self.innernodes)}, '
                f'#leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n')

    def _p(self, parent, indent):
        if parent is None:
            return "{}None\n".format(" " * indent)
        return "{}{}\n{}".format(" " * indent,
                                 str(parent),
                                 ''.join([self._p(r, indent + 5) for r in ifnone(parent.children, [])]))

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
        if sum(d is not None for d in (data, rows, columns)) != 1:
            raise ValueError('Only either of the three is allowed.')

        if data:
            rows = data

        if type(rows) is list and rows or type(rows) is np.ndarray and rows.shape:  # Transpose the rows
            columns = [[row[i] for row in rows] for i in range(len(self.variables))]

        if type(columns) is list and columns or type(columns) is np.ndarray and columns.shape:
            shape = len(columns[0]), len(columns)
        else:
            raise ValueError('No data given.')

        data = np.ndarray(shape=shape, dtype=np.float32)
        for i, (var, col) in enumerate(zip(self.variables, columns)):
            data[:, i] = col if var.numeric else [var.domain.labels.index(v) for v in col]

        global _data
        _data = data

        # --------------------------------------------------------------------------------------------------------------
        # Determine the prior distributions
        self.priors = {var: var.dist(data=_data[:, i]) for i, var in enumerate(self.variables)}

        # --------------------------------------------------------------------------------------------------------------
        # Start the training

        started = datetime.datetime.now()
        JPT.logger.info('Started learning of %s x %s at %s' % (data.shape[0], data.shape[1], started))
        # build up tree
        self.c45queue.put_nowait((tuple(range(_data.shape[0])), None, None))
        while self.c45queue.qsize():
            self.c45(*self.c45queue.get())

        if self.innernodes:
            self.root = self.innernodes[0]
        elif self.leaves:
            self.root = self.leaves[0]
        else:
            out('NO INNER NODES!', self.innernodes, self.leaves)
            self.root = None

        # --------------------------------------------------------------------------------------------------------------
        # Print the statistics

        JPT.logger.info('Learning took %s' % (datetime.datetime.now() - started))
        # if logger.level >= 20:
        JPT.logger.debug(self)

    def infer(self, query, evidence=None):
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
                p_m *= leaf.distributions[var].p(evidence_val)

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
                    p_m *= leaf.distributions[var].p(query_val)
                p_q += p_m

                r.candidates.append(leaf)
                r.weights.append(p_m)

        r.result = p_q / p_e
        r.weights = [w / p_e for w in r.weights]
        return r

    def expectation(self, variables=None, evidence=None, confidence_level=None):
        '''
        Compute the expected value of all ``variables``. If no ``variables`` are passed,
        it defaults to all variables not passed as ``evidence``.
        '''
        evidence_ = self._prepropress_query(ifnone(evidence, {}))
        conf_level = ifnone(confidence_level, .95)
        variables = ifnone(variables, set(self.variables) - set(evidence_))
        distributions = {var: deque() for var in variables}

        result = {var: ExpectationResult(var, evidence_, confidence_level) for var in variables}

        for leaf in self.apply(evidence_):
            p_m = 1
            for var in set(evidence_.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                elif var.symbolic and var in leaf.path:
                    continue
                p_m *= leaf.distributions[var].p(evidence_val)

            for var in variables:
                distributions[var].append((leaf.distributions[var], p_m))

        posteriors = {var: var.domain.merge([d for d, _ in distributions[var]],
                                            normalized([w for _, w in distributions[var]])) for var in distributions}
        expectations = {var: dist.expectation() for var, dist in posteriors.items()}

        for var, dist in posteriors.items():
            result[var].result = expectations[var]
            if var.numeric:
                result[var].lower = dist.ppf.eval(max(0, (dist.cdf.eval(expectations[var]) - conf_level / 2)))
                result[var].upper = dist.ppf.eval(min(1, (dist.cdf.eval(expectations[var]) + conf_level / 2)))

        return list(result.values())

    def mpe(self, evidence=None):
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
                p_m *= leaf.distributions[var].p(evidence_val)

            for var in self.variables:
                distributions[var].append((leaf.distributions[var], p_m))

        posteriors = {var: var.domain.merge([d for d, _ in distributions[var]],
                                            normalized([w for _, w in distributions[var]])) for var in distributions}

        for var, dist in posteriors.items():
            r.path.update({var: dist.mpe()})
        return r

    def _prepropress_query(self, query):
        '''
        Transform a query entered by a user into an internal representation
        that can be further processed.
        '''
        # Transform lists into a numeric interval:
        query_ = {var: list2interval(val) if type(val) in (list, tuple) else val for var, val in query.items()}
        # Transform single numeric values in to intervals given by the haze
        # parameter of the respective variable:
        for var, val in list(query_.items()):
            if var.numeric and isinstance(val, numbers.Number):
                prior = self.priors[var]
                quantile = prior.cdf.eval(val)
                query_[var] = ContinuousSet(prior.ppf.eval(max(0, quantile - var.haze / 2)),
                                            prior.ppf.eval(min(1, quantile + var.haze / 2)))
        # Transform into internal values (symbolic values to their indices):
        query_ = {var: val if var.numeric else var.domain.labels.index(val) for var, val in query_.items()}
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
        query_ = {var: val if var.numeric else set(var.domain.labels.index(v) for v in val) for var, val in query.items()}
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

    def plot(self, filename=None, directory='/tmp', plotvars=None, view=True):
        '''Generates an SVG representation of the generated regression tree.

        :param filename: the name of the JPT (will also be used as filename; extension will be added automatically)
        :type filename: str
        :param directory: the location to save the SVG file to
        :type directory: str
        :param plotvars: the variables to be plotted in the graph
        :type plotvars: <jpt.variables.Variable>
        :param view: whether the generated SVG file will be opened automatically
        :type view: bool
        '''
        if plotvars == None:
            plotvars = []

        if not os.path.exists(directory):
            os.makedirs(directory)

        dot = Digraph(format='svg', name=filename or self.name,
                      directory=directory,
                      filename=f'{filename or self.name}')

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
                    n.distributions[pvar].plot(title=html.escape(pvar.name), fname=img_name, directory=directory, view=False)
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
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{html.escape(v.name)}=' + (f'{html.escape(str(v.domain.labels[dist.expectation()]))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items()])}</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{f"{land}".join([html.escape(var.str(val)) for var, val in n.path.items()])}</TD>
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
        JPT.logger.info(f'Saving rendered image to {os.path.join(directory, filename or self.name)}.svg')

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


class Result:

    def __init__(self, query, evidence, res=None, cand=None, w=None):
        self.query = query
        self.evidence = evidence
        self._res = ifnone(res, [])
        self._cand = ifnone(cand, [])
        self._w = ifnone(w, [])

    def __str__(self):
        return self.format_result()

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
        self.lower = lower
        self.upper = upper

    def format_result(self):
        left = 'E(%s%s%s; %s = %.3f)' % (self.query,
                                         ' | ' if self.evidence else '',
                                         ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
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
        self.path={}

    def format_result(self):
        return f'MPE({self.evidence}) = {format_path(self.path)}'
