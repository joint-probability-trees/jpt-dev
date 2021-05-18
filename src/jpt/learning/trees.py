'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import html
import math
import operator
import os
import pickle
from collections import defaultdict, deque, ChainMap, OrderedDict
import datetime

import numpy as np
from dnutils.stats import stopwatch
from graphviz import Digraph
from matplotlib import style
from sklearn.metrics import mean_squared_error

import dnutils
from dnutils import first, out, ifnone, stop
from sklearn.tree import DecisionTreeRegressor

from .distributions import Distribution
from intervals import ContinuousSet as Interval, EXC, INC, R

from .impurity import Impurity
from ..constants import plotstyle, orange, green
from ..utils import rel_entropy, list2interval

logger = dnutils.getlogger(name='TreeLogger', level=dnutils.DEBUG)

style.use(plotstyle)


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
        return ' ^ '.join([var.str(val, fmt='logic') for var, val in self.path.items()])

    # def set_trainingssamples(self, examples, variables):
    #     '''
    #     :param examples:    tba
    #     :type examples:     tba
    #     :param variables:        tba
    #     :type variables:         tba
    #     '''
    #     self.samples = len(examples)
    #     for i, v in enumerate(variables):
    #         self.distributions[v] = v.dist().set_data(examples[:, i])

    @property
    def str_node(self):
        return self.dec_criterion.name

    def __str__(self):
        return (f'Node<ID: {self.idx}; '
                f'CRITERION: {self.dec_criterion.__class__.__name__}; '
                f'PARENT: {f"Node<ID: {self.parent.idx}>" if self.parent else None}; '
                f'#CHILDREN: {len(self.children)}>')

    def __repr__(self):
        return f'Node<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class DecisionNode(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''

    def __init__(self, idx, splits, dec_criterion, parent=None, treename=None):
        '''
        :param threshold:       the threshold at which the data for the decision criterion are separated
        :type threshold:        float
        :param dec_criterion:   the split feature name
        :type dec_criterion:    jpt.variables.Variable
        '''
        self.splits = splits
        self.dec_criterion = dec_criterion
        self.dec_criterion_val = None
        self.splits = splits
        super().__init__(idx, parent=parent, treename=treename)
        self.children = [None] * len(self.splits)

    def set_child(self, idx, node):
        self.children[idx] = node
        node._path = list(self._path)
        node._path.append((self.dec_criterion, self.splits[idx]))

    def str_edge(self, idx):
        return str(self.dec_criterion.str(self.splits[idx], fmt='logic'))


# ----------------------------------------------------------------------------------------------------------------------


class Leaf(Node):
    '''
    Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.
    '''
    def __init__(self, idx, parent=None, treename=None):
        super().__init__(idx, parent=parent, treename=treename)
        self.distributions = defaultdict(Distribution)

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
        return (f'Leaf<ID: {self.idx}; '
                f'VALUE: {",".join([f"{var.name}: {str(dist)}" for var, dist in self.distributions.items()])}; '
                f'PARENT: {f"Node<ID: {self.parent.idx}>" if self.parent else None}>')

    def __repr__(self):
        return f'Leaf<{self.idx}> object at {hex(id(self))}'


# ----------------------------------------------------------------------------------------------------------------------


class JPT:

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
        self.data = None
        self.c45queue = deque()

    @property
    def variables(self):
        return self._variables

    def impurity(self, indices, tgt):
        r'''Calculate the mean squared error for the data set `xmpls`, i.e.

        .. math::
            MSE = \frac{1}{n} · \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        '''
        if not indices:
            return 0.

        # assuming all Examples have identical indices for their targets
        tgt_idx = self._variables.index(tgt)

        if tgt.symbolic:  # case categorical targets
            tgts_plain = np.array([self.data[i, tgt_idx] for i in indices])
            # count occurrences for each value of target tgt and determine their probability
            _, counts = np.unique(tgts_plain, return_counts=True)
            probs = counts / tgts_plain.shape[0]
            # calculate actual impurity target tgt
            return rel_entropy(probs)

        elif tgt.numeric:  # case numeric targets
            tgts_sklearn = np.array([self.data[i, tgt_idx] for i in indices])
            # calculate mean for target tgt
            ft_mean = np.mean(tgts_sklearn)
            # calculate normalized mean squared error for target tgt
            sqerr = mean_squared_error(tgts_sklearn, [ft_mean] * len(tgts_sklearn))

            # calculate actual impurity for target tgt
            return sqerr

    def gains(self, indices, ft, tgt):
        r'''Calculate the impurity for the data set after selection of feature `ft`, i.e.

        :param xmpls:
        :type xmpls:
        :param ft:
        :type ft:
        :param tgt:

        .. math::
            R(ft) = \sum_{i=1}^{v}\frac{p_i + n_i}{p+n} · I(\frac{p_i}{p_i + n_i}, \frac{n_i}{p_i + n_i})

        '''

        if not indices:
            return {None: 0.}

        impurity = self.impurity(indices, tgt)

        # assuming all Examples have identical indices for their features
        ft_idx = self.variables.index(ft)

        # sort by ft values for easier dataset split
        indices = sorted(indices, key=lambda i: self.data[i, ft_idx])
        fts_plain = np.array([self.data[i, ft_idx] for i in indices])
        distinct, counts = np.unique(fts_plain, return_counts=True)

        if self.variables[ft_idx].symbolic:
            probs = counts / len(indices)
            # divide examples into distinct sets for each value of ft [[Example]]
            partition = [[i for i in indices if self.data[i, ft_idx] == val] for val in distinct]
            if not all(len(p) >= self.min_samples_leaf for p in partition):
                return {None: 0}
            # determine overall impurity after selection of ft by multiplying probability for each feature value with its impurity,
            gain = {None: impurity - sum([p * self.impurity(subset, tgt) for p, subset in zip(probs, partition)])}

        if self.variables[ft_idx].numeric:
            # determine split points of dataset
            opts = [(a + b) / 2 for a, b in zip(distinct[:-1], distinct[1:])] if len(distinct) > 1 else distinct

            # divide examples into distinct sets for the left (ft <= value) and right (ft > value) of ft [[Example]]
            examples_ft = [(spp,
                            [i for i in indices if self.data[i, ft_idx] <= spp],
                            [i for i in indices if self.data[i, ft_idx] > spp]) for spp in opts]

            # the squared errors for the left and right datasets of each split value
            gain = {mv: impurity - (self.impurity(left, tgt) * len(left) / len(indices) +
                                    self.impurity(right, tgt) * len(right)) / len(indices) if len(left) >= self.min_samples_leaf and len(right) >= self.min_samples_leaf else 0
                    for mv, left, right in examples_ft}

        return gain

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

    def c45(self, indices, parent, child_idx):
        '''
        Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
        ``indices``. The created node is put as a child with index ``child_idx`` to the children of
        node ``parent``, if any.
        '''
        # --------------------------------------------------------------------------------------------------------------
        min_impurity_improvement = ifnone(self.min_impurity_improvement, 0)
        # --------------------------------------------------------------------------------------------------------------
        data = self.data[indices, :]

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

    def _p(self, root, indent):
        return "{}{}\n{}".format(" " * indent,
                                 str(root),
                                 ''.join([self._p(r, indent + 5) for r in ifnone(root.children, [])]))

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

        self.data = data

        # --------------------------------------------------------------------------------------------------------------
        # Start the training

        started = datetime.datetime.now()
        logger.info('Started learning of %s x %s at %s' % (data.shape[0], data.shape[1], started))
        # build up tree
        self.c45queue.append((list(range(len(data))), None, None))
        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        if self.innernodes:
            self.root = self.innernodes[0]
        elif self.leaves:
            self.root = self.leaves[0]
        else:
            self.root = None

        # --------------------------------------------------------------------------------------------------------------
        # Print the statistics

        logger.info('Learning took %s' % (datetime.datetime.now() - started))
        if logger.level >= 20:
            out(self)

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
        evidence = ifnone(evidence, {})
        query = {var: list2interval(val) if type(val) in (list, tuple) else val for var, val in query.items()}
        evidence = {var: list2interval(val) if type(val) in (list, tuple) else val for var, val in evidence.items()}
        # Transform into internal values (symbolic values to their indices)
        evidence_ = {var: val if var.numeric else var.domain.labels.index(val) for var, val in evidence.items()}
        query_ = {var: val if var.numeric else var.domain.labels.index(val) for var, val in query.items()}

        r = Result(query_, evidence_)

        p_q = 0.
        p_e = 0.

        for leaf in self.apply(evidence_):
            # out(leaf.format_path(), 'applies', ' ^ '.join([var.str_by_idx(val) for var, val in evidence_.items()]))
            p_m = 1
            for var in set(evidence_.keys()).intersection(leaf.path.keys()):
                evidence_val = evidence_[var]
                if var.numeric and var in leaf.path:
                    evidence_val = evidence_val.intersection(leaf.path[var])
                p_m *= leaf.distributions[var].p(evidence_val)

            w = leaf.samples / self.root.samples
            p_m *= w
            p_e += p_m

            if leaf.applies(query_):
                for var in set(query_.keys()).intersection(leaf.path.keys()):
                    query_val = query_[var]
                    if var.numeric:
                        query_val = query_val.intersection(leaf.path[var])
                    p_m *= leaf.distributions[var].p(query_val)
                p_q += p_m

                r.candidates.append(leaf)
                r.weights.append(p_m)

        r.result = p_q / p_e
        return r

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

    def reverse(self, query):
        '''Determines the leaf nodes that match query best and returns their respective paths to the root node.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :type query: dict
        :returns: a mapping from probabilities to lists of matcalo.core.algorithms.JPT.Node (path to root)
        :rtype: dict
        '''

        # if none of the target variables is present in the query, there is no match possible
        if set(query.keys()).isdisjoint(set(self.targets)):
            return []

        numerics = []
        cat = []
        for i, tgt in enumerate(self.targets):
            if issubclass(self._variables[tgt], Distribution):
                lower = np.NINF
                upper = np.PINF

                if tgt in query:
                    t_ = query[tgt]
                    # value is a function, e.g. min, meaning that we are looking for the minimal value of this variable
                    if callable(t_):
                        numerics.append(t_)
                    else:
                        numerics.append(Interval(t_.lower if isinstance(t_, Interval) else t_[0], t_.upper if isinstance(t_, Interval) else t_[1]))
                else:
                    numerics.append(Interval(lower, upper))
            else:
                # TODO: categorical query vars given as set (or iterable), then replace query by either one-element set or find all possible values for cat variables
                val = None
                if tgt in query:
                    if hasattr(query[tgt], '__iter__'):
                        val = query[tgt]
                    else:
                        val = {query[tgt]}
                else:
                    # if no value constraints for this categorical feature is set, set all possible values of this feature
                    # as allowed
                    val = self._catvalues[tgt]
                cat.append(val)

        # find the leaf (or the leaves) that matches the query best
        sims = defaultdict(float)

        for k, l in self.leaves.items():
            # TODO: FIX THIS TO USE DISTRIBUTIONS
            sims[l] += l.gbf.query(cat, numerics)

        candidates = sorted(sims, key=lambda l: sims[l], reverse=True)

        # for the chosen candidate determine the path to the root
        paths = []
        for c in candidates:
            p = []
            curcand = c
            while curcand is not None:
                p.append([curcand, [s.identifier for s in curcand.samples]])
                curcand = curcand.parent
            paths.append([sims[c], p])

        return paths

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
                    img_name = f'{pvar.name}-{idx}'
                    n.distributions[pvar].plot(title=pvar.name, fname=img_name, directory=directory, view=False)
                    img += (f'''{"<TR>" if i % rc == 0 else ""}
                                        <TD><IMG SCALE="TRUE" SRC="{os.path.join(directory, f"{img_name}.png")}"/></TD>
                                {"</TR>" if i % rc == rc-1 or i == len(plotvars) - 1 else ""}
                                ''')

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

            land = '<BR/>\u2227'
            element = ' \u2208 '

            # content for node labels
            nodelabel = f'''<TR>
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{"Leaf" if isinstance(n, Leaf) else "Node"} #{n.idx}</B><BR/>{html.escape(n.str_node)}</TD>
                            </TR>'''

            if isinstance(n, Leaf):
                nodelabel = f'''{nodelabel}{imgs}
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{len(n.samples) if isinstance(n.samples, list) else n.samples}</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>Expectation:</B></TD>
g                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{v.name}=' + (f'{v.domain.labels[dist.expectation()]!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items()])}</TD>
                                </TR>
                                <TR>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                                    <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{f" {land} ".join([(var.str(val) if var.symbolic else f'{var.str(val)}') for var, val in n.path.items()])}</TD>
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
                dot.edge(str(n.idx), str(c.idx), label=n.str_edge(i))

        # show graph
        logger.info(f'Saving rendered image to {os.path.join(directory, filename or self.name)}.svg')
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
                logger.info(f'Loading JPT {os.path.abspath(fpath)}')
                return pickle.load(f)
            except ModuleNotFoundError:
                logger.error(f'Could not load file {os.path.abspath(fpath)}')
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

    def sklearn_tree(self):
        assert self.data is not None, 'call learn() first to preprocess the data.'

        tree = DecisionTreeRegressor(min_samples_leaf=self.min_samples_leaf,
                                     min_impurity_decrease=self.min_impurity_improvement,
                                     random_state=0)
        with stopwatch('/sklearn/decisiontree'):
            tree.fit(self.data, self.data)


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
        return ('P(%s%s%s) = %.3f %%' % (', '.join([var.str(val, fmt="logic") for var, val in self.query.items()]),
                                         ' | ' if self.evidence else '',
                                         ', '.join([var.str(val, fmt='logic') for var, val in self.evidence.items()]),
                                         self.result * 100))

    def explain(self):
        result = self.format_result()
        for weight, leaf in sorted(zip(self.weights, self.candidates), key=operator.itemgetter(0), reverse=True):
            result += '.3f %%: %s\n' % leaf.format_path()
        return result
