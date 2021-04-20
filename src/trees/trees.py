import html
import os
import pickle
import traceback
from collections import defaultdict

import numpy as np
from graphviz import Digraph
from matplotlib import style

import dnutils
from dnutils import edict, first, out
from dnutils.stats import Gaussian
from .intervals import Interval, EXC, INC
from .probs import GenericBayesFoo
from .example import SymbolicFeature, BooleanFeature, Example, NumericFeature
from ..constants import plotstyle, orange, green

logger = dnutils.getlogger(name='TreeLogger', level=dnutils.ERROR)

style.use(plotstyle)


class Node:
    """Represents an internal decision node of the regression tree."""

    def __init__(self, idx, threshold, leftchild, rightchild, dec_criterion, parent=None, treename=None):
        """Each Node stores a Gaussian distribution that allows for making a statement about the confidence of
        a prediction.

        :param idx:             the identifier of a node
        :type idx:              int
        :param threshold:       the threshold at which the data for the decision criterion are separated
        :type threshold:        float
        :param leftchild        the left subtree of this node, None if Leaf
        :type leftchild:        matcalo.core.algorithms.Tree.{Node,Leaf}
        :param rightchild       the right subtree of this node, None if Leaf
        :type rightchild:       matcalo.core.algorithms.Tree.{Node,Leaf}
        :param dec_criterion:   the split feature name
        :type dec_criterion:    str
        :param parent:          the parent node
        :type parent:           matcalo.core.algorithms.Tree.Node
        :param treename:        the name of the decision tree
        :type treename:         str
        """
        self.idx = idx
        self.threshold = threshold
        self.dec_criterion = dec_criterion
        self.leftchild = leftchild
        self.rightchild = rightchild
        self.parent = parent
        self.path = edict({})
        self.training_dist = Gaussian()  # TODO: was CALOGaussian before, change back if something breaks!
        self.samples = []
        self.treename = treename

    def add_trainingsample(self, example):
        self.training_dist.update(example.tsklearn())
        self.samples.append(example)

    @property
    def value(self):
        return self.training_dist.mean


class Leaf(Node):
    """Represents a leaf node of the tree."""

    def __init__(self, idx, parent, treename=None):
        Node.__init__(self, idx, None, None, None, None, parent=parent, treename=treename)


class Tree:
    """Custom wrapper around regression trees. We store a Gaussian distribution
    induced by its training samples in the nodes so we can later make statements
    about the confidence of the prediction."""

    def __init__(self, regressor, min_samples_leaf=5, name=None):
        self.features = None
        self.targets = None
        self.min_samples_leaf = min_samples_leaf
        self.regressor = regressor
        self.samples = 0
        self.leaves = {}
        self.innernodes = {}
        self.allnodes = {}
        self.root = None
        self._X = None
        self._T = None
        self.name = name or self.__class__.__name__

    def learn(self, data=None):
        """Fits the ``data`` into a tree.

        :param data: The training examples containing features and targets
        :type data: matcalo.utils.utils.Example
        """
        if self.features is None:
            self.features = [f.name for f in first(data).x]
        if self.targets is None:
            self.targets = [f.name for f in first(data).t]
        self._X = np.array([d.xsklearn() for d in data])
        self._T = np.array([d.tsklearn() for d in data])
        try:
            self.regressor.fit(self._X, self._T)
        except FloatingPointError:
            traceback.print_exc()

    def predict(self, sample):
        raise NotImplemented

    def plot(self):
        raise NotImplemented

    def pickle(self, fpath):
        """Pickles the fitted regression tree to a file at the given location ``fpath``.

        :param fpath: the location for the pickled file
        :type fpath: str
        """
        with open(os.path.abspath(fpath), 'wb') as f:
            pickle.dump(self, f)

    def reverse(self, query):
        raise NotImplemented

    @staticmethod
    def load(fpath):
        """Loads the pickled regression tree from the file at the given location ``fpath``.

        :param fpath: the location of the pickled file
        :type fpath: str
        """
        with open(os.path.abspath(fpath), 'rb') as f:
            try:
                logger.info(f'Loading Tree {os.path.abspath(fpath)}')
                return pickle.load(f)
            except ModuleNotFoundError:
                logger.error(f'Could not load file {os.path.abspath(fpath)}')
                raise Exception(f'Could not load file {os.path.abspath(fpath)}. Probably deprecated.')


class StructNode(Node):
    """Represents an internal decision node of the matcalo.core.algorithms.StructRegTree."""

    def __init__(self, idx, threshold, dec_criterion, parent=None, treename=None):
        """
        `gbf' represents the distributions in this node that allows for reasoning over the data
        :param idx:
        :param threshold:
        :param dec_criterion:
        :param parent:
        :param treename:
        """
        super(StructNode, self).__init__(idx, threshold, None, None, dec_criterion, parent=parent, treename=treename)
        self.dec_criterion = dec_criterion
        self.dec_criterion_val = None
        self.t_dec_criterion = None
        self.children = []
        self.gbf = GenericBayesFoo(leaf=idx)

    def add_trainingsample(self, example):
        self.gbf.add_trainingssample(example)

        # store sample for debugging
        self.samples.append(example)

    @property
    def str_node(self):
        return f'{self.dec_criterion}' if self.t_dec_criterion in [SymbolicFeature,
                                                                   BooleanFeature] else f'{self.dec_criterion}<={self.dec_criterion_val:.2f}'

    @property
    def str_edge(self):
        return f'{self.threshold}'

    @property
    def value(self):
        return self.gbf.training_dists

    @property
    def value_str(self):
        return ",\n".join(
            [f'{tuple([f"{sfname}={sfval}" for sfname, sfval in zip(self.gbf.symbolics, k)])}: {str(v)}' for k, v in
             self.gbf.training_dists.items()])

    def __str__(self):
        return f'StructNode<id: {self.idx}; criterion: {self.dec_criterion}; parent: {f"StructNode< id:{self.parent.idx}>" if self.parent else None}; #children: {len(self.children)}>'

    def __repr__(self):
        return f'StructNode<{self.idx}> object at {hex(id(self))}'


class StructLeaf(StructNode):
    """Represents a leaf node of the matcalo.core.algorithms.StructRegTree."""

    def __init__(self, idx, parent, treename=None):
        StructNode.__init__(self, idx, None, None, parent=parent, treename=treename)

    @property
    def str_node(self):
        return ""

    @property
    def str_edge(self):
        return f'{self.threshold}'

    def __str__(self):
        return f'StructLeaf<ID: {self.idx}; THR: {self.threshold}; VALUE: {self.gbf}; parent: {f"StructNode<id: {self.parent.idx}>" if self.parent else None}>'

    def __repr__(self):
        return f'StructLeaf<{self.idx}> object at {hex(id(self))}'


class StructRegTree(Tree):
    """Custom wrapper around regression trees. We store multiple multivariate Gaussian distribution
    induced by its training samples in the nodes so we can later make statements
    about the confidence of the prediction.
    """

    def __init__(self, min_samples_leaf=1, name=None, ignore='?'):
        self.ignore = ignore
        self.t_types = []
        self.f_types = []
        self._catvalues = defaultdict(set)
        Tree.__init__(self, None, min_samples_leaf=min_samples_leaf, name=name)

    def gains(self, xmpls, ft, tgt):
        r"""Calculate the impurity for the data set after selection of feature `ft`, i.e.

        .. math::
            R(ft) = \sum_{i=1}^{v}\frac{p_i + n_i}{p+n} · I(\frac{p_i}{p_i + n_i}, \frac{n_i}{p_i + n_i})

        """

        if not xmpls:
            return {None: 0.}

        impurity = Example.impurity(xmpls, tgt)

        # assuming all Examples have identical indices for their features
        ft_idx = xmpls[0].features.index(ft)

        # sort by ft values for easier dataset split
        xmpls = sorted(xmpls, key=lambda xmpl: xmpl.x[ft_idx].value)
        fts_plain = np.array([xmpl.xplain() for xmpl in xmpls]).T[ft_idx]
        distinct = sorted(list(set(fts_plain)))

        if self.f_types[ft] in (SymbolicFeature, BooleanFeature):
            # count occurrences for each value of given feature and determine their probability; [(ftval, count)]
            probs_ft = [(distinctfeatval, float(list(fts_plain).count(distinctfeatval)) / len(fts_plain)) for distinctfeatval in distinct]

            # divide examples into distinct sets for each value of ft [[Example]]
            datasets_ft = [[e for e in xmpls if ft[0] in (e.x[ft_idx].value, str(e.x[ft_idx].value))] for ft in probs_ft]

            # determine overall impurity after selection of ft by multiplying probability for each feature value with its impurity,
            r_a = {None: impurity - sum([ft[1] * Example.impurity(ds, tgt) for ft, ds in zip(probs_ft, datasets_ft)])}
        else:
            distinct = np.array(distinct, dtype=np.float32)
            # determine split points of dataset
            opts = [(a + b) / 2 for a, b in zip(sorted(distinct)[:-1], sorted(distinct)[1:])] if len(distinct) > 1 else distinct

            # divide examples into distinct sets for the left (ft <= value) and right (ft > value) of ft [[Example]]
            examples_ft = [(spp,
                            [e for e in xmpls if e.x[ft_idx].value <= spp],
                            [e for e in xmpls if e.x[ft_idx].value > spp]) for spp in opts]

            # the squared errors for the left and right datasets of each split value
            r_a = {mv: impurity - (Example.impurity(left, tgt) * len(left) + Example.impurity(right, tgt) * len(right))/len(xmpls) for mv, left, right in examples_ft}
        return r_a

    def c45(self, data, parent, ft_idx=None, tr=0.8):
        if not data:
            logger.warning('No data left. Returning parent', parent)
            return parent

        # calculate gains for each feature/target combination and normalize over targets
        gains_tgt = defaultdict(dict)
        for tgt in self.targets:
            maxval = 0.
            for ft in self.features:
                gains_tgt[tgt][ft] = self.gains(data, ft, tgt)
                maxval = max(maxval, *gains_tgt[tgt][ft].values())

            # normalize gains for comparability
            gains_tgt[tgt] = {ft: {v: g/maxval if maxval > 0. else 0 for v, g in gains_tgt[tgt][ft].items()} for ft in self.features}

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

        ft_best_idx = self.features.index(ft_best)

        # BASE CASE 1: all samples belong to the same class --> create leaf node for these targets
        # BASE CASE 2: None of the features provides any information gain --> create leaf node higher up in tree using expected value of the class
        # BASE CASE 3: Instance of previously-unseen class encountered --> create a decision node higher up the tree using the expected value
        if max_gain < tr:
            node = StructLeaf(idx=len(self.allnodes), parent=parent, treename=self.name)

            for s in data:
                node.add_trainingsample(s)

            # inherit path from parent
            node.path = node.parent.path.copy()

            # as datasets have been split before, take an arbitrary example and look up the value
            node.threshold = None if parent is None else data[0].x[self.features.index(parent.dec_criterion)].value if parent.t_dec_criterion in (SymbolicFeature, BooleanFeature) else data[0].x[self.features.index(parent.dec_criterion)].value <= parent.dec_criterion_val

            # update path
            self._update_path(node, data)

            self.allnodes[node.idx] = node

            if logger.level >= 20:
                if max_gain < tr:
                    logger.warning('BASE CASE 2: None of the features provides a high enough information gain. Returning leaf node', node.idx)
                else:
                    logger.warning('BASE CASE 3: Instance of previously-unseen class encountered. Returning leaf node', node.idx)

            return node

        # divide examples into distinct sets for each value of ft_best
        split_data = defaultdict(list)
        if self.f_types[ft_best] in (SymbolicFeature, BooleanFeature):
            # CASE SPLIT VARIABLE IS SYMBOLIC
            thresh = None if parent is None else data[0].x[self.features.index(parent.dec_criterion)].value if parent.t_dec_criterion in (SymbolicFeature, BooleanFeature) else data[0].x[self.features.index(parent.dec_criterion)].value <= parent.dec_criterion_val

            # split examples into distinct sets for each value of the selected feature
            for d in data:
                split_data[d.x[ft_best_idx].value].append(d)

            dec_criterion_val = list(split_data.keys())
        else:
            # CASE SPLIT VARIABLE IS NUMERIC
            dec_criterion_val = sp_best
            thresh = None if parent is None else data[0].x[self.features.index(parent.dec_criterion)].value if parent.t_dec_criterion in (SymbolicFeature, BooleanFeature) else data[0].x[self.features.index(parent.dec_criterion)].value <= parent.dec_criterion_val

            # split examples into distinct sets for smaller and higher values of the selected feature than the selected split value
            for d in data:
                split_data[f'{ft_best}{"<=" if d.x[ft_best_idx].value <= sp_best else ">"}{sp_best}'].append(d)

        # create decision node splitting on ft_best or leaf node if min_samples_leaf criterion is not met
        if any([len(d) < self.min_samples_leaf for d in split_data.values()]):
            node = StructLeaf(idx=len(self.allnodes), parent=parent, treename=self.name)
            node.threshold = None if parent is None else data[0].x[ft_idx].value if parent.t_dec_criterion in (SymbolicFeature, BooleanFeature) else data[0].x[ft_idx].value <= parent.dec_criterion_val

            for s in data:
                node.add_trainingsample(s)

            # update path
            self._update_path(node, data)
            self.allnodes[node.idx] = node

        else:
            node = StructNode(idx=len(self.allnodes), threshold=thresh, dec_criterion=ft_best, parent=parent, treename=self.name)
            node.t_dec_criterion = self.f_types[node.dec_criterion]
            node.dec_criterion_val = dec_criterion_val

            # update path
            self._update_path(node, data)
            self.allnodes[node.idx] = node

            # recurse on sublists
            for _, d_ft in split_data.items():
                node.children.append(self.c45(d_ft, node, ft_idx=ft_best_idx))

        return node

    def _update_path(self, node, data):
        if node.parent is None:
            return

        node.path = node.parent.path.copy()
        if node.parent.t_dec_criterion in [SymbolicFeature, BooleanFeature]:
            node.path[node.parent.dec_criterion] = node.threshold
        else:
            low = all([d.x[d.features.index(node.parent.dec_criterion)].value <= node.parent.dec_criterion_val for d in data])
            i = Interval(-np.Inf if low else node.parent.dec_criterion_val, node.parent.dec_criterion_val if low else np.Inf, left=EXC, right=INC if low else EXC)
            if node.parent.dec_criterion in node.path:
                node.path[node.parent.dec_criterion] = node.path[node.parent.dec_criterion].intersection(i)
            else:
                node.path[node.parent.dec_criterion] = i

    def __str__(self):
        return f'Tree<{self.name}>:\n{"="*(len(self.name)+7)}\n\n{self._p(self.root, 0)}\nTree stats: #innernodes = {len(self.innernodes)}, #leaves = {len(self.leaves)} ({len(self.allnodes)} total)\n'

    def _p(self, root, indent):
        return "{}{}\n{}".format(" " * indent, str(root), ''.join([self._p(r, indent + 5) for r in root.children]) if hasattr(root, 'children') else 'None')

    def learn(self, data=None, tr=0.8):
        """Fits the ``data`` into a regression tree.

        :param data: The training examples containing features and targets
        :type data: list of matcalo.utils.example.Example
        """

        # assuming that features and targets in dataset have identical structure
        if self.features is None:
            self.features = data[0].features
        if self.targets is None:
            self.targets = data[0].targets

        # determine the types (Numeric or Symbolic) of the features and targets
        self.f_types = {f: t for f, t in zip(data[0].features, data[0].ft_types)}
        self.t_types = {f: t for f, t in zip(data[0].targets, data[0].tgt_types)}

        # determine all possible values for each categorical variable
        for d in data:
            for var in d.x + d.t:
                if self.f_types.get(var.name) == NumericFeature or self.t_types.get(var.name) == NumericFeature: continue
                self._catvalues[var.name].add(var.value)

        self.c45(data, None, ft_idx=None, tr=tr)

        # build up tree
        self.innernodes = {n.idx: n for i, n in self.allnodes.items() if type(n) == StructNode}
        self.leaves = {n.idx: n for i, n in self.allnodes.items() if type(n) == StructLeaf}
        if self.innernodes:
            self.root = self.innernodes[0]
        elif self.leaves:
            self.root = self.leaves[0]
        else:
            self.root = None

        if logger.level >= 20:
            out(self)

    def predict(self, sample):
        """Predicts value of ``sample`` for the learned tree.

        :param sample: if dict: {featname: featvalue} or {featname: Interval.fromstring([x,y])}
        :type sample: dict or matcalo.utils.example.Example
        :returns: a mapping of target name to target value
        :rtype: dict
        """
        if isinstance(sample, dict):
            sample = Example(x=[ftype(value=self.sample(sample, feat), name=feat) for feat, ftype in self.f_types.items()])
        cand = self.apply(sample)
        if cand is not None:
            return cand.idx, cand.value  # TODO: do not return cand.idx. Only for debugging!
        else:
            return {}

    def predict_us(self, sample):
        """predict underspecified: only certain variables along the path are given; multiple answers are therefore
        possible"""
        cands = self.apply_us(sample)
        if cands:
            return {c: dict([(t, v) for t, v in zip(self.targets, c.value)]) for c in cands}
        else:
            return {}

    def apply(self, sample):
        # if the sample doesn't match the features of the tree, there is no valid prediction possible
        if set(sample.features).isdisjoint(set(self.features)):
            return None

        # find the leaf (or the leaves) that have each variable either
        # - not occur in the path to this node OR
        # - match the boolean/symbolic value in the path OR
        # - lie in the interval of the numeric value in the path
        # -> return leaf that matches query
        for k, l in self.leaves.items():
            if all([True if feat.name not in l.path else l.path.get(feat.name) == feat.value if ftype in (SymbolicFeature, BooleanFeature) else l.path.get(feat.name).contains(feat.value) for feat, ftype in zip(sample.x, sample.ft_types)]):
                return l

        # if no leaf matches query, return None
        return None

    def apply_us(self, sample):
        # if the sample doesn't match the features of the tree, there is no valid prediction possible
        if set(sample.keys()).isdisjoint(set(self.features)):
            return []

        # find the leaf (or the leaves) that have each variable either
        # - not occur in the path to this node OR
        # - match the boolean/symbolic value in the path OR
        # - lie in the interval of the numeric value in the path
        sims = []
        for k, l in self.leaves.items():
            if all([True if feat not in l.path else l.path.get(feat) == val if ftype in (SymbolicFeature, BooleanFeature) else l.path.get(feat).contains(val) for feat, val, ftype in zip(sample.keys(), sample.values(), [self.f_types[k] for k in sample.keys()])]):
                sims.append(l)

        # return (possibly empty) list of matching leaves
        return sims

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
        """Determines the leaf nodes that match query best and returns their respective paths to the root node.

        :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
        :type query: dict
        :returns: a mapping from probabilities to lists of matcalo.core.algorithms.Tree.Node (path to root)
        :rtype: dict
        """

        # if none of the target variables is present in the query, there is no match possible
        if set(query.keys()).isdisjoint(set(self.targets)):
            return []

        numerics = []
        cat = []
        for i, tgt in enumerate(self.targets):
            if self.t_types[tgt] == NumericFeature:
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

    @staticmethod
    def calcnorm(sigma, mu, intervals):
        """Computes the CDF for a multivariate normal distribution.

        :param sigma: the standard deviation
        :param mu: the expected value
        :param intervals: the boundaries of the integral
        :type sigma: float
        :type mu: float
        :type intervals: list of matcalo.utils.utils.Interval
        """
        from scipy.stats import mvn
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], mu, sigma))

    @property
    def fitted(self):
        return hasattr(self.regressor, 'tree_')

    def plot(self, filename='regtree', directory=None, view=True):
        """Generates an SVG representation of the generated regression tree.

        :param filename: the name of the Tree (will also be used as filename; extension will be added automatically)
        :type filename: str
        :param directory: the location to save the SVG file to
        :type directory: str
        :param view: whether the generated SVG file will be opened automatically
        :type view: bool
        """

        if directory is None:
            directory = os.path.join('..', 'plots')
        dot = Digraph(format='svg', name=self.name,
                      directory=directory,
                      filename=filename)

        # create nodes
        sep = ",<BR/>"
        for idx, n in self.allnodes.items():

            # plot and save distributions for later use in tree plot
            if isinstance(n, StructLeaf):
                n.gbf.plot(directory=directory, view=False)

            # content for node labels
            nodelabel = f"""<TR>
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><B>{"Leaf" if isinstance(n, StructLeaf) else "Node"} #{n.idx}</B><BR/>{html.escape(n.str_node)}</TD>
                            </TR>"""

            # content for leaf labels
            numsamples = len(n.samples) if isinstance(n.samples, list) else n.samples
            samples = f"""<TR>
                              <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>samples:</B></TD>
                              <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{sep.join([s.identifier for s in n.samples])}</TD>
                          </TR>"""

            leaflabel = f"""{nodelabel}
                            <TR>
                                <TD ALIGN="CENTER" VALIGN="MIDDLE" COLSPAN="2"><IMG SCALE="TRUE" SRC="{os.path.join(directory, f'{n.gbf.name}.png')}"/></TD>
                            </TR>
                            <TR>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>#samples:</B></TD>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{len(n.samples) if isinstance(n.samples, list) else n.samples}</TD>
                            </TR>
                            {samples if numsamples < 5 else ""}
                            <TR>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE"><B>value:</B></TD>
                                <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{n.gbf.pred_val(precision=2)}</TD>
                            </TR>
                            <TR>
                                <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE"><B>path:</B></TD>
                                <TD BORDER="1" ROWSPAN="{len(n.path)}" ALIGN="CENTER" VALIGN="MIDDLE">{sep.join([f"{k}: {v}" for k, v in n.path.items()])}</TD>
                            </TR>
                            """

            # stitch together
            lbl = f"""<<TABLE ALIGN="CENTER" VALIGN="MIDDLE" BORDER="0" CELLBORDER="0" CELLSPACING="0">
                            {leaflabel if isinstance(n, StructLeaf) else nodelabel}
                      </TABLE>>"""

            if isinstance(n, StructLeaf):
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
        for idx, n in self.allnodes.items():
            for c in n.children:
                dot.edge(str(n.idx), str(c.idx), label=c.str_edge)

        # show graph
        logger.info(f'Saving rendered image to {os.path.join(directory, filename)}')
        dot.render(view=view, cleanup=False)

    @staticmethod
    def calcnorm(sigma, mu, intervals):
        """Computes the CDF for a multivariate normal distribution.

        :param sigma: the standard deviation
        :param mu: the expected value
        :param intervals: the boundaries of the integral
        :type sigma: float
        :type mu: float
        :type intervals: list of matcalo.utils.utils.Interval
        """
        from scipy.stats import mvn
        return first(mvn.mvnun([x.lower for x in intervals], [x.upper for x in intervals], mu, sigma))
