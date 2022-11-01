import datetime
from collections import OrderedDict, ChainMap, deque
from typing import List, Dict, Tuple, Any

import dnutils
import os
import math
import html
import matplotlib.pyplot as plt
from itertools import zip_longest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import jpt.trees
from jpt.variables import VariableMap, Variable, NumericVariable, SymbolicVariable
import numpy as np
import pandas as pd
from graphviz import Digraph

try:
    from .base.quantiles import __module__
    from .base.intervals import __module__
    from .learning.impurity import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import ContinuousSet as Interval, EXC, INC, R, ContinuousSet, RealSet
    from .learning.impurity import PCAImpurity, Impurity
    from .base.constants import plotstyle, orange, green, SYMBOL

# ----------------------------------------------------------------------------------------------------------------------

class PCADecisionNode(jpt.trees.Node):
    """
    Represents an inner (decision) node of the the :class:`jpt.trees.Tree`.
    """

    def __init__(self, idx: int,
                 variables: List[jpt.variables.Variable],
                 weights: np.ndarray,
                 split_value: float,
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
    def __init__(self, idx: int,
                 parent: jpt.trees.Node,
                 prior: float,
                 scaler: StandardScaler,
                 decomposer: PCA):
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

    @property
    def str_node(self) -> str:
        return ""

    @property
    def value(self):
        return self.distributions

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
                                  decomposer=decomposer)

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
                standardized_transformation_split = np.dot(standardized_transformation_eigen, eigen_transformation_split)

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
                data_split_value = float(np.dot(-data_normal_of_split.T, data_origin_of_split))

                # create numeric decision node
                node = PCADecisionNode(idx=len(self.allnodes),
                                       variables=self.numeric_variables,
                                       weights=data_normal_of_split,
                                       split_value=data_split_value,
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
                                    <TD BORDER="1" ALIGN="CENTER" VALIGN="MIDDLE">{',<BR/>'.join([f'{"<B>" + html.escape(v.name) + "</B>"  if self.targets is not None and v in self.targets else html.escape(v.name)}=' + (f'{html.escape(str(dist.expectation()))!s}' if v.symbolic else f'{dist.expectation():.2f}') for v, dist in n.value.items()])}</TD>
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
