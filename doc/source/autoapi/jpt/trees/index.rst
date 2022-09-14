:py:mod:`jpt.trees`
===================

.. py:module:: jpt.trees

.. autoapi-nested-parse::

   © Copyright 2021, Mareike Picklum, Daniel Nyga.



Module Contents
---------------

.. py:class:: Node(idx: int, parent: Union[None, DecisionNode] = None)

   Wrapper for the nodes of the :class:`jpt.learning.trees.Tree`.

   :param idx:             the identifier of a node
   :type idx:              int
   :param parent:          the parent node
   :type parent:           jpt.learning.trees.Node

   .. py:method:: consistent_with(self, evidence: jpt.variables.VariableMap) -> bool

      Check if the node is consistend with the variable assignments in evidence.

      :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
      :type evidence: VariableMap



.. py:class:: DecisionNode(idx: int, variable: jpt.variables.Variable, parent: DecisionNode = None)



   .. autoapi-inheritance-diagram:: jpt.trees.DecisionNode
      :parts: 1
      :private-bases:

   Represents an inner (decision) node of the the :class:`jpt.learning.trees.Tree`.

   :param idx:             the identifier of a node
   :type idx:              int
   :param variable:   the split feature name
   :type variable:    jpt.variables.Variable

   .. py:method:: consistent_with(self, evidence: jpt.variables.VariableMap) -> bool

      Check if the node is consistend with the variable assignments in evidence.

      :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
      :type evidence: VariableMap



.. py:class:: Leaf(idx: int, parent: Node or None = None, prior=None)



   .. autoapi-inheritance-diagram:: jpt.trees.Leaf
      :parts: 1
      :private-bases:

   Represents a leaf node of the :class:`jpt.learning.trees.Tree`.

   :param idx:             the identifier of a node
   :type idx:              int
   :param parent:          the parent node
   :type parent:           jpt.learning.trees.Node

   .. py:method:: applies(self, query: jpt.variables.VariableMap) -> bool

      Checks whether this leaf is consistent with the given ``query``.


   .. py:method:: consistent_with(self, evidence: jpt.variables.VariableMap) -> bool

      Check if the node is consistend with the variable assignments in evidence.

      :param evidence: A VariableMap that maps to singular values (numeric or symbolic)
      :type evidence: VariableMap



.. py:class:: JPT(variables, targets=None, min_samples_leaf=0.01, min_impurity_improvement=None, max_leaves=None, max_depth=None, variable_dependencies=None)

   Joint Probability Trees.

   Implementation of Joint Probability Tree (JPT) learning. We store multiple distributions
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

   .. py:method:: infer(self, query, evidence=None, fail_on_unsatisfiability=True) -> Result

      For each candidate leaf ``l`` calculate the number of samples in which `query` is true:

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


   .. py:method:: posterior(self, variables, evidence, fail_on_unsatisfiability=True) -> PosteriorResult

      :param variables:        the query variables of the posterior to be computed
      :type variables:         list of jpt.variables.Variable
      :param evidence:    the evidence given for the posterior to be computed
      :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                       likelihood of the evidence is 0.
      :type fail_on_unsatisfiability:  bool
      :return:            jpt.trees.InferenceResult containing distributions, candidates and weights


   .. py:method:: expectation(self, variables=None, evidence=None, confidence_level=None, fail_on_unsatisfiability=True) -> ExpectationResult

      Compute the expected value of all ``variables``. If no ``variables`` are passed,
      it defaults to all variables not passed as ``evidence``.


   .. py:method:: mpe(self, evidence=None, fail_on_unsatisfiability=True) -> MPEResult

      Compute the (conditional) MPE state of the model.


   .. py:method:: _preprocess_query(self, query, transform_values=True, remove_none=True) -> jpt.variables.VariableMap

      Transform a query entered by a user into an internal representation
      that can be further processed.


   .. py:method:: c45(self, data, start, end, parent, child_idx, depth) -> None

      Creates a node in the decision tree according to the C4.5 algorithm on the data identified by
      ``indices``. The created node is put as a child with index ``child_idx`` to the children of
      node ``parent``, if any.

      :param data:        the indices for the training samples used to calculate the gain.
      :param start:       the starting index in the data.
      :param end:         the stopping index in the data.
      :param parent:      the parent node of the current iteration, initially ``None``.
      :param child_idx:   the index of the child in the current iteration.
      :param depth:       the depth of the tree in the current recursion level.


   .. py:method:: pfmt(self) -> str

      Return a pretty-format string representation of this JPT.


   .. py:method:: _preprocess_data(self, data=None, rows=None, columns=None) -> numpy.ndarray

      Transform the input data into an internal representation.


   .. py:method:: learn(self, data=None, rows=None, columns=None) -> JPT

      Fits the ``data`` into a regression tree.

      :param data:    The training examples (assumed in row-shape)
      :type data:     [[str or float or bool]]; (according to `self.variables`)
      :param rows:    The training examples (assumed in row-shape)
      :type rows:     [[str or float or bool]]; (according to `self.variables`)
      :param columns: The training examples (assumed in row-shape)
      :type columns:  [[str or float or bool]]; (according to `self.variables`)


   .. py:method:: likelihood(self, queries: numpy.ndarray, durac_scaling=2.0, min_distances=None) -> numpy.ndarray

      Get the probabilities of a list of worlds. The worlds must be fully assigned with
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


   .. py:method:: reverse(self, query, confidence=0.5) -> List[Tuple[Dict, List[Node]]]

      Determines the leaf nodes that match query best and returns their respective paths to the root node.

      :param query: a mapping from featurenames to either numeric value intervals or an iterable of categorical values
      :type query: dict
      :param confidence:  the confidence level for this MPE inference
      :type confidence: float
      :returns: a mapping from probabilities to lists of matcalo.core.algorithms.JPT.Node (path to root)
      :rtype: dict


   .. py:method:: plot(self, title=None, filename=None, directory='/tmp', plotvars=None, view=True, max_symb_values=10)

      Generates an SVG representation of the generated regression tree.

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


   .. py:method:: pickle(self, fpath) -> None

      Pickles the fitted regression tree to a file at the given location ``fpath``.

      :param fpath: the location for the pickled file
      :type fpath: str


   .. py:method:: load(fpath)
      :staticmethod:

      Loads the pickled regression tree from the file at the given location ``fpath``.

      :param fpath: the location of the pickled file
      :type fpath: str


   .. py:method:: calcnorm(sigma, mu, intervals)
      :staticmethod:

      Computes the CDF for a multivariate normal distribution.

      :param sigma: the standard deviation
      :param mu: the expected value
      :param intervals: the boundaries of the integral
      :type sigma: float
      :type mu: float
      :type intervals: list of matcalo.utils.utils.Interval


   .. py:method:: conditional_jpt(self, evidence: jpt.variables.VariableMap, keep_evidence: bool = False)

      Apply evidence on a JPT and get a new JPT that represent P(x|evidence).
      The new JPT contains all variables that are not in the evidence and is a 
      full joint probability distribution over those variables.

      :param evidence: A variable Map mapping the observed variables to there observed,
          single values (not intervals)
      :type evidence: ``VariableMap``

      :param keep_evidence: Rather to keep the evidence variables in the new
          JPT or not. If kept, their PDFs are replaced with Durac impulses.
      :type keep_evidence: bool


   .. py:method:: save(self, file) -> None

      Write this JPT persistently to disk.

      ``file`` can be either a string or file-like object.


   .. py:method:: load(file)
      :staticmethod:

      Load a JPT from disk.



