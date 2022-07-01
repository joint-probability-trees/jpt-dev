:orphan:

:py:mod:`jpt.sequential_jpt`
============================

.. py:module:: jpt.sequential_jpt


Module Contents
---------------

.. py:class:: SequentialJPT(template_variables: List[jpt.variables.Variable], timesteps: int = 2, **kwargs)

   
   Create a JPT template that can be expanded across multiple time dimensions.

   :param template_variables: The template variables of the sequence model. (The variables that are within each
   timestep)
   :type template_variables: [jpt.variables.Variable]
   :param timesteps: The amount of timesteps that are jointly modelled together.
   :type timesteps: int

   .. py:method:: learn(self, data: numpy.ndarray)

      Fit the joint distribution.

      :param data: The training sequences. This needs to be a list of 2d ndarrays
          where each element in the list describes an observed time series. These
          time series was observed contiguously. The inner ndarray describes the
          length of the time series in the first dimension and the observed values
          for the variables (in order of self.template_variables) in the second
          dimension.
      :type data: np.ndarray


   .. py:method:: integrate(self) -> float

      Calculate the overall probability mass that is obtained by expanding the template to one more timestep
      than it is modelled in the template tree.
      For example if timesteps = 2, then the mass for timesteps = 3 is calculated.
      For the probability mass Z it holds that 0 < Z <= 1.


   .. py:method:: fill_shared_dimensions(self)

      Generate distributions for all shared dimensions. Shared dimensions are dimensions that occur in more
      than one factor.


   .. py:method:: likelihood(self, queries: List) -> numpy.ndarray

      Get the probabilities of a list of worlds. The worlds must be fully assigned with
          single numbers (no intervals).

      :param queries: A list containing the sequences. The shape is (num_sequences, num_timesteps, len(variables)).
      :type queries: list
      Returns: An np.array with shape (num_sequences, ) containing the probabilities.


   .. py:method:: infer(self, queries: List[jpt.variables.VariableMap], evidences: List[jpt.variables.VariableMap]) -> float

      Return the probability of a sequence taking values specified in queries given ranges specified in evidences


   .. py:method:: prob_leaf_combination(self, leaf_0: jpt.trees.Leaf, leaf_1: jpt.trees.Leaf, query_0: jpt.variables.VariableMap, query_1: jpt.variables.VariableMap) -> float

      Calculate the probability mass of a query in a product of 2 leaves.
      It is assumed that they are in a sequence context where leaf_0 is the leaf in the predecessing factor and leaf_1
      is the leaf in the succeeding factor.

      @param: leaf_0, the predecessor leaf
      @param: leaf_1, the successor leaf
      @param: query, the query for the variables
      @rtype: float



.. py:function:: integrate_continuous_pdfs(pdf1: jpt.base.quantiles.PiecewiseFunction, pdf2: jpt.base.quantiles.PiecewiseFunction, interval: jpt.base.intervals.ContinuousSet) -> float

   Calculate the volume that is covered by the integral of pdf1 * pdf2 in the range of the interval.
   Those pdfs are distributions of the same continuous random variable.

   :param pdf1: The first probability density function
   :param pdf2: The second probability density function
   :param interval: The integral boundaries


.. py:function:: integrate_continuous_distributions(distribution1: jpt.learning.distributions.Numeric, distribution2: jpt.learning.distributions.Numeric, normalize=False) -> jpt.learning.distributions.Numeric

   Calculate the cdf that is obtained by integrating pdf1(x) * pdf2(x) and normalize it s. t. the joint
   cdf converges to 1 if wanted.

   :param distribution1: The first distribution
   :type distribution1: jpt.learning.distributions.Numeric
   :param distribution2: The second distribution
   :type distribution2: jpt.learning.distributions.Numeric
   :param normalize: Rather to normalize the distribution or not
   :type normalize: bool


.. py:function:: integrate_discrete_distribution(distribution1: jpt.learning.distributions.Multinomial, distribution2: jpt.learning.distributions.Multinomial, normalize=False) -> jpt.learning.distributions.Multinomial

   Calculate the cdf that is obtained by integrating pdf1(x) * pdf2(x) and normalize it s. t. the joint
   cdf converges to 1 if wanted.

   :param distribution1: The first distribution
   :type distribution1: jpt.learning.distributions.Multinomial
   :param distribution2: The second distribution
   :type distribution2: jpt.learning.distributions.Multinomial
   :param normalize: Rather to normalize the distribution or not
   :type normalize: bool


