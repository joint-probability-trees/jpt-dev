:py:mod:`jpt.learning.distributions`
====================================

.. py:module:: jpt.learning.distributions

.. autoapi-nested-parse::

   © Copyright 2021, Mareike Picklum, Daniel Nyga.



Module Contents
---------------

.. py:class:: Gaussian(mean=None, cov=None, data=None, weights=None)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.Gaussian
      :parts: 1
      :private-bases:

   Extension of :class:`dnutils.stats.Gaussian`

   Creates a new Gaussian distribution.

   :param mean:    the mean of the Gaussian
   :type mean:     float if multivariate else [float] if multivariate
   :param cov:     the covariance of the Gaussian
   :type cov:      float if multivariate else [[float]] if multivariate
   :param data:    if ``mean`` and ``cov`` are not provided, ``data`` may be a data set (matrix) from which the
                   parameters of the distribution are estimated.
   :type data:     [[float]]
   :param weights:  **[optional]** weights for the data points. The weight do not need to be normalized.
   :type weights:  [float]

   .. py:method:: deviation(self, x)

      Computes the deviation of ``x`` in multiples of the standard deviation.

      :param x:
      :type x:
      :returns:


   .. py:method:: sample(self, n)

      Return `n` samples from the distribution subject to the parameters.
      .. warning::
          This method requires the ``numpy`` package installed.


   .. py:method:: linreg(self)

      Compute a 4-tuple ``<m, b, rss, noise>`` of a linear regression represented by this Gaussian.

      :return:    ``m`` - the slope of the line
                  ``b`` - the intercept of the line
                  ``rss`` - the residual sum-of-squares error
                  ``noise`` - the square of the sample correlation coefficient ``r^2``

      References:
          - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#In_least_squares_regression_analysis
          - https://milnepublishing.geneseo.edu/natural-resources-biometrics/chapter/chapter-7-correlation-and-simple-linear-regression/
          - https://en.wikipedia.org/wiki/Residual_sum_of_squares
          - https://en.wikipedia.org/wiki/Explained_sum_of_squares


   .. py:method:: update_all(self, data, weights=None)

      Update the distribution with new data points given in ``data``.


   .. py:method:: estimate(self, data, weights=None)

      Estimate the distribution parameters with subject to the given data points.


   .. py:method:: update(self, x, w=1)

      update the Gaussian distribution with a new data point ``x`` and weight ``w``.


   .. py:method:: retract(self, x, w=1)

      Retract the a data point `x` with eight `w` from the Gaussian distribution.

      In case the data points are being kept in the distribution, it must actually exist and have the right
      weight associated. Otherwise, a ValueError will be raised.


   .. py:method:: kldiv(self, g2)

      Compute the KL-divergence of two multivariate Gaussian distributions.

      :param g1: instance of ``dnutils.Gaussian``
      :param g2: instance of ``dnutils.Gaussian``
      :return:



.. py:class:: MultiVariateGaussian(mean=None, cov=None, data=None, ignore=-6000000)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.MultiVariateGaussian
      :parts: 1
      :private-bases:

   Extension of :class:`dnutils.stats.Gaussian`

   A Multivariate Gaussian distribution that can be incrementally updated with new samples
           

   .. py:method:: cdf(self, intervals)

      Computes the CDF for a multivariate normal distribution.

      :param intervals: the boundaries of the integral
      :type intervals: list of matcalo.utils.utils.Interval


   .. py:method:: mvg(self)
      :property:

      Computes the multivariate Gaussian distribution.
              


   .. py:method:: dim(self)
      :property:

      Returns the dimension of the distribution.
              


   .. py:method:: cov_(self)
      :property:

      Returns the covariance matrix for prettyprinting (precision .2).
              


   .. py:method:: mean_(self)
      :property:

      Returns the mean vector for prettyprinting (precision .2).
              


   .. py:method:: conditional(self, evidence)

      Returns a distribution conditioning on the variables in ``evidence`` following the calculations described
      in `Conditional distributions <https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions>`_,
      i.e., after determining the partitions of :math:`\mu`, i.e. :math:`\mu_{1}` and :math:`\mu_{2}` as well as
      the partitions of :math:`\Sigma`, i.e. :math:`\Sigma_{11}, \Sigma_{12}, \Sigma_{21} \text{ and } \Sigma_{22}`, we
      calculate the multivariate normal :math:`N(\overline\mu,\overline\Sigma)` using

      .. math::
          \overline\mu = \mu_{1} + \Sigma_{12}\Sigma_{22}^{-1}(a-\mu_{2})
          :label: mu

      .. math::
          \overline\Sigma = \Sigma_{11} + \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
          :label: sigma

      :param evidence: the variables the returned distribution conditions on (mapping indices to values or Intervals of values)
      :type evidence: dict


   .. py:method:: plot(self)

      .. highlight:: python
      .. code-block:: python

          import sys
          self.dim==1


   .. py:method:: deviation(self, x)

      Computes the deviation of ``x`` in multiples of the standard deviation.

      :param x:
      :type x:
      :returns:


   .. py:method:: sample(self, n)

      Return `n` samples from the distribution subject to the parameters.
      .. warning::
          This method requires the ``numpy`` package installed.


   .. py:method:: linreg(self)

      Compute a 4-tuple ``<m, b, rss, noise>`` of a linear regression represented by this Gaussian.

      :return:    ``m`` - the slope of the line
                  ``b`` - the intercept of the line
                  ``rss`` - the residual sum-of-squares error
                  ``noise`` - the square of the sample correlation coefficient ``r^2``

      References:
          - https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#In_least_squares_regression_analysis
          - https://milnepublishing.geneseo.edu/natural-resources-biometrics/chapter/chapter-7-correlation-and-simple-linear-regression/
          - https://en.wikipedia.org/wiki/Residual_sum_of_squares
          - https://en.wikipedia.org/wiki/Explained_sum_of_squares


   .. py:method:: update_all(self, data, weights=None)

      Update the distribution with new data points given in ``data``.


   .. py:method:: estimate(self, data, weights=None)

      Estimate the distribution parameters with subject to the given data points.


   .. py:method:: update(self, x, w=1)

      update the Gaussian distribution with a new data point ``x`` and weight ``w``.


   .. py:method:: retract(self, x, w=1)

      Retract the a data point `x` with eight `w` from the Gaussian distribution.

      In case the data points are being kept in the distribution, it must actually exist and have the right
      weight associated. Otherwise, a ValueError will be raised.


   .. py:method:: kldiv(self, g2)

      Compute the KL-divergence of two multivariate Gaussian distributions.

      :param g1: instance of ``dnutils.Gaussian``
      :param g2: instance of ``dnutils.Gaussian``
      :return:



.. py:class:: Distribution

   Abstract supertype of all domains and distributions

   .. py:method:: plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, **kwargs)
      :abstractmethod:

      Generates a plot of the distribution.

      :param title:       the name of the variable this distribution represents
      :type title:        str
      :param fname:       the name of the file
      :type fname:        str
      :param directory:   the directory to store the generated plot files
      :type directory:    str
      :param pdf:         whether to store files as PDF. If false, a png is generated by default
      :type pdf:          bool
      :param view:        whether to display generated plots, default False (only stores files)
      :type view:         bool
      :return:            None



.. py:class:: DataScaler(data=None)

   A numeric data transformation that represents data points in form of a translation
   by their mean and a scaling by their variance. After the transformation, the transformed
   input data have zero mean and unit variance.


.. py:class:: Identity

   Simple identity mapping that mimics the __getitem__ protocol of dicts.


.. py:class:: Numeric(quantile=None)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.Numeric
      :parts: 1
      :private-bases:

   Wrapper class for numeric domains and distributions.

   .. py:method:: plot(self, title=None, fname=None, xlabel='value', directory='/tmp', pdf=False, view=False, **kwargs)

      Generates a plot of the piecewise linear function representing
      the variable's cumulative distribution function

      :param title:       the name of the variable this distribution represents
      :type title:        str
      :param fname:       the name of the file to be stored
      :type fname:        str
      :param xlabel:      the label of the x-axis
      :type xlabel:       str
      :param directory:   the directory to store the generated plot files
      :type directory:    str
      :param pdf:         whether to store files as PDF. If false, a png is generated by default
      :type pdf:          bool
      :param view:        whether to display generated plots, default False (only stores files)
      :type view:         bool
      :return:            None



.. py:class:: ScaledNumeric(quantile=None)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.ScaledNumeric
      :parts: 1
      :private-bases:

   Scaled numeric distribution represented by mean and variance.

   .. py:method:: plot(self, title=None, fname=None, xlabel='value', directory='/tmp', pdf=False, view=False, **kwargs)

      Generates a plot of the piecewise linear function representing
      the variable's cumulative distribution function

      :param title:       the name of the variable this distribution represents
      :type title:        str
      :param fname:       the name of the file to be stored
      :type fname:        str
      :param xlabel:      the label of the x-axis
      :type xlabel:       str
      :param directory:   the directory to store the generated plot files
      :type directory:    str
      :param pdf:         whether to store files as PDF. If false, a png is generated by default
      :type pdf:          bool
      :param view:        whether to display generated plots, default False (only stores files)
      :type view:         bool
      :return:            None



.. py:class:: HashableOrderedDict(*args, **kwargs)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.HashableOrderedDict
      :parts: 1
      :private-bases:

   Ordered dict that can be hashed.

   Initialize self.  See help(type(self)) for accurate signature.

   .. py:method:: clear()

      D.clear() -> None.  Remove all items from D.


   .. py:method:: copy()

      D.copy() -> a shallow copy of D


   .. py:method:: get()

      Return the value for key if key is in the dictionary, else default.


   .. py:method:: items()

      D.items() -> a set-like object providing a view on D's items


   .. py:method:: keys()

      D.keys() -> a set-like object providing a view on D's keys


   .. py:method:: pop()

      D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
      If key is not found, d is returned if given, otherwise KeyError is raised


   .. py:method:: popitem()

      Remove and return a (key, value) pair as a 2-tuple.

      Pairs are returned in LIFO (last-in, first-out) order.
      Raises KeyError if the dict is empty.


   .. py:method:: setdefault()

      Insert key with a value of default if key is not in the dictionary.

      Return the value for key if key is in the dictionary, else default.


   .. py:method:: update()

      D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
      If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
      If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
      In either case, this is followed by: for k in F:  D[k] = F[k]


   .. py:method:: values()

      D.values() -> an object providing a view on D's values



.. py:class:: OrderedDictProxy(*args, **kwargs)

   This is a proxy class that mimics the interface of a regular dict
   without inheriting from dict.


.. py:class:: Multinomial(params=None)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.Multinomial
      :parts: 1
      :private-bases:

   Abstract supertype of all symbolic domains and distributions.

   .. py:method:: items(self)

      Return a list of (probability, label) pairs representing this distribution.


   .. py:method:: sample(self, n)

      Returns ``n`` sample `values` according to their respective probability


   .. py:method:: sample_one(self)

      Returns one sample `value` according to its probability


   .. py:method:: sample_labels(self, n)

      Returns ``n`` sample `labels` according to their respective probability


   .. py:method:: sample_one_label(self)

      Returns one sample `label` according to its probability


   .. py:method:: _expectation(self)

      Returns the value with the highest probability for this variable


   .. py:method:: crop(self, incl_values=None, excl_values=None)

      Compute the posterior of the multinomial distribution.

      ``values`` and ``exclude`` are indices of the values (labels) that are admitted and/or excluded.


   .. py:method:: plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, horizontal=False, max_values=None)

      Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

      :param title:       the name of the variable this distribution represents
      :type title:        str
      :param fname:       the name of the file to be stored
      :type fname:        str
      :param directory:   the directory to store the generated plot files
      :type directory:    str
      :param pdf:         whether to store files as PDF. If false, a png is generated by default
      :type pdf:          bool
      :param view:        whether to display generated plots, default False (only stores files)
      :type view:         bool
      :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
      :type horizontal:   bool
      :param max_values:  maximum number of values to plot
      :type max_values:   int
      :return:            None



.. py:class:: Bool(params=None)



   .. autoapi-inheritance-diagram:: jpt.learning.distributions.Bool
      :parts: 1
      :private-bases:

   Wrapper class for Boolean domains and distributions.

   .. py:method:: items(self)

      Return a list of (probability, label) pairs representing this distribution.


   .. py:method:: sample(self, n)

      Returns ``n`` sample `values` according to their respective probability


   .. py:method:: sample_one(self)

      Returns one sample `value` according to its probability


   .. py:method:: sample_labels(self, n)

      Returns ``n`` sample `labels` according to their respective probability


   .. py:method:: sample_one_label(self)

      Returns one sample `label` according to its probability


   .. py:method:: _expectation(self)

      Returns the value with the highest probability for this variable


   .. py:method:: crop(self, incl_values=None, excl_values=None)

      Compute the posterior of the multinomial distribution.

      ``values`` and ``exclude`` are indices of the values (labels) that are admitted and/or excluded.


   .. py:method:: plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, horizontal=False, max_values=None)

      Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

      :param title:       the name of the variable this distribution represents
      :type title:        str
      :param fname:       the name of the file to be stored
      :type fname:        str
      :param directory:   the directory to store the generated plot files
      :type directory:    str
      :param pdf:         whether to store files as PDF. If false, a png is generated by default
      :type pdf:          bool
      :param view:        whether to display generated plots, default False (only stores files)
      :type view:         bool
      :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
      :type horizontal:   bool
      :param max_values:  maximum number of values to plot
      :type max_values:   int
      :return:            None



