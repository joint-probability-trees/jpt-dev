:py:mod:`jpt.variables`
=======================

.. py:module:: jpt.variables

.. autoapi-nested-parse::

   © Copyright 2021, Mareike Picklum, Daniel Nyga.



Module Contents
---------------

.. py:class:: Variable(name, domain, min_impurity_improvement=None)

   Abstract class for a variable name along with its distribution class type.

   :param name:    name of the variable
   :type name:     str
   :param domain:  the class type (not an instance!) of the represented Distribution
   :type domain:   class type of jpt.learning.distributions.Distribution
   :param min_impurity_improvement:
   :type min_impurity_improvement: float

   .. py:method:: dist(self, params=None, data=None, rows=None, col=None)

      Create and return a new instance of the distribution type attached to this variable.

      Either the distribution ``params`` can be passed or the ``data`` the distribution parameters
      are to be determined from.


   .. py:method:: _dist(self, params)

      Create and return a new instance of the distribution associated with this type of variable.



.. py:class:: NumericVariable(name, domain=Numeric, min_impurity_improvement=None, haze=None, max_std=None, precision=None)



   .. autoapi-inheritance-diagram:: jpt.variables.NumericVariable
      :parts: 1
      :private-bases:

   Represents a continuous variable.

   :param name:    name of the variable
   :type name:     str
   :param domain:  the class type (not an instance!) of the represented Distribution
   :type domain:   class type of jpt.learning.distributions.Distribution
   :param min_impurity_improvement:
   :type min_impurity_improvement: float

   .. py:method:: dist(self, params=None, data=None, rows=None, col=None)

      Create and return a new instance of the distribution type attached to this variable.

      Either the distribution ``params`` can be passed or the ``data`` the distribution parameters
      are to be determined from.


   .. py:method:: _dist(self, params)

      Create and return a new instance of the distribution associated with this type of variable.



.. py:class:: SymbolicVariable(name, domain, min_impurity_improvement=None)



   .. autoapi-inheritance-diagram:: jpt.variables.SymbolicVariable
      :parts: 1
      :private-bases:

   Represents a symbolic variable.

   :param name:    name of the variable
   :type name:     str
   :param domain:  the class type (not an instance!) of the represented Distribution
   :type domain:   class type of jpt.learning.distributions.Distribution
   :param min_impurity_improvement:
   :type min_impurity_improvement: float

   .. py:method:: dist(self, params=None, data=None, rows=None, col=None)

      Create and return a new instance of the distribution type attached to this variable.

      Either the distribution ``params`` can be passed or the ``data`` the distribution parameters
      are to be determined from.


   .. py:method:: _dist(self, params)

      Create and return a new instance of the distribution associated with this type of variable.



.. py:function:: infer_from_dataframe(df, scale_numeric_types=True, min_impurity_improvement=None, haze=None, max_std=None, precision=None)

   Creates the ``Variable`` instances from column types in a Pandas or Spark data frame.

   :param df:  the data frame object to generate the variables from.
   :type df:   ``pandas.DataFrame``

   :param scale_numeric_types: Whether of not to use scaled types for the numeric variables.
   :type scale_numeric_types: bool

   :param min_impurity_improvement:   the minimum imrovement that a split must induce to be acceptable.
   :type min_impurity_improvement: ``float``

   :param haze:
   :type haze:         ``float``

   :param max_std:
   :type max_std:      ``float``

   :param precision:
   :type precision:    ``float`` in ``[0, 1]``


.. py:class:: VariableMap(data: List[Tuple] = None)

   Convenience class for mapping a ``Variable`` object to anything else. This special map, however,
   supports accessing the image set both by the variable object instance itself _and_ its name.

   ``data`` may be an iterable of (variable, value) pairs.


