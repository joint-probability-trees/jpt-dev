:orphan:

:py:mod:`jpt.base.utils`
========================

.. py:module:: jpt.base.utils


Module Contents
---------------

.. py:exception:: Unsatisfiability



   .. autoapi-inheritance-diagram:: jpt.base.utils.Unsatisfiability
      :parts: 1
      :private-bases:

   Error that is raised on logically unsatisfiable inferences.

   Initialize self.  See help(type(self)) for accurate signature.

   .. py:method:: with_traceback()

      Exception.with_traceback(tb) --
      set self.__traceback__ to tb and return self.



.. py:function:: pairwise(seq)

   Iterate over all consecutive pairs in ``seq``.


.. py:function:: mapstr(seq, fmt=None, limit=None)

   Convert the sequence ``seq`` into a list of strings by applying ``str`` to each of its elements.


.. py:function:: to_json(obj)

   Recursively generate a JSON representation of the object ``obj``.

   Non-natively supported data types must provide a ``to_json()`` method that
   returns a representation that is in turn jsonifiable.


.. py:function:: format_path(path)

   Returns a readable string representation of a conjunction of variable assignments,
   given by the dictionary ``path``.


.. py:function:: entropy(p)

   Compute the entropy of the multinomial probability distribution ``p``.
   :param p:   the probabilities
   :type p:    [float] or {str:float}
   :return:


.. py:function:: max_entropy(n)

   Compute the maximal entropy that a multinomial random variable with ``n`` states can have,
   i.e. the entropy value assuming a uniform distribution over the values.
   :param p:
   :return:


.. py:function:: rel_entropy(p)

   Compute the entropy of the multinomial probability distribution ``p`` normalized
   by the maximal entropy that a multinomial distribution of the dimensionality of ``p``
   can have.
   :type p: distribution


.. py:function:: gini(p)

   Compute the Gini impurity for the distribution ``p``.


.. py:function:: classproperty(func)

   This decorator allows to define class properties in the same way as normal object properties.

   https://stackoverflow.com/questions/5189699/how-to-make-a-class-property


.. py:function:: list2interval(l)

   Converts a list representation of an interval to an instance of type


.. py:function:: normalized(dist, identity_on_zeros=False, allow_neg=False)

   Returns a modification of ``seq`` in which all elements sum to 1, but keep their proportions.


.. py:class:: CSVDialect



   .. autoapi-inheritance-diagram:: jpt.base.utils.CSVDialect
      :parts: 1
      :private-bases:

   Describe the usual properties of Excel-generated CSV files.


