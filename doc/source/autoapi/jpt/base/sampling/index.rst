:py:mod:`jpt.base.sampling`
===========================

.. py:module:: jpt.base.sampling

.. autoapi-nested-parse::

   © Copyright 2021, Mareike Picklum, Daniel Nyga.



Module Contents
---------------

.. py:class:: RouletteWheelSampler(elements, weights, normalize=False)

   Roulette wheel proportional sampler

   .. py:method:: index(self, x)

      Returns the index of the element, which corresponds to the "roulette" field, ``x`` falls into.
      :param x:
      :return:


   .. py:method:: sample(self, n=1)

      Sample ``n`` values from the the roulette wheel.
      :param n:
      :return:


   .. py:method:: samplei(self, n=1)

      Same as ``sample()``, but returns a list of indices of selected elements.
      :param n:
      :return:



.. py:function:: wchoice(population, weights)

   Choose one element from the ``population`` proportionally to their ``weights``.


.. py:function:: wchoiced(dist)

   Choose from the dict ``dist`` one element from key set proportionally to the weights given as values


.. py:function:: wchoicei(population, weights)

   Choose one element from the ``population`` proportionally to their ``weights`` and return its index.


.. py:function:: wsample(population, weights, k)

   Obtain a sample of the ``population`` of length ``k``.

   The probability of each element in ``population`` to be sampled is proportional to its weight in ``weights``
   vector. ``len(population)`` must equal to ``len(weights)``.

   :param population:
   :param weights:
   :param k:
   :return:


.. py:function:: wsamplei(population, weights, k)

   Equivalent to ``wsample``, but returns a tuples of the elements chosen and their index in the population.
   :param population:
   :param weights:
   :param k:
   :return:


