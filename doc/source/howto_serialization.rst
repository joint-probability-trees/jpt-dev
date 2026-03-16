How to Save and Load a JPT
==========================

Trained models can be persisted to disk in two formats:

* **pickle** (default) — fastest, most compact, Python-only
* **json** — human-readable, portable across Python versions

Saving with Pickle (default)
-----------------------------

.. code-block:: python

    model.save('my_model.pkl')
    # equivalent:
    model.save('my_model.pkl', protocol='pickle')

Loading with Pickle
-------------------

.. code-block:: python

    from jpt.trees import JPT

    model = JPT.load('my_model.pkl')

Saving and Loading with JSON
-----------------------------

JSON serialisation preserves the full model in a text file that can
be inspected or version-controlled alongside your code:

.. code-block:: python

    model.save('my_model.json', protocol='json')

    model2 = JPT.load('my_model.json', protocol='json')

Serialising to a Bytes Buffer
------------------------------

:py:meth:`~jpt.trees.JPT.dumps` / :py:meth:`~jpt.trees.JPT.loads`
work with in-memory buffers, useful for caching or network transfer:

.. code-block:: python

    blob = model.dumps(protocol='json')     # → bytes
    model3 = JPT.loads(blob, protocol='json')

Choosing a Format
-----------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 15 50

   * - Format
     - File size
     - Speed
     - Use when …
   * - pickle
     - small
     - fast
     - local caching, pipelines
   * - json
     - larger
     - slower
     - portability, inspection, version control

.. note::

    Pickle files are tied to the Python and ``pyjpt`` version that
    created them.  Use JSON for long-term storage or cross-version
    compatibility.

.. seealso::

    :doc:`notebooks/tutorial_learning` — shows a full fit-save-load
    round-trip.
