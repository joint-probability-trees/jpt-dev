How to Visualise a JPT
======================

.. seealso::

    :doc:`notebooks/tutorial_visualization` — the full illustrated
    tutorial with rendered plots for every example on this page.

``pyjpt`` ships with two visualisation backends selectable via an
``engine`` parameter:

* ``'matplotlib'`` — static PNG / SVG / PDF output
* ``'plotly'`` — interactive HTML figures; static export via kaleido

Install the backend(s) you need:

.. code-block:: bash

    pip install pyjpt[matplotlib]
    pip install pyjpt[plotly]          # includes kaleido for static export
    sudo apt-get install graphviz      # required for the tree plot

Plotting the Tree Structure
----------------------------

:py:meth:`~jpt.trees.JPT.plot` renders the decision tree as a Graphviz
SVG.  The distribution mini-plots inside each leaf are drawn by the
selected engine:

.. code-block:: python

    from IPython.display import SVG, display

    path = model.plot(
        title='Iris JPT',
        filename='iris',
        directory='/tmp',
        engine='matplotlib',   # or 'plotly'
        view=False,
    )
    display(SVG(path))

Pass ``plotvars`` to restrict leaf plots to a subset of variables, and
``nodefill`` / ``leaffill`` for custom node colours:

.. code-block:: python

    model.plot(
        filename='iris_petals',
        directory='/tmp',
        engine='matplotlib',
        plotvars=[varnames['petal length (cm)'], varnames['species']],
        nodefill='#d0e8ff',
        leaffill='#ffe8d0',
        view=False,
    )

Plotting Individual Distributions
-----------------------------------

Every distribution object exposes ``dist.plot(engine=...)``.  Pass it a
distribution retrieved from a leaf or from a posterior query:

.. code-block:: python

    # Numeric CDF — Matplotlib returns a Figure
    fig = numeric_dist.plot(
        engine='matplotlib',
        title='Petal length',
        xlabel='petal length (cm)',
        fname='petal_length',
        directory='/tmp',
    )

    # Symbolic bar chart — Plotly returns a go.Figure
    fig = symbolic_dist.plot(
        engine='plotly',
        title='Species posterior',
        horizontal=True,
        color='rgb(15, 21, 110)',
    )
    fig.show()

Accepted options for the Plotly engine:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``color``
     - CSS colour, ``rgb(r,g,b)``, or hex string
   * - ``horizontal``
     - Horizontal bars for symbolic/integer dists
   * - ``max_values``
     - Show only the top-N values
   * - ``alphabet``
     - Sort bars alphabetically instead of by probability
   * - ``fill``
     - Plotly fill style, e.g. ``'tozeroy'``
   * - ``dim``
     - Gaussian 2-D: ``2`` = heatmap, ``3`` = surface

Saving Figures
--------------

**Matplotlib** — pass ``fname`` and ``directory`` to any ``.plot()``
call; the PNG is written automatically.

**Plotly — interactive HTML:**

.. code-block:: python

    dist.plot(engine='plotly', fname='out.html', directory='/tmp')

**Plotly — static image** (PNG, SVG, JPEG, WebP via kaleido):

.. code-block:: python

    dist.plot(engine='plotly', fname='out.svg', directory='/tmp')
    # or directly on the returned figure:
    fig.write_image('/tmp/out.png', scale=2)

Custom Rendering Engine
------------------------

Subclass :py:class:`~jpt.plotting.engines.rendering.DistributionRendering`
and pass the instance as ``engine`` to any ``plot()`` call:

.. code-block:: python

    from jpt.plotting.engines.rendering import DistributionRendering

    class MyRenderer(DistributionRendering):
        def plot_numeric(self, dist, **kwargs):
            ...   # custom rendering logic

    model.plot(engine=MyRenderer(), directory='/tmp', view=False)
