|project| Command-line Tools
============================

|project| comes with a set of convenience tools for learning, reasoning and testing. The tools can be started from the
command line.

|project|
---------

Start the tool from the command line e.g. ::

    $ matcalo --query '{"Radius": "[0.664, 3.19]", "Electronegativity": "[0.9,2.4]"}' --trees alloys.tree,alloysv2.tree -v 1


Parameters
~~~~~~~~~~

There are a couple parameters that have significant influence on the results of the query:

* **verbose** will tell the algorithms to display nicely formatted
  progress information during runtime, a summary of all parameters passed
  to the algorithms, the final learnt model or nicely formatted inference results.
  The verbosity level can have values from {0..5}. Default is 1.

* **mongo** is the key of the mongo settings in the config file

* **strategy** The strategy for the inference (0=Depth-first search, 1=Breadth-first search). Default is 1.

* **threshold** The threshold for the query results. Default is 0.

* **trees** The trees to query, e.g. 'alloys.tree,alloysv2.tree'

* **query** The query

* **interactive** Starts |project| in the webbrowser.

matcalolearn
------------

Start the tool from the command line e.g. ::

    $ matcalolearn -h

.. note::
    This page is under construction


Parameters
~~~~~~~~~~

There are a couple parameters that both the query tool and the learning
tool have in common, which can be set with the respective checkboxes:

* **param1** is mapped to ``param1=True`` argument for algorithm ``XY``.

matcalotest
-----------

Start the tool from the command line e.g. ::

    $ matcalotest -h

.. note::
    This page is under construction


Parameters
~~~~~~~~~~

There are a couple parameters that both the query tool and the learning
tool have in common, which can be set with the respective checkboxes:

* **param1** is mapped to ``param1=True`` argument for algorithm ``XY``.

