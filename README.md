# Joint Probability Trees

Joint Probability Trees (short JPTs) are a formalism for learning of and reasoning about joint probability
distributions, which is tractable for practical applications. JPTs support both symbolic and subsymbolic variables in a single
hybrid model, and they do not rely on prior knowledge about variable dependencies or families of distributions.
JPT representations build on tree structures that partition the problem space into relevant subregions that are elicited
from the training data instead of postulating a rigid dependency model prior to learning. Learning and reasoning scale
linearly in JPTs, and the tree structure allows white-box reasoning about any posterior probability :math:`P(Q\mid E)`,
such that interpretable explanations can be provided for any inference result. This documentation introduces the
code base of the ``pyjpt`` library, which is implemented in Python/Cython, and showcases the practical
applicability of JPTs in high-dimensional heterogeneous probability spaces, making it
a promising alternative to classic probabilistic graphical models.

## Install using pip
Install the pyjpt package:
```console
pip install pyjpt
```

## Documentation
The documentation is hosted via readthedocs.org [here](https://joint-probability-trees.readthedocs.io/en/latest/).