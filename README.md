# Joint Probability Trees

Joint Probability Trees (JPTs) are a formalism for learning and reasoning
about joint probability distributions that is tractable for practical
applications. JPTs support both symbolic and subsymbolic variables in a
single hybrid model without relying on prior knowledge about variable
dependencies or families of distributions.

JPT representations build on tree structures that partition the probability
space into relevant subregions elicited from training data, rather than
postulating a rigid dependency model prior to learning. Learning and
inference scale linearly, and the tree structure enables white-box
reasoning about any posterior probability P(Q|E), providing interpretable
explanations for every inference result.

## Installation

Install the core package from PyPI:

```console
pip install pyjpt
```

### Optional dependencies

Depending on your use case, install one or more optional dependency
groups:

```console
pip install pyjpt[matplotlib]   # matplotlib and graphviz plotting
pip install pyjpt[plotly]       # interactive plotly plotting
pip install pyjpt[seq]          # sequential/temporal models
pip install pyjpt[mlflow]       # MLflow experiment tracking
```

Multiple groups can be combined:

```console
pip install pyjpt[matplotlib,mlflow]
```

## Development setup

Clone the repository and install in editable mode with all development
dependencies:

```console
git clone https://github.com/joint-probability-trees/jpt-dev
cd jpt-dev
pip install -e ".[dev]"
```

By default, Cython extensions are compiled during installation. For
development, you can skip pre-compilation and let `pyximport` handle
on-the-fly compilation at runtime instead:

```console
JPT_NO_CYTHON=1 pip install -e ".[dev]"
```

With on-the-fly compilation, changes to `.pyx` files are picked up 
automatically on the next import without requiring a rebuild.

### Running tests

```console
cd test
python -m unittest discover
```

### Building distributions

```console
python -m build            # sdist + wheel
python -m build --sdist    # source distribution only
python -m build --wheel    # wheel only
```

## Documentation

The documentation is hosted on
[Read the Docs](https://joint-probability-trees.readthedocs.io/en/latest/).

### Building the documentation locally

Install the documentation dependencies:

```console
pip install -r doc/requirements.txt
```

Then build the HTML documentation from the `doc/` directory:

```console
cd doc
make html
```

The output is written to `doc/build/html/`. Open `doc/build/html/index.html`
in a browser to view it. The `html` target automatically cleans previous
build artifacts before rebuilding.

## Citation

If you use JPTs in your research, please cite:

```bibtex
@inproceedings{nyga23jpts,
    title     = {{Joint Probability Trees}},
    author    = {Daniel Nyga and Mareike Picklum and
                 Tom Schierenbeck and Michael Beetz},
    year      = {2023},
    booktitle = {arxiv.org},
    url       = {http://arxiv.org/abs/2302.07167}
}
```
