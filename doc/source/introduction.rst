Getting Started
===============

Overview
********
The JPT package brings reliable, expressive, efficient and interpretable joint probability distributions to everyone.

Supported types of inference are:
    - Full evidence queries
    - Marginal queries
    - Conditional queries
    - Expectations
    - Most probable explanations
    - Confidence intervals

Installation
************

Install the package with

``pip install pyjpt``

or clone the repository and install with

``python setup.py install``

An additional requirements is GraphViz. GraphViz can be installed with

``sudo apt install graphviz``

To build the documentation clone the repository and switch to the doc folder

``cd doc``

install the documentation requirements

``pip install -r requirements.txt``

and additionally install pandoc

``sudo apt install pandoc``

After everything is successfully installed build it sphinx style, for example use

``make html``

Currently the package is only supported and tested for Ubuntu 18+, but *theoretically* it should be working
for other operating systems too.