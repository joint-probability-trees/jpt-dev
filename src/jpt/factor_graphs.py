import jpt.variables
import jpt.trees
from typing import List, Dict, Set
import networkx
from jpt.learning.distributions import SymbolicType
import itertools
import matplotlib.pyplot as plt


class Name:
    """ Represent a wrapper around names, where prefixes are usable"""
    def __init__(self, prefix: str, name: str):
        self.prefix = prefix
        self.name = name

    def __str__(self):
        return self.prefix + "." + self.name

    def __repr__(self):
        return str(self)


class JPTFactor:
    """ Represent a Factor containing a JPT. """
    def __init__(self, name: str, distribution: jpt.trees.JPT):
        self.name: str = name
        self.distribution = distribution

        leaf_variable = jpt.variables.SymbolicVariable("Leaf", SymbolicType("Leaf", distribution.leaves.keys()))
        self.variables: List[jpt.variables.Variable] = list(distribution.variables) + [leaf_variable]


class LatentFactor:
    """ Represent a Factor connecting the latent "Leaf" variables from JPTs."""
    def __init__(self, name: str, factors: List[JPTFactor]):
        self.name = name
        self.factors = factors


class FactorGraph:
    """ Implementation of FactorGraphs utilizing JPTs"""
    def __init__(self, factors: List[LatentFactor or JPTFactor]):

        self.factors: List[LatentFactor or JPTFactor] = factors
        self.jpt_factors = [f for f in factors if isinstance(f, JPTFactor)]
        self.latent_factors = [f for f in factors if isinstance(f, LatentFactor)]

        self.graph: networkx.Graph = networkx.Graph()
        [self.graph.add_node(f.name, factor=False) for f in self.jpt_factors]

        for factor in self.latent_factors:
            self.graph.add_edges_from(itertools.product([f.name for f in factor.factors],
                                                        [f.name for f in factor.factors]))

