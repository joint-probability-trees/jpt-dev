from typing import List, Dict

import numpy as np
import numpy.lib.stride_tricks
import pgmpy
import pgmpy.factors.discrete
import pgmpy.models
import pgmpy.inference

import networkx
import matplotlib.pyplot as plt

import jpt.trees


class SequentialJPT:
    def __init__(self, template_tree):
        self.template_tree: jpt.trees.JPT = template_tree
        self.transition_model: np.array = None

    def fit(self, sequences: List):
        """ Fits the transition and emission models. """
        data = np.concatenate(sequences)
        self.template_tree.fit(data)

        transition_data = None

        for sequence in sequences:

            # encode the samples to 'leaf space'
            encoded = self.template_tree.encode(sequence)

            # convert to 2 sizes sliding window
            transitions = numpy.lib.stride_tricks.sliding_window_view(encoded, (2,), axis=0)

            # concatenate transitions
            if transition_data is None:
                transition_data = transitions
            else:
                transition_data = np.concatenate((transition_data, transitions))

        # load number of leaves
        num_leaves = len(self.template_tree.leaves)

        # calculate factor values for transition model
        values = np.zeros((num_leaves * num_leaves,))
        for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
            for jdx, leaf_jdx in enumerate(self.template_tree.leaves.keys()):
                state_index = jdx + idx * num_leaves
                count = sum((transition_data[:, 0] == leaf_idx) & (transition_data[:, 1] == leaf_jdx))
                values[state_index] = count/len(transition_data)

        self.transition_model = values

    def preprocess_sequence_map(self, evidence: List[jpt.variables.VariableMap]):
        """ Preprocess a list of variable maps to be used in JPTs. """
        return [self.template_tree._prepropress_query(e,) for e in evidence]

    def ground(self, evidence: List[jpt.variables.VariableMap]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """

        # create factorgraph
        factor_graph = pgmpy.models.FactorGraph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        factor_graph.add_nodes_from(timesteps)

        # create transition factors
        factors = []

        # for each transition
        for idx in range(len(evidence)-1):

            # get the variable names
            state_names = {"t%s" % idx: list(self.template_tree.leaves.keys()),
                           "t%s" % (idx+1): list(self.template_tree.leaves.keys())}

            # create factor with values from transition model
            factor = pgmpy.factors.discrete.DiscreteFactor(list(state_names.keys()),
                                                           [len(self.template_tree.leaves),
                                                            len(self.template_tree.leaves)],
                                                           self.transition_model, state_names)
            factors.append(factor)

        # add factors
        factor_graph.add_factors(*factors)

        # add edges for state variables and transition variables
        for idx, factor in enumerate(factors):
            factor_graph.add_edges_from([("t%s" % idx, factor),
                                         (factor, "t%s" % (idx+1))])

        # create prior factors
        for timestep, e in zip(timesteps, evidence):

            # create values of current variable
            state_names = {timestep: list(self.template_tree.leaves.keys())}

            # apply the evidence
            conditional_jpt = self.template_tree.conditional_jpt(e)

            # create the prior distribution from the conditional tree
            values = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    values[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            factor = pgmpy.factors.discrete.DiscreteFactor([timestep], [len(self.template_tree.leaves)],
                                                           values, state_names)

            # add factor and edge from variable to prior
            factor_graph.add_factors(factor)
            factor_graph.add_edge(timestep, factor)

        return factor_graph

    def mpe(self, evidence):
        raise NotImplementedError("Not yet implemented")

    def probability(self, query, evidence) -> float:
        query = self.preprocess_sequence_map(query)
        evidence = self.preprocess_sequence_map(evidence)

        factor_graph = self.ground(evidence)
        bp = pgmpy.inference.BeliefPropagation(factor_graph)
        return 1.

    def independent_marginals(self, evidence: List[jpt.variables.VariableMap]) -> List[jpt.trees.JPT]:
        """ Return the independent marginal distributions of all variables in this sequence along all
        timesteps.

        @param evidence: The evidence observed in every timesteps. The length of this list determines the length
            of the whole sequence
        """
        # preprocess evidence
        evidence = self.preprocess_sequence_map(evidence)

        # ground factor graph
        factor_graph = self.ground(evidence)

        # create result list
        result = []

        # create belief propagation class
        bp = pgmpy.inference.BeliefPropagation(factor_graph)

        # infer the latent distribution
        latent_distribution: Dict[str, pgmpy.factors.discrete.DiscreteFactor] = \
            bp.query(factor_graph.get_variable_nodes(), joint=False)

        # transform trees
        for name, distribution in sorted(latent_distribution.items()):
            prior = dict(zip(distribution.state_names[name], distribution.values))
            adjusted_tree = self.template_tree.multiply_by_leaf_prior(prior)
            adjusted_tree.plot(directory="/tmp/%s" % name)
            result.append(adjusted_tree)

        return result
