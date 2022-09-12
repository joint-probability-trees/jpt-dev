from typing import List, Dict

import numpy as np
import numpy.lib.stride_tricks
import factorgraph

import jpt.trees


class SequentialJPT:
    def __init__(self, template_tree):
        self.template_tree: jpt.trees.JPT = template_tree
        self.transition_model: np.array = None

    def fit(self, sequences: List):
        """ Fits the transition and emission models. """
        data = np.concatenate(sequences)
        self.template_tree.learn(data)

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
        values = np.zeros((num_leaves, num_leaves))
        for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
            for jdx, leaf_jdx in enumerate(self.template_tree.leaves.keys()):
                count = sum((transition_data[:, 0] == leaf_idx) & (transition_data[:, 1] == leaf_jdx))
                values[idx, jdx] = count/len(transition_data)

        self.transition_model = values

    def preprocess_sequence_map(self, evidence: List[jpt.variables.VariableMap]):
        """ Preprocess a list of variable maps to be used in JPTs. """
        return [self.template_tree._preprocess_query(e) for e in evidence]

    def ground(self, evidence: List[jpt.variables.VariableMap]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """

        # create factorgraph
        factor_graph = factorgraph.Graph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        [factor_graph.rv(timestep, len(self.template_tree.leaves)) for timestep in timesteps]

        # for each transition
        for idx in range(len(evidence)-1):

            # get the variable names
            state_names = ["t%s" % idx, "t%s" % (idx+1)]

            # create factor with values from transition model
            factor_graph.factor(state_names, potential=self.transition_model)

        # create prior factors
        for timestep, e in zip(timesteps, evidence):

            # apply the evidence
            conditional_jpt = self.template_tree.conditional_jpt_safe(e)

            # create the prior distribution from the conditional tree
            prior = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    prior[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            factor_graph.factor([timestep], potential=prior)

        return factor_graph

    def mpe(self, evidence):
        raise NotImplementedError("Not yet implemented")

    def probability(self, query, evidence) -> float:
        """
        Calculate the probability of sequence 'query' given sequence 'evidence'.

        @param query: The question
        @param evidence: The evidence
        @return: probability (float)
        """
        # apply evidence
        independent_marginals = self.independent_marginals(evidence)

        # initialize probability
        probability = 1.

        # calculate q|e for every adjusted tree
        for q, adjusted_tree in zip(query, independent_marginals):

            # multiply results
            probability *= adjusted_tree.infer(q).result

        return probability

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

        # Run (loopy) belief propagation (LBP)
        iters, converged = factor_graph.lbp(max_iters=100, progress=True)
        latent_distribution = factor_graph.rv_marginals()

        # transform trees
        for name, distribution in sorted(latent_distribution, key=lambda x: x[0].name):
            prior = dict(zip(self.template_tree.leaves.keys(), distribution))
            adjusted_tree = self.template_tree.replace_leaf_prior(prior)
            result.append(adjusted_tree)

        return result

