import datetime
from operator import itemgetter
from typing import List, Dict

import numpy as np
import numpy.lib.stride_tricks
import fglib
import factorgraph
from dnutils import out, getlogger

from jpt import JPT
from jpt.base.errors import Unsatisfiability
from jpt.base.utils import normalized
from jpt.variables import LabelAssignment, VariableAssignment, VariableMap, Variable


class SequentialJPT:

    logger = getlogger('/jpt/seq')

    def __init__(self, template_tree: JPT):
        self.template_tree: JPT = template_tree
        self.transition_model: np.array or None = None

    def fit(self, sequences: List[np.ndarray], timesteps: int = 2):
        """ Fits the transition and emission models. The emission model is fitted
         with respect to the variables in the next timestep, but it doesn't use them.

         @param sequences: The sequences to learn from
         @param timesteps: The timesteps to jointly model (minimum of 2 required) """

        # extract copies of variables for the expanded tree
        expanded_variables = [var.copy() for var in self.template_tree.variables]

        # keep track of which dimensions to include in the training process
        data_indices = list(range(len(expanded_variables)))

        # extract target indices from the
        if self.template_tree.targets:
            target_indices = [idx for idx, var in enumerate(self.template_tree.variables)
                              if var in self.template_tree.targets]
        else:
            target_indices = list(range(len(self.template_tree.variables)))

        # create variables for jointly modelled timesteps
        for timestep in range(1, timesteps):
            expanded_variables += [
                self._shift_variable_to_timestep(self.template_tree.variables[idx], timestep)
                for idx in target_indices
            ]

            # append targets to data index
            data_indices += [idx + timestep * len(self.template_tree.variables) for idx in target_indices]

        # create expanded tree
        expanded_template_tree = JPT(
            variables=expanded_variables,
            targets=expanded_variables[len(self.template_tree.variables):],
            min_samples_leaf=self.template_tree.min_samples_leaf,
            min_impurity_improvement=self.template_tree.min_impurity_improvement,
            max_leaves=self.template_tree.max_leaves,
            max_depth=self.template_tree.max_depth
        )

        # initialize data
        data = None

        # for every sequence
        for sequence in sequences:

            # unfold the timesteps such that they are expanded to jointly model all timesteps
            unfolded = np.lib.stride_tricks.sliding_window_view(sequence, (timesteps, ), axis=0)
            unfolded = unfolded.reshape((len(unfolded), len(self.template_tree.variables) * timesteps), order="F")

            unfolded = unfolded[:, data_indices]

            # append or set data
            if data is None:
                data = unfolded
            else:
                data = np.concatenate((data, unfolded), axis=0)

        # fit joint timesteps tree
        expanded_template_tree.learn(data=data)

        # create template tree from learnt joint tree
        self.template_tree.root = expanded_template_tree.root
        self.template_tree.innernodes = expanded_template_tree.innernodes
        for idx, leaf in expanded_template_tree.leaves.items():
            leaf.distributions = VariableMap([(v, d) for v, d in leaf.distributions.items()
                                             if v.name in self.template_tree.varnames.keys()])
            self.template_tree.leaves[idx] = leaf
        self.template_tree.priors = VariableMap({
                v.name: prior for v, prior in expanded_template_tree.priors.items()
                if v.name in self.template_tree.varnames
            },
            variables=self.template_tree.variables
        )

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

    def _shift_variable_to_timestep(self,
                                    variable: Variable,
                                    timestep: int = 1) -> Variable:
        """ Create a new variable where the name is shifted by +n and the domain remains the same.

        @param variable: The variable to shift
        @param timestep: timestep in the future, i.e. timestep >= 1
        """
        variable_ = variable.copy()
        variable_._name = "%s+%s" % (variable_.name, timestep)
        return variable_

    # def preprocess_sequence_map(self,
    #                             evidence: List[LabelAssignment],
    #                             allow_singular_values: bool = True):
    #     """ Preprocess a list of variable maps to be used in JPTs. """
    #     return [
    #         self.template_tree._preprocess_query(
    #             e,
    #             allow_singular_values=allow_singular_values
    #         ) for e in evidence
    #     ]

    def ground(self, evidence: List[VariableAssignment]) -> (factorgraph.Graph, List[JPT]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """
        evidence_ = []
        for e in evidence:
            if isinstance(e, LabelAssignment):
                e = e.value_assignment()
            evidence_.append(e)
        evidence = evidence_

        # create factorgraph
        factor_graph = factorgraph.Graph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        [factor_graph.rv(timestep, len(self.template_tree.leaves)) for timestep in timesteps]

        altered_jpts = []

        # for each transition
        for idx in range(len(evidence)-1):

            # get the variable names
            state_names = ["t%s" % idx, "t%s" % (idx+1)]

            # create factor with values from transition model
            factor_graph.factor(state_names, potential=self.transition_model)

        # create prior factors
        start = datetime.datetime.now()
        for timestep, e in zip(timesteps, evidence):
            # apply the evidence
            try:
                conditional_jpt = self.template_tree.conditional_jpt(e)
                self.logger.debug(
                    'Conditional JPT from %s to %s nodes.' % (
                        len(self.template_tree.allnodes), len(conditional_jpt.allnodes)
                    )
                )
            except Unsatisfiability as e:
                raise Unsatisfiability(
                    'Unsatisfiable evidence at time step %s.' % timestep
                )

            # append altered jpt
            altered_jpts.append(conditional_jpt)

            # create the prior distribution from the conditional tree
            prior = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    prior[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            factor_graph.factor([timestep], potential=prior)
        now = datetime.datetime.now()
        self.logger.debug(
            'Conditional jpt computation '
            '(length %s) took %s.' % (len(evidence), now - start)
        )

        return factor_graph, altered_jpts

    def ground_fglib(self, evidence: List[VariableAssignment]) -> (factorgraph.Graph, List[JPT]):
        """Ground a factor graph where inference can be done. The factor graph is grounded with
        one variable for each timestep, one prior node as factor for each timestep and one factor node for each
        transition.

        @param evidence: A list of VariableMaps that describe evidence in the given timesteps.
        """

        # create factorgraph
        factor_graph = fglib.graphs.FactorGraph()

        # add variable nodes for timesteps
        timesteps = ["t%s" % t for t in range(len(evidence))]
        fg_variables = [fglib.rv.Discrete()]

        altered_jpts = []

        # for each transition
        for idx in range(len(evidence)-1):

            # get the variable names
            state_names = ["t%s" % idx, "t%s" % (idx+1)]

            # create factor with values from transition model
            factor_graph.factor(state_names, potential=self.transition_model)

        # create prior factors
        for timestep, e in zip(timesteps, evidence):

            # apply the evidence
            conditional_jpt = self.template_tree.conditional_jpt(e)

            # append altered jpt
            altered_jpts.append(conditional_jpt)

            # create the prior distribution from the conditional tree
            prior = np.zeros((len(self.template_tree.leaves), ))

            # fill the distribution with the correct values
            for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
                if leaf_idx in conditional_jpt.leaves.keys():
                    prior[idx] = conditional_jpt.leaves[leaf_idx].prior

            # create a factor from it
            factor_graph.factor([timestep], potential=prior)

        return factor_graph, altered_jpts

    def mpe(self, evidence):
        raise NotImplementedError("Not yet implemented")

    def probability(self, query, evidence) -> float:
        """
        Calculate the probability of sequence 'query' given sequence 'evidence'.

        @param query: The question
        @param evidence: The evidence
        @return: probability (float)
        """
        raise NotImplementedError("Not yet implemented")

    def independent_marginals(self, evidence: List[LabelAssignment or Dict]) -> List[JPT]:
        """ Return the independent marginal distributions of all variables in this sequence along all
        timesteps.

        @param evidence: The evidence observed in every timesteps. The length of this list determines the length
            of the whole sequence
        """
        # preprocess evidence
        evidence_ = []
        for e in evidence:
            if e is None or isinstance(e, dict):
                e = self.template_tree.bind(e, allow_singular_values=False)
            if isinstance(e, LabelAssignment):
                e = e.value_assignment()
            evidence_.append(e)

        self.logger.debug('Evidence sequence preproceessing finished.')

        # ground factor graph
        start = datetime.datetime.now()
        factor_graph, altered_jpts = self.ground(evidence_)
        now = datetime.datetime.now()
        self.logger.debug(
            'Grounding of conditional JPT sequence '
            '(length %s) took %s.' % (len(evidence_), now - start)
        )

        # create result list
        result = []

        # Run (loopy) belief propagation (LBP)
        start = datetime.datetime.now()
        factor_graph.lbp(max_iters=100, progress=True)
        leaf_distribution = {
            var.name: normalized(dist, zeros=.1)
            for var, dist in factor_graph.rv_marginals()
        }

        now = datetime.datetime.now()
        self.logger.debug(
            'Leaf prior computations '
            '(length %s) took %s.' % (len(evidence_), now - start)
        )

        # transform trees
        for ((tree_name, distribution), tree) in zip(
            sorted(leaf_distribution.items(), key=itemgetter(0)),
            altered_jpts
        ):
            prior = dict(
                zip(
                    sorted(self.template_tree.leaves.keys()),
                    distribution
                )
            )

            adjusted_tree = tree.multiply_by_leaf_prior(prior)
            result.append(adjusted_tree)

        self.logger.debug('Independent marginals computation finished.')
        return result

    def to_json(self):
        return {
            "template_tree": self.template_tree.to_json(),
            "transition_model": self.transition_model.tolist()
        }

    @staticmethod
    def from_json(data):
        template_tree = JPT.from_json(data["template_tree"])
        result = SequentialJPT(template_tree)
        result.transition_model = np.array(data["transition_model"])
        return result
