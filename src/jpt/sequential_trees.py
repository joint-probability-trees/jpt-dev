from typing import List, Dict, Union, Optional, Tuple, Any

import numpy as np
import numpy.lib.stride_tricks
import pandas as pd
import pgmpy
import pgmpy.factors.discrete
import pgmpy.inference
import pgmpy.models
from dnutils import getlogger

from . import JPT
from .variables import LabelAssignment, VariableAssignment, VariableMap, Variable
from .base.errors import Unsatisfiability
from .base.utils import format_path

class SequentialJPT:
    logger = getlogger('/jpt/seq')

    def __init__(self, template_tree: JPT):

        self.template_tree: JPT = template_tree
        """
        The JPT used as emission template in the timesteps.
        """

        self.transition_model: Optional[np.array] = None
        """
        The transition model used to transfer from one leaf of the template tree to another leaf of the template tree
        in the next timestep. The rows enumerate the first timestep, the columns enumerate the second timestep.
        """

    def fit(self, sequences: List[np.ndarray], timesteps: int = 2):
        """ Fit the transition and emission models. The emission model is fitted
         with respect to the variables in the next timestep and then marginalizes them.

         :param sequences: The sequences to learn from
         :param timesteps: The timesteps to jointly model (only 2 supported for now) """

        if timesteps != 2:
            raise ValueError("Only jointly modelling 2 timesteps is supported.")

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
            unfolded = np.lib.stride_tricks.sliding_window_view(sequence, (timesteps,), axis=0)
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
                values[idx, jdx] = count / len(transition_data)

        self.transition_model = values

    def _shift_variable_to_timestep(self,
                                    variable: Variable,
                                    timestep: int = 1) -> Variable:
        """ Create a new variable where the name is shifted by +n and the domain remains the same.

        :param variable: The variable to shift
        :param timestep: timestep in the future, i.e. timestep >= 1
        """
        variable_ = variable.copy()
        variable_._name = "%s+%s" % (variable_.name, timestep)
        return variable_

    def preprocess_sequence_map(self, query: Union[dict, VariableMap],
                                remove_none: bool = True,
                                skip_unknown_variables: bool = False,
                                allow_singular_values: bool = False) -> List[LabelAssignment]:
        """
        Preprocess a list of dictionaries to be used in JPTs.
        """
        return [self.template_tree._preprocess_query(q, remove_none=remove_none,
                                                     skip_unknown_variables=skip_unknown_variables,
                                                     allow_singular_values=allow_singular_values, ) for q in query]

    def bind(self, queries: List[dict], *args, **kwargs) -> List[LabelAssignment]:
        """
        Returns a list of ``LabelAssignment`` objects with the assignments passed.

        This method accepts one optional positional argument, which -- if passed -- must be a dictionary
        of the desired variable assignments.

        Keyword arguments may specify additional variable, value pairs.

        If a positional argument is passed, the following options may be passed in addition
        as keyword arguments:

        :param queries: A list of query like objects that will be bound to the template tree.
        """
        return [self.template_tree.bind(query, *args, **kwargs) for query in queries]

    def priors_of_tree(self, tree: JPT) -> np.ndarray:
        """
        Get the distribution of the leaf priors of a JPT.
        :param tree: The tree to get the distribution from
        :return: A numpy array containing the probability distribution of leaves using the original jpts definition.
        """
        result = np.zeros(len(self.template_tree.leaves))

        # fill the distribution with the correct values
        for idx, leaf_idx in enumerate(self.template_tree.leaves.keys()):
            if leaf_idx in tree.leaves.keys():
                result[idx] = tree.leaves[leaf_idx].prior
        return result

    @property
    def conditional_transition_model(self):
        """
        :return: The conditional probability distribution P(timestep_1 | timestep_0)
        The rows enumerate over the states of timestep 0, the column over the states of timestep 1
        """
        return (self.transition_model.T / self.transition_model.sum(axis=1)).T

    def maximum_encoding(self, tree: Optional[JPT] = None, use_leaf_prior = False) -> Tuple[Dict[float, int],
                                                                                            Dict[int, Optional[LabelAssignment]]]:
        """
        Map the index of each leaf in the template tree to the likelihood of its mpe state.

        :param tree: The tree to get the likelihoods from
        :param use_leaf_prior: Rather to use the prior of the leaf for the maxima or to cancel it

        :return: The sorted dictionary with the mapping
        """
        result = {leaf.idx: 0. for leaf in self.template_tree.leaves.values()}
        states = {leaf.idx: None for leaf in self.template_tree.leaves.values()}
        if tree is None:
            tree = self.template_tree

        for leaf in tree.leaves.values():
            state, likelihood = leaf.mpe(self.template_tree.minimal_distances)
            states[leaf.idx] = state
            result[leaf.idx] = likelihood
            if not use_leaf_prior:
                result[leaf.idx] /= leaf.prior

        return dict(sorted(result.items())), dict(sorted(states.items()))


    def ground(self, evidence: List[VariableAssignment], fail_on_unsatisfiability: bool = True) -> \
            Tuple[pgmpy.models.BayesianNetwork, List[JPT], List[pgmpy.factors.discrete.TabularCPD]]:

        # convert evidence
        evidence = [e.value_assignment() if isinstance(e, LabelAssignment) else e for e in evidence]

        # create names for timesteps
        timesteps = [f"t{t}" for t in range(len(evidence))]

        bayes_network = pgmpy.models.BayesianNetwork([*zip(timesteps[:-1], timesteps[1:])])

        cpd_tables = \
            [pgmpy.factors.discrete.TabularCPD(variable="t0",
                                               variable_card=len(self.template_tree.leaves),
                                               values=np.array([self.priors_of_tree(self.template_tree)]).T)] + \
            [pgmpy.factors.discrete.TabularCPD(variable=curr,
                                               variable_card=len(self.template_tree.leaves),
                                               values=self.conditional_transition_model,
                                               evidence=[prev], evidence_card=[len(self.template_tree.leaves)]) for
             prev, curr in zip(timesteps[:-1], timesteps[1:])]

        bayes_network.add_cpds(*cpd_tables)

        conditional_trees = []
        virtual_evidences = []

        for index, evidence_ in enumerate(evidence):
            conditional_tree = self.template_tree.conditional_jpt(evidence_)
            conditional_trees.append(conditional_tree)

            virtual_evidence = pgmpy.factors.discrete.TabularCPD(variable=timesteps[index],
                                                                 variable_card=len(self.template_tree.leaves),
                                                                 values=np.array(
                                                                     [self.priors_of_tree(conditional_tree)]).T)
            virtual_evidences.append(virtual_evidence)

        return bayes_network, conditional_trees, virtual_evidences

    def likelihood(self, queries: List[Union[np.ndarray, pd.DataFrame]], dirac_scaling: float = 2.,
                   min_distances: Dict = None) -> List[np.ndarray]:
        """
        Get the probabilities of a list of worlds. The worlds must be fully assigned with
        single numbers (no sets).

        :param queries: A list containing sequences describing the worlds.
            The shape is of each element in the outermost list is (len(timesteps), len(variables)).
        :param dirac_scaling: the minimal distance between the samples within a dimension are multiplied by this factor
            if a durac impulse is used to model the variable.
        :param min_distances: A dict mapping the variables to the minimal distances between the observations.
            This can be useful to use the same likelihood parameters for different test sets for example in cross
            validation processes.
        :returns: A list of numpy arrays with the same length as the input. The numpy arrays in the list contain the
            likelihoods for each element. The first element in each array contains the likelihood of the first element a
            priori. All successive elements contain the likelihood given the previous elements. Hence, the total
            likelihood can be computed by a simple multiplication of all elements.
        """
        # initialize probabilities
        sequence_likelihoods = []

        # for every sequence
        for sequence_idx, sequence in enumerate(queries):

            # encode the entire sequence using the leaf indices
            encoded_sequence = self.template_tree.encode(sequence)

            # initialize the likelihood for each element in the sequence
            sequence_likelihood = np.zeros(len(sequence))

            # for each element index, the element and the encoded element (leaf index)
            for element_idx, (element, encoded_element) in enumerate(zip(sequence, encoded_sequence)):
                # if it is the first element in a sequence, use P(Q) directly
                if element_idx == 0:
                    sequence_likelihood[element_idx] = self.template_tree.likelihood(element.reshape(1, -1),
                                                                                     dirac_scaling, min_distances)
                    continue

                # if it is not the first element calculate likelihood in leaf
                likelihood_in_leaf = self.template_tree.leaves[encoded_element]. \
                    parallel_likelihood(element.reshape(1, -1), dirac_scaling, min_distances)

                # and multiply it by the transition probability
                transition_likelihood = self.transition_likelihood(encoded_sequence[element_idx - 1], encoded_element)
                sequence_likelihood[element_idx] = transition_likelihood * likelihood_in_leaf

            sequence_likelihoods.append(sequence_likelihood)

        return sequence_likelihoods

    def leaf_idx_to_transition_idx(self, leaf_idx: int) -> int:
        """
        Get the index in the transition model of a leaf.
        :param leaf_idx: The leaf index in the template tree
        :return: The row/column index of the transition model
        """
        index_map = {li: ti for ti, li in enumerate(self.template_tree.leaves.keys())}
        return index_map[leaf_idx]

    def transition_likelihood(self, leaf_1: int, leaf_2: int) -> float:
        """
        Calculate the probability P(leaf_2|p_leaf_1) using the transition model.
        :param leaf_1: The index of the evidence leaf
        :param leaf_2: The index of the query leaf
        :return: The probability
        """
        transition_index_1 = self.leaf_idx_to_transition_idx(leaf_1)
        transition_index_2 = self.leaf_idx_to_transition_idx(leaf_2)
        return self.transition_model[transition_index_1, transition_index_2] / \
            np.sum(self.transition_model[transition_index_1, :])

    def posterior(self, evidence: List[LabelAssignment or Dict], fail_on_unsatisfiability: bool = True,) -> \
            Optional[List[JPT]]:
        """ Return the independent marginal distributions of all variables in this sequence along all
        timesteps.

        :param evidence: The evidence observed in every timesteps. The length of this list determines the length
            of the whole sequence
        :param fail_on_unsatisfiability: Rather or not an ``Unsatisfiability`` error is raised if the
            likelihood of the evidence is 0.
        """
        # preprocess evidence
        evidence_ = []
        for e in evidence:
            if e is None or isinstance(e, dict):
                e = self.template_tree.bind(e, allow_singular_values=False)
            if isinstance(e, LabelAssignment):
                e = e.value_assignment()
            evidence_.append(e)

        # ground factor graph
        bayes_network, altered_jpts, virtual_evidences = self.ground(evidence_)

        # Run belief propagation

        belief_propagation = pgmpy.inference.BeliefPropagation(bayes_network)
        result = belief_propagation.query(list(bayes_network.nodes), virtual_evidence=virtual_evidences, joint=False)

        # calculate the new priors
        for (variable, factor), conditional_jpt in zip(result.items(), altered_jpts):
            if np.any(np.isnan(factor.values)):
                if fail_on_unsatisfiability:
                    raise Unsatisfiability('Evidence %s is unsatisfiable.' % [format_path(e) for e in evidence_],)
                else:
                    return None

            conditional_jpt.multiply_by_leaf_prior(dict(zip(self.template_tree.leaves.keys(), factor.values)))

        return altered_jpts

    def probability(self, event: List[Union[Dict[Union[Variable, str], Any], VariableAssignment]]) -> float:
        """
        Calculate the probability of an event.
        :param event: The event as a list of VariableAssignment like objects
        :return: The probability
        """
        probability = 1.

        message = self.transition_model.sum(axis=1)

        for event_ in event:
            current_probability = 0.
            for idx, leaf in enumerate(self.template_tree.leaves.values()):
                leaf_probability = leaf.probability(event_)
                current_probability += message[idx] * leaf.probability(event_)
                message[idx] *= leaf_probability

            probability *= current_probability
            message = message@self.transition_model
            message_sum = sum(message)

            if message_sum == 0:
                return  0.
            else:
                message /= message_sum

            if probability == 0:
                return probability
        return probability

    def infer(
            self,
            query: List[Union[Dict[Union[Variable, str], Any], VariableAssignment]],
            evidence: List[Union[Dict[Union[Variable, str], Any], VariableAssignment]] = None,
            fail_on_unsatisfiability: bool = True) -> Optional[float]:


        p_evidence = self.probability(evidence)
        if p_evidence == 0:
            if fail_on_unsatisfiability:
                raise Unsatisfiability()
            else:
                return None

        query_and_evidence = [q.intersection(e) for q,e in zip(query, evidence)]
        p_query_and_evidence = self.probability(query_and_evidence)
        return p_query_and_evidence/p_evidence

    def mpe(self, evidence: List[Union[Dict[Union[Variable, str], Any], VariableAssignment]] = None,
            fail_on_unsatisfiability: bool = True) -> Optional[Tuple[List[LabelAssignment], float]]:

        # initialize the trellis for backtracking
        trellis = np.full((len(evidence) - 1, len(self.template_tree.leaves),), -1, dtype=int)
        mpe_states = []

        # initialize message as unity
        message = None

        for timestep, evidence_ in enumerate(evidence):

            # apply evidence
            conditional_tree = self.template_tree.conditional_jpt(evidence_,
                                                                  fail_on_unsatisfiability=fail_on_unsatisfiability)

            # if the evidence is impossible return None
            if conditional_tree is None:
                return None

            # if it's the first iteration
            if timestep == 0:

                # get maximum likelihoods and respective states
                maxima, states = self.maximum_encoding(conditional_tree, use_leaf_prior=True)

                # append states for reconstruction
                mpe_states.append(states)

                # initialize the message
                message = np.array(list(maxima.values()))
                continue

            # if it's not the first iteration
            # get maximum likelihoods and respective states without leave priors
            maxima, states = self.maximum_encoding(conditional_tree)

            # append states for reconstruction
            mpe_states.append(states)

            # calculate conditional transition matrix P(t_1 | argmax(t_0))
            conditional_transition = self.conditional_transition_model * message

            # get the maxima of the current timestep
            maxima = np.array(list(maxima.values()))

            # multiply them with the maxima of each leaf
            joint_distribution = (maxima * conditional_transition.T).T

            # get the maximum for the new message
            message = joint_distribution.max(axis=1)

            # save the argmax into the trellis
            trellis[timestep-1, :] = joint_distribution.argmax(axis=1)

            # quite the MPE if it becomes impossible
            if max(message) == 0:
                if fail_on_unsatisfiability:
                    raise Unsatisfiability()
                else:
                    return None

        # get most probable state sequence
        previous_mpe_state = message.argmax()

        # reconstruct most probable path
        most_probable_sequence = [previous_mpe_state]

        # reverse through trellis to get the mpe
        for trellis_step in reversed(trellis):
            previous_mpe_state = trellis_step[previous_mpe_state]
            most_probable_sequence.append(previous_mpe_state)

        # flip to the correct order
        most_probable_sequence.reverse()

        # replace indices with events
        result = [list(state.values())[r_] for r_, state in zip(most_probable_sequence, mpe_states)]

        return result, max(message)

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
