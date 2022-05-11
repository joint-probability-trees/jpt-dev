from copy import deepcopy
from typing import Dict, List

import jpt.variables
import jpt.trees
import jpt.base.quantiles
import jpt.base.intervals

import numpy as np
import numpy.lib.stride_tricks

import tqdm

import variables


class SequentialJPT:
    
    def __init__(self, template_variables:List[jpt.variables.Variable], 
                        timesteps:int = 2, 
                        **kwargs) -> None:
        '''
        Create a JPT template that can be expanded across multiple time dimensions.

        :param template_variables:
        :type template_variables: [jpt.variables.Variable]
        :param timesteps: The amount of timesteps that are modelled together.
        :type timesteps:
        '''

        self.template_variables = template_variables
        self.timesteps = timesteps
        
        # a dict mapping the template variable to its time expensions
        self.template_variable_map:Dict[jpt.variables.Variable, 
                            List[jpt.variables.Variable]] = dict()
        
        # just a plain list of all "grounded" variables
        self.variables:List[jpt.variables.Variable] = []

        # generate the time expanded variables from the template
        for template_variable in template_variables:
            self.template_variable_map[template_variable] = []
            for timestep in range(timesteps):
                variable_ = deepcopy(template_variable)
                variable_._name = "%s+%s" % (template_variable.name,timestep)
                self.template_variable_map[template_variable].append(variable_)
                self.variables.append(variable_)

        #inverse of the template_variable_map
        self.variable_template_map = dict()
        for variable in self.variables:
            for template_variable, variables in self.template_variable_map.items():
                if variable in variables:
                    self.variable_template_map[variable] = template_variable


        variable_dependencies = dict()
        for variable in self.variables:
            variable_dependencies[variable] = [v for v in self.variables if v != variable]

        variable_dependencies = jpt.variables.VariableMap(variable_dependencies.items())

        #generate the template tree to copy among timesteps
        self.template_tree:jpt.trees.JPT = jpt.trees.JPT(self.variables, 
            variable_dependencies=variable_dependencies,
            **kwargs)

        #forward the plot method from the original jpts
        self.plot = self.template_tree.plot

        self.probability_mass_ = 1.


    def learn(self, data:np.ndarray):
        '''
        Fit the joint distribution.

        :param data: The training sequences. This needs to be a list of 2d ndarrays 
            where each element in the list describes an observed time series. These
            time series was observed contiguously. The inner ndarray describes the 
            length of the time series in the first dimension and the observed values
            for the variables (in order of self.template_variables) in the second 
            dimension.
        :type data: np.ndarray
        '''
        
        samples = np.zeros((sum([len(timeseries)-self.timesteps+1 for timeseries in data]),
                    len(self.variables)))
        begin = 0
        for timeseries in tqdm.tqdm(data, desc="Windowing timeseries"):
            end = begin + len(timeseries) + 1 - self.timesteps
            windowed = numpy.lib.stride_tricks.sliding_window_view(timeseries, 
                                                        window_shape=(2,), 
                                                        axis=0)
            samples[begin:end] = windowed.reshape(samples[begin:end].shape)
            begin = end

        self.template_tree.learn(samples)
        self.probability_mass_ = self.integrate()

    def integrate(self):
        probability_mass = 0.

        for n, leaf_n in self.template_tree.leaves.items():
            for m, leaf_m in self.template_tree.leaves.items():
                intersecting_mass = leaf_m.prior * leaf_n.prior

                # for all template variables get their groundings
                for template_variable, grounded_variables in self.template_variable_map.items():

                    intersecting_mass *= integrate_pdfs(leaf_n.distributions[grounded_variables[1]].pdf,
                                   leaf_m.distributions[grounded_variables[0]].pdf,
                                   jpt.base.intervals.ContinuousSet(-float("inf"), float("inf")))

                probability_mass += intersecting_mass
        return probability_mass


    def likelihood(self, queries: List) -> np.ndarray:
        '''Get the probabilities of a list of worlds. The worlds must be fully assigned with
                single numbers (no intervals).

                :param queries: A list containing the sequences. The shape is (num_sequences, num_timesteps, len(variables)).
                :type queries: list
                Returns: An np.array with shape (num_sequences, ) containing the probabilities.
        '''

        result = np.zeros(len(queries))

        for idx, sequence in enumerate(tqdm.tqdm(queries, desc="Processing Time Series")):
            windowed = numpy.lib.stride_tricks.sliding_window_view(sequence, window_shape=(2,), axis=0)
            probs = self.template_tree.likelihood(windowed[:, 0, :])
            result[idx] = np.prod(probs)
        return result/self.probability_mass_


    def infer(self, queries:List[jpt.variables.VariableMap], evidences:List[jpt.variables.VariableMap]) -> float:
        '''
        Return the probability of a sequence taking values specified in queries given ranges specified in evidences
        '''
        if len(queries) != len(evidences):
            raise Exception("Queries and Evidences need to be sequences of same length.")

        for timestep in range(len(queries)-2):
            complete_query_phi_0 = dict()
            complete_evidence_phi_0 = dict()
            complete_query_phi_1 = dict()
            complete_evidence_phi_1 = dict()

            for template_variable, ground_variables in self.template_variable_map.items():

                # get the currently relevant queries for phi_0 and phi_1
                if template_variable in queries[timestep]:
                    complete_query_phi_0[ground_variables[0]] = queries[timestep][template_variable]
                if template_variable in queries[timestep+1]:
                    complete_query_phi_0[ground_variables[1]] = queries[timestep+1][template_variable]
                    complete_query_phi_1[ground_variables[0]] = queries[timestep+1][template_variable]
                if template_variable in queries[timestep+2]:
                    complete_query_phi_1[ground_variables[1]] = queries[timestep+2][template_variable]

                # get the currently relevant evidences for phi_0 and phi_1
                if template_variable in evidences[timestep]:
                    complete_evidence_phi_0[ground_variables[0]] = evidences[timestep][template_variable]
                if template_variable in evidences[timestep + 1]:
                    complete_evidence_phi_0[ground_variables[1]] = evidences[timestep + 1][template_variable]
                    complete_evidence_phi_1[ground_variables[0]] = evidences[timestep + 1][template_variable]
                if template_variable in evidences[timestep + 2]:
                    complete_evidence_phi_1[ground_variables[1]] = evidences[timestep + 2][template_variable]

            # apply evidences
            phi_0 = self.template_tree.conditional_jpt(complete_evidence_phi_0)
            phi_1 = self.template_tree.conditional_jpt(complete_evidence_phi_1)

            # preprocess queries
            complete_query_phi_0 = self.template_tree._prepropress_query(complete_query_phi_0)
            complete_query_phi_1 = self.template_tree._prepropress_query(complete_query_phi_1)

            # initialize result
            probability = 0.

            # for all leaf combinations
            for leaf_0 in phi_0.leaves.values():
                for leaf_1 in phi_1.leaves.values():
                    probability += self.prob_leaf_combination(leaf_0, leaf_1, complete_query_phi_0, complete_query_phi_1)
        print(probability, pow(self.probability_mass_, len(queries)-2))
        # return 1/Z * model(x)
        return probability / pow(self.probability_mass_, len(queries)-2)

    def prob_leaf_combination(self, leaf_0: jpt.trees.Leaf, leaf_1: jpt.trees.Leaf, query_0: jpt.variables.VariableMap,
                              query_1: jpt.variables.VariableMap) -> float:
        """Calculate the probability mass of a query in a product of 2 leaves.
        It is assumed that they are in a sequence context where leaf_0 is the leaf in the predecessing factor and leaf_1
        is the leaf in the successing factor.

        @param: leaf_0, the predecessor leaf
        @param: leaf_1, the successor leaf
        @param: query, the query for the variables
        @rtype: float
        """

        if not(leaf_0.applies(query_0) and leaf_1.applies(query_1)):
            return 0.

        probability = leaf_0.prior * leaf_1.prior

        for template_variable, ground_variables in self.template_variable_map.items():

            # handle "complex" continuous case
            if template_variable.numeric:

                # integral over x_dt0
                if ground_variables[0] in query_0.keys():
                    interval_q0 = query_0[ground_variables[0]]
                    probability *= leaf_0.distributions[ground_variables[0]].cdf.eval(interval_q0.upper)\
                                   - leaf_0.distributions[ground_variables[0]].cdf.eval(interval_q0.lower)

                # integral over f_0(x_dt1) * f1(x_xdt1)
                if ground_variables[0] in query_1.keys() and ground_variables[1] in query_0.keys():
                    probability *= integrate_pdfs(leaf_0.distributions[ground_variables[1]].pdf,
                                                  leaf_1.distributions[ground_variables[0]].pdf,
                                                  query_0[ground_variables[1]])

                # integral over x_dt2
                if ground_variables[1] in query_1.keys():
                    interval_q1 = query_1[ground_variables[1]]
                    probability *= leaf_1.distributions[ground_variables[1]].cdf.eval(interval_q1.upper)\
                                   - leaf_1.distributions[ground_variables[1]].cdf.eval(interval_q1.lower)

            # handle "easy" symbolic case
            elif template_variable.symbolic:
                for ground_variable in ground_variables:
                    if ground_variable in query_0.keys():
                        probability *= leaf_0.distributions[ground_variable].pdf.eval(query_0[ground_variable])
                    if ground_variable in query_1.keys():
                        probability *= leaf_1.distributions[ground_variable].pdf.eval(query_1[ground_variable])

        return probability


def integrate_pdfs(pdf1: jpt.base.quantiles.PiecewiseFunction, pdf2: jpt.base.quantiles.PiecewiseFunction,
                   interval: jpt.base.intervals.ContinuousSet) -> float:
    """Calculate the volume that is covered by the integral of pdf1 * pdf2 in the range of the interval."""

    mass = 0.
    for interval_1, function_1 in zip(pdf1.intervals, pdf1.functions):
        for interval_2, function_2 in zip(pdf2.intervals, pdf2.functions):
            # syntax highlighting
            interval_1: jpt.base.intervals.ContinuousSet
            interval_2: jpt.base.intervals.ContinuousSet

            # form overall intersection
            intersection = interval_1.intersection(interval_2).intersection(interval)

            # skip non-intersecting areas
            if intersection.isempty() or function_1.value == 0 or function_2.value == 0:
                continue

            mass += pow((intersection.upper - intersection.lower), 2) * function_1.value * function_2.value

    return mass

