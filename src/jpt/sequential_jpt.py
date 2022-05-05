from copy import deepcopy
from typing import Dict, List

import jpt.variables
import jpt.trees

import numpy as np
import numpy.lib.stride_tricks

import tqdm

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
                intersecting_mass = 0.

                # for all template variables get their groundings
                for template_variable, grounded_variables in self.template_variable_map.items():

                    for interval_n, function_n in zip(leaf_n.distributions[grounded_variables[1]].pdf.intervals,
                                                      leaf_n.distributions[grounded_variables[1]].pdf.functions):
                        for interval_m, function_m in zip(leaf_m.distributions[grounded_variables[0]].pdf.intervals,
                                                          leaf_m.distributions[grounded_variables[0]].pdf.functions):

                            if function_m.value == 0 or function_n.value == 0:
                                continue

                            intersection = interval_n.intersection(interval_m)
                            if intersection.isempty() == 0 \
                                    and intersection.lower != -np.inf \
                                    and intersection.upper != np.inf:
                                intersecting_mass += ((intersection.upper - intersection.lower) * function_m.value) * \
                                                    ((intersection.upper - intersection.lower) * function_n.value) * \
                                                     leaf_n.prior * leaf_m.prior

                    probability_mass += intersecting_mass

        return probability_mass


    def likelihood(self, queries: List, minimum_probability=pow(10, -10)) -> np.ndarray:
        '''Get the probabilities of a list of worlds. The worlds must be fully assigned with
                single numbers (no intervals).

                :param queries: A list containing the sequences. The shape is (num_sequences, num_timesteps, len(variables)).
                :type queries: list
                :param minimum_probability: If the probability of a sample is 0 (which is possible
                    due to numeric approximations) the probability will be replaced with
                    'minimum_probability'.
                :type minimum_probability: float
                Returns: An np.array with shape (num_sequences, ) containing the probabilities.
        '''

        result = np.zeros(len(queries))

        for idx, sequence in enumerate(tqdm.tqdm(queries, desc="Processing Time Series")):
            windowed = numpy.lib.stride_tricks.sliding_window_view(sequence, window_shape=(2,), axis=0)
            probs = self.template_tree.likelihood(windowed[:, 0, :])
            result[idx] = np.prod(probs)
        return result


    def infer(self, queries, evidences):
        '''
        Return the probability of a sequence taking values specified in queries given ranges specified in evidences
        '''
        if len(queries) != len(evidences):
            raise Exception("Queries and Evidences need to be sequences of same length.")

        for timestep in range(len(queries)):
            current_tree_query= queries[timestep]
            current_tree_evidence= evidences[timestep]
            joint_query_map = dict()
            joint_evidence_map = dict()