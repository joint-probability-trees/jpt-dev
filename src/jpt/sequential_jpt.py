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
            end = begin + len(timeseries) +1 -self.timesteps
            windowed = numpy.lib.stride_tricks.sliding_window_view(timeseries, 
                                                        window_shape=(2,), 
                                                        axis=0)
            samples[begin:end] = windowed.reshape(samples[begin:end].shape)
            begin = end

        self.template_tree.learn(samples)
        self.probability_mass_ = self.integrate()

    def integrate(self):
        probability_mass = 0.
        leaf_masses = []
        for n, leaf_n in self.template_tree.leaves.items():
            for m, leaf_m in self.template_tree.leaves.items():
                leaf_mass = leaf_n.prior * leaf_m.prior
                integral = 0.
                # for all template variables get their groundings
                for template_variable, grounded_variables in self.template_variable_map.items():

                    for interval_n, function_n in zip(leaf_n.distributions[grounded_variables[1]].pdf.intervals,
                                                      leaf_n.distributions[grounded_variables[1]].pdf.functions):
                        for interval_m, function_m in zip(leaf_m.distributions[grounded_variables[0]].pdf.intervals,
                                                          leaf_m.distributions[grounded_variables[0]].pdf.functions):
                            if function_m.value > 0 and function_n.value > 0:
                                leaf_mass *= (interval_n.upper - interval_n.lower) \
                                    * (interval_m.upper - interval_m.lower) \
                                    * (function_n.value * function_m.value)
                leaf_masses.append(leaf_mass)

            probability_mass += leaf_mass

        return probability_mass