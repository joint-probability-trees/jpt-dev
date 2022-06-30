from copy import deepcopy
from typing import Dict, List
import tqdm

import jpt.variables
import jpt.trees
import jpt.base.quantiles
import jpt.base.intervals
import jpt.learning.distributions

import numpy as np
import numpy.lib.stride_tricks


class SequentialJPT:

    def __init__(self, template_variables: List[jpt.variables.Variable], timesteps: int = 2, **kwargs) -> None:
        """
        Create a JPT template that can be expanded across multiple time dimensions.

        :param template_variables: The template variables of the sequence model. (The variables that are within each
        timestep)
        :type template_variables: [jpt.variables.Variable]
        :param timesteps: The amount of timesteps that are jointly modelled together.
        :type timesteps: int
        """

        self.template_variables = template_variables
        self.timesteps = timesteps

        # a dict mapping the template variable to its time expansions
        self.template_variable_map: Dict[jpt.variables.Variable,
                                         List[jpt.variables.Variable]] = dict()

        # just a plain list of all "grounded" variables
        self.variables: List[jpt.variables.Variable] = []

        # generate the time expanded variables from the template
        for template_variable in template_variables:
            self.template_variable_map[template_variable] = []
            for timestep in range(timesteps):
                variable_ = deepcopy(template_variable)
                variable_._name = "%s+%s" % (template_variable.name, timestep)
                self.template_variable_map[template_variable].append(variable_)
                self.variables.append(variable_)

        # inverse of the template_variable_map
        self.variable_template_map = dict()
        for variable in self.variables:
            for template_variable, variables in self.template_variable_map.items():
                if variable in variables:
                    self.variable_template_map[variable] = template_variable

        variable_dependencies = dict()
        for variable in self.variables:
            variable_dependencies[variable] = [v for v in self.variables if v != variable]

        variable_dependencies = jpt.variables.VariableMap(variable_dependencies.items())

        # generate the template tree to copy among timesteps
        self.template_tree: jpt.trees.JPT = jpt.trees.JPT(self.variables,
                                                          variable_dependencies=variable_dependencies,
                                                          **kwargs)

        # forward the plot method from the original jpts
        self.plot = self.template_tree.plot

        # initialize probability mass
        self.probability_mass_ = 1.

        # this map will store a distribution for each shared dimension
        self.shared_dimensions_integral = dict()

    def learn(self, data: np.ndarray):
        """
        Fit the joint distribution.

        :param data: The training sequences. This needs to be a list of 2d ndarrays
            where each element in the list describes an observed time series. These
            time series was observed contiguously. The inner ndarray describes the
            length of the time series in the first dimension and the observed values
            for the variables (in order of self.template_variables) in the second
            dimension.
        :type data: np.ndarray
        """

        samples = np.zeros((sum([len(timeseries) - self.timesteps + 1 for timeseries in data]),
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
        self.fill_shared_dimensions()
        self.probability_mass_ = self.integrate()

    def integrate(self) -> float:
        """
        Calculate the overall probability mass that is obtained by expanding the template to one more timestep
        than it is modelled in the template tree.
        For example if timesteps = 2, then the mass for timesteps = 3 is calculated.
        For the probability mass Z it holds that 0 < Z <= 1.
        """

        # reset probability mass
        probability_mass = 0.

        # for all leaf combinations
        for n, leaf_n in self.template_tree.leaves.items():
            for m, leaf_m in self.template_tree.leaves.items():

                # start with mass as prior
                intersecting_mass = leaf_m.prior * leaf_n.prior

                # for all template variables get their groundings
                for template_variable, grounded_variables in self.template_variable_map.items():

                    # integrate numeric variables on shared dimensions
                    if template_variable.numeric:
                        intersecting_mass *= \
                            self.shared_dimensions_integral[(n, m)][template_variable].cdf.functions[-1].c

                    # integrate symbolic variables on shared dimensions
                    elif template_variable.symbolic:
                        intersecting_mass *= sum(self.shared_dimensions_integral[(n, m)][template_variable]._params)

                # sum the mass of all partitions
                probability_mass += intersecting_mass

        return probability_mass

    def fill_shared_dimensions(self):
        """Generate distributions for all shared dimensions. Shared dimensions are dimensions that occur in more
        than one factor."""

        # for all leaf combinations
        for n, leaf_n in self.template_tree.leaves.items():
            for m, leaf_m in self.template_tree.leaves.items():

                # create distributions for every variable as dict
                distributions = dict()

                # for every variable
                for template_variable, ground_variables in self.template_variable_map.items():

                    # calculate the integral of numeric continuous variables
                    if template_variable.numeric:
                        dist = integrate_continuous_distributions(leaf_n.distributions[ground_variables[1]],
                                                                  leaf_m.distributions[ground_variables[0]])

                    # calculate the integral of symbolic variables
                    elif template_variable.symbolic:
                        dist = integrate_discrete_distribution(leaf_n.distributions[ground_variables[1]],
                                                               leaf_m.distributions[ground_variables[0]])

                    # save the cdf
                    distributions[template_variable] = dist

                # convert dict to variable map and save it
                self.shared_dimensions_integral[(n, m)] = jpt.variables.VariableMap(distributions.items())

    def likelihood(self, queries: List) -> np.ndarray:
        """
        Get the probabilities of a list of worlds. The worlds must be fully assigned with
            single numbers (no intervals).

        :param queries: A list containing the sequences. The shape is (num_sequences, num_timesteps, len(variables)).
        :type queries: list
        Returns: An np.array with shape (num_sequences, ) containing the probabilities.
        """

        result = np.zeros(len(queries))

        for idx, sequence in enumerate(tqdm.tqdm(queries, desc="Processing Time Series")):
            windowed = numpy.lib.stride_tricks.sliding_window_view(sequence, window_shape=(2,), axis=0)
            probs = self.template_tree.likelihood(windowed[:, 0, :])
            result[idx] = np.prod(probs)
        return result / self.probability_mass_

    def infer(self, queries: List[jpt.variables.VariableMap], evidences: List[jpt.variables.VariableMap]) -> float:
        '''
        Return the probability of a sequence taking values specified in queries given ranges specified in evidences
        '''
        if len(queries) != len(evidences):
            raise Exception("Queries and Evidences need to be sequences of same length.")

        for timestep in range(len(queries) - 2):
            complete_query_phi_0 = dict()
            complete_evidence_phi_0 = dict()
            complete_query_phi_1 = dict()
            complete_evidence_phi_1 = dict()

            for template_variable, ground_variables in self.template_variable_map.items():

                # get the currently relevant queries for phi_0 and phi_1
                if template_variable in queries[timestep]:
                    complete_query_phi_0[ground_variables[0]] = queries[timestep][template_variable]
                if template_variable in queries[timestep + 1]:
                    complete_query_phi_0[ground_variables[1]] = queries[timestep + 1][template_variable]
                    complete_query_phi_1[ground_variables[0]] = queries[timestep + 1][template_variable]
                if template_variable in queries[timestep + 2]:
                    complete_query_phi_1[ground_variables[1]] = queries[timestep + 2][template_variable]

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
                    probability += self.prob_leaf_combination(leaf_0, leaf_1,
                                                              complete_query_phi_0, complete_query_phi_1)

        # scale the resulting potential by the overall probability mass
        return probability / self.probability_mass_

    def prob_leaf_combination(self, leaf_0: jpt.trees.Leaf, leaf_1: jpt.trees.Leaf, query_0: jpt.variables.VariableMap,
                              query_1: jpt.variables.VariableMap) -> float:
        """
        Calculate the probability mass of a query in a product of 2 leaves.
        It is assumed that they are in a sequence context where leaf_0 is the leaf in the predecessing factor and leaf_1
        is the leaf in the succeeding factor.

        @param: leaf_0, the predecessor leaf
        @param: leaf_1, the successor leaf
        @param: query, the query for the variables
        @rtype: float
        """
        if not (leaf_0.applies(query_0) and leaf_1.applies(query_1)):
            return 0.

        probability = leaf_0.prior * leaf_1.prior

        for template_variable, ground_variables in self.template_variable_map.items():
            # handle "complex" continuous case
            if template_variable.numeric:

                # integral over x_dt0
                if ground_variables[0] in query_0.keys():
                    interval_q0 = query_0[ground_variables[0]]
                    probability *= leaf_0.distributions[ground_variables[0]].cdf.eval(interval_q0.upper) \
                                   - leaf_0.distributions[ground_variables[0]].cdf.eval(interval_q0.lower)

                # integral over f_0(x_dt1) * f1(x_xdt1)
                if ground_variables[0] in query_1.keys() and ground_variables[1] in query_0.keys():
                    distribution = self.shared_dimensions_integral[(leaf_0.idx, leaf_1.idx)][template_variable]
                    p = distribution.cdf.eval(query_0[ground_variables[1]].upper) - \
                        distribution.cdf.eval(query_0[ground_variables[1]].lower)
                    probability *= p

                # integral over x_dt2
                if ground_variables[1] in query_1.keys():
                    interval_q1 = query_1[ground_variables[1]]
                    probability *= leaf_1.distributions[ground_variables[1]].cdf.eval(interval_q1.upper) \
                                   - leaf_1.distributions[ground_variables[1]].cdf.eval(interval_q1.lower)

            # handle "easy" symbolic case
            elif template_variable.symbolic:
                for ground_variable in ground_variables:
                    if ground_variable in query_0.keys():
                        probability *= leaf_0.distributions[ground_variable]._p(query_0[ground_variable])
                    if ground_variable in query_1.keys():
                        probability *= leaf_1.distributions[ground_variable]._p(query_1[ground_variable])

        return probability


def integrate_continuous_pdfs(pdf1: jpt.base.quantiles.PiecewiseFunction, pdf2: jpt.base.quantiles.PiecewiseFunction,
                              interval: jpt.base.intervals.ContinuousSet) -> float:
    """
    Calculate the volume that is covered by the integral of pdf1 * pdf2 in the range of the interval.
    Those pdfs are distributions of the same continuous random variable.

    :param pdf1: The first probability density function
    :param pdf2: The second probability density function
    :param interval: The integral boundaries
    """

    # reset mass
    mass = 0.

    # for all combinations of all intervals
    for interval_1, function_1 in zip(pdf1.intervals, pdf1.functions):
        for interval_2, function_2 in zip(pdf2.intervals, pdf2.functions):

            # form intersection of function 1, 2 and the integral boundaries
            intersection = interval_1.intersection(interval_2).intersection(interval)

            # skip non-intersecting areas
            if intersection.isempty() or function_1.value == 0 or function_2.value == 0:
                continue

            # calculate volume of hypercube
            mass += pow((intersection.upper - intersection.lower), 2) * function_1.value * function_2.value

    return mass


def integrate_continuous_distributions(distribution1: jpt.learning.distributions.Numeric,
                                       distribution2: jpt.learning.distributions.Numeric,
                                       normalize=False) \
        -> jpt.learning.distributions.Numeric:
    """
    Calculate the cdf that is obtained by integrating pdf1(x) * pdf2(x) and normalize it s. t. the joint
    cdf converges to 1 if wanted.

    :param distribution1: The first distribution
    :type distribution1: jpt.learning.distributions.Numeric
    :param distribution2: The second distribution
    :type distribution2: jpt.learning.distributions.Numeric
    :param normalize: Rather to normalize the distribution or not
    :type normalize: bool
    """

    # calculate the overall probability mass obtained by multiplying these functions
    if normalize:
        probability_mass = integrate_continuous_pdfs(distribution1.pdf, distribution2.pdf,
                                                     jpt.base.intervals.ContinuousSet(-float("inf"), float("inf")))
    else:
        probability_mass = 1.

    result = jpt.base.quantiles.PiecewiseFunction()

    # if there is no probability mass in the product return a constant 0 function that can be plotted :)
    if probability_mass == 0:
        result.intervals = [jpt.base.intervals.ContinuousSet(-float("inf"), -10.),
                            jpt.base.intervals.ContinuousSet(-10., 10.),
                            jpt.base.intervals.ContinuousSet(10., float("inf"))]
        result.functions = [jpt.base.quantiles.ConstantFunction(0.)] * 3
        result = jpt.base.quantiles.QuantileDistribution.from_cdf(result)
        result = jpt.learning.distributions.Numeric(result)
        return result

    current_integral_value = 0

    joint_intervals = []
    joint_functions = []

    # for all combinations of all intervals
    for interval_1, function_1 in zip(distribution1.pdf.intervals, distribution1.pdf.functions):
        for interval_2, function_2 in zip(distribution2.pdf.intervals, distribution2.pdf.functions):

            # form intersection
            intersection = interval_1.intersection(interval_2)

            # skip non-intersecting intervals
            if intersection.isempty():
                continue

            joint_intervals.append(intersection)
            intersection_value = function_1.value * function_2.value / probability_mass

            constant = 0. if intersection.lower == -float("inf") else \
                current_integral_value - (intersection_value * intersection.lower)
            joint_functions.append(jpt.base.quantiles.LinearFunction(intersection_value, constant))

            if intersection_value > 0:
                current_integral_value += pow(intersection.upper - intersection.lower, 2) * intersection_value

    result.intervals = joint_intervals
    result.functions = joint_functions
    result = jpt.base.quantiles.QuantileDistribution.from_cdf(result)
    result = jpt.learning.distributions.Numeric(result)
    return result


def integrate_discrete_distribution(distribution1: jpt.learning.distributions.Multinomial,
                                    distribution2: jpt.learning.distributions.Multinomial,
                                    normalize=False) -> jpt.learning.distributions.Multinomial:
    """
    Calculate the cdf that is obtained by integrating pdf1(x) * pdf2(x) and normalize it s. t. the joint
    cdf converges to 1 if wanted.

    :param distribution1: The first distribution
    :type distribution1: jpt.learning.distributions.Multinomial
    :param distribution2: The second distribution
    :type distribution2: jpt.learning.distributions.Multinomial
    :param normalize: Rather to normalize the distribution or not
    :type normalize: bool
    """

    # copy labels since they share the same dimension
    labels = distribution1.labels

    # initialize values
    _params = np.zeros(len(labels))

    # calculate values as product of both probabilities
    for idx, value in enumerate(distribution1._params):
        _params[idx] = value * distribution2._params[idx]

    # normalize if wanted
    if normalize:
        _params /= sum(_params)

    # create resulting distribution
    result = type(distribution1)()
    result._params = _params
    return result

