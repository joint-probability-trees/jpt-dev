from typing import Dict, List

import jpt.variables
import jpt.trees
import jpt.base.quantiles
import jpt.base.intervals
import jpt.learning.distributions
from jpt.learning.distributions import Multinomial, Numeric


class JPTLike:
    """This one implements an interface for both sum product joint probability trees.
    To be used to construct a new JPT it is necessary that independent marginals and impurities are implemented
    """

    def __init__(self, variables: List[jpt.variables.Variable], jpts: List[jpt.trees.JPT]):
        self.variables = variables
        self.jpts = jpts

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> jpt.trees.PosteriorResult or None:
        """ Compute the marginal distribution of every varialbe in 'variables' assuming independence.
        Unlike JPT.posterior, this method also can compute marginals on variables that are in the evidence.

        :param variables:        the query variables of the posterior to be computed
        :type variables:         list of jpt.variables.Variable
        :param evidence:    the evidence given for the posterior to be computed
        :param fail_on_unsatisfiability: wether or not an ``Unsatisfiability`` error is raised if the
                                         likelihood of the evidence is 0.
        :type fail_on_unsatisfiability:  bool
        :return:            jpt.trees.InferenceResult containing distributions, candidates and weights
        """
        raise NotImplementedError("This is an abstract class.")


class SumJPT(JPTLike):
    """ Represent a sum of JPTs that can be used as a training basis for new JPTs.
        This is needed in the variable nodes of factor graphs.
     """

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> jpt.trees.PosteriorResult or None:

        # construct a list with the independent marginals
        independent_marginals = []

        for tree in self.jpts:
            marginals = tree.independent_marginals(variables, evidence, False)
            if marginals is not None:
                independent_marginals.append(marginals)

        result = jpt.trees.PosteriorResult(independent_marginals[0].query, independent_marginals[0].evidence)
        result.result = sum(r.result for r in independent_marginals) / len(independent_marginals)

        for variable in variables:
            if variable.numeric:
                result.distributions[variable] = Numeric.merge([r.distributions[variable]
                                                                for r in independent_marginals],
                                                               weights=[r.result
                                                                        for r in independent_marginals], )
            elif variable.symbolic:
                result.distributions[variable] = Multinomial.merge([r.distributions[variable]
                                                                    for r in independent_marginals],
                                                                   weights=[r.result
                                                                            for r in independent_marginals], )

        return result


class ProductJPT(JPTLike):
    """ Represent a product of JPTs that can be used as a training basis for new JPTs.
        This is needed in the factor nodes of factor graphs.
     """

    def independent_marginals(self, variables: List[jpt.variables.Variable], evidence: jpt.variables.VariableMap,
                              fail_on_unsatisfiability=True) -> jpt.trees.PosteriorResult or None:
        # construct a list with the independent marginals
        independent_marginals = []

        for tree in self.jpts:
            marginals = tree.independent_marginals(variables, evidence, False)
            if marginals is not None:
                independent_marginals.append(marginals)

        result = jpt.trees.PosteriorResult(independent_marginals[0].query, independent_marginals[0].evidence)
        result.result = sum(r.result for r in independent_marginals) / len(independent_marginals)

        for variable in variables:
            if variable.numeric:
                result.distributions[variable] = Numeric.merge([r.distributions[variable]
                                                                for r in independent_marginals],
                                                               weights=[r.result
                                                                        for r in independent_marginals], )
            elif variable.symbolic:
                result.distributions[variable] = Multinomial.merge([r.distributions[variable]
                                                                    for r in independent_marginals],
                                                                   weights=[r.result
                                                                            for r in independent_marginals], )

        return result