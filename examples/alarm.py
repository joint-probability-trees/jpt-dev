"""Bayesian network reasoning: the alarm network.

Implements the classic alarm network with five Boolean
variables (Earthquake, Burglary, Alarm, MaryCalls,
JohnCalls). Conditional distributions are defined
manually, then training data is sampled and a JPT is
learned. The resulting model is queried for conditional
probabilities and human-readable explanations.

Demonstrates:
    - Conditional distributions for data sampling
    - Bool variables for binary events
    - Inference with evidence via ``infer()``
    - Human-readable explanations via ``explain()``
    - Posterior distributions via ``posterior()``
"""
import tempfile

import numpy as np

from jpt.distributions import Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


# -------------------------------------------------------


class Conditional:
    """A simple conditional distribution table for
    sampling from a Bayesian network.

    Maps tuples of parent values to distribution
    instances that can be sampled.
    """

    def __init__(self, typ, conditionals):
        self.type = typ
        self.conditionals = conditionals
        self.p = {}

    def __setitem__(self, evidence, dist):
        if not isinstance(evidence, tuple):
            evidence = (evidence,)
        self.p[evidence] = dist

    def sample_one(self, evidence):
        if not isinstance(evidence, (tuple, list)):
            evidence = (evidence,)
        return self.p[tuple(evidence)].sample_one()


# -------------------------------------------------------


def main(visualize=True):
    """Define the alarm network, sample data, learn a
    JPT, and run inference queries.

    :param visualize: whether to show interactive plots
    """
    # Define the five Boolean variables
    E = SymbolicVariable('Earthquake', Bool)
    B = SymbolicVariable('Burglary', Bool)
    A = SymbolicVariable('Alarm', Bool)
    M = SymbolicVariable('MaryCalls', Bool)
    J = SymbolicVariable('JohnCalls', Bool)

    # Define the conditional distribution for Alarm
    # given Burglary and Earthquake
    A_ = Conditional(Bool, [B.domain, E.domain])
    A_[True, True] = Bool().set(.95)
    A_[True, False] = Bool().set(.94)
    A_[False, True] = Bool().set(.29)
    A_[False, False] = Bool().set(.001)

    # Define the conditional distributions for Mary
    # and John calling given Alarm
    M_ = Conditional(Bool, [A])
    M_[True] = Bool().set(.7)
    M_[False] = Bool().set(.01)

    J_ = Conditional(Bool, [A])
    J_[True] = Bool().set(.9)
    J_[False] = Bool().set(.05)

    # Sample training data from the network
    data = []
    for _ in range(10000):
        e = E.distribution().set(.2).sample_one()
        b = B.distribution().set(.1).sample_one()
        a = A_.sample_one([e, b])
        m = M_.sample_one(a)
        j = J_.sample_one(a)
        data.append([e, b, a, m, j])

    # Verify sampled marginal distributions
    print('Marginal distributions from sampled data:')
    d = np.array(data).T
    for var, x in zip([E, B, A, M, J], d):
        unique, counts = np.unique(x, return_counts=True)
        print(
            f'  {var.name}: '
            + ', '.join(
                f'{u}={c / sum(counts):.3f}'
                for u, c in zip(unique, counts)
            )
        )

    # Learn the JPT from the sampled data
    tree = JPT(
        variables=[E, B, A, M, J],
        min_impurity_improvement=0
    )
    tree.learn(np.array(data))

    # Query: P(Alarm=True | MaryCalls=True)
    q = {A: True}
    e = {M: True}
    prob = tree.infer(q, e)
    print(f'\nP(Alarm=True | MaryCalls=True)'
          f' = {prob:.4f}')

    # Posterior distribution of Alarm given Mary calls
    posterior = tree.posterior(
        [A], evidence={M: True}
    )[A].probabilities
    print(f'\nP(Alarm | MaryCalls=True) = {posterior}')

    # Plot the learned tree
    if visualize:
        out_dir = tempfile.mkdtemp(prefix='jpt-alarm-')
        tree.plot(
            plotvars=[E, B, A, M, J],
            directory=out_dir,
            view=True
        )


if __name__ == '__main__':
    main(visualize=True)
