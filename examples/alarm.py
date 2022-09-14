import os
from datetime import datetime

import numpy as np

from dnutils import out
from matplotlib import pyplot as plt

from jpt.base.utils import Conditional
from jpt.distributions import Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


def alarm(verbose=True):
    plt.close()
    E = SymbolicVariable('Earthquake', Bool)  # .02
    B = SymbolicVariable('Burglary', Bool)  # Bool(.01)
    A = SymbolicVariable('Alarm', Bool)
    M = SymbolicVariable('MaryCalls', Bool)
    J = SymbolicVariable('JohnCalls', Bool)

    A_ = Conditional(Bool, [B.domain, E.domain])
    A_[True, True] = Bool().set(.95)
    A_[True, False] = Bool().set(.94)
    A_[False, True] = Bool().set(.29)
    A_[False, False] = Bool().set(.001)

    M_ = Conditional(Bool, [A])
    M_[True] = Bool().set(.7)
    M_[False] = Bool().set(.01)

    J_ = Conditional(Bool, [A])
    J_[True] = Bool().set(.9)
    J_[False] = Bool().set(.05)

    c = 0.
    t = 1
    for i in range(t):

        # Construct the CSV for learning
        data = []
        for i in range(10000):
            e = E.distribution().set(.2).sample_one()
            b = B.distribution().set(.1).sample_one()
            a = A_.sample_one([e, b])
            m = M_.sample_one(a)
            j = J_.sample_one(a)

            data.append([e, b, a, m, j])

        # sample check
        if verbose:
            out('Probabilities as determined by sampled data')
        d = np.array(data).T
        for var, x in zip([E, B, A, M, J], d):
            unique, counts = np.unique(x, return_counts=True)
            if verbose:
                out(var.name, list(zip(unique, counts, counts/sum(counts))))

        tree = JPT(variables=[E, B, A, M, J], min_impurity_improvement=0)
        tree.learn(data)

        # diagnostic
        q = {A: True}
        e = {M: True}

        c += tree.infer(q, e).result

    res = tree.infer(q, e)
    explanation = res.explain()
    posterior = tree.posterior([A], evidence={M: True})[A].probabilities

    if verbose:
        print(explanation)
        print(posterior)
        print('AVG', c / t)
        tree.plot(plotvars=[E, B, A, M, J],
                  directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Alarm'))


def main(*args):
    alarm()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
