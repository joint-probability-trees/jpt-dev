import pyximport
pyximport.install()
import os
from datetime import datetime

import numpy as np
from numpy import iterable

from dnutils import out
from jpt.base.utils import Conditional
from jpt.learning.distributions import Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


def alarm():

    E = SymbolicVariable('Earthquake', Bool)  # .02
    B = SymbolicVariable('Burglary', Bool)  # Bool(.01)
    A = SymbolicVariable('Alarm', Bool)
    M = SymbolicVariable('MaryCalls', Bool)
    J = SymbolicVariable('JohnCalls', Bool)

    A_ = Conditional(Bool, [E.domain, B.domain])
    A_[True, True] = Bool(.95)
    A_[True, False] = Bool(.94)
    A_[False, True] = Bool(.29)
    A_[False, False] = Bool(.001)

    M_ = Conditional(Bool, [A])
    M_[True] = Bool(.7)
    M_[False] = Bool(.01)

    J_ = Conditional(Bool, [A])
    J_[True] = Bool(.9)
    J_[False] = Bool(.05)

    c = 0.
    t = 10
    for i in range(t):

        # Construct the CSV for learning
        data = []
        for i in range(10000):
            e = E.dist(.2).sample_one()
            b = B.dist(.1).sample_one()
            a = A_.sample_one([e, b])
            m = M_.sample_one(a)
            j = J_.sample_one(a)

            data.append([e, b, a, m, j])

        # sample check
        out('Probabilities as determined by sampled data')
        d = np.array(data).T
        for var, x in zip([E, B, A, M, J], d):
            unique, counts = np.unique(x, return_counts=True)
            out(var.name, list(zip(unique, counts, counts/sum(counts))))

        tree = JPT(variables=[E, B, A, M, J], min_impurity_improvement=0)
        tree.learn(data)
        # tree.sklearn_tree()
        # tree.plot(plotvars=[E, B, A, M, J])
        # conditional
        # q = {A: True}
        # e = {E: False, B: True}

        # joint
        # q = {A: True, E: False, B: True}
        # e = {}

        # diagnostic
        q = {A: True}
        e = {M: True}

        c += tree.infer(q, e).result

    # tree = JPT(variables=[E, B, A, M, J], name='Alarm', min_impurity_improvement=0)
    # tree.learn(data)
    # out(tree)
    res = tree.infer(q, e)
    print(res.explain())

    # print_stopwatches()
    # print('AVG', c/t)
    tree.plot(plotvars=[E, B, A, M, J], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Alarm'))


def main(*args):
    alarm()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
