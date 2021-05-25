import numpy as np
from numpy import iterable

from dnutils import out
from jpt.learning.distributions import Bool
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


class Conditional:

    def __init__(self, typ, conditionals):
        self.type = typ
        self.conditionals = conditionals
        self.p = {}

    def __getitem__(self, values):
        if not iterable(values):
            values = (values,)
        return self.p[tuple(values)]

    def __setitem__(self, evidence, dist):
        if not iterable(evidence):
            evidence = (evidence,)
        self.p[evidence] = dist

    def sample(self, evidence, n):
        if not iterable(evidence):
            evidence = (evidence,)
        return self.p[tuple(evidence)].sample(n)

    def sample_one(self, evidence):
        if not iterable(evidence):
            evidence = (evidence,)
        return self.p[tuple(evidence)].sample_one()


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
        for i in range(1000):
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

        tree = JPT(variables=[E, B, A, M, J], name='Alarm', min_impurity_improvement=0)
        tree.learn(data)
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

    tree = JPT(variables=[E, B, A, M, J], name='Alarm', min_impurity_improvement=0)
    tree.learn(data)
    out(tree)
    res = tree.infer(q, e)
    res.explain()
    print('AVG', c/t)
    tree.plot(plotvars=[E, B, A, M, J])


def main(*args):
    alarm()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
