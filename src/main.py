from collections import defaultdict

from dnutils import out
from numpy import iterable

from sampling import wsample, wchoice
from variables import MultinomialRV, Uniform, Domain, SymbolicDistribution, P

BOOL = Domain(['T', 'F'])


def sample(dist):
    pass


class Multinomial:

    def __init__(self, p, values):
        if not iterable(p):
            raise ValueError('Probabilities must be an iterable with at least 2 elements, got %s' % p)
        if len(values) != len(p):
            raise ValueError('Number of values and probabilities must coincide.')
        self.p = p
        self.values = values

    def sample(self, n):
        return wsample(self.values, self.p, n)

    def sample_one(self):
        return wchoice(self.values, self.p)

    def __getitem__(self, value):
        return self.p[self.values.index(value)]

    def __setitem__(self, value, p):
        self.p[self.values.index(value)] = p


class Bool(Multinomial):

    def __init__(self, p):
        if not iterable(p):
            p = [p, 1 - p]
        values = [True, False]
        super().__init__(p, values)


class Conditional:

    def __init__(self, conditionals):
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


def main(*args):
    E = Bool(.002)
    B = Bool(.001)

    A = Conditional([E, B])
    A[True, True] = Bool(.95)
    A[True, False] = Bool(.94)
    A[False, True] = Bool(.29)
    A[False, False] = Bool(.001)

    M = Conditional([A])
    M[True] = Bool(.7)
    M[False] = Bool(.01)

    J = Conditional([A])
    J[True] = Bool(.9)
    J[False] = Bool(.05)

    for i in range(1000):
        e = E.sample_one()
        b = B.sample_one()
        a = A.sample_one([e, b])
        m = M.sample_one(a)
        j = J.sample_one(a)
        print(e, b, a, m, j)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
