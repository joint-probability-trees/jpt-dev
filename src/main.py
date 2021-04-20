from collections import deque

from dnutils import out
from numpy import iterable

from jpt.sampling import wsample, wchoice
from jpt.learning.example import Example, BooleanFeature
from jpt.learning.trees import StructRegTree


class Multinomial:

    values = None

    def __init__(self, p):
        if not iterable(p):
            raise ValueError('Probabilities must be an iterable with at least 2 elements, got %s' % p)
        # if len(values) != len(p):
        #     raise ValueError('Number of values and probabilities must coincide.')
        self.p = p
        # self.values = values

    def sample(self, n):
        return wsample(self.values, self.p, n)

    def sample_one(self):
        return wchoice(self.values, self.p)

    def __getitem__(self, value):
        return self.p[self.values.index(value)]

    def __setitem__(self, value, p):
        self.p[self.values.index(value)] = p


def SymbolicType(name, values):
    t = type(name, (Multinomial,), {})
    t.values = list(values)
    return t


class Bool(Multinomial):

    values = [True, False]

    def __init__(self, p):
        if not iterable(p):
            p = [p, 1 - p]
        super().__init__(p)

    def __getitem__(self, v):
        return self.p[v]

    def __setitem__(self, v, p):
        self.p[v] = p
        self.p[1 - v] = 1 - p


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


def main(*args):

    YesNo = SymbolicType('YesNo', ['Y', 'N'])

    v = YesNo([.1, .9])
    out(v)

    E = Bool(.02)
    B = Bool(.01)

    A = Conditional(Bool, [E, B])
    A[True, True] = Bool(.95)
    A[True, False] = Bool(.94)
    A[False, True] = Bool(.29)
    A[False, False] = Bool(.001)

    M = Conditional(Bool, [A])
    M[True] = Bool(.7)
    M[False] = Bool(.01)

    J = Conditional(Bool, [A])
    J[True] = Bool(.9)
    J[False] = Bool(.05)

    # Construct the CSV for learning
    print(E, B, A, M, J)
    data = deque()
    for i in range(1000):
        e = E.sample_one()
        b = B.sample_one()
        a = A.sample_one([e, b])
        m = M.sample_one(a)
        j = J.sample_one(a)

        vector = [BooleanFeature(e, 'E'),
                  BooleanFeature(b, 'B'),
                  BooleanFeature(a, 'A'),
                  BooleanFeature(m, 'M'),
                  BooleanFeature(j, 'J')]
        data.append(Example(
            x=vector,
            t=vector
        ))
        print(data[-1])

    tree = StructRegTree()
    tree.learn(data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
