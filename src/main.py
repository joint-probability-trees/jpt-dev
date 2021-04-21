import csv
import os
import re
import traceback
from collections import deque, defaultdict

from numpy import iterable

from dnutils import out
from jpt.learning.distributions import Multinomial, Bool, Histogram
from jpt.learning.intervals import Interval

from jpt.sampling import wsample, wchoice
from jpt.learning.example import Example, BooleanFeature, SymbolicFeature, NumericFeature
from jpt.learning.trees import StructRegTree


def SymbolicType(name, values):
    t = type(name, (Multinomial,), {})
    t.values = list(values)
    return t


def HistogramType(name, values, d=1):
    t = type(name, (Histogram,), {"d": d})
    t.values = list(values)
    return t


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
            t=vector,
            identifier=str(i)
        ))
        print(data[-1])

    tree = StructRegTree()
    tree.learn(data)
    tree.plot(view=True)

    # ----------------------

    X = HistogramType('YesNo', ['Y', 'N'])
    mn1 = X([20, 30], d=100)
    out('MN1', mn1, mn1.p, mn1.d)

    mn2 = X([10, 12], d=200)
    out('MN2', mn2, mn2.p, mn2.d)

    mn3 = mn1 + mn2
    out('MN3 as merge of MN1 and MN2', mn3, mn3.p, mn3.d)

    mn2 += mn1
    out('MN2 after adding MN1', mn2, mn2.p, mn2.d)
    out(mn2 == mn3)

    # ----------------------


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
