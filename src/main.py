from collections import deque

from numpy import iterable

from dnutils import out
from jpt.learning.distributions import Multinomial, Bool, Histogram
from jpt.learning.example import Example, BooleanFeature
from jpt.learning.trees import JPT


def SymbolicType(name, values):
    t = type(name, (Multinomial,), {})
    t.values = list(values)
    return t


def BoolType(name):
    t = type(name, (Bool,), {})
    return t


def HistogramType(name, values):
    t = type(name, (Histogram,), {})
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

    # YesNo = SymbolicType('YesNo', ['Y', 'N'])
    #
    # v = YesNo([.1, .9])
    # out(v)
    #
    # E = Bool(.02)
    # B = Bool(.01)
    #
    # A = Conditional(Bool, [E, B])
    # A[True, True] = Bool(.95)
    # A[True, False] = Bool(.94)
    # A[False, True] = Bool(.29)
    # A[False, False] = Bool(.001)
    #
    # M = Conditional(Bool, [A])
    # M[True] = Bool(.7)
    # M[False] = Bool(.01)
    #
    # J = Conditional(Bool, [A])
    # J[True] = Bool(.9)
    # J[False] = Bool(.05)
    #
    # # Construct the CSV for learning
    # print(E, B, A, M, J)
    # data = deque()
    # for i in range(1000):
    #     e = E.sample_one()
    #     b = B.sample_one()
    #     a = A.sample_one([e, b])
    #     m = M.sample_one(a)
    #     j = J.sample_one(a)
    #
    #     vector = [BooleanFeature(e, 'E'),
    #               BooleanFeature(b, 'B'),
    #               BooleanFeature(a, 'A'),
    #               BooleanFeature(m, 'M'),
    #               BooleanFeature(j, 'J')]
    #     data.append(Example(
    #         x=vector,
    #         t=vector,
    #         identifier=str(i)
    #     ))
    #     print(data[-1])
    #
    # tree = JPT([YesNo])
    # tree.learn(data)
    # tree.plot(directory='/home/mareike/Desktop/sebaimages', view=True)

    # ---------------------- TEST MERGE HISTOGRAMS

    # X = HistogramType('YesNo', ['Y', 'N'])
    # mn1 = X([20, 30])
    # out('MN1', mn1, mn1.p, mn1.d)
    #
    # mn2 = X([10, 12])
    # out('MN2', mn2, mn2.p, mn2.d)
    #
    # mn3 = mn1 + mn2
    # out('MN3 as merge of MN1 and MN2', mn3, mn3.p, mn3.d)
    #
    # mn2 += mn1
    # out('MN2 after adding MN1', mn2, mn2.p, mn2.d)
    # out(mn2 == mn3)

    # ---------------------- TEST PLOT HISTOGRAM/MULTINOMIAL

    # # YN = SymbolicType('YN', ['Y', 'N'])
    # YN = HistogramType('YN', ['Y', 'N'])
    #
    # # yn = YN([.1, .9])
    # yn = YN([20, 30])
    # yn.plot(name='TestPlot Histogram', directory='/home/mareike/Desktop/sebaimages', pdf=False, view=True, horizontal=True)

    # ---------------------- New Tree structure

    # declare variable types
    Alternative = BoolType('Alternative')
    Bar = BoolType('Bar')
    Friday = BoolType('Friday')
    Hungry = BoolType('Hungry')
    Patrons = SymbolicType('Patrons', [3, 10, 20])
    Price = SymbolicType('Price', ['$', '$$', '$$$'])
    Rain = BoolType('Rain')
    Reservation = BoolType('Reservation')
    Food = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEst = SymbolicType('WaitEstimate', [10, 30, 60, 120])

    # define probs
    al = Alternative(.4)
    ba = Bar(.2)
    fr = Friday(1/7.)
    hu = Hungry(.8)
    pa = Patrons([.2, .6, .2])
    pr = Price([.1, .7, .2])
    ra = Rain(.3)
    re = Reservation(.1)
    fo = Food([.1, .2, .4, .3])
    wa = WaitEst([.3, .4, .2, .1])

    numsamples = 1000
    # variables = [Alternative, Bar, Friday, Hungry, Patrons, Price, Rain, Reservation, Food, WaitEst]
    variables = [al, ba, fr, hu, pa, pr, ra, re, fo, wa]

    data = [[al.sample_one(), ba.sample_one(), fr.sample_one(), hu.sample_one(), pa.sample_one(),
            pr.sample_one(), ra.sample_one(), re.sample_one(), fo.sample_one(), wa.sample_one()] for _ in range(numsamples)]

    # data = [al.sample(numsamples), ba.sample(numsamples), fr.sample(numsamples), hu.sample(numsamples), pa.sample(numsamples),
    #         pr.sample(numsamples), ra.sample(numsamples), re.sample(numsamples), fo.sample(numsamples), wa.sample(numsamples)]

    jpt = JPT(variables, min_samples_leaf=5)
    jpt.learn(data)
    out(jpt)
    jpt.plot(directory='/home/mareike/Desktop/sebaimages', view=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
