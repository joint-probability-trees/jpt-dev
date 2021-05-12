import pyximport
pyximport.install()

import os
import pickle

import numpy as np
from numpy import iterable

from dnutils import out
from jpt.learning.distributions import Bool, Numeric, HistogramType, SymbolicType
from intervals import ContinuousSet as Interval
from jpt.learning.trees import JPT
from jpt.variables import Variable, SymbolicVariable, NumericVariable
from quantiles import Quantiles


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


def restaurant():
    # declare variable types
    PatronsType = SymbolicType('Patrons', ['some', 'full', 'none'])
    PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
    FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = SymbolicType('WaitEstimate', ['0-10', '10-30', '30-60', '>60'])

    # create variables
    minimp = 0.9
    al = SymbolicVariable('Alternatives', Bool, min_impurity_improvement=minimp)
    ba = SymbolicVariable('Bar', Bool, min_impurity_improvement=minimp)
    fr = SymbolicVariable('Friday', Bool, min_impurity_improvement=minimp)
    hu = SymbolicVariable('Hungry', Bool, min_impurity_improvement=minimp)
    pa = SymbolicVariable('Patrons', PatronsType, min_impurity_improvement=minimp)
    pr = SymbolicVariable('Price', PriceType, min_impurity_improvement=minimp)
    ra = SymbolicVariable('Rain', Bool, min_impurity_improvement=minimp)
    re = SymbolicVariable('Reservation', Bool, min_impurity_improvement=minimp)
    fo = SymbolicVariable('Food', FoodType, min_impurity_improvement=minimp)
    we = SymbolicVariable('WaitEst', WaitEstType, min_impurity_improvement=minimp)
    wa = SymbolicVariable('WillWait', Bool, min_impurity_improvement=minimp)

    # define probs
    numsamples = 500
    data = [[al.dist(6/12.).sample_one_label(),
             ba.dist(6/12.).sample_one_label(),
             fr.dist(5/12.).sample_one_label(),
             hu.dist(7/12.).sample_one_label(),
             pa.dist([4/12., 6/12., 2/12.]).sample_one_label(),
             pr.dist([7/12., 2/12., 3/12.]).sample_one_label(),
             ra.dist(4/12.).sample_one_label(),
             re.dist(5/12.).sample_one_label(),
             fo.dist([2/12., 4/12., 4/12., 2/12.]).sample_one_label(),
             we.dist([6/12., 2/12., 2/12., 2/12.]).sample_one_label(),
             wa.dist(.5).sample_one_label()] for _ in range(numsamples)]

    variables = [al, ba, fr, hu, pa, pr, ra, re, fo, we, wa]
    jpt = JPT(variables, name='Restaurant', min_samples_leaf=30, min_impurity_improvement=0)
    jpt.learn(data)
    out(jpt)
    jpt.plot(plotvars=variables, view=True)
    # candidates = jpt.apply({ba: True, re: False})
    q = {ba: True, re: False}
    e = {ra: False}
    res = jpt.infer(q, e)
    print(res.explain())


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


def test_merge():
    X = HistogramType('YesNo', ['Y', 'N'])
    mn1 = X([20, 30])
    out('MN1', mn1, mn1.p, mn1.d)

    mn2 = X([10, 12])
    out('MN2', mn2, mn2.p, mn2.d)

    mnmerged = X([30, 42])
    out('MNMERGED', mnmerged, mnmerged.p, mnmerged.d)

    mn3 = mn1 + mn2
    out('MN3 as merge of MN1 and MN2', mn3, mn3.p, mn3.d, mn3==mnmerged)

    mn2 += mn1
    out('MN2 after adding MN1', mn2, mn2.p, mn2.d, mn2 == mnmerged)


def test_dists():
    a = Bool()  # initialize empty then set data
    a.set_data([True, False, False, False, False, False, False, False, False, False])
    b = Bool([1, 9])  # set counts
    c = Bool(.1)  # set probability
    d = Bool([.1, .9])  # set both probabilities; not supposed to be used like that
    out(a)
    out(b)
    out(c)
    out(d)
    out(a == b, c == d)

    # prettyprinting tests for str und repr
    FoodType = HistogramType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    fo = Variable('Food', FoodType)
    dist = fo.dist([.1, .1, .1, .7])

    # should print without probs
    out('\n')
    print(dist)
    print(repr(dist))

    # should print with probs
    out('\n')
    print(a)
    print(repr(a))
    print(repr(fo.dist()))
    print(repr(Bool()))


def test_muesli():
    f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = pickle.load(fi)

    quantiles = Quantiles(data[0], epsilon=.0001)
    cdf_ = quantiles.cdf()
    d = Numeric(cdf=cdf_)

    interval = Interval(-2.05, -2.0)
    p = d.p(interval)
    out('query', interval, p)

    print(d.cdf.pfmt())
    d.plot(name='Müsli Beispiel', view=True)


def muesli_tree():
    f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = pickle.load(fi)

    unique, counts = np.unique(data[2], return_counts=True)

    ObjectType = SymbolicType('ObjectType', unique)

    x = NumericVariable('X', Numeric)
    y = NumericVariable('Y', Numeric)
    o = SymbolicVariable('Object', ObjectType)

    jpt = JPT([x, y, o], name="Müslitree", min_samples_leaf=10)
    jpt.learn(list(zip(*data)))

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    jpt.plot(plotvars=[x, y, o])


def main(*args):

    # test_merge()
    # test_dists()
    # test_muesli()
    restaurant()  # for bools and strings
    # muesli_tree()  # for numerics and strings
    # alarm()  # for bools


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
