import pyximport
pyximport.install()

import os
import pickle
import pprint

import numpy as np
from matplotlib import pyplot as plt
from numpy import iterable

from dnutils import out
from jpt.learning.distributions import Multinomial, Bool, Histogram, Numeric
from jpt.learning.intervals import Interval
from jpt.learning.trees import JPT
from jpt.variables import Variable
from quantiles import Quantiles


def SymbolicType(name, values):
    t = type(name, (Multinomial,), {})
    t.values = list(values)
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


def restaurant():
    # declare variable types
    PatronsType = HistogramType('Patrons', ['3', '10', '20'])
    PriceType = HistogramType('Price', ['$', '$$', '$$$'])
    FoodType = HistogramType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = HistogramType('WaitEstimate', ['10', '30', '60', '120'])

    # define probs
    al = Variable('Alternatives', Bool)  # Alternatives(.2)
    ba = Variable('Bar', Bool)  # Bar(.2)
    fr = Variable('Friday', Bool)  # Friday(1/7.)
    hu = Variable('Hungry', Bool)  # Hungry(.8)
    pa = Variable('Patrons', PatronsType)  # PatronsType([.2, .6, .2])
    pr = Variable('Price', PriceType)  # Price([.1, .7, .2])
    ra = Variable('Rain', Bool)  # Rain(.3)
    re = Variable('Reservation', Bool)  # Reservation(.1)
    fo = Variable('Food', FoodType)  # Food([.1, .2, .4, .3])
    wa = Variable('WaitEst', WaitEstType)  # WaitEst([.3, .4, .2, .1])

    numsamples = 500
    variables = [ba, ra, re]  # [al, ba, fr, hu, pa, pr, ra, re, fo, wa]

    data = [[ba.dist(.2).sample_one(), ra.dist(.3).sample_one(), re.dist(.2).sample_one()] for _ in range(numsamples)]
    # data = [[ba.dist(.2).sample_one(), pa.dist([.2, .6, .2]).sample_one(), pr.dist([.1, .7, .2]).sample_one(), ra.dist(.3).sample_one(),
    #          re.dist(.1).sample_one(), fo.dist([.1, .2, .4, .3]).sample_one(), wa.dist([.3, .4, .2, .1]).sample_one()] for _ in range(numsamples)]

    jpt = JPT(variables, min_samples_leaf=5)
    jpt.learn(data)
    out(jpt)
    jpt.plot(directory='/home/mareike/Desktop/sebaimages', plotvars=[ba, ra, re], view=True)
    # candidates = jpt.apply({ba: True, re: False})
    candidates = jpt.infer({ba: True, re: False}, {ra: False})
    out(candidates)


def alarm():

    E = Variable('Earthquake', Bool)  # .02
    B = Variable('Burglary', Bool)  # Bool(.01)
    A = Variable('Alarm', Bool)
    M = Variable('MaryCalls', Bool)
    J = Variable('JohnCalls', Bool)

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

    tree = JPT([E, B, A, M, J])
    tree.learn(data)
    out(tree)
    # tree.plot(directory=os.path.abspath('/tmp'), plotvars=[E, B, A, M, J], view=False)

    # conditional
    # q = {A: True}
    # e = {E: False, B: True}

    # joint
    # q = {A: True, E: False, B: True}
    # e = {}

    # diagnostic
    q = {A: True}
    e = {M: True}
    out(f'P({",".join([f"{k.name}={v}" for k, v in q.items()])}{" | " if e else ""}{",".join([f"{k.name}={v}" for k, v in e.items()])})', tree.infer(q, e))


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
        data = np.array(pickle.load(fi))
    data_ = np.array(sorted([float(x) for x in data.T[0]]))

    quantiles = Quantiles(data_, epsilon=.0001)
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

    ObjectType = HistogramType('ObjectType', unique)

    x = Variable('PosX', Numeric)
    y = Variable('PosY', Numeric)
    o = Variable('Object', ObjectType)

    out('got data', data)

    jpt = JPT([x, y, o], name="Müslitree")
    jpt.learn(list(zip(*data)))

    jpt.plot()


def picklemuesli():
    f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    with open(f, 'rb') as fi:
        data = np.array(pickle.load(fi))

    transformed = []
    for c in data.T:
        try:
            transformed.append(np.array(c, dtype=float))
        except:
            transformed.append(np.array(c))

    print(transformed)
    with open(f, 'wb+') as fi:
        pickle.dump(transformed, fi)


def main(*args):

    # test_merge()
    # test_dists()
    # restaurant()
    # test_muesli()
    muesli_tree()
    # picklemuesli()
    # alarm()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
