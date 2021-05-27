import pprint

import pandas as pd
import pyximport
from matplotlib import pyplot as plt

pyximport.install()

import os
import pickle

import numpy as np
from numpy import iterable

from dnutils import out, first
from jpt.learning.distributions import Bool, Numeric, HistogramType, SymbolicType
from jpt.base.intervals import ContinuousSet as Interval
from jpt.trees import JPT
from jpt.variables import Variable, SymbolicVariable, NumericVariable
from jpt.base.quantiles import Quantiles


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
    PatronsType = SymbolicType('Patrons', ['Some', 'Full', 'None'])
    PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
    FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

    # create variables
    al = SymbolicVariable('Alternatives', Bool)
    ba = SymbolicVariable('Bar', Bool)
    fr = SymbolicVariable('Friday', Bool)
    hu = SymbolicVariable('Hungry', Bool)
    pa = SymbolicVariable('Patrons', PatronsType)
    pr = SymbolicVariable('Price', PriceType)
    ra = SymbolicVariable('Rain', Bool)
    re = SymbolicVariable('Reservation', Bool)
    fo = SymbolicVariable('Food', FoodType)
    we = SymbolicVariable('WaitEstimate', WaitEstType)
    wa = SymbolicVariable('WillWait', Bool)

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
    out(f'P({",".join([f"{k.name}={v}" for k, v in q.items()])}{" | " if e else ""}'
        f'{",".join([f"{k.name}={v}" for k, v in e.items()])}) = {res.result}')
    print(res.explain())


def restaurantsample():
    import pandas as pd
    d = pd.read_csv(os.path.join('../', 'examples', 'data', 'restaurant.csv'))

    # declare variable types
    PatronsType = SymbolicType('Patrons', ['Some', 'Full', 'None'])
    PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
    FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

    # create variables
    al = SymbolicVariable('Alternatives', Bool)
    ba = SymbolicVariable('Bar', Bool)
    fr = SymbolicVariable('Friday', Bool)
    hu = SymbolicVariable('Hungry', Bool)
    pa = SymbolicVariable('Patrons', PatronsType)
    pr = SymbolicVariable('Price', PriceType)
    ra = SymbolicVariable('Rain', Bool)
    re = SymbolicVariable('Reservation', Bool)
    fo = SymbolicVariable('Food', FoodType)
    we = SymbolicVariable('WaitEstimate', WaitEstType)
    wa = SymbolicVariable('WillWait', Bool)

    variables = [pa, hu, fo, fr, al, ba, pr, ra, re, we, wa]

    def rec(vars, vals):
        if not vars:
            return [v[1] for v in vals]

        d = hilfsfunktion(vars[0], vals)
        sample = wchoice(vars[0].domain.labels, d)
        return rec(vars[1:], vals+[(vars[0], sample)])

    def hilfsfunktion(var, vals):
        d_ = d
        for v, val in vals:
            d_ = d_[d_[v.name] == val]

        dist = [len(d_[d_[var.name] == l])/len(d_) for l in var.domain.labels]
        return dist

    data = [rec(variables, []) for _ in range(500)]

    jpt = JPT(variables, name='Restaurant', min_samples_leaf=30, min_impurity_improvement=0)
    jpt.learn(rows=data)
    out(jpt)
    jpt.plot(plotvars=variables, view=True)
    # candidates = jpt.apply({ba: True, re: False})
    q = {ba: True, re: False}
    e = {ra: False}
    res = jpt.infer(q, e)
    out(f'P({",".join([f"{k.name}={v}" for k, v in q.items()])}{" | " if e else ""}'
        f'{",".join([f"{k.name}={v}" for k, v in e.items()])}) = {res.result}')
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

        tree = JPT(variables=[E, B, A, M, J], name='Alarm', min_impurity_improvement=0)
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
    tree.plot()


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
    # f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    # with open(f, 'rb') as fi:
    #     data = pickle.load(fi)
    data = pd.read_pickle('../examples/data/human_muesli.dat')
    # unique, counts = np.unique(data[2], return_counts=True)
    print(data)
    ObjectType = SymbolicType('ObjectType', data['Class'].unique())

    x = NumericVariable('X', Numeric)
    y = NumericVariable('Y', Numeric)
    o = SymbolicVariable('Object', ObjectType)

    jpt = JPT([x, y, o], name="Müslitree", min_samples_leaf=5)
    jpt.learn(columns=data.values.T)

    for clazz in data['Class'].unique():
        print(jpt.infer(query={o: clazz}, evidence={x: [.9, None], y: [None, .45]}))

    for clazz in data['Class'].unique():
        for exp in jpt.expectation([x, y], evidence={o: clazz}, confidence_level=.1):
            print(exp)

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    jpt.plot(plotvars=[x, y, o])

    # q = {o: ("BowlLarge_Bdvg", "JaNougatBits_UE0O"), x: [.812, .827]}
    # r = jpt.reverse(q)
    # out('Query:', q, 'result:', pprint.pformat(r))


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

    with open(f, 'wb+') as fi:
        pickle.dump(transformed, fi)


def tourism():
    df = pd.read_csv('../examples/data/tourism.csv')
    df['Price'] *= 2000
    df['DoY'] *= 710

    DestinationType = SymbolicType('DestinationType', df['Destination'].unique())
    PersonaType = SymbolicType('PersonaType', df['Persona'].unique())

    price = NumericVariable('Price', Numeric)
    t = NumericVariable('Time', Numeric, haze=.1)
    d = SymbolicVariable('Destination', DestinationType)
    p = SymbolicVariable('Persona', PersonaType)

    jpt = JPT(variables=[price, t, d, p], name="Tourism", min_samples_leaf=15)
    # out(df.values.T)
    jpt.learn(columns=df.values.T[1:])

    for clazz in df['Destination'].unique():
        print(jpt.infer(query={d: clazz}, evidence={t: 300}))
        for exp in jpt.expectation([t, price], evidence={d: clazz}, confidence_level=.95):
            print(exp)
    for persona in df['Persona'].unique():
        for exp in jpt.expectation([t, price], evidence={p: persona}, confidence_level=.95):
            print(exp)
    jpt.plot()  # plotvars=[price, t]


def regression():

    def f(x):
        """The function to predict."""
        return x * np.sin(x)

    # ----------------------------------------------------------------------
    #  First the noiseless case
    POINTS = 1000
    X = np.atleast_2d(np.random.uniform(0, 10.0, size=int(POINTS / 2))).T
    X = np.vstack((np.atleast_2d(np.random.uniform(-20, 0.0, size=int(POINTS / 2))).T, X))
    X = X.astype(np.float32)

    # Observations
    y = f(X).ravel()

    dy = 1.5 + .5 * np.random.random(y.shape)
    noise = np.random.normal(0, dy)
    y += noise
    y = y.astype(np.float32)

    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
    xx = np.atleast_2d(np.linspace(-25, 20, 500)).T
    xx = xx.astype(np.float32)

    varx = NumericVariable('x', Numeric)
    vary = NumericVariable('y', Numeric)

    jpt = JPT(variables=[varx, vary], min_samples_leaf=10)
    jpt.learn(columns=[X.ravel(), y])

    my_predictions = [first(jpt.expectation([vary], evidence={varx: x_})) for x_ in xx.ravel()]
    y_pred_ = [p.result for p in my_predictions]
    y_lower_ = [p.lower for p in my_predictions]
    y_upper_ = [p.upper for p in my_predictions]


    # Plot the function, the prediction and the 90% confidence interval based on
    # the MSE
    fig = plt.figure()
    plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
    plt.plot(X, y, '.', color='gray', markersize=5, label=u'Observations')
    plt.plot(xx, y_pred_, 'm-', label=u'My Prediction')
    plt.plot(xx, y_lower_, 'y--')
    plt.plot(xx, y_upper_, 'y--')
    # plt.plot(xx, mytree.regressor.predict(xx), 'b-')
    # plt.fill(np.concatenate([xx, xx[::-1]]),
    #          np.concatenate([y_upper_, y_lower_[::-1]]),
    #          alpha=.5, fc='y', ec='None', label='my 90% prediction interval')
    # plt.plot(xx, y_pred, 'r-', label=u'Prediction')
    # plt.plot(xx, y_upper, 'k-')
    # plt.plot(xx, y_lower, 'k-')
    # plt.fill(np.concatenate([xx, xx[::-1]]),
    #          np.concatenate([y_upper, y_lower[::-1]]),
    #          alpha=.5, fc='b', ec='None', label='90% prediction interval')
    # mytree.draw('bla')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.show()


def main(*args):

    # test_merge()
    # test_dists()
    # restaurant()  # for bools and strings
    # test_muesli()
    # muesli_tree()  # for numerics and strings
    # picklemuesli()
    # alarm()  # for bools
    # tourism()
    regression()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
