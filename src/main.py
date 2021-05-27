from datetime import datetime

import pandas as pd
import pyximport

from jpt.base.sampling import wchoice

pyximport.install()

import os
import pickle

import numpy as np
from numpy import iterable

from dnutils import out
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
    jpt.plot(plotvars=variables, view=True, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Restaurant'))
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
    jpt.plot(plotvars=variables, view=True, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Restaurant'))
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
    tree.plot(plotvars=[E, B, A, M, J], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Alarm'))


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

    data = pd.read_csv('../examples/data/muesli.csv')
    d = np.array(sorted(data['X']), dtype=np.float64)

    quantiles = Quantiles(d, epsilon=.0001)
    cdf_ = quantiles.cdf()
    d = Numeric(quantile=quantiles)

    interval = Interval(-2.05, -2.0)
    p = d.p(interval)
    out('query', interval, p)

    print(d.cdf.pfmt())
    d.plot(title='Piecewise Linear CDF of Breakfast Data', fname='BreakfastPiecewise', xlabel='X', view=True, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))


def muesli_tree():
    # f = os.path.join('../' 'examples', 'data', 'human_muesli.pkl')

    data = []
    # with open(f, 'rb') as fi:
    #     data = pickle.load(fi)
    # data = pd.read_pickle('../examples/data/human_muesli.dat')
    data = pd.read_csv('../examples/data/muesli.csv')
    # unique, counts = np.unique(data[2], return_counts=True)
    print(data)
    ObjectType = SymbolicType('ObjectType', data['Class'].unique())

    x = NumericVariable('X', Numeric)
    y = NumericVariable('Y', Numeric)
    o = SymbolicVariable('Object', ObjectType)
    s = SymbolicVariable('Success', Bool)

    jpt = JPT([x, y, o, s], name="Breakfasttree", min_samples_leaf=5)
    jpt.learn(columns=data.values.T)
    jpt.plot(plotvars=[x, y, o], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Muesli'))

    for clazz in data['Class'].unique():
        out(jpt.infer(query={o: clazz}, evidence={x: [.9, None], y: [None, .45]}))
    print()

    for clazz in data['Class'].unique():
        for exp in jpt.expectation([x, y], evidence={o: clazz}, confidence_level=.1):
            out(exp)

    # plotting vars does not really make sense here as all leaf-cdfs of numeric vars are only piecewise linear fcts
    # --> only for testing
    # jpt.plot(plotvars=[x, y, o, s])

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
    jpt.plot(plotvars=[price, t, d, p], directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Tourism'))  # plotvars=[price, t]


def neemdata():
    # location of NEEM_SetTable_Breakfast.tar.xz
    ltargz = 'https://seafile.zfn.uni-bremen.de/f/fa5a760d89234cfc83ad/?dl=1'
    df = pd.read_csv(ltargz, compression='xz', delimiter=';', sep=';', skip_blank_lines=True, header=0,
                     index_col=False,
                     names=['id', 'type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'parent', 'next', 'previous', 'object_acted_on', 'object_type', 'bodyPartsUsed', 'arm', 'grasp', 'effort'],
                     usecols=['type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'object_acted_on', 'bodyPartsUsed', 'arm'],
                     na_values=['type', 'startTime', 'endTime', 'duration', 'success', 'failure', 'object_acted_on', 'bodyPartsUsed', 'arm', np.inf])

    # set default values for empty, infinity or nan values and remove nan rows
    df = df[df['endTime'].notna()]  # this not only removes the lines with endTime=NaN, but in particular all lines where each feature is NaN
    df.replace([np.inf, -np.inf], -1, inplace=True)
    df['object_acted_on'] = df['object_acted_on'].fillna('DEFAULTOBJECT')
    df['bodyPartsUsed'] = df['bodyPartsUsed'].fillna('DEFAULTBP')
    df['arm'] = df['arm'].fillna('DEFAULTARM')
    df['failure'] = df['failure'].fillna('DEFAULTFAIL')
    df['startTime'] = df['startTime'].fillna(-1)
    df['endTime'] = df['endTime'].fillna(-1)
    df['duration'] = df['duration'].fillna(-1)

    # type declarations
    tpTYPE = SymbolicType('type', df['type'].unique())
    failTYPE = SymbolicType('failure', df['failure'].unique())
    oaoTYPE = SymbolicType('object_acted_on', df['object_acted_on'].unique())
    bpuTYPE = SymbolicType('bodyPartsUsed', df['bodyPartsUsed'].unique())
    armTYPE = SymbolicType('arm', df['arm'].unique())

    # variable declarations
    tp = SymbolicVariable('type', tpTYPE)
    st = NumericVariable('startTime', Numeric, haze=.1)
    et = NumericVariable('endTime', Numeric, haze=.1)
    dur = NumericVariable('duration', Numeric, haze=.1)
    succ = SymbolicVariable('success', Bool)
    fail = SymbolicVariable('failure', failTYPE)
    oao = SymbolicVariable('object_acted_on', oaoTYPE)
    bpu = SymbolicVariable('bodyPartsUsed', bpuTYPE)
    arm = SymbolicVariable('arm', armTYPE)

    vars = [tp, st, et, dur, succ, fail, oao, bpu, arm]
    jpt = JPT(variables=vars, name="NEEMs", min_samples_leaf=500)
    out(f'Learning sebadata-Tree...')
    jpt.learn(columns=df.values.T)
    out(f'Done! Plotting...')
    jpt.plot(filename=jpt.name, plotvars=vars, directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-NEEMdata'), view=True)


def main(*args):

    # test_merge()
    # test_dists()
    # restaurant()  # for bools and strings
    # test_muesli()
    # muesli_tree()  # for numerics and strings
    # picklemuesli()
    # alarm()  # for bools
    # fraport()
    neemdata()  # for dataset from sebastian (fetching milk from fridge)
    # tourism()  # for flight data


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
