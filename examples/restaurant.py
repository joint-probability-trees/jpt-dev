import os
from datetime import datetime

import pandas as pd

from jpt.base.sampling import wchoice
from jpt.distributions import Bool, SymbolicType, IntegerType
from jpt.trees import JPT
from jpt.variables import SymbolicVariable, IntegerVariable


def restaurant_manual_sample(visualize=True):
    # generate JPT from data based on manually set distributions
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
    data = [[al.distribution().set(6/12.).sample_one(),
             ba.distribution().set(6/12.).sample_one(),
             fr.distribution().set(5/12.).sample_one(),
             hu.distribution().set(7/12.).sample_one(),
             pa.distribution().set([4/12., 6/12., 2/12.]).sample_one(),
             pr.distribution().set([7/12., 2/12., 3/12.]).sample_one(),
             ra.distribution().set(4/12.).sample_one(),
             re.distribution().set(5/12.).sample_one(),
             fo.distribution().set([2/12., 4/12., 4/12., 2/12.]).sample_one(),
             we.distribution().set([6/12., 2/12., 2/12., 2/12.]).sample_one(),
             wa.distribution().set(.5).sample_one()] for _ in range(numsamples)]

    variables = [al, ba, fr, hu, pa, pr, ra, re, fo, we, wa]
    jpt = JPT(variables, min_samples_leaf=30, min_impurity_improvement=0)
    jpt.learn(data)

    jpt.plot(plotvars=variables,
             view=visualize,
             directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Restaurant'))

    q = {ba: True, re: False}
    e = {ra: False}

    res = jpt.infer(q, e)
    print(res)


def restaurant_mixed_type_variables(visualize=True):
    PatronsType = IntegerType('Patrons', lmin=0, lmax=2)
    PriceType = IntegerType('Price', lmin=1, lmax=3)
    FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = SymbolicType('WaitEstimate', ['0--10', '10--30', '30--60', '>60'])

    # create variables
    al = SymbolicVariable('Alternatives', Bool)
    ba = SymbolicVariable('Bar', Bool)
    fr = SymbolicVariable('Friday', Bool)
    hu = SymbolicVariable('Hungry', Bool)
    pa = IntegerVariable('Patrons', PatronsType)
    pr = IntegerVariable('Price', PriceType)
    ra = SymbolicVariable('Rain', Bool)
    re = SymbolicVariable('Reservation', Bool)
    fo = SymbolicVariable('Food', FoodType)
    we = SymbolicVariable('WaitEstimate', WaitEstType)
    wa = SymbolicVariable('WillWait', Bool)

    # define probs
    numsamples = 500
    data = [[
        al.distribution().set(6/12.).sample_one(),
        ba.distribution().set(6/12.).sample_one(),
        fr.distribution().set(5/12.).sample_one(),
        hu.distribution().set(7/12.).sample_one(),
        pa.distribution().set([4/12., 6/12., 2/12.]).sample_one(),
        pr.distribution().set([7/12., 2/12., 3/12.]).sample_one(),
        ra.distribution().set(4/12.).sample_one(),
        re.distribution().set(5/12.).sample_one(),
        fo.distribution().set([2/12., 4/12., 4/12., 2/12.]).sample_one(),
        we.distribution().set([6/12., 2/12., 2/12., 2/12.]).sample_one(),
        wa.distribution().set(.5).sample_one()
    ] for _ in range(numsamples)]

    variables = [al, ba, fr, hu, pa, pr, ra, re, fo, we, wa]
    jpt = JPT(variables, min_samples_leaf=30, min_impurity_improvement=0)
    jpt.learn(data)

    jpt.plot(
        plotvars=variables,
        view=visualize,
        directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Restaurant')
    )

    q = {ba: True, re: False}
    e = {ra: False}

    res = jpt.infer(q, e)

    print(f'P({",".join([f"{k.name}={v}" for k, v in q.items()])}{" | " if e else ""}'
          f'{",".join([f"{k.name}={v}" for k, v in e.items()])}) = {res.result}')

    print(res.explain())


def preprocess_restaurant():
    f_csv = '../examples/data/restaurant.csv'
    data = pd.read_csv(f_csv, sep=',').fillna(value='???')
    return data


def restaurant_auto_sample(visualize=True):
    # generate JPT from data sampled based on distributions from lecture data
    df = pd.read_csv(os.path.join('../', 'examples', 'data', 'restaurant.csv'))

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

    def rec(var, vals):
        if not var:
            return [v[1] for v in vals]
            return
        d = dist(var[0], vals)
        sample = wchoice(var[0].domain.labels, d)
        return rec(var[1:], vals + [(var[0], sample)])

    def dist(var, vals):
        d_ = df
        for v, val in vals:
            d_ = d_[d_[v.name] == val]
        return [len(d_[d_[var.name] == l]) / len(d_) for l in var.domain.labels.values()] if len(d_) else None

    data = [rec(variables, []) for _ in range(500)]

    jpt = JPT(variables, min_samples_leaf=30, min_impurity_improvement=0)
    jpt.learn(rows=data)

    jpt.plot(
        plotvars=variables,
        view=visualize,
        directory=os.path.join('/tmp', f'{datetime.now().strftime("%d.%m.%Y-%H:%M:%S")}-Restaurant')
    )

    q = {ba: True, re: False}
    e = {ra: False}

    res = jpt.infer(q, e)


def main(*args):
    # restaurant()
    restaurant_mixed_type_variables()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
