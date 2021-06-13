import os
from datetime import datetime

from dnutils import out
from jpt.base.sampling import wchoice
from jpt.learning.distributions import Bool, SymbolicType
from jpt.trees import JPT
from jpt.variables import SymbolicVariable


def restaurant():
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
    jpt = JPT(variables, min_samples_leaf=30, min_impurity_improvement=0)
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
    # generate JPT from data sampled based on distributions from lecture data
    import pandas as pd
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

    def rec(vars, vals):
        if not vars:
            return [v[1] for v in vals]
            return
        d = dist(vars[0], vals)
        sample = wchoice(vars[0].domain.labels, d)
        return rec(vars[1:], vals+[(vars[0], sample)])

    def dist(var, vals):
        d_ = df
        for v, val in vals:
            d_ = d_[d_[v.name] == val]
        return [len(d_[d_[var.name] == l]) / len(d_) for l in var.domain.labels.values()] if len(d_) else None

    data = [rec(variables, []) for _ in range(500)]
    out(data)

    jpt = JPT(variables, min_samples_leaf=30, min_impurity_improvement=0)
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


def main(*args):
    # restaurant()
    restaurantsample()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
