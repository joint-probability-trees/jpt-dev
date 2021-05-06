from dnutils import out
from jpt.learning.distributions import Bool, SymbolicType
from jpt.learning.trees import JPT
from jpt.variables import SymbolicVariable


def restaurant():
    # declare variable types
    PatronsType = SymbolicType('Patrons', ['some', 'full', 'none'])
    PriceType = SymbolicType('Price', ['$', '$$', '$$$'])
    FoodType = SymbolicType('Food', ['French', 'Thai', 'Burger', 'Italian'])
    WaitEstType = SymbolicType('WaitEstimate', ['0-10', '10-30', '30-60', '>60'])

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
    we = SymbolicVariable('WaitEst', WaitEstType)
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


def main(*args):
    restaurant()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
