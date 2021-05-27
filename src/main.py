import pyximport

from examples import tourism, quantile, restaurant, neems, alarm, muesli

pyximport.install()

from dnutils import out
from jpt.learning.distributions import Bool, HistogramType
from jpt.variables import Variable


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


def main(*args):

    # test_merge()
    # test_dists()

    # call imported examples
    alarm.main()
    # muesli.main()
    # neems.main()
    # quantile.main()
    # restaurant.main()
    # tourism.main()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
