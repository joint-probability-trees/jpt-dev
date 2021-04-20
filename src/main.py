from collections import defaultdict

from variables import MultinomialRV, Uniform, Domain, SymbolicDistribution, P

BOOL = Domain(['T', 'F'])


def sample(dist):
    pass


def main(*args):
    # E = MultinomialRV('E', BOOL)
    # B = MultinomialRV('B', BOOL)
    # A = MultinomialRV('A', BOOL)
    # J = MultinomialRV('J', BOOL)
    # M = MultinomialRV('M', BOOL)
    P_E = {}
    P_B = {}
    P_A_EB = defaultdict(dict)
    P_M_A = defaultdict(dict)
    P_J_A = defaultdict(dict)

    P_E['T'] = .002
    P_E['F'] = .998

    P_B['T'] = .001
    P_B['F'] = .999

    P_A_EB['T', 'T']['T'] = .95
    P_A_EB['T', 'T']['F'] = .05
    P_A_EB['T', 'F']['T'] = .94
    P_A_EB['T', 'F']['F'] = .06
    P_A_EB['F', 'T']['T'] = .29
    P_A_EB['F', 'T']['F'] = .71
    P_A_EB['F', 'F']['T'] = .001
    P_A_EB['F', 'F']['F'] = .999

    P_M_A['T']['T'] = .7
    P_M_A['T']['F'] = .3
    P_M_A['F']['T'] = .01
    P_M_A['F']['F'] = .99

    P_J_A['T']['T'] = .9
    P_J_A['T']['F'] = .1
    P_J_A['F']['T'] = .05
    P_J_A['F']['F'] = .95

    for i in range(100):
        pass

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main('PyCharm')
