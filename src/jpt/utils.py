import math
from functools import reduce

from dnutils import ifnone


class SYMBOL:
    LAND = '\u2227'
    IN = '\u2208'


def mapstr(seq, format=None):
    '''Convert the sequence ``seq`` into a list of strings by applying ``str`` to each of its elements.'''
    return [format(e) for e in seq] if callable(format) else [ifnone(format, '%s') % (e,) for e in seq]


def prod(iterable):
    return reduce(lambda x, y: x * y, iterable)


def tojson(obj):
    """Recursively generate a JSON representation of the object ``obj``."""
    if hasattr(obj, 'tojson'):
        return obj.tojson()
    if type(obj) in (list, tuple):
        return [tojson(e) for e in obj]
    elif isinstance(obj, dict):
        return {str(k): tojson(v) for k, v in obj.items()}
    return obj


def entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p``.
    :param p:   a ``list`` of probabilities or a ``dict``.
    :return:
    '''
    if isinstance(p, dict):
        p = list(p.values())
    return abs(-sum([0 if p_i == 0 else math.log(p_i, 2) * p_i for p_i in p]))


def max_entropy(n):
    '''Compute the maximal entropy that a multinomial random variable with ``n`` states can have,
    i.e. the entropy value assuming a uniform distribution over the values.
    :param p:
    :return:
    '''
    return entropy([1 / n for _ in range(n)])


def rel_entropy(p):
    '''Compute the entropy of the multinomial probability distribution ``p`` normalized
    by the maximal entropy that a multinomial distribution of the dimensionality of ``p``
    can have.
    :type p: distribution'''
    if len(p) == 1:
        return 0
    return entropy(p) / max_entropy(len(p))