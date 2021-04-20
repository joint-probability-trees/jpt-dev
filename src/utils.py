from functools import reduce

from dnutils import ifnone


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
