
__module__ = 'intervals'

try:
    from .base import __module__
    from .intset import __module__
    from .contset import __module__
    from .unionset import __module__
except ModuleNotFoundError:
    from jpt.base import pyximporter
    pyximporter.install()
finally:
    from .base import (
        NumberSet,
        Interval,
        INC,
        EXC,
        _CHAR_EMPTYSET as STR_EMPTYSET,
        _CHAR_INFTY as STR_INFTY
    )
    from .contset import ContinuousSet
    from .intset import IntSet
    from .unionset import UnionSet

import numpy as np


EMPTY = ContinuousSet(0, 0, EXC, EXC)
R = ContinuousSet(-np.inf, np.inf, EXC, EXC)
Z = IntSet(-np.inf, np.inf)
