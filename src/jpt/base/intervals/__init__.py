
__module__ = 'intervals'

try:
    from .base import __module__
    from .contset import __module__
    from .intset import __module__
    from .unionset import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base import NumberSet, Interval, INC, EXC
    from .contset import ContinuousSet
    from .intset import IntSet
    from .unionset import UnionSet

import numpy as np


EMPTY = ContinuousSet(0, 0, EXC, EXC)
R = ContinuousSet(np.NINF, np.PINF, EXC, EXC)
Z = IntSet(np.NINF, np.PINF)

from .contset import INC, EXC
from .base import _CHAR_EMPTYSET as STR_EMPTYSET
from .base import _CHAR_INFTY as STR_INFTY
