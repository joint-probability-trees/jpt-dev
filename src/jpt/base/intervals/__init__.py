
__module__ = 'intervals'

from ..pyximporter import pyx_import

pyx_import(
    '.base',
    '.intset',
    '.contset',
    '.unionset'
)

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
R = ContinuousSet(np.NINF, np.PINF, EXC, EXC)
Z = IntSet(np.NINF, np.PINF)
