from unittest import TestCase

import numpy as np
from ddt import ddt, data, unpack

from jpt.base.functions import Undefined


# ----------------------------------------------------------------------

@ddt
class UndefinedFunctionTest(TestCase):

    @data((1,), (-1,), (100,))
    @unpack
    def test_eval(self, x):
        """Verify Undefined function evaluates to NaN for any input."""
        f = Undefined()
        self.assertTrue(np.isnan(f.eval(x)))
