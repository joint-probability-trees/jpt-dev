'''© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error

import dnutils
from jpt.base.utils import to_json

UNKNOWN_FLOAT32 = 1e7
logger = dnutils.getlogger(name='ExampleLogger', level=dnutils.ERROR)


class Feature:
    """Represents a feature consisting of the value and unit """

    def __init__(self, value, name=None, description=None):
        self.value = value
        self.name = name
        self.description = description

    def tojson(self):
        return to_json({'name': self.name, 'value': self.value, 'description': self.description})

    def __repr__(self):
        return str(self)

    def __str__(self):
        return '{}({}, name={})'.format(type(self).__name__, self.value, self.name)


class NumericFeature(Feature):
    """Every numeric feature is an instance of this class"""

    def __init__(self, value, name=None, unit=None, description=None):
        Feature.__init__(self, value, name=name, description=description)
        self.unit = unit

    def tojson(self):
        return to_json({'name': self.name, 'value': self.value, 'description': self.description, 'unit': self.unit})

    def __str__(self):
        return '{}({}, name={}, unit={})'.format(type(self).__name__, self.value, self.name, self.unit)

    def __add__(self, other):
        return NumericFeature(self.value + other.value, name=self.name, unit=self.unit, description=self.description)

    def __sub__(self, other):
        return NumericFeature(self.value - other.value, name=self.name, unit=self.unit, description=self.description)

    def __mul__(self, other):
        return NumericFeature(self.value * other.value, name=self.name, unit=self.unit, description=self.description)

    def __truediv__(self, other):
        if isinstance(other, NumericFeature):
            return NumericFeature(self.value / other.value, name=self.name, unit=self.unit, description=self.description)
        else:
            return NumericFeature(self.value / other, name=self.name, unit=self.unit, description=self.description)


class SymbolicFeature(Feature):
    """Every symbolic feature must be an instance of this class.
    NB: The values of symbolic features must be enum values"""

    def __init__(self, value, name=None, description=None):
        Feature.__init__(self, value, name, description)


class BooleanFeature(SymbolicFeature):
    """Every boolean feature is an instance of this class. A boolean feature is a special case of
    a symbolic feature which accepts only two the values True and False"""

    def __init__(self, value, name=None, description=None):
        SymbolicFeature.__init__(self, value, name, description)


class Example:
    """Custom Example class for more intuitive data handling"""

    def __init__(self, x, t=None, identifier=None):
        """

        :param x: the features
        :param t: the targets
        :param identifier: can be used for debugging - identifies the training example
        :type x: list of matcalo.database.models.Feature
        :type t: list of matcalo.database.models.Feature
        :type identifier: str
        """
        self.identifier = identifier
        self.x = x
        self.t = t
        self._xtypes = None
        self._ttypes = None
        self._xsklearn = None
        self._tsklearn = None
        self._xtable = None
        self._ttable = None

    @property
    def x(self):
        return self._x #if len(self._x) > 1 else first(self._x)

    @x.setter
    def x(self, x):
        self._xsklearn = None
        self._x = x if type(x) is list else [x]

    @property
    def t(self):
        return self._t
        # if self._t is None:
        #     return
        # else:
        #     return self._t #if len(self._t) > 1 else first(self._t)

    @t.setter
    def t(self, t):
        self._tsklearn = None
        if type(t) is list:
            self._t = t
        elif t is None:
            self._t = None
        elif type(t) in [int, float]:
            self._t = [t]
        else:
            raise ValueError('Target type undefined', t)

    @property
    def features(self):
        return [self.x.name] if isinstance(self.x, Feature) else [f.name for f in self.x] if self.x is not None else []

    @property
    def targets(self):
        return [self.t.name] if isinstance(self.t, Feature) else [f.name for f in self.t] if self.t is not None else []

    @property
    def ft_types(self):
        return [type(ft) for ft in self._x]

    @property
    def tgt_types(self):
        return [type(tgt) for tgt in self._t]

    def xplain(self):
        return [f.value for f in self.x] if type(self.x) is list else [self.x.value]

    def tplain(self):
        return [t.value for t in self.t] if type(self.t) is list else [self.t.value]

    def shape(self):
        return len(self.x), len(self.t)

    def xsklearn(self):
        """Return a representation of the ``x`` vector that is compatible with sklearn."""
        if self._xsklearn is None:
            self._xsklearn = Example._tosklearn(self.x, np.float64)
        return self._xsklearn

    def tsklearn(self):
        """Return a representation of the ``t`` vector that is compatible with sklearn."""
        if self._tsklearn is None:
            self._tsklearn = Example._tosklearn(self.t, np.float64)
        return self._tsklearn

    def tosklearn(self):
        """Returns a representation of the ``x`` and ``t`` vectors that is compatible with sklearn."""
        return self.xsklearn(), self.tsklearn()

    @staticmethod
    def _tosklearn(v, numt=np.float32):
        if v is None: return None
        if type(v) is not list:
            v = [v]
        vec = [None] * len(v) if type(v) is list else 1
        for i, x in enumerate(v):
            if x.value is not None:
                vec[i] = numt(x.value) if not isinstance(x, SymbolicFeature) else numt(x.value.value)
            else:
                if isinstance(x, NumericFeature):
                    vec[i] = UNKNOWN_FLOAT32
                elif isinstance(x, SymbolicFeature) or isinstance(x, BooleanFeature):
                    vec[i] = numt(-1)
        return vec

    @staticmethod
    def count(data, ft=None, tgt=None):
        cnt = 0
        if tgt is None and tgt is None:
            return cnt
        if ft:
            for d in data:
                pass

    @staticmethod
    def impurity(xmpls, tgt):
        r"""Calculate the mean squared error for the data set `xmpls`, i.e.

        .. math::
            MSE = \frac{1}{n} · \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
        """
        if not xmpls:
            return 0.

        # assuming all Examples have identical indices for their targets
        tgt_idx = xmpls[0].targets.index(tgt)

        if xmpls[0].tgt_types[tgt_idx] in (SymbolicFeature, BooleanFeature):
            # case categorical targets
            tgts_plain = np.array([xmpl.tplain() for xmpl in xmpls]).T[tgt_idx]

            # count occurrences for each value of target tgt and determine their probability
            prob = [float(list(tgts_plain).count(distincttgtval)) / len(tgts_plain) for distincttgtval in list(set(tgts_plain))]

            # calculate actual impurity target tgt
            return entropy(prob, base=2)
        else:
            # case numeric targets
            tgts_sklearn = np.array(np.array([xmpl.tplain() for xmpl in xmpls]).T[tgt_idx], dtype=np.float32)

            # calculate mean for target tgt
            ft_mean = np.mean(tgts_sklearn)

            # calculate normalized mean squared error for target tgt
            sqerr = mean_squared_error(tgts_sklearn, [ft_mean]*len(tgts_sklearn))

            # calculate actual impurity for target tgt
            return sqerr

    def __str__(self):
        return 'Example<{}>\nx:[{}]\nt:[{}]'.format(self.identifier, ',\n'.join([str(x) for x in self._x]), ',\n'.join([str(x) for x in self._t]) if self._t is not None else '')

    def __repr__(self):
        return 'id:{}, x:{}, t:{}'.format(self.identifier, self.x, self.t)


if __name__ == '__main__':
    labelsx = ['x', 'y']#, 'z']
    labelst = ['t1', 't2']#, 't3']

    # X = [
    #     [True, '1', 'b'],
    #     [False, '2', 'a'],
    #     [True, '3', 'a'],
    # ]
    # X = [
    #     [4, 1],
    #     [5, 2],
    #     [7, 3],
    # ]
    X = [
        ['a', '0'],
        ['a', '1'],
        ['b', '1']
    ]

    # Y = [['a', 'i', True], ['b', 'o', False], ['b', 'o', True]]
    # Y = [[3, 2, 1], [1, 3, 2], [2, 1, 3]]
    Y = [
        ['x', 'a'],
        ['y', 'a'],
        ['y', 'b']
    ]

    examples = []
    for i, x in enumerate(X):
        examples.append(Example(x=[SymbolicFeature(v, name=ft) for ft, v in zip(labelsx, X[i])], t=[SymbolicFeature(v, name=tgt) for tgt, v in zip(labelst, Y[i])], identifier=f'e_{i}'))
        # examples.append(Example(x=[NumericFeature(v, name=ft) for ft, v in zip(labelsx, X[i])], t=[NumericFeature(v, name=tgt) for tgt, v in zip(labelst, Y[i])], identifier=f'e_{i}'))

    print(Example.impurity(examples, ft='x'))
    # print(Example.impurity(examples))