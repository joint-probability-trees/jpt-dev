'''
© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import hashlib
import math
import numbers
from collections import OrderedDict
from typing import List, Tuple, Any, Union, Dict, Iterator

import numpy as np
from dnutils import first, ifnone, out

from jpt.base.utils import mapstr, to_json

try:
    from jpt.base.intervals import INC, EXC, ContinuousSet
    from jpt.base.constants import SYMBOL
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
    from jpt.base.intervals import INC, EXC
    from jpt.base.constants import SYMBOL


from jpt.learning.distributions import Multinomial, Numeric, ScaledNumeric, Distribution, SymbolicType, NumericType


# ----------------------------------------------------------------------------------------------------------------------
# Generic variable classes


class Variable:
    '''
    Abstract class for a variable name along with its distribution class type.
    '''

    def __init__(self, name, domain, min_impurity_improvement=None):
        '''
        :param name:    name of the variable
        :type name:     str
        :param domain:  the class type (not an instance!) of the represented Distribution
        :type domain:   class type of jpt.learning.distributions.Distribution
        :param min_impurity_improvement:
        :type min_impurity_improvement: float
        '''
        self._name = name
        self._domain = domain
        self.min_impurity_improvement = min_impurity_improvement or 0.
        if not issubclass(type(self), Variable) or type(self) is Variable:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    def dist(self, params=None, data=None, rows=None, col=None):
        '''
        Create and return a new instance of the distribution type attached to this variable.

        Either the distribution ``params`` can be passed or the ``data`` the distribution parameters
        are to be determined from.
        '''
        if data is None:
            return self._dist(params)
        elif data is not None:
            if col is None and len(data.shape) > 1:
                raise ValueError('In multi-dimensional matrices (dim=%d), a col index must be passed.' % data.ndim)
            dist = self.dist(params=params)
            return dist.fit(data, rows, col if data.ndim > 1 else 0, **self.params)

    @property
    def params(self):
        return {}

    def _dist(self, params):
        '''Create and return a new instance of the distribution associated with this type of variable.'''
        return self._domain(params)

    def __str__(self):
        return f'{self.name}[{self.domain.__name__}(%s)]' % {0: 'SYM', 1: 'NUM'}[self.numeric]

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.name == other.name and
                # self.domain.equiv(other.domain) and
                self.min_impurity_improvement == other.min_impurity_improvement)

    def __hash__(self):
        return hash((hashlib.md5(self.name.encode()).hexdigest(), self.domain))

    @property
    def symbolic(self):
        return issubclass(self.domain, Multinomial)

    @property
    def numeric(self):
        return issubclass(self.domain, Numeric)

    def str(self, assignment, **kwargs):
        raise NotImplemented()

    def to_json(self):
        return {'name': self.name,
                'type': 'numeric' if self.numeric else 'symbolic',
                'domain': self.domain.to_json(),
                'min_impurity_improvement': self.min_impurity_improvement}

    @staticmethod
    def from_json(data):
        if data['type'] == 'numeric':
            return NumericVariable.from_json(data)
        elif data['type'] == 'symbolic':
            return SymbolicVariable.from_json(data)
        else:
            raise TypeError('Unknown distribution type: %s' % data['type'])

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = Variable.from_json(state).__dict__


# ----------------------------------------------------------------------------------------------------------------------
# Numeric variables


class NumericVariable(Variable):
    '''
    Represents a continuous variable.
    '''

    def __init__(self, name, domain=Numeric, min_impurity_improvement=None, haze=None, max_std=None, precision=None):
        super().__init__(name, domain, min_impurity_improvement=min_impurity_improvement)
        self.haze = ifnone(haze, .05)
        self._max_std_lbl = ifnone(max_std, 0.)
        self.precision = ifnone(precision, .01)

    def __eq__(self, o):
        return (super().__eq__(o) and
                self.haze == o.haze and
                self._max_std_lbl == o._max_std_lbl and
                self.precision == o.precision)

    def __hash__(self):
        return hash((NumericVariable,
                     hashlib.md5(self.name.encode()).hexdigest(),
                     self.domain,
                     self.haze,
                     self.max_std,
                     self.precision))

    @Variable.params.getter
    def params(self):
        return {'epsilon': self.precision}

    def to_json(self):
        result = super().to_json()
        result['max_std'] = self._max_std_lbl
        result['precision'] = self.precision
        return result

    @staticmethod
    def from_json(data):
        domain = Distribution.type_from_json(data['domain'])
        return NumericVariable(name=data['name'],
                               domain=domain,
                               max_std=data.get('max_std'),
                               precision=data.get('precision'))

    @property
    def _max_std(self):
        if issubclass(self.domain, ScaledNumeric):
            return self._max_std_lbl / math.sqrt(self.domain.values.datascaler.variance)
        else:
            return self._max_std_lbl

    @property
    def max_std(self):
        return self._max_std_lbl

    def str(self, assignment, **kwargs):
        fmt = kwargs.get('fmt', 'set')
        precision = kwargs.get('precision', 3)
        lower = '%%.%df %%s ' % precision
        upper = ' %%s %%.%df' % precision
        if type(assignment) is set:
            if len(assignment) == 1:
                valstr = str(first(assignment))
            else:
                valstr = ', '.join([self.str(a, fmt=fmt) for a in assignment])
        else:
            valstr = str(ContinuousSet(self.domain.labels[assignment.lower],
                                       self.domain.labels[assignment.upper],
                                       assignment.left,
                                       assignment.right))
        if isinstance(assignment, numbers.Number):
            return '%s = %s' % (self.name, self.domain.labels[assignment])
        if fmt == 'set':
            return f'{self.name} {SYMBOL.IN} {valstr}'
        elif fmt == 'logic':
            return '%s%s%s' % (lower % (self.domain.labels[assignment.lower],
                                        {INC: SYMBOL.LTE,
                                         EXC: SYMBOL.LT}[assignment.left]) if assignment.lower != np.NINF else '',
                               self.name,
                               upper % ({INC: SYMBOL.LTE,
                                         EXC: SYMBOL.LT}[assignment.right],
                                        self.domain.labels[assignment.upper]) if assignment.upper != np.PINF else '')
        else:
            raise ValueError('Unknown format for numeric variable: %s.' % fmt)


# ----------------------------------------------------------------------------------------------------------------------
# Classes to represent symbolic variables


class SymbolicVariable(Variable):
    '''
    Represents a symbolic variable.
    '''

    def __init__(self, name, domain, min_impurity_improvement=None):
        super().__init__(name,
                         domain,
                         min_impurity_improvement=min_impurity_improvement)

    @staticmethod
    def from_json(data):
        domain = Distribution.type_from_json(data['domain'])
        return SymbolicVariable(name=data['name'], domain=domain)

    def str(self, assignment, **kwargs):
        fmt = kwargs.get('fmt', 'set')
        limit = kwargs.get('limit', 10)
        if type(assignment) is set:
            if len(assignment) == 1:
                return self.str(first(assignment), fmt=fmt)
            elif fmt == 'set':
                valstr = ', '.join(mapstr([self.domain.labels[a] for a in assignment], limit=limit))
                return f'{self.name} {SYMBOL.IN} {{{valstr}}}'
            elif fmt == 'logic':
                return ' v '.join([self.str(a, fmt=fmt) for a in assignment])
        if isinstance(assignment, numbers.Number):
            return '%s = %s' % (self.name, self.domain.labels[assignment])
        else:
            return '%s = %s' % (self.name, str(assignment))


# ----------------------------------------------------------------------------------------------------------------------
# Convenience functions and classes

def infer_from_dataframe(df, scale_numeric_types=True, min_impurity_improvement=None,
                         haze=None, max_std=None, precision=None):
    '''
    Creates the ``Variable`` instances from column types in a Pandas or Spark data frame.

    :param df:  the data frame object to generate the variables from.
    :type df:   ``pandas.DataFrame``

    :param scale_numeric_types: Whether of not to use scaled types for the numeric variables.
    :type scale_numeric_types: bool

    :param min_impurity_improvement:   the minimum imrovement that a split must induce to be acceptable.
    :type min_impurity_improvement: ``float``

    :param haze:
    :type haze:         ``float``

    :param max_std:
    :type max_std:      ``float``

    :param precision:
    :type precision:    ``float`` in ``[0, 1]``
    '''

    variables = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype in (str, object, bool):
            dom = SymbolicType('%s_TYPE' % col.upper(), labels=df[col].unique())
            var = SymbolicVariable(col,
                                   dom,
                                   min_impurity_improvement=min_impurity_improvement)

        elif dtype in (np.float64, np.int64, np.float32, np.int32):
            if scale_numeric_types:
                dom = NumericType('%s_TYPE' % col.upper(), df[col].unique())
            else:
                dom = Numeric
            var = NumericVariable(col,
                                  dom,
                                  min_impurity_improvement=min_impurity_improvement,
                                  haze=haze,
                                  max_std=max_std,
                                  precision=precision)
        else:
            raise TypeError('Unknown column type:', col, '[%s]' % dtype)
        variables.append(var)
    return variables


class VariableMap:
    '''
    Convenience class for mapping a ``Variable`` object to anything else. This special map, however,
    supports accessing the image set both by the variable object instance itself _and_ its name.
    '''

    def __init__(self, data: List[Tuple] = None):
        '''
        ``data`` may be an iterable of (variable, value) pairs.
        '''
        super().__init__()
        self._variables = {}
        self._map = OrderedDict()
        if data:
            for var, value in data:
                self[var] = value

    @property
    def map(self) -> OrderedDict:
        return self._map

    def __getitem__(self, key: Union[str, Variable]) -> Any:
        if isinstance(key, Variable):
            return self.__getitem__(key.name)
        return self._map.__getitem__(key)

    def __setitem__(self, variable: Union[str, Variable], value: Any) -> None:
        if not isinstance(variable, Variable):
            raise ValueError('Illegal argument value: expected Variable, got %s.' % type(variable).__name__)
        self._map[variable.name] = value
        self._variables[variable.name] = variable

    def __delitem__(self, key: Union[str, Variable]) -> None:
        if isinstance(key, Variable):
            self.__delitem__(key.name)
            return
        del self._map[key]
        del self._variables[key]

    def __contains__(self, item: Union[str, Variable]) -> bool:
        if isinstance(item, Variable):
            return self.__contains__(item.name)
        return item in self._map

    def __iter__(self):
        return iter((self._variables[name] for name in self._map))

    def __len__(self):
        return len(self._map)

    def __bool__(self):
        return len(self)

    def __eq__(self, o):
        return (type(o) is VariableMap and
                list(self._map.items()) == list(o._map.items()) and
                list(self._variables.items()) == list(o._variables.items()))

    def get(self, key: Union[str, Variable], default=None) -> Any:
        if key not in self:
            return default
        return self[key]

    def keys(self) -> Iterator[str]:
        yield from (self._variables[name] for name in self._map.keys())

    def values(self) -> Iterator[Any]:
        yield from self._map.values()

    def items(self) -> Iterator[Tuple]:
        yield from ((self._variables[name], value) for name, value in self._map.items())

    def to_json(self) -> Dict[str, Any]:
        return {var.name: to_json(value) for var, value in self.items()}

    def update(self, varmap: 'VariableMap') -> 'VariableMap':
        self._map.update(varmap._map)
        self._variables.update(varmap._variables)
        return self

    @staticmethod
    def from_json(variables: List[Variable], d: Dict[str, Any], typ=None, args=()) -> 'VariableMap':
        vmap = VariableMap()
        varbyname = {var.name: var for var in variables}
        for vname, value in d.items():
            vmap[varbyname[vname]] = (typ.from_json(value, *args)
                                      if typ is not None and hasattr(typ, 'from_json')
                                      else value)
        return vmap

    def __repr__(self):
        return '<VariableMap {%s}>' % ','.join(['%s: %s' % (var.name, repr(val)) for var, val in self.items()])
