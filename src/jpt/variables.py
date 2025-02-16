'''
© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import hashlib
import math
import numbers
import uuid

from typing import List, Tuple, Any, Union, Dict, Iterator, Set, Iterable, Type, Optional
import collections.abc

import numpy as np
from dnutils import first, edict, ifnone

from jpt.base.utils import mapstr, to_json, list2interval, setstr, setstr_int
from jpt.base.constants import SYMBOL

from jpt.distributions import Multinomial, Numeric, ScaledNumeric, Distribution, SymbolicType, NumericType, Integer, \
    IntegerType, Bool

try:
    from .base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from .base.intervals import INC, EXC, ContinuousSet, RealSet, NumberSet


# ----------------------------------------------------------------------------------------------------------------------
# Generic variable classes

class Variable:
    '''
    Abstract class for a variable name along with its distribution class type.
    '''

    MIN_IMPURITY_IMPROVEMENT = 'min_impurity_improvement'

    SETTINGS = {
        MIN_IMPURITY_IMPROVEMENT: 0
    }

    def __init__(
            self,
            name: str,
            domain: Optional[Type[Distribution]] = None,
            **settings
    ):
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
        if not issubclass(type(self), Variable) or type(self) is Variable:
            raise TypeError(
                f'Instantiation of abstract class {type(self).__name__} is not allowed!'
            )
        self.settings = type(self).SETTINGS.copy()
        for attr in type(self).SETTINGS:
            try:
                super().__getattribute__(attr)
            except AttributeError:
                pass
            else:
                raise AttributeError(
                    'Attribute ambiguity: Object of type "%s" '
                    'already has an attribute with name "%s"' % (
                        type(self).__name__,
                        attr
                    )
                )
        for attr, value in settings.items():
            if attr not in self.settings:
                raise AttributeError(
                    'Unknown settings "%s": expected one of {%s}' % (
                        attr,
                        setstr(type(self).SETTINGS)
                    )
                )
            if value is not None:
                self.settings[attr] = value

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in type(self).SETTINGS:
                return self.settings[name]
            else:
                raise

    @property
    def name(self) -> str:
        return self._name

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, d):
        self._domain = d

    def distribution(self) -> Distribution:
        '''
        Create and return a new instance of the distribution type attached to this variable.
        '''
        return self._domain(**{
            k: self.settings[k]
            for k in self._domain.SETTINGS
            if k in self.settings
        })

    def __str__(self):
        return f'{self.name}[{self.domain.__name__}]'

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return (
            type(self) == type(other)
            and self.name == other.name
            and self.domain.equiv(other.domain)
            and self.settings == other.settings
        )

    def __hash__(self):
        return hash((
            type(self),
            hashlib.md5(
                self.name.encode()
            ).hexdigest(),
            tuple(
                sorted(
                    self.settings.items()
                )
            ),
            self.domain.hash() if self.domain is not None else None
        ))

    @property
    def symbolic(self) -> bool:
        return issubclass(self.domain, Multinomial)

    @property
    def numeric(self) -> bool:
        return issubclass(self.domain, Numeric)

    @property
    def integer(self) -> bool:
        return issubclass(self.domain, Integer)

    def str(self, assignment, **kwargs) -> str:
        raise NotImplementedError()

    def to_json(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'domain': None if self.domain is None else self.domain.to_json(),
            'settings': self.settings
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> Union['NumericVariable', 'SymbolicVariable', 'IntegerVariable']:
        if data['type'] == 'numeric':
            return NumericVariable.from_json(data)
        elif data['type'] == 'symbolic':
            return SymbolicVariable.from_json(data)
        elif data['type'] == 'integer':
            return IntegerVariable.from_json(data)
        else:
            raise TypeError('Unknown distribution type: %s' % data['type'])

    def __getstate__(self):
        return self.to_json()

    def __setstate__(self, state):
        self.__dict__ = Variable.from_json(state).__dict__

    def copy(self):
        return Variable.from_json(self.to_json())

    def assignment2set(self, assignment: Any):
        '''
        Return a canonical representation of the variable ``assignment`` as a set
        in the corresponding type of set.

        For a ``NumericVariable``, a scalar ``assignment`` will be converted to
        a ``ContinuousSet`` instance, for a ``SymbolicVariable``, a single value will
        be converted to a ``set`` collection.

        If ``assignment`` is alreay in its canonical set representation, it
        will not be modified and returned as passed.
        '''
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------------------------------------
# Numeric variables


class NumericVariable(Variable):
    '''
    Represents a continuous variable.
    '''

    BLUR = 'blur'
    MAX_STDEV = 'max_std_lbl'
    PRECISION = 'precision'

    SETTINGS = edict(Variable.SETTINGS) + {
        BLUR: 0,
        MAX_STDEV: .0,
        PRECISION: .01
    }

    def __init__(
            self,
            name: str,
            domain: Optional[Type[Numeric]] = Numeric,
            min_impurity_improvement: Optional[float] = None,
            blur: Optional[float] = None,
            max_std: Optional[float] = None,
            precision: Optional[float] = None
    ):
        settings = {
            Variable.MIN_IMPURITY_IMPROVEMENT: min_impurity_improvement,
            NumericVariable.BLUR: blur,
            NumericVariable.MAX_STDEV: max_std,
            NumericVariable.PRECISION: precision
        }
        super().__init__(
            name,
            domain,
            **settings
        )

    def to_json(self) -> Dict[str, Any]:
        return edict(super().to_json()) + {
            'type': 'numeric'
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'NumericVariable':
        domain = Distribution.from_json(data['domain'])
        return NumericVariable(
            name=data['name'],
            domain=domain,
            min_impurity_improvement=data.get(Variable.MIN_IMPURITY_IMPROVEMENT),
            max_std=data.get(NumericVariable.MAX_STDEV),
            precision=data.get(NumericVariable.PRECISION),
            blur=data.get(NumericVariable.BLUR)
        )

    @property
    def _max_std(self):
        if issubclass(self.domain, ScaledNumeric):
            return self.max_std_lbl / math.sqrt(self.domain.values.datascaler.scale)
        else:
            return self.max_std_lbl

    @property
    def max_std(self):
        return self.max_std_lbl

    # noinspection PyIncorrectDocstring
    def str(
        self,
        assignment: Union[List, Set, numbers.Number, NumberSet],
        **kwargs
    ) -> str:
        '''
        Construct a pretty-formatted string representation of the respective
        variable assignment.

        :param assignment:        the value(s) assigned to this variable.
        :param fmt:               ["set" | "logic"] use either set or logical notation.
        :param precision:         (int) the number of decimals to use for rounding.
        '''
        fmt = kwargs.get('fmt', 'set')
        precision = kwargs.get('precision', 3)
        prec = '%%s = %%.%df' % precision
        lower = '%%.%df %%s ' % precision
        upper = ' %%s %%.%df' % precision

        if type(assignment) is list:
            assignment = list2interval(assignment)
        elif type(assignment) is set:
            intervals = []
            for s in assignment:
                if isinstance(s, ContinuousSet):
                    intervals.append(s)
                elif type(s) is tuple:
                    intervals.append(list2interval(s))
                elif isinstance(s, numbers.Number):
                    intervals.append(ContinuousSet(s, s))
                else:
                    raise TypeError('Expected number of ContinuousSet, got %s.' % type(s).__name__)
            assignment = RealSet(intervals).simplify()
        if isinstance(assignment, ContinuousSet):
            assignment = RealSet([assignment])
        if isinstance(assignment, numbers.Number):
            return prec % (self.name, self.domain.labels[assignment])
        if fmt == 'set':
            return f'{self.name} {SYMBOL.IN} {str(assignment)}'
        elif fmt == 'logic':
            s = []
            for i in assignment.intervals:
                if i.size() == 1:
                    s.append(
                        '%s = %s' % (self.name, self.domain.labels[i.lower])
                    )
                else:
                    s.append(
                        '%s%s%s' % (lower % (
                            self.domain.labels[i.lower],
                            {INC: SYMBOL.LTE, EXC: SYMBOL.LT}[i.left]) if i.lower != -np.inf else '',
                             self.name,
                             upper % (
                                 {INC: SYMBOL.LTE, EXC: SYMBOL.LT}[i.right],
                                 self.domain.labels[i.upper]
                             ) if i.upper != np.inf else ''
                        )
                    )
            return ' v '.join(s)
        else:
            raise ValueError('Unknown format for numeric variable: %s.' % fmt)

    def assignment2set(self, assignment: Union[float, NumberSet]) -> NumberSet:
        if isinstance(assignment, numbers.Number):
            return ContinuousSet(assignment, assignment)
        return assignment


# ----------------------------------------------------------------------------------------------------------------------

class IntegerVariable(Variable):
    '''
    Represents an integer-valued variable.
    '''

    def __init__(
            self,
            name: str,
            domain: Optional[Type[Integer]],
            min_impurity_improvement: Optional[float] = None
    ):
        settings = {
            Variable.MIN_IMPURITY_IMPROVEMENT: min_impurity_improvement
        }
        super().__init__(name, domain, **settings)

    def str(self, assignment, **kwargs) -> str:
        fmt = kwargs.get('fmt', 'set')
        if type(assignment) is set:
            if len(assignment) == 1:
                return self.str(first(assignment), fmt=fmt)
            elif fmt == 'set':
                valstr = setstr_int(assignment)
                return f'{self.name} {SYMBOL.IN} {{{valstr}}}'
            elif fmt == 'logic':
                return ' v '.join([self.str(a, fmt=fmt) for a in assignment])
        if isinstance(assignment, numbers.Number):
            return '%s = %s' % (self.name, self.domain.labels[assignment])
        else:
            return '%s = %s' % (self.name, str(assignment))

    def assignment2set(self, assignment: Union[int, Set[int]]) -> Set[int]:
        if isinstance(assignment, numbers.Integral):
            return {assignment}
        return assignment

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'IntegerVariable':
        domain = Distribution.from_json(data['domain'])
        return IntegerVariable(
            name=data['name'],
            domain=domain,
            min_impurity_improvement=data.get(Variable.MIN_IMPURITY_IMPROVEMENT)
        )

    def to_json(self) -> Dict[str, Any]:
        return edict(super().to_json()) + {
            'type': 'integer'
        }


# ----------------------------------------------------------------------------------------------------------------------
# Classes to represent symbolic variables

INVERT_IMPURITY = 'invert_impurity'


class SymbolicVariable(Variable):
    '''
    Represents a symbolic variable.
    '''

    SETTINGS = edict(Variable.SETTINGS) + {
        INVERT_IMPURITY: False,
    }

    def __init__(
            self,
            name: str,
            domain: Optional[Type[Multinomial]],
            min_impurity_improvement: Optional[float] = None,
            invert_impurity: Optional[bool] = None
    ):
        super().__init__(
            name,
            domain,
            min_impurity_improvement=min_impurity_improvement,
            invert_impurity=invert_impurity
        )

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'SymbolicVariable':
        return SymbolicVariable(
            name=data['name'],
            domain=Distribution.from_json(data['domain'])
        )

    def to_json(self) -> Dict[str, Any]:
        return edict(super().to_json()) + {
            'type': 'symbolic'
        }

    def str(self, assignment: Union[set, numbers.Number], **kwargs) -> str:
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

    def assignment2set(self, assignment: Any):
        if not isinstance(assignment, set):
            return {assignment}
        return assignment


# ----------------------------------------------------------------------------------------------------------------------
# Convenience functions and classes

def infer_from_dataframe(df,
                         scale_numeric_types: bool = True,
                         min_impurity_improvement: float = None,
                         blur: float = None,
                         max_std: float = None,
                         precision: float = None,
                         unique_domain_names: bool = False,
                         excluded_columns: Dict[str, type] = None,
                         remove_nan: bool = False):
    '''
    Creates the ``Variable`` instances from column types in a Pandas or Spark data frame.

    :param df:  the data frame object to generate the variables from.
    :type df:   ``pandas.DataFrame``

    :param scale_numeric_types: Whether of not to use scaled types for the numeric variables.
    :type scale_numeric_types: bool

    :param min_impurity_improvement:   the minimum improvement that a split must induce to be acceptable.
    :type min_impurity_improvement: ``float``

    :param blur:
    :type blur:         ``float``

    :param max_std:
    :type max_std:      ``float``

    :param precision:
    :type precision:    ``float`` in ``[0, 1]``

    :param unique_domain_names:     for multiple calls of infer_from_dataframe containing duplicate column names the
                                    generated domain names will be unique
    :type unique_domain_names:    ``bool``

    :param excluded_columns:     user-provided domains for specific columns
    :type excluded_columns:    ``Dict[str, type]``

    :param remove_nan:  skip all ``None`` or ``NaN`` or ``Inf`` values in the data to construct the
                        numeric variable domains.
    '''
    variables = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype in (str, object, bool):
            if excluded_columns is not None and col in excluded_columns:
                dom = excluded_columns[col]
            elif dtype in (bool, np.bool_):
                dom = Bool
            else:
                dom = SymbolicType(
                    '%s%s_TYPE_S' % (col.upper(), '_' + str(uuid.uuid4()) if unique_domain_names else ''),
                    labels=df[col].unique()
                )
            var = SymbolicVariable(col,
                                   dom,
                                   min_impurity_improvement=min_impurity_improvement)

        elif dtype in (np.float64, np.float32):
            if excluded_columns is not None and col in excluded_columns:
                dom = excluded_columns[col]
            elif scale_numeric_types:
                values = df[col]
                if remove_nan:
                    values = values[~values.isin([np.nan, np.inf])]
                dom = NumericType(
                    '%s%s_TYPE_N' % (col.upper(), '_' + str(uuid.uuid4()) if unique_domain_names else ''),
                    values.unique()
                )
            else:
                dom = Numeric
            var = NumericVariable(col,
                                  dom,
                                  min_impurity_improvement=min_impurity_improvement,
                                  blur=blur,
                                  max_std=max_std,
                                  precision=precision)
        elif dtype in (np.int32, np.int64):
            if excluded_columns is not None and col in excluded_columns:
                dom = excluded_columns[col]
            else:
                dom = IntegerType(
                    '%s%s_TYPE_I' % (col.upper(), '_' + str(uuid.uuid4()) if unique_domain_names else ''),
                    lmin=df[col].min(),
                    lmax=df[col].max()
                )
            var = IntegerVariable(col, dom)
        else:
            raise TypeError('Unknown column type:', col, '[%s]' % dtype)
        variables.append(var)
    return variables


class VariableMap:
    '''
    Convenience class for mapping a ``Variable`` object to anything else. This special map, however,
    supports accessing the image set both by the variable object instance itself _and_ its name.
    '''

    def __init__(self,
                 data: List[Tuple] or Dict = None,
                 variables: Iterable[Variable] = None):
        '''
        ``data`` may be an iterable of (variable, value) pairs.
        '''
        super().__init__()
        self._variables = {v.name: v for v in ifnone(variables, ())}
        self._map = {}
        if data:
            if isinstance(data, dict):
                tuples = data.items()
            else:
                tuples = data
            for var, value in tuples:
                self[var] = value

    @property
    def variables(self) -> Set[Variable]:
        return set(self._variables.values())

    @property
    def varnames(self) -> Dict[str, Variable]:
        return {v.name: v for v in self.variables}

    @property
    def map(self) -> {}:
        return self._map

    def __getitem__(self, key: Union[str, Variable]) -> Any:
        if isinstance(key, Variable):
            return self.__getitem__(key.name)
        return self._map.__getitem__(key)

    def __setitem__(self, variable: Union[str, Variable], value: Any) -> None:
        if type(variable) is str:
            if variable not in self._variables:
                raise ValueError(
                    'Variable "%s" not available in this %s object. '
                    'Set "variables" in the constructor '
                    'or use a Variable object.' % (variable, type(self).__name__))
            variable = self._variables[variable]
        if not isinstance(variable, Variable):
            raise ValueError('Illegal argument value: '
                             'expected Variable, got %s.' % type(variable).__name__)
        self._map[variable.name] = value
        if variable.name not in self._variables:
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
        return bool(len(self))

    def __eq__(self, o: 'VariableMap'):
        return (
            type(o) is type(self) and
            self._map == o._map and
            self.variables == o.variables
        )

    def __hash__(self):
        return hash((
            type(self),
            tuple(
                sorted(
                    [(var, tuple(sorted(val)) if type(val) is set else val)
                     for var, val in self.items()], key=lambda t: t[0].name
                )
            )
        ))

    def __isub__(self, other):
        if isinstance(other, VariableMap):
            for v in other:
                if v in self:
                    del self[v]
        else:
            del self[other]
        return self

    def __iadd__(self, other):
        if isinstance(other, VariableMap):
            for var, val in other.items():
                self[var] = val
            return self
        else:
            raise TypeError('Expected VariableMap, got %s' % type(other).__name__)

    def get(self, key: Union[str, Variable], default=None) -> Any:
        if key not in self:
            return default
        return self[key]

    def keys(self) -> Iterator[Variable]:
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

    def copy(self, deep: bool = False) -> 'VariableMap':
        if not deep:
            return type(self)([(var, val) for var, val in self.items()],
                              variables=self._variables.values())

        vmap = type(self)(variables=self._variables.values())
        for vname, value in self.items():
            if isinstance(value, (numbers.Number, str)):
                vmap[vname] = value
            elif isinstance(value, (set, list, tuple, NumberSet)):
                vmap[vname] = value.copy()
        return vmap

    @classmethod
    def from_json(cls,
                  variables: Iterable[Variable],
                  d: Dict[str, Any],
                  typ=None,
                  args=()) -> 'VariableMap':
        vmap = cls()
        varbyname = {var.name: var for var in variables}
        for vname, value in d.items():
            vmap[varbyname[vname]] = (
                typ.from_json(value, *args)
                if typ is not None and hasattr(typ, 'from_json')
                else value
            )
        return vmap

    def __repr__(self):
        return '<%s {%s}>' % (type(self).__name__,
                              ', '.join(['%s: %s' % (var.name, repr(val)) for var, val in self.items()]))


# ----------------------------------------------------------------------------------------------------------------------

class VariableAssignment(VariableMap):
    '''
    Specialization of a ``VariableMap`` that maps a set of variables
    to values of the respective variables. This is an abstract base class
    that cannot be instantiated.
    There exist two specializations ``LabelAssignment`` and ``ValueAssignment``
    that are supposed to be used instead.
    '''

    def __init__(self,
                 data: Iterable[Tuple] = None,
                 variables: Iterable[Variable] = None):
        super().__init__(data, variables=variables)
        if type(self) is VariableAssignment:
            raise TypeError('Abstract super class %s cannot be instantiated.' % type(self).__name__)

    def scalar2sets(self):
        copy = self.copy()
        for var, val in copy.items():
            copy[var] = var.assignment2set(val)
        return copy

    @classmethod
    def from_json(cls,
                  variables: Iterable[Variable],
                  d: Dict[str, Any],
                  typ=None,
                  args=()) -> 'VariableMap':
        vmap = cls()
        var_by_name = {var.name: var for var in variables}
        for v_name, value in d.items():
            variable = var_by_name[v_name]

            if variable.symbolic or variable.integer:
                value_ = set(value)
            elif variable.numeric:
                value_ = ContinuousSet.from_json(value)
            else:
                raise NotImplementedError('Unknown variable type: %s' % variable.__class__.__name__)

            vmap[variable] = value_
        return vmap


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode
class LabelAssignment(VariableAssignment):
    '''
    Maps a set of variables to values represented by their exterior representation, i.e.
    the perspective of a user.
    '''

    def __setitem__(self,
                    variable: Variable,
                    value: Union[Set[int],
                                 Set[str],
                                 NumberSet,
                                 numbers.Number,
                                 str]) -> None:
        if isinstance(variable, NumericVariable) and not isinstance(value, (numbers.Number, NumberSet)):
            raise TypeError('Expected value of type numbers.Number or NumberSet, got %s.' % type(value).__name__)
        elif isinstance(variable, SymbolicVariable):
            if type(value) is not set:
                value_ = {value}
            else:
                value_ = value
            for v in value_:
                if v not in set(variable.domain.labels.values()):
                    raise TypeError('Value %s is not in the labels of domain %s.' % (v, variable.domain.__name__))
        super().__setitem__(variable, value)

    def value_assignment(self) -> 'ValueAssignment':
        return ValueAssignment([(var, var.domain.label2value(val)) for var, val in self.items()])

    def to_json(self) -> Dict[str, Any]:
        """ Convert this LabelAssignment to a json serializable dictionary. To achieve that sets are replaced
        with lists."""
        result = dict()
        for variable, value in self.items():
            if variable.symbolic or variable.integer:
                if isinstance(value, collections.abc.Iterable):
                    value = list(value)

            result[variable.name] = to_json(value)

        return result


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode
class ValueAssignment(VariableAssignment):
    '''
    Maps a set of variables to values represented by their interior representation, i.e.
    the internal value representation used by JPTs.
    '''

    def __setitem__(self, variable: Variable, value: Union[Set[int], NumberSet, numbers.Number]) -> None:
        if isinstance(variable, NumericVariable) and not isinstance(value, (numbers.Number, NumberSet)):
            raise TypeError('Expected value of type numbers.Number or NumberSet, got %s.' % type(value).__name__)
        elif isinstance(variable, SymbolicVariable) and type(value) not in (numbers.Integral, set):
            if type(value) is not set:
                value_ = {value}
            else:
                value_ = value
            for v in value_:
                if v not in set(variable.domain.values.values()):
                    raise TypeError('Value %s is not in the values of domain %s.' % (v, variable.domain.__name__))
        super().__setitem__(variable, value)

    def label_assignment(self) -> LabelAssignment:
        return LabelAssignment([(var, var.domain.value2label(val)) for var, val in self.items()])
