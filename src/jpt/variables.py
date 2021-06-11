'''
© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
import numbers

import numpy as np
from dnutils import first, ifnone

from jpt.base.intervals import INC, EXC
from jpt.learning.distributions import Multinomial, Numeric
from jpt.base.constants import SYMBOL


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

    def dist(self, params=None, data=None):
        '''
        Create and return a new instance of the distribution type attached to this variable.

        Either the distribution ``params`` can be passed or the ``data`` the distribution parameters
        are to be determined from.
        '''
        if data is None:
            return self._domain(params)
        elif data is not None:
            dist = self.dist(params=params)
            dist.set_data(data)
            return dist

    def __str__(self):
        return f'{self.name}[{self.domain.__name__}]'

    def __repr__(self):
        return str(self)

    @property
    def symbolic(self):
        return issubclass(self.domain, Multinomial)

    @property
    def numeric(self):
        return issubclass(self.domain, Numeric)

    def str(self, assignment, **kwargs):
        raise NotImplemented()


class NumericVariable(Variable):

    def __init__(self, name, domain, min_impurity_improvement=None, haze=None):
        super().__init__(name, domain, min_impurity_improvement=min_impurity_improvement)
        self.haze = ifnone(haze, .05)

    def str(self, assignment, **kwargs):
        fmt = kwargs.get('fmt', 'set')
        precision = kwargs.get('precision', 3)
        lower = '%%.%df %%s ' % precision
        upper = ' %%s %%.%df' % precision
        if type(assignment) is set:
            if len(assignment) == 1:
                valstr = str(first(assignment))
            else:
                valstr = ', '.join([self.str(a, fmt) for a in assignment])
        else:
             valstr = str(assignment)
        if isinstance(assignment, numbers.Number):
            return '%s = %s' % (self.name, assignment)
        if fmt == 'set':
            return f'{self.name} {SYMBOL.IN} {valstr}'
        elif fmt == 'logic':
            return '%s%s%s' % (lower % (assignment.lower,
                                        {INC: SYMBOL.LTE,
                                         EXC: SYMBOL.LT}[assignment.left]) if assignment.lower != np.NINF else '',
                               self.name,
                               upper % ({INC: SYMBOL.LTE,
                                         EXC: SYMBOL.LT}[assignment.right],
                                         assignment.upper) if assignment.upper != np.PINF else '')
        else:
            raise ValueError('Unknown format for numeric variable: %s.' % fmt)


class SymbolicVariable(Variable):

    def __init__(self, name, domain, min_impurity_improvement=None):
        super().__init__(name, domain, min_impurity_improvement=min_impurity_improvement)

    def str(self, assignment, **kwargs):
        fmt = kwargs.get('fmt', 'set')
        if type(assignment) is set:
            if len(assignment) == 1:
                return self.str(first(assignment), fmt=fmt)
            elif fmt == 'set':
                valstr = ', '.join([self.str(a, fmt=fmt) for a in assignment])
                return f'{self.name} {SYMBOL.IN} {{{valstr}}}'
            elif fmt == 'logic':
                return ' v '.join([self.str(a, fmt=fmt) for a in assignment])
        if isinstance(assignment, numbers.Integral):
            return '%s = %s' % (self.name, self.domain.labels[assignment])

