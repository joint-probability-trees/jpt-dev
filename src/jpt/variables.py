'''
© Copyright 2021, Mareike Picklum, Daniel Nyga.
'''
from dnutils import first

from jpt.learning.distributions import Multinomial, Numeric


class Variable:
    '''Wrapper class for a variable name along with its distribution class type.
    '''
    def __init__(self, name, domain):
        '''
        :param name:    name of the variable
        :type name:     str
        :param domain:  the class type (not an instance!) of the represented Distribution
        :type domain:   class type of jpt.learning.distributions.Distribution
        '''
        self._name = name
        self._domain = domain

    @property
    def name(self):
        return self._name

    @property
    def domain(self):
        return self._domain

    def dist(self, params=None):
        return self._domain(params)

    def __str__(self):
        return f'{self.name}[{self.domain.__name__}]'

    def __repr__(self):
        return str(self)  # '<Variable %s @%x>' % (self, id(self))

    @property
    def symbolic(self):
        return issubclass(self.domain, Multinomial)

    @property
    def numeric(self):
        return issubclass(self.domain, Numeric)

    # def tostr(self, val_idx):
    #     if self.symbolic:
    #         valstr = str(self.domain.values[val_idx])
    #     return '%s=%s' % (self.name, valstr)


class SymbolicVariable(Variable):

    def __init__(self, name, domain):
        super().__init__(name, domain)

    def str(self, assignment):
        if type(assignment) is set:
            if len(assignment) == 1:
                valstr = str(first(assignment))
            else:
                valstr = '{%s}' % ','.join(map(str, assignment))
        else:
            valstr = str(assignment)
        return '%s=%s' % (self.name, valstr)

    def str_by_idx(self, assignment):
        if type(assignment) is not set:
            assignment = {assignment}
        return self.str({self.domain.values[a] for a in assignment})


#
#
# def P(space):
#     if type(space) is MultinomialRV:
#         return JPT([space])
#     elif type(space) is Marginal:
#         return JPT(space)
#
#
# class Distribution:
#     '''Abstract base class for all distributions.'''
#
#     def __init__(self, variables):
#         self.variables = variables
#         self.var2idx = {v: i for i, v in enumerate(variables)}
#
#     def to_variable_vector(self, vars):
#         result = [None] * len(self.variables)
#         for v in vars:
#             result[self.var2idx[v]] = v
#         return result
#
#     def to_value_vector(self, vars):
#         result = [None] * len(self.variables)
#         for v in vars:
#             result[self.var2idx[v]] = v.value
#         return result
#
#
# class Conditional(Distribution):
#     '''Conditional distributions'''
#
#     def __init__(self, variables, evidence):
#         super().__init__(variables)
#         self.evidence = evidence
#         self.evidence2idx = {v: i for i, v in enumerate(evidence)}
#
#     def to_evidence_vector(self, vars):
#         result = [None] * len(self.evidence)
#         if any(r is None for r in result):
#             raise ValueError('Not all variables are given: %s' % ';'.join(set(self.evidence).difference((set(vars)))))
#         for v in vars:
#             result[self.evidence2idx[v]] = v
#         return result
#
#     def to_evidence_value_vector(self, vars):
#         result = [None] * len(self.variables)
#         for v in vars:
#             result[self.var2idx[v]] = v.value
#         return result
#
#     def p(self, q, e):
#         return
#
#     def __str__(self):
#         return 'P(%s | %s)' % (self.variables, self.evidence)
#
#
# class Marginal(Distribution):
#     '''Marginal distributions'''
#
#     def __init__(self, variables):
#         super().__init__(variables)
#
#     def p(self, q):
#         raise NotImplemented()
#
#     def __str__(self):
#         return 'P(%s)' % self.variables
#
#
# class SymbolicDistribution(Distribution):
#     '''Base class for symbolic distributions.'''
#
#
# class JPT(SymbolicDistribution):
#     '''Symbolic joint probability table.'''
#
#     def __init__(self, variables):
#         super().__init__(variables)
#         self._jpt = None
#
#     @property
#     def jpt(self):
#         return self._jpt
#
#     @jpt.setter
#     def jpt(self, jpt):
#         self._jpt = jpt
#
#     def p(self, q):
#         return self._jpt[self.to_value_vector(q)]
#
#     def __getitem__(self, query):
#         return self.p(query)
#
#     def __setitem__(self, values, prob):
#         self._jpt[self.to_value_vector(values)] = prob
#
#
# class CPT(Conditional, SymbolicDistribution):
#     '''Symbolic conditional probability table.'''
#
#     def __init__(self, variables):
#         super().__init__(variables)
#         self._cpt = None
#
#     @property
#     def cpt(self):
#         return self._cpt
#
#     @cpt.setter
#     def cpt(self, cpt):
#         self._cpt = cpt
#
#     def p(self, e):
#         return self._cpt[self.to_evidence_value_vector(e)]
#
#     def __getitem__(self, values):
#         return self.p(values)
#
#     def __setitem__(self, evidence, dist):
#         self._cpt[evidence] = dist
#
#
# class Uniform(SymbolicDistribution):
#     '''A uniform distribution for symbolic variables.'''
#
#     def __init__(self, space):
#         super().__init__(space, None)
#         self._size = space.size()
#
#     def prob(self, world):
#         return 1 / self._size()
#
#
# class NumericDistribution:
#     '''Abstract base class for numeric distributions.'''
#
#
# class DiscreteDistribution:
#     '''Base class for discrete numeric distributions.'''
#
#
# class ContinuousDistribution:
#     '''Base class for continuous numeric distributions.'''
#
#
# class Domain:
#     '''
#     Base class that represents the domain of a random variable.
#
#     Values are ordered so they can be identified by their index.
#     '''
#
#     def __init__(self, values=None):
#         self.values = ifnone(values, [], list)
#         self.val2index = {v: i for i, v in enumerate(self.values)}
#
#     def __contains__(self, item):
#         return item in self.values
#
#     def __iter__(self):
#         return iter(self.values)
#
#     def __len__(self):
#         return len(self.values)
#
#     def __str__(self):
#         return '{%s}' % ', '.join(self.values)
#
#     def __repr__(self):
#         return '<Domain @%x>' % id(self)


# class ProbabilitySpace:
#     '''
#     Abstract base class for all probability spaces. A probability
#     space can be either a joint or a conditional.'''


# class Conditional(ProbabilitySpace):
#     '''
#     Represents a conditional set of random variables.
#     '''

    # def __init__(self, variables):
    #     if isinstance(variables, MultinomialRV):
    #         self.variables = ProbabilitySpace([variables])
    #     elif isinstance(variables, (tuple, list)):
    #         self.variables = ProbabilitySpace(variables)
    #     else:
    #         self.variables = variables
    #
    # def size(self):
    #     return self.variables.size()
    #
    # def __str__(self):
    #     return ('%s' % self.variables)

#
# class ProbabilitySpace:
#     '''
#     This class represents the joint of a set of random variables.
#     '''
#
#     def __init__(self, variables):
#         self.variables = variables
#
#     def __str__(self):
#         return ' ^ '.join(mapstr(self.variables))
#
#     def _value_tuples(self, values, variables):
#         if not variables:
#             yield tuple(values)
#             return
#         var = variables.pop(0)
#         for value in var.domain:
#             yield from self._value_tuples(values + [value], list(variables))
#
#     def value_tuples(self):
#         yield from self._value_tuples([], list(self.variables))
#
#     def size(self):
#         return prod(len(v.domain) for v in self.variables)
#
#     def __and__(self, other):
#         return ProbabilitySpace(self.variables + (other.variables if isinstance(other, ProbabilitySpace) else [other]))
#
#     # def __or__(self, other):
#     #     return ProbabilitySpace(self, other)
#
#
# class RV:
#     pass
#
#
# class MultinomialRV(RV):
#     '''
#     Base class for a multinomial symbolic random variable.
#     '''
#
#     def __init__(self, name, domain, value=None):
#         self.name = name
#         self.domain = domain
#         self._value = None
#         if value is not None:
#             self.value = value
#
#     @property
#     def value(self):
#         return self._value
#
#     @value.setter
#     def value(self, v):
#         if v not in self.domain:
#             raise ValueError('Value must be in {%s}, but got %s.' % (str(self.domain), v))
#         self._value = v
#
#     def set(self, v):
#         self.value = v
#         return self
#
#     def copy(self, *value):
#         result = MultinomialRV(self.name, self.domain, self.value)
#         if value:
#             result.value = first(value)
#         return result
#
#     def draw_one(self, pdf):
#         return first(self.draw(pdf, 1))
#
#     def draw(self, pdf, n):
#         '''
#         Returns ``n`` copies of this random variables, each of which has
#         a value assigned drawn from the distribution ``pdf``.
#
#         :param pdf:
#         :param n:
#         :return:
#         '''
#         yield from (self.copy(v) for v in wsample(self.domain.values, pdf, n))
#
#     def __or__(self, other):
#         return Conditional([self], other)
#
#     def __and__(self, other):
#         return Conditional([self] + (other if isinstance(other, list) else [other]))
#
#     def __str__(self):
#         return ('%s' % self.name) + ('=%s' % self.value if self.value is not None else '')
#
#
