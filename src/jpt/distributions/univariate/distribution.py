'''© Copyright 2021, Mareike Picklum, Daniel Nyga.'''
import numbers
from collections.abc import Hashable
from typing import Set, Iterable, Type, Any, Dict, Union

import numpy as np

from jpt.base.utils import setstr


# ----------------------------------------------------------------------------------------------------------------------
# Constant symbols

SYMBOLIC = 'symbolic'
NUMERIC = 'numeric'
CONTINUOUS = 'continuous'
DISCRETE = 'discrete'


# ----------------------------------------------------------------------------------------------------------------------

class ValueMap(Hashable):

    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, label: Any):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()

    @property
    def map(self):
        return lambda x: self[x]

    def __eq__(self, other):
        raise NotImplementedError()

    def __contains__(self, item):
        raise NotImplementedError()


# ----------------------------------------------------------------------------------------------------------------------

class Identity(ValueMap):
    '''
    Simple identity mapping that mimics the __getitem__ protocol of dicts.
    '''

    def __getitem__(self, item):
        return item

    @property
    def map(self):
        return lambda a: self[a]

    def __eq__(self, o):
        return type(o) is Identity

    def __hash__(self):
        return hash(Identity)

    def __contains__(self, item):
        return True


# ----------------------------------------------------------------------------------------------------------------------

class Distribution:
    '''
    Abstract supertype of all domains and distributions
    '''
    values: ValueMap = None
    labels: ValueMap = None

    SETTINGS = {
    }

    def __init__(self, **settings):
        # used for str and repr methods to be able to print actual type
        # of Distribution when created with jpt.variables.Variable
        self._cl = self.__class__.__qualname__
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
                    'Unknown settings "%s": '
                    'expected one of {%s}' % (
                        attr,
                        setstr(type(self).SETTINGS)
                    )
                )
            self.settings[attr] = value

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name in type(self).SETTINGS:
                return self.settings[name]
            else:
                raise

    @classmethod
    def hash(cls):
        raise NotImplementedError()

    def __hash__(self):
        return hash((type(self), self.values, self.labels))

    def __getitem__(self, value):
        return self.p(value)

    @classmethod
    def value2label(cls, value):
        raise NotImplementedError()

    @classmethod
    def label2value(cls, label):
        raise NotImplementedError()

    def _sample(self, n: int) -> Iterable:
        raise NotImplementedError()

    def _sample_one(self):
        raise NotImplementedError()

    def sample(self, n: int) -> Iterable:
        yield from (self.value2label(v) for v in self._sample(n))

    def sample_one(self) -> Any:
        return self.value2label(self._sample_one())

    def p(self, value) -> float:
        raise NotImplementedError()

    def _p(self, value) -> float:
        raise NotImplementedError()

    def mpe(self):
        raise NotImplementedError()

    def crop(self, restriction: Set) -> 'Distribution':
        raise NotImplementedError()

    def _crop(self, restriction: Set) -> 'Distribution':
        raise NotImplementedError()

    @staticmethod
    def merge(
            distributions: Iterable['Distribution'],
            weights: Iterable[numbers.Real]
    ) -> 'Distribution':
        raise NotImplementedError()

    def update(
            self,
            dist: 'Distribution',
            weight: float
    ) -> 'Distribution':
        raise NotImplementedError()

    def fit(self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: numbers.Integral = None) -> 'Distribution':
        raise NotImplementedError()

    def _fit(self,
             data: np.ndarray,
             rows: np.ndarray = None,
             col: numbers.Integral = None) -> 'Distribution':
        raise NotImplementedError()

    def set(self, params: Any) -> 'Distribution':
        raise NotImplementedError()

    def kl_divergence(self, other: 'Distribution'):
        raise NotImplementedError()

    def number_of_parameters(self) -> int:
        raise NotImplementedError()

    @staticmethod
    def jaccard_similarity(
            d1: 'Distribution',
            d2: 'Distribution'
    ) -> float:
        raise NotImplementedError()

    def plot(self, title=None, fname=None, directory='/tmp', pdf=False, view=False, **kwargs):
        '''Generates a plot of the distribution.

        :param title:       the name of the variable this distribution represents
        :type title:        str
        :param fname:       the name of the file
        :type fname:        str
        :param directory:   the directory to store the generated plot files
        :type directory:    str
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :type pdf:          bool
        :param view:        whether to display generated plots, default False (only stores files)
        :type view:         bool
        :return:            None
        '''
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()

    def __reduce__(self):
        return (
            Distribution.from_json,
            (type(self).type_to_json(), self.to_json()),
        )

    @staticmethod
    def type_from_json(data: Dict[str, Any]) -> Type['Distribution']:
        from .numeric import Numeric, ScaledNumeric
        from .multinomial import Multinomial
        from .integer import Integer

        DISTRIBUTION_TYPES = {
            'numeric': Numeric,
            'scaled-numeric': ScaledNumeric,
            'symbolic': Multinomial,
            'integer': Integer
        }

        cls = DISTRIBUTION_TYPES.get(data['type'])
        if cls is None:
            raise TypeError('Unknown distribution type: %s' % data['type'])
        return cls.type_from_json(data)

    @staticmethod
    def from_json(
            dtype: Dict[str, Any],
            dinst: Dict[str, Any] = None
    ) -> Union['Distribution', Type['Distribution']]:
        clazz = Distribution.type_from_json(dtype)
        if dinst is not None:
            return clazz.from_json(dinst)
        return clazz
