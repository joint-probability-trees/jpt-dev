'''© Copyright 2021, Mareike Picklum, Daniel Nyga.'''
import numbers
from collections import Counter
from itertools import tee
from operator import itemgetter
from types import FunctionType
from typing import Union, Any, Set, Optional, List, Tuple, Iterable, Type, Collection

import numpy as np
from deprecated import deprecated
from dnutils import ifnone, first
from plotly.graph_objs import Figure

from . import Distribution
from ..utils import OrderedDictProxy
from ...base.errors import Unsatisfiability
from ...base.sampling import wsample, wchoice
from ...base.utils import mapstr, classproperty, Symbol, Collections
from ...plotting.rendering import DistributionRendering


# noinspection DuplicatedCode
class Multinomial(Distribution):
    '''
    Abstract supertype of all symbolic domains and distributions.
    '''

    values: OrderedDictProxy = None
    labels: OrderedDictProxy = None

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Multinomial) or type(self) is Multinomial:
            raise Exception(f'Instantiation of abstract class {type(self)} is not allowed!')

        self._params: Optional[np.ndarray] = None
        self.to_json: FunctionType = self.inst_to_json

    @classmethod
    def hash(cls):
        return hash((
            cls.__qualname__,
            cls.values,
            cls.labels,
            tuple(
                sorted(cls.SETTINGS.items())
            )
        ))

    # noinspection DuplicatedCode
    @classmethod
    def value2label(
            cls,
            value: Union[int, Iterable[int]]
    ) -> Union[Symbol, Collection[Symbol]]:
        if not isinstance(value, Collections):
            value_ = {value}
        else:
            value_ = value
        if not all(
            [v in cls.labels for v in value_]
        ):
            raise ValueError(
                '%s not among the values of domain %s.' % (
                    first([v for v in value_ if v not in cls.labels]),
                    cls.__qualname__
                )
            )
        if not isinstance(value, Collections):
            return cls.labels[value]
        # noinspection PyArgumentList
        return type(value)([cls.labels[v] for v in value_])

    # noinspection DuplicatedCode
    @classmethod
    def label2value(
            cls,
            label: Union[Symbol, Collection[Symbol]]
    ) -> Union[int, Collection[int]]:
        if not isinstance(label, Collections):
            label_ = {label}
        else:
            label_ = label
        if not all(
            [l in cls.values for l in label_]
        ):
            raise ValueError(
                '%s not among the labels of domain %s.' % (
                    first([l for l in label_ if l not in cls.labels]),
                    cls.__qualname__
                )
            )
        if not isinstance(label, Collections):
            return cls.values[label]
        # noinspection PyArgumentList
        return type(label)([cls.values[l] for l in label_])

    @classmethod
    def pfmt(cls, max_values=10, labels_or_values='labels') -> str:
        '''
        Returns a pretty-formatted string representation of this class.

        By default, a set notation with value labels is used. By setting
        ``labels_or_values`` to ``"values"``, the internal value representation
        is used. If the domain comprises more than ``max_values`` values,
        the middle part of the list of values is abbreviated by "...".
        '''
        if labels_or_values not in ('labels', 'values'):
            raise ValueError(
                'Illegal Value for "labels_or_values": Expected one out of '
                '{"labels", "values"}, got "%s"' % labels_or_values
            )
        return '%s = {%s}' % (
            cls.__name__,
            ', '.join(
                mapstr(
                    cls.values.values() if labels_or_values == 'values'
                    else cls.labels.values(),
                    limit=max_values
                )
            )
        )

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls) -> int:
        return len(cls.values)

    def __contains__(self, item):
        return item in self.values

    @classmethod
    def equiv(cls, other):
        if not issubclass(other, Multinomial):
            return False
        return cls.__name__ == other.__name__ and cls.labels == other.labels and cls.values == other.values

    @staticmethod
    def jaccard_similarity(
            *d: 'Multinomial'
    ) -> float:
        '''Calculate the similarity of two or more Multinomial distributions,

            \text{sim}(D_1, ..., D_n) = \frac{\sum_{x \in dom(D)} min(p_i(x))}{\sum_{x \in dom(D)} max(p_i(x))}

        adapted from the Jaccard coefficient:

            \text{sim}(S_1, ..., S_n) = \frac{\mid \bigcap_{i}^{n} S_i \mid}{\mid \bigcup_{i}^{n} S_i \mid}
         '''
        # if the domains of the given distributions are not identical, they are considered maximally dissimilar
        if any([len(set(i)) != 1 for i in zip(*map(lambda x: x.values, d))]): return 0.

        intersect = sum([min(a) for a in zip(*map(lambda x: x.probabilities, d))])
        union = sum([max(a) for a in zip(*map(lambda x: x.probabilities, d))])

        return intersect / union

    def mover_dist(
            self,
            other: 'Multinomial'
    ) -> float:
        # if the domains of the given distributions are not identical, they are considered maximally distant
        if any([len(set(i)) != 1 for i in zip(self.values, other.values)]): return np.PINF

        # otherwise determine the "effort" that is required to transform one distribution into
        # the other, defined as the sum of the absolute values of the differences in relative frequency of each
        # label, e.g.:
        # d1 = [.1, .4, .5]
        # d2 = [.2, .4, .4]
        # mover_dist(d1, d2) = |.1 - .2 | + | .4 - .4 | + | .5 - .4 | = .2
        return sum([abs(a-b) for a, b in zip(self.probabilities, other.probabilities)])

    def similarity(
            self,
            other: 'Multinomial'
    ) -> float:
        return self.mover_dist(other)
        # return Multinomial.jaccard_similarity(self, other)

    def distance(
            self,
            other: 'Multinomial',
    ) -> float:
        return 1-Multinomial.jaccard_similarity(self, other)

    def __getitem__(self, value):
        return self.p([value])

    def __setitem__(self, label, p):
        self._params[self.values[label]] = p

    def __eq__(self, other):
        return type(self).equiv(
            type(other)
        ) and (
            self.probabilities == other.probabilities
        ).all()

    def __str__(self):
        if self._p is None:
            return f'<{self.__class__.__qualname__} p=n/a>'
        return '<%s p=[%s]>' % (
            self.__class__.__qualname__,
            ";".join(
               [f"{v}={p:.3f}" for v, p in zip(self.values, self.probabilities)]
            )
        )

    def __repr__(self):
        return str(self)

    def sorted(self) -> Iterable[Tuple[float, Symbol]]:
        '''
        Generate a sequence of (prob, label) pairs representing this distribution,
        ordered by descending probability.
        :return:
        '''
        yield from sorted([
            (p, l) for p, l in zip(self._params, self.labels.values())],
            key=itemgetter(0),
            reverse=True
        )

    def _items(self) -> Iterable[Tuple[float, int]]:
        '''Generate a sequence of (probability, value) pairs representing this distribution.'''
        yield from ((p, v) for p, v in zip(
            self.probabilities,
            self.values.values()
        ))

    def items(self) -> Iterable[Tuple[float, Symbol]]:
        '''Generate a sequence of (probability, label) pairs representing this distribution.'''
        yield from ((p, l) for p, l in zip(
            self.probabilities,
            self.labels.values()
        ))

    def copy(self):
        return type(self)(**self.settings).set(params=self._params)

    def _pdf(self, value: int) -> float:
        return self._p(value)

    def pdf(self, label: Symbol) -> float:
        return self.p(label)

    def p(
        self,
        event: Union[Symbol, Set[Symbol], List[Symbol], Tuple[Symbol], np.ndarray]
    ) -> float:
        '''
        Compute the probability of a certain ``event`` given this
        multinomial distribution.

        An event can be atomic random event, or a disjunction thereof, e.g. given
        the domain values {'Head', 'Tail'}, ``event`` may be

            dist.p('Head')
            dist.p({'Tail'})
            dist.p({'Head', 'Tail'})

        :param event:   the event in label space, the prob' of which is to be computed.
        :return:        the probability of the ``event``
        '''
        return self._p(
            self.label2value(event)
        )

    def _p(
        self,
        event: Union[int, Set[int], List[int], Tuple[int], np.ndarray]
    ) -> float:
        '''
        Compute the probability of a certain ``event`` given this
        multinomial distribution.

        See also ``Multinomial.p()``

        :param event:   the event int value space, the prob' of which is to be computed.
        :return:        the probability of the ``event``
        '''
        if isinstance(event, numbers.Integral):
            event = {event}
        elif not isinstance(event, set):
            event = set(event)
        i1, i2 = tee(event, 2)
        if not all(isinstance(v, numbers.Integral) for v in i1):
            raise TypeError('All arguments must be integers: %s' % event)

        return sum(
            self.probabilities[v] for v in i2
        )

    def create_dirac_impulse(
            self,
            value: int
    ) -> 'Multinomial':
        '''
        Create a singular modification of this distribution object, in which the
        ``value`` has probability ``1``, whereas all other events have prob ``0``.

        :param value:    the singular value to get assigned prob ``1``.
        :return:         the created distribution object
        '''
        result = self.copy()
        result._params = np.zeros(
            shape=result.n_values,
            dtype=np.float64
        )
        result._params[value] = 1
        return result

    def _sample(self, n: int) -> Iterable[int]:
        '''Returns ``n`` sample `values` according to their respective probability'''
        return wsample(
            list(self.values.values()),
            self.probabilities,
            n
        )

    def _sample_one(self) -> Symbol:
        '''Returns one sample `value` according to its probability'''
        return wchoice(
            list(self.values.values()),
            self._params
        )

    @deprecated('Expectation is undefined in symbolic domains. Use Multinomial._mode() instead.')
    def _expectation(self) -> Set[int]:
        '''Returns the value with the highest probability for this variable'''
        return self._mpe()[0]

    @deprecated('Expectation is undefined in symbolic domains. Use Multinomial._mode() instead.')
    def expectation(self) -> Set[Symbol]:
        """
        For symbolic variables the expectation is equal to the mpe.
        :return: The set of all most likely values
        """
        return self.value2label(
            self._expectation()
        )

    def mpe(self) -> Tuple[Set[Symbol], float]:
        states, p_max = self._mpe()
        return self.value2label(states), p_max

    def _mpe(self) -> Tuple[Set[int], float]:
        """
        Calculate the most probable configuration of this distribution in value space.

        :return: The likelihood of the mpe itself as Set and the likelihood of the mpe as float
        """
        return first(self._k_mpe(k=1))

    def _k_mpe(self, k: int = None) -> List[Tuple[Set[Symbol], float]]:
        likelihoods = {p for p in self.probabilities if p}
        sorted_likelihood = sorted(
            likelihoods,
            reverse=True
        )[:ifnone(k, len(likelihoods))]
        result = []

        for likelihood in sorted_likelihood:
            result.append(
                (
                    {value for value, p in zip(self.values.values(), self.probabilities) if p == likelihood},
                    likelihood
                )
            )

        return result

    def k_mpe(self, k: Optional[int] = None) -> List[Tuple[Set[Symbol], float]]:
        return [
            (
                self.value2label(state),
                likelihood
            ) for state, likelihood in self._k_mpe(k=k)
        ]

    mode = mpe
    _mode = _mpe

    def kl_divergence(self, other: 'Multinomial') -> float:
        '''
        Compute the KL-divergence of this distribution to the ``other`` distribution.
        :param other:
        :return:
        '''
        if type(other) is not type(self):
            raise TypeError(
                'Can only compute KL divergence between '
                'distributions of the same type, got %s' % type(other)
            )
        result = 0
        for v in range(self.n_values):
            result += self.probabilities[v] * abs(self.probabilities[v] - other.probabilities[v])
        return result

    def _crop(
            self,
            restriction: Union[int, Collection[int]],
    ) -> 'Multinomial':
        if not isinstance(restriction, Collections):
            return self.create_dirac_impulse(restriction)

        result = self.copy()
        for idx, value in enumerate(result.values.values()):
            if value not in restriction:
                result.probabilities[idx] = 0

        if sum(result._params) == 0:
            raise Unsatisfiability(
                'All values have zero probability [%s].' % type(result).__name__
            )
        else:
            result._params = result.probabilities / sum(result.probabilities)
        return result

    def crop(
            self,
            restriction: Union[Symbol, Collection[Symbol]]
    ) -> 'Multinomial':
        """
        Apply a restriction to this distribution such that all values are in the given set.

        :param restriction: The values to remain
        :return: Copy of self that is consistent with the restriction
        """
        return self._crop(
            self.label2value(restriction)
        )

    def _fit(
            self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: int = None
        ) -> 'Multinomial':

        self._params = np.zeros(shape=self.n_values, dtype=np.float64)
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(len(data))):
            self._params[int(data[row, col])] += 1 / n_samples
        return self

    def set(
            self,
            params: Iterable[numbers.Real]
    ) -> 'Multinomial':
        if len(self.values) != len(params):
            raise ValueError('Number of values and probabilities must coincide.')
        self._params = np.array(params, dtype=np.float64)
        return self

    def update(
            self,
            dist: 'Multinomial',
            weight: float
    ) -> 'Multinomial':
        '''
        Update this multinomial distribution with ``dist`` and ``weight``.

        The resulting distribution will be a weighted mean of ``self`` and ``dist``,
        where ``self`` will have a weight of ``(1-weight)``, and ``dist`` will
        have a weight of ``weight``.

        :param dist:     the update distribution
        :param weight:   the weight
        :return:
        '''
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1], got %s' % weight)
        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight

        return self

    @staticmethod
    def merge(
            distributions: Iterable['Multinomial'],
            weights: Iterable[float]
    ) -> 'Multinomial':
        '''
        Merge the ``distributions`` under consideration of ``weights``.

        :param distributions:
        :param weights:
        :return:
        '''
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError(
                'Only distributions of the same type can be merged.'
            )

        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError(
                'Weights must sum to 1 (but is %s).' % sum(weights)
            )

        params = np.zeros(distributions[0].n_values)
        for d, w in zip(distributions, weights):
            params += d.probabilities * w

        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability(
                'Sum of weights must not be zero.'
            )

        return type(distributions[0])().set(params)

    @classmethod
    def type_to_json(cls):
        return {
            'type': 'symbolic',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values())
        }

    def inst_to_json(self):
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    to_json = type_to_json

    @staticmethod
    def type_from_json(data):
        return SymbolicType(data['class'], data['labels'])

    @classmethod
    def from_json(cls, data):
        return cls(**data['settings']).set(data['params'])

    def is_dirac_impulse(self):
        for p in self._params:
            if p == 1:
                return True
        return False

    def number_of_parameters(self) -> int:
        """
        :return: The number of relevant parameters in this decision node.
                 1 if this is a dirac impulse, number of parameters else
        """
        if self.is_dirac_impulse():
            return 1
        return len(self._params)

    def plot(
            self,
            engine: str,
            **kwargs
    ) -> Figure:

        return DistributionRendering(engine).plot_multinomial(
            self,
            **kwargs
        )

# ----------------------------------------------------------------------------------------------------------------------

class Bool(Multinomial):
    '''
    Wrapper class for Boolean domains and distributions.
    '''

    values = OrderedDictProxy([(False, 0), (True, 1)])
    labels = OrderedDictProxy([(0, False), (1, True)])

    def __init__(self, **settings):
        super().__init__(**settings)

    def set(self, params: Union[np.ndarray, float]) -> 'Bool':
        if params is not None and not isinstance(params, Iterable):
            params = [1 - params, params]
        super().set(params)
        return self

    # def __str__(self):
    #     if self.p is None:
    #         return f'{self._cl}<p=n/a>'
    #     return f'{self._cl}<p=[{",".join([f"{v}={p:.3f}" for v, p in zip(self.labels, self._params)])}]>'

    def __setitem__(self, v, p):
        if not isinstance(p, Iterable):
            p = np.array([p, 1 - p])
        super().__setitem__(v, p)


# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming
def SymbolicType(name: str, labels: Iterable[Any]) -> Type[Multinomial]:
    if len(labels) < 1:
        raise ValueError('At least one value is needed for a symbolic type.')
    if len(set(labels)) != len(labels):
        duplicates = [item for item, count in Counter(labels).items() if count > 1]
        raise ValueError('List of labels  contains duplicates: %s' % duplicates)
    t = type(name, (Multinomial,), {})
    t.values = OrderedDictProxy([(lbl, int(val)) for val, lbl in zip(range(len(labels)), labels)])
    t.labels = OrderedDictProxy([(int(val), lbl) for val, lbl in zip(range(len(labels)), labels)])
    return t