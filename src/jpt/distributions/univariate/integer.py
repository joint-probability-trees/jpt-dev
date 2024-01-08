'''© Copyright 2021, Mareike Picklum, Daniel Nyga.'''
import numbers
import re
from itertools import tee, count
from operator import itemgetter, attrgetter
from types import FunctionType
from typing import Optional, Type, Dict, Any, Union, Set, Iterable, Tuple, List

import numpy as np
from deprecated.classic import deprecated
from dnutils import edict, ifnone, project, first
from matplotlib import pyplot as plt

from . import Distribution
from .distribution import ValueMap
from ...base.errors import Unsatisfiability
from ...base.sampling import wsample, wchoice
from utils import setstr, normalized, classproperty, save_plot, Collections

from functions import PiecewiseFunction, Undefined, ConstantFunction
from intervals import ContinuousSet, IntSet, RealSet, NumberSet


# ----------------------------------------------------------------------------------------------------------------------

class IntegerMap(ValueMap):
    '''A mapping of external integers to their internal representation.'''

    def __init__(self, lmin: Optional[int] = ..., lmax: Optional[int] = ...):
        self.min = int(lmin) if lmin is not ... else ...
        self.max = int(lmax) if lmax is not ... else ...

    def __iter__(self):
        return iter(
            IntSet(
                np.NINF if self.min is ... else self.min,
                np.PINF if self.max is ... else self.max
            )
        )

    def __len__(self):
        if self.min is ... and self.max is ...:
            return np.inf
        elif self.min is ... and self.max is not ...:
            return np.inf
        elif self.min is not ... and self.max is ...:
            return np.inf
        else:
            return self.max - self.min + 1

    def __hash__(self):
        return hash((
            type(self),
            self.min,
            self.max
        ))

class IntegerLabelMap(IntegerMap):

    def __getitem__(self, label: int) -> int:
        if self.min is not ...:
            result = label - self.min
        elif self.max is not ...:
            result = self.max - label
        else:
            result = label
        if not ((np.NINF if self.min is ... else self.min)
                <= label <=
                (np.PINF if self.max is ... else self.max)
        ):
            raise ValueError(
                f'Label {label} is out of domain {{{self.min}..{self.max}}}'
            )
        return result


class IntegerValueMap(IntegerMap):

    def __getitem__(self, value: int) -> int:
        if self.min is not ...:
            result = value + self.min
        elif self.max is not ...:
            result = value + self.max
        else:
            result = value
        if not (
                (np.NINF if self.min is ... else self.min)
                <= result <=
                (np.PINF if self.max is ... else self.max)
        ):
            raise ValueError(
                f'Value {value} is out of domain [{self.min}, {self.max}]'
            )
        return result


# ----------------------------------------------------------------------------------------------------------------------

# noinspection DuplicatedCode
class Integer(Distribution):

    values: Optional[IntegerLabelMap]
    labels = Optional[IntegerValueMap]

    OPEN_DOMAIN = 'open_domain'
    AUTO_DOMAIN = 'auto_domain'

    SETTINGS = edict(Distribution.SETTINGS) + {
        OPEN_DOMAIN: False,
        AUTO_DOMAIN: False
    }

    @classproperty
    @deprecated
    def lmin(cls):
        return cls.values.min

    @classproperty
    @deprecated
    def lmax(cls):
        return cls.values.max

    @classproperty
    def min(cls) -> Optional[int]:
        return cls.values.min

    @classproperty
    def max(cls) -> Optional[int]:
        return cls.values.max

    @classproperty
    def _min(cls) -> Optional[int]:
        return cls.values[cls.values.min]

    @classproperty
    def _max(cls) -> Optional[int]:
        return cls.values[cls.values.max]

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Integer) or type(self) is Integer:
            raise Exception(
                f'Instantiation of abstract class {type(self)} is not allowed!'
            )
        self._params: Optional[Dict[int, float]] = None
        self.to_json: FunctionType = self.inst_to_json

    @classmethod
    def hash(cls):
        return hash((
            cls.__qualname__,
            cls.min,
            cls.max,
            tuple(
                sorted(cls.SETTINGS.items())
            )
        ))

    def __add__(
            self,
            other: 'Integer'
    ) -> 'Integer':
        return self.add(other)

    @property
    def cdf(self) -> PiecewiseFunction:
        cdf = PiecewiseFunction()
        vals = list(self.values.values())

        if self._params is None:
            raise Exception(f'Fit!')

        for v1, v2, p in zip(['-inf']+vals[:-1], vals, self._params):
            # constant for each value and undefined for everything in between
            cdf.append(interval=ContinuousSet.parse(f']{v1},{v2}['), f=Undefined())
            cdf.append(interval=ContinuousSet.parse(f'[{v2},{v2}]'), f=ConstantFunction(p))
        # undefined for everythin after last value
        cdf.append(interval=ContinuousSet.parse(f']{v2},inf['), f=Undefined())

        return cdf

    def add(
            self,
            other: 'Integer',
            name: Optional[str] = None
    ) -> 'Integer':
        res_t = IntegerType(
            name=f'{name if name is not None else self.__class__.__name__}+{other.__class__.__name__}',
            lmin=(type(self).min + type(other).min) if ... not in (type(self).min, type(other).min) else ...,
            lmax=(type(self).max + type(other).max) if ... not in (type(self).max, type(other).max) else ...
        )
        res = res_t()
        z_min = (
            min(project(self.items(False), 1))
            +
            min(project(other.items(False), 1))
        )
        z_max = (
            max(project(self.items(False), 1))
            +
            max(project(other.items(False), 1))
        )

        dist = {}
        for z in range(z_min, z_max + 1):
            p = 0
            for p_k, k in self.items():
                p += p_k * (
                    other.p(z - k)
                    if ((type(other).min if type(other).min is not ... else np.NINF)
                        <= z - k <=
                        (type(other).max if type(other).max is not ... else np.PINF))
                    else 0
                )
            if not p:
                continue
            dist[z] = p
        res.set(dist)
        return res

    @classmethod
    def equiv(cls, other: Type[Distribution]) -> bool:
        if not issubclass(other, Integer):
            return False
        return all((
            cls.__name__ == other.__name__,
            cls.min == other.min,
            cls.max == other.max,
            cls.SETTINGS == other.SETTINGS
        ))

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        return {
            'type': 'integer',
            'class': cls.__qualname__,
            'min': int(cls.min) if cls.min is not ... else None,
            'max': int(cls.max) if cls.max is not ... else None,
            'settings': cls.SETTINGS
        }

    to_json = type_to_json

    def inst_to_json(self) -> Dict[str, Any]:
        return {
            'class': type(self).__qualname__,
            'params': self._params,
            'settings': self.settings
        }

    @staticmethod
    def type_from_json(data):
        i_type = IntegerType(
            data['class'],
            data.get('min', data.get('lmin')),
            data.get('max', data.get('lmax'))
        )
        i_type.SETTINGS = data.get(
            'settings',
            Integer.SETTINGS
        )
        return i_type

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Integer':
        return cls(**data['settings']).set(data['params'])

    def copy(self):
        result = type(self)(**self.settings)
        result._params = self._params.copy() if self._params is not None else None
        return result

    @property
    def probabilities(self) -> Dict[int, float]:
        return self._params

    @classproperty
    def n_values(cls) -> Optional[int]:
        if ... not in (cls.min, cls.max):
            return cls.max - cls.min + 1
        else:
            return None

    # noinspection DuplicatedCode
    @classmethod
    def value2label(
            cls,
            value: Union[int, Iterable[int], IntSet, RealSet]
    ) -> Union[int, Iterable[int], IntSet, RealSet]:
        if isinstance(value, Collections):
            return type(value)([cls.value2label(v) for v in value])
        elif isinstance(value, IntSet):
            return IntSet(cls.value2label(value.lower), cls.value2label(value.upper))
        elif isinstance(value, RealSet):
            return RealSet(
                cls.value2label(v) for v in value.intervals
            )
        else:
            return cls.labels[value]

    # noinspection DuplicatedCode
    @classmethod
    def label2value(
            cls,
            label: Union[int, Iterable[int], IntSet, RealSet]
    ) -> Union[int, Iterable[int], IntSet, RealSet]:
        if isinstance(label, Collections):
            return type(label)([cls.label2value(l) for l in label])
        elif isinstance(label, IntSet):
            return IntSet(cls.label2value(label.lower), cls.label2value(label.upper))
        elif isinstance(label, RealSet):
            return RealSet(
                cls.label2value(l) for l in label.intervals
            )
        else:
            return cls.values[label]

    def _sample(self, n: int) -> Iterable[int]:
        items = self.probabilities.items()
        return wsample(
            project(items, 0),
            weights=project(items, 1),
            k=n
        )

    def _sample_one(self) -> int:
        items = self.probabilities.items()
        return wchoice(
            project(items, 0),
            weights=project(items, 1),
        )

    def sample(self, n: int) -> Iterable[int]:
        return [self.value2label(v) for v in self._sample(n)]

    def sample_one(self) -> int:
        return self.value2label(self._sample_one())

    @property
    def _pdf(self) -> FunctionType:
        return self._p

    @property
    def pdf(self) -> FunctionType:
        return self.p

    def p(self, labels: Union[int, Iterable[int]]) -> float:
        return self._p(self.label2value(labels))

    def _p(self, values: Union[int, Iterable[int]]) -> float:
        if isinstance(values, numbers.Integral):
            values = {values}
        if isinstance(values, Collections):
            values = set(values)
        if isinstance(values, set):
            values = IntSet.from_set(values)
        if isinstance(values, RealSet):
            return sum(self._p(i) for i in values.simplify().intervals)
        if isinstance(values, IntSet):
            return sum(p for v, p in self.probabilities.items() if v in values and p)

    def expectation(self) -> float:
        return sum(
            p * self.value2label(v) for v, p in self.probabilities.items() if p
        )

    def _expectation(self) -> float:
        return sum(
            p * v for v, p in self.probabilities.items() if p
        )

    def variance(self) -> float:
        e = self.expectation()
        return sum(
            (self.value2label(v) - e) ** 2 * p for v, p in self.probabilities.items()
        )

    def _variance(self) -> float:
        e = self._expectation()
        return sum(
            (v - e) ** 2 * p for v, p in self.probabilities.items()
        )

    def _k_mpe(self, k: Optional[int] = None) -> Iterable[Tuple[NumberSet, float]]:
        """
        Calculate the ``k`` most probable explanation states.

        :param k: The number of solutions to generate
        :return: An list containing a tuple containing the likelihood and state in descending order.
        """
        likelihoods = {p for p in self.probabilities.values() if p}
        sorted_likelihood = sorted(
            likelihoods,
            reverse=True
        )[:ifnone(k, len(likelihoods))]

        for likelihood in sorted_likelihood:
            yield IntSet.from_set(
                {v for v, p in self.probabilities.items() if p == likelihood}
            ), likelihood

    def k_mpe(self, k: int = None) -> Iterable[Tuple[NumberSet, float]]:
        return ((self.value2label(state), likelihood) for state, likelihood in self._k_mpe(k=k))

    def mpe(self) -> (NumberSet, float):
        state, p_max = self._mpe()
        return self.value2label(state), p_max

    def _mpe(self) -> (Set[int], float):
        return first(self._k_mpe(k=1))

    mode = mpe
    _mode = _mpe

    def crop(self, restriction: Union[NumberSet, int]) -> 'Integer':
        if isinstance(restriction, numbers.Integral):
            restriction = IntSet.from_set({restriction})
        return self._crop(self.label2value(restriction))

    def _crop(self, restriction: Union[NumberSet, int]) -> 'Integer':
        if isinstance(restriction, numbers.Integral):
            restriction = IntSet.from_set({restriction})
        result = self.copy()
        try:
            params = normalized({
                v: p for v, p in self.probabilities.items() if v in restriction
            })
        except ValueError:
            raise Unsatisfiability(
                'Restriction unsatisfiable: probabilities must sum to 1.'
            )
        else:
            return result.set(params=params)

    @staticmethod
    def merge(
            distributions: Iterable['Integer'],
            weights: Iterable[numbers.Real]
    ) -> 'Integer':
        if not all(type(distributions[0]).equiv(type(d)) for d in distributions):
            raise TypeError('Only distributions of the same type can be merged.')

        if abs(1 - sum(weights)) > 1e-10:
            raise ValueError('Weights must sum to 1 (but is %s).' % sum(weights))

        params = {}

        for d, w in zip(distributions, weights):
            for v, p in d.probabilities.items():
                params[v] = params.get(v, 0) + d.probabilities.get(v, 0) * w

        if abs(sum(params.values())) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')

        return type(distributions[0])().set(params)

    def update(
            self,
            dist: 'Integer',
            weight: int
    ) -> 'Integer':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')

        params = {}
        for v in set(self.probabilities.keys()).union(dist.probabilities.keys()):
            params[v] = (1 - weight) * self._p(v) + weight * dist._p(v)
        self._params = params
        return self

    def fit(
            self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: int = None
    ) -> 'Integer':
        if rows is None:
            rows = range(data.shape[0])

        data_ = np.array(
            [self.label2value(int(data[row][col])) for row in rows],
            dtype=data.dtype
        )
        return self._fit(
            data_.reshape(-1, 1),
            None,
            0
        )

    def _fit(
            self,
            data: np.ndarray,
            rows: np.ndarray = None,
            col: int = None
    ) -> 'Integer':
        self._params = {}
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(data.shape[0])):
            v = int(data[row, col])
            self._params[v] = self._params.get(v, 0) + 1 / n_samples

        return self

    def set(self, params: Dict[int, float] or Iterable[float]) -> 'Integer':
        if isinstance(params, dict):
            probabilities = params.copy()
        else:
            if not self.finite:
                raise ValueError(
                    'Unable to set unbounded integer distributions '
                    'with object of type %s' % type(params).__qualname__
                )
            if ifnone(self.n_values, np.PINF) != len(params):
                raise ValueError(
                    'Number of values and probabilities must coincide.'
                )
            probabilities = {i: p for i, p in enumerate(params)}
        if abs(sum(probabilities.values()) - 1) > 1e-8:
            raise ValueError(
                'Probabilities must sum to 1, got %s' % sum(probabilities.values())
            )
        self._params = probabilities
        return self

    def __eq__(self, other) -> bool:
        return (
            type(self).equiv(type(other)) and
            self.probabilities == other.probabilities
        )

    def __str__(self):
        if self._p is None:
            return f'<{type(self).__qualname__} p=n/a>'
        return '<%s p=[%s]>' % (
            self.__class__.__qualname__,
            "; ".join([f"{v}: {p:.3f}" for v, p in zip(self.labels.values(), self.probabilities)])
        )

    def __repr__(self):
        return str(self)

    @classproperty
    def finite(cls) -> bool:
        return ... not in (cls.min, cls.max)

    @classproperty
    def infinite(cls) -> bool:
        return not cls.finite

    def _sorted(
            self,
            exhaustive: bool = True,
            reverse: bool = False
    ) -> Iterable[Tuple[int, float]]:
        probabilities = self.probabilities.copy()
        if self.infinite and exhaustive:
            raise ValueError(
                'Unable to exhaustively enumerate items in an infinite domain.'
            )
        if exhaustive and ... not in (self.min, self.max):
            probabilities.update({
                v: 0 for v in range(
                    self.label2value(type(self).min),
                    self.label2value(type(self).max) + 1
                ) if v not in self.probabilities
            })
        return sorted(
            [(v, p) for v, p in probabilities.items() if p or exhaustive],
            key=itemgetter(1),
            reverse=not reverse
        )

    def sorted(
            self,
            exhaustive: bool = True,
            reverse: bool = False
    ) -> Iterable[Tuple[int, float]]:
        return ((self.value2label(v), p)
            for v, p in self._sorted(
                exhaustive=exhaustive,
                reverse=reverse
            )
        )

    def _items(self, exhaustive: bool = False) -> Iterable[Tuple[float, int]]:
        '''Return a list of (probability, value) pairs representing this distribution.'''
        if not self.finite and exhaustive:
            raise ValueError(
                'Unable to exhaustively enumerate items in an infinite domain.'
            )
        probabilities = self.probabilities.copy()
        if exhaustive and ... not in (self.min, self.max):
            probabilities.update({
                v: 0 for v in range(
                    self.label2value(type(self).min),
                    self.label2value(type(self).max) + 1
                ) if v not in self.probabilities
            })
        yield from sorted(
            ((p, v) for v, p in probabilities.items() if p or exhaustive),
            key=itemgetter(1)
        )

    def items(self, exhaustive: bool = True) -> Iterable[Tuple[float, int]]:
        '''Return a list of (probability, label) pairs representing this distribution.'''
        yield from (
            (p, self.value2label(v)) for p, v in self._items(exhaustive=exhaustive)
        )

    def kl_divergence(self, other: 'Integer') -> float:
        if type(other) is not type(self):
            raise TypeError(
                'Can only compute KL divergence between '
                'distributions of the same type, got %s' % type(other)
            )
        values = set(self.probabilities.keys()).union(other.probabilities.keys())
        result = 0
        for v in values:
            result += self._p(v) * abs(self._p(v) - other._p(v))
        return result

    def number_of_parameters(self) -> int:
        return len(self._params)

    def moment(self, order: int = 1, center: float = 0) -> float:
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param center: The constant to subtract in the basis of the exponent
        """
        result = 0
        for value, probability in self.probabilities.items():
            result += pow(self.value2label(value) - center, order) * probability
        return result

    @staticmethod
    def jaccard_similarity(
            d1: 'Integer',
            d2: 'Integer'
    ) -> float:
        values = set(d1.probabilities.keys()).union(d2.probabilities.keys())
        intersect = sum([min(d1._p(v), d2._p(v)) for v in values])
        union = sum([max(d1._p(v), d2._p(v)) for v in values])
        return intersect/union

    def plot(
            self,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            pdf: bool = False,
            view: bool = False,
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False
    ):
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the name of the variable this distribution represents
        :param fname:       the name of the file to be stored
        :param directory:   the directory to store the generated plot files
        :param pdf:         whether to store files as PDF. If false, a png is generated by default
        :param view:        whether to display generated plots, default False (only stores files)
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :param max_values:  maximum number of values to plot
        :param alphabet:    whether the bars are sorted in alphabetical order of the variable names. If False, the bars
                            are sorted by probability (descending); default is False
        :return:            None
        '''
        # Only save figures, do not show
        if not view:
            plt.ioff()

        # generate data
        max_values = min(
            ifnone(max_values, len(self.probabilities)),
            len(self.probabilities)
        )

        # prepare prob-label pairs containing only the first `max_values` highest probability tuples
        # pairs = sorted(
        #     [
        #         self.sorted(exhaustive=False)
        #     ],
        #     key=itemgetter(0),
        #     reverse=True
        # )[:max_values]
        #
        # if alphabet:
        #     # re-sort remaining values alphabetically
        #     pairs = sorted(pairs, key=itemgetter(1))
        if alphabet:
            data = list(self.items())[:max_values]
        else:
            data = list(self.sorted())[:max_values]
        labels = project(data, 0)
        probs = project(data, 1)
        vals = [re.escape(str(x)) for x in labels]

        x = np.arange(max_values)  # the label locations
        # width = .35  # the width of the bars
        err = [.015] * max_values

        fig, ax = plt.subplots()
        ax.set_title(f'{title or f"Distribution of {self._cl}"}')
        if horizontal:
            ax.barh(x, probs, xerr=err, color='cornflowerblue', label='P', align='center')
            ax.set_xlabel('P')
            ax.set_yticks(x)
            ax.set_yticklabels(vals)
            ax.invert_yaxis()
            ax.set_xlim(left=0., right=1.)

            for p in ax.patches:
                h = p.get_width() - .09 if p.get_width() >= .9 else p.get_width() + .03
                plt.text(h, p.get_y() + p.get_height() / 2,
                         f'{p.get_width():.2f}',
                         fontsize=10, color='black', verticalalignment='center')
        else:
            ax.bar(x, probs, yerr=err, color='cornflowerblue', label='P')
            ax.set_ylabel('P')
            ax.set_xticks(x)
            ax.set_xticklabels(vals)
            ax.set_ylim(bottom=0., top=1.)

            # print precise value labels on bars
            for p in ax.patches:
                h = p.get_height() - .09 if p.get_height() >= .9 else p.get_height() + .03
                plt.text(p.get_x() + p.get_width() / 2, h,
                         f'{p.get_height():.2f}',
                         rotation=90, fontsize=10, color='black', horizontalalignment='center')

        fig.tight_layout()

        save_plot(fig, directory, fname or self.__class__.__name__, fmt='pdf' if pdf else 'svg')

        if view:
            plt.show()



# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming,PyTypeChecker
def IntegerType(name: str, lmin: Optional[int] = ..., lmax: Optional[int] = ...) -> Type[Integer]:
    lmin = ifnone(lmin, ...)
    lmax = ifnone(lmax, ...)
    if (..., ...) != (lmin, lmax) and lmin > lmax:
        raise ValueError(
            'Min label is greater tham max value: %s > %s' % (lmin, lmax)
        )
    t: Type[Integer] = type(name, (Integer,), {})
    t.labels = IntegerValueMap(lmin, lmax)
    t.values = IntegerLabelMap(lmin, lmax)
    return t
