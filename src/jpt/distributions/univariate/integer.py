'''© Copyright 2021, Mareike Picklum, Daniel Nyga.'''
import numbers
import os
from itertools import tee
from operator import itemgetter
from types import FunctionType
from typing import Optional, Type, Dict, Any, Union, Set, Iterable, Tuple

import numpy as np
from dnutils import edict, ifnone, project, first
from plotly.graph_objs import Figure
import plotly.graph_objects as go

from . import Distribution
from ..utils import OrderedDictProxy
from ...base.errors import Unsatisfiability
from ...base.sampling import wsample, wchoice
from ...base.utils import setstr, normalized, classproperty, save_plot

try:
    from ...base.functions import __module__
    from ...base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from ...base.functions import PiecewiseFunction, Undefined, ConstantFunction
    from ...base.intervals import ContinuousSet


# noinspection DuplicatedCode
class Integer(Distribution):

    lmin = None
    lmax = None
    vmin = None
    vmax = None
    values = None
    labels = None

    OPEN_DOMAIN = 'open_domain'
    AUTO_DOMAIN = 'auto_domain'

    SETTINGS = edict(Distribution.SETTINGS) + {
        OPEN_DOMAIN: False,
        AUTO_DOMAIN: False
    }

    def __init__(self, **settings):
        super().__init__(**settings)
        if not issubclass(type(self), Integer) or type(self) is Integer:
            raise Exception(
                f'Instantiation of abstract class {type(self)} is not allowed!'
            )
        self._params: Optional[np.ndarray] = None
        self.to_json: FunctionType = self.inst_to_json

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
            lmin=self.lmin + other.lmin,
            lmax=self.lmax + other.lmax
        )
        res = res_t()
        dist = []
        for z in res.labels.values():
            p = 0
            for k in self.labels.values():
                p += self.p(k) * (other.p(z-k) if other.lmin <= z-k <= other.lmax else 0)
            dist.append(p)
        res.set(dist)
        return res

    @classmethod
    def equiv(cls, other: Type[Distribution]) -> bool:
        if not issubclass(other, Integer):
            return False
        return all((
            cls.__name__ == other.__name__,
            cls.labels == other.labels,
            cls.values == other.values,
            cls.lmin == other.lmin,
            cls.lmax == other.lmax,
            cls.vmin == other.vmin,
            cls.vmax == other.vmax
        ))

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        return {
            'type': 'integer',
            'class': cls.__qualname__,
            'labels': list(cls.labels.values()),
            'vmin': int(cls.vmin),
            'vmax': int(cls.vmax),
            'lmin': int(cls.lmin),
            'lmax': int(cls.lmax)
        }

    to_json = type_to_json

    def inst_to_json(self) -> Dict[str, Any]:
        return {
            'class': type(self).__qualname__,
            'params': list(self._params),
            'settings': self.settings
        }

    @staticmethod
    def type_from_json(data):
        return IntegerType(data['class'], data['lmin'], data['lmax'])

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'Integer':
        return cls(**data['settings']).set(data['params'])

    def copy(self):
        result = type(self)(**self.settings)
        result._params = np.array(self._params)
        return result

    @property
    def probabilities(self):
        return self._params

    @classproperty
    def n_values(cls) -> int:
        return cls.lmax - cls.lmin + 1

    # noinspection DuplicatedCode
    @classmethod
    def value2label(
            cls,
            value: Union[int, Iterable[int]]
    ) -> Union[int, Iterable[int]]:
        if not isinstance(value, Iterable):
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
        if not isinstance(value, Iterable):
            return cls.labels[value]
        # noinspection PyArgumentList
        return type(value)([cls.labels[v] for v in value_])

    # noinspection DuplicatedCode
    @classmethod
    def label2value(
            cls,
            label: Union[int, Iterable[int]]
    ) -> Union[int, Iterable[int]]:
        if not isinstance(label, Iterable):
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
        if not isinstance(label, Iterable):
            return cls.values[label]
        # noinspection PyArgumentList
        return type(label)([cls.values[l] for l in label_])

    def _sample(self, n: int) -> Iterable[int]:
        return wsample(
            list(self.values.values()),
            weights=self.probabilities,
            k=n
        )

    def _sample_one(self) -> int:
        return wchoice(
            list(self.values.values()),
            weights=self.probabilities
        )

    def sample(self, n: int) -> Iterable[int]:
        return [self.value2label(v) for v in self._sample(n)]

    def sample_one(self) -> int:
        return self.value2label(self._sample_one())

    def p(self, labels: Union[int, Iterable[int]]) -> float:
        return self._p(self.label2value(labels))

    def _p(self, values: Union[int, Iterable[int]]) -> float:
        if not isinstance(values, Iterable):
            values = {values}
        elif not isinstance(values, set):
            values = set(values)
        i1, i2 = tee(values, 2)
        if not all(isinstance(v, numbers.Integral) and self.vmin <= v <= self.vmax for v in i1):
            raise ValueError(
                'Arguments must be in %s' % setstr(self.values.values(), limit=5)
            )

        return sum(self._params[v] for v in i2)

    def expectation(self) -> float:
        return sum(p * v for p, v in zip(self.probabilities, self.values))

    def _expectation(self) -> float:
        return sum(p * v for p, v in zip(self.probabilities, self.labels))

    def variance(self) -> float:
        e = self.expectation()
        return sum((l - e) ** 2 * p for l, p in zip(self.labels.values(), self.probabilities))

    def _variance(self) -> float:
        e = self._expectation()
        return sum((v - e) ** 2 * p for v, p in zip(self.values.values(), self.probabilities))

    def mpe(self) -> (float, Set[int]):
        p_max, lbls = self._mpe()
        return p_max, self.value2label(lbls)

    def _mpe(self) -> (float, Set[int]):
        p_max = max(self.probabilities)
        return (
            p_max,
            {
                l for l, p in zip(
                    self.values.values(),
                    self.probabilities
                ) if p == p_max
             }
        )

    mode = mpe
    _mode = _mpe

    def crop(self, restriction: Union[Iterable[int], int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        return self._crop([self.label2value(l) for l in restriction])

    def _crop(self, restriction: Union[Iterable[int], int]) -> 'Distribution':
        if isinstance(restriction, numbers.Integral):
            restriction = {restriction}
        result = self.copy()
        try:
            params = normalized([
                p if v in restriction else 0
                for p, v in zip(self.probabilities, self.values.values())
            ])
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

        params = np.zeros(distributions[0].n_values)

        for d, w in zip(distributions, weights):
            params += d.probabilities * w

        if abs(sum(params)) < 1e-10:
            raise Unsatisfiability('Sum of weights must not be zero.')

        return type(distributions[0])().set(params)

    def update(
            self,
            dist: 'Integer',
            weight: int
    ) -> 'Integer':
        if not 0 <= weight <= 1:
            raise ValueError('Weight must be in [0, 1]')

        if self._params is None:
            self._params = np.zeros(self.n_values)
        self._params *= 1 - weight
        self._params += dist._params * weight

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
        self._params = np.zeros(
            shape=self.n_values,
            dtype=np.float64
        )
        n_samples = ifnone(rows, len(data), len)
        col = ifnone(col, 0)
        for row in ifnone(rows, range(data.shape[0])):
            self._params[int(data[row, col])] += 1 / n_samples

        return self

    def set(self, params: Iterable[float]) -> 'Integer':
        if len(self.values) != len(params):
            raise ValueError(
                'Number of values and probabilities must coincide.'
            )
        self._params = np.array(params, dtype=np.float64)
        return self

    def __eq__(self, other) -> bool:
        return (
            type(self).equiv(type(other)) and
            (self.probabilities == other.probabilities).all()
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

    def sorted(self) -> Iterable[Tuple[float, int]]:
        return sorted(
            [(p, l) for p, l in zip(self._params, self.labels.values())],
            key=itemgetter(0),
            reverse=True
        )

    def _items(self) -> Iterable[Tuple[float, int]]:
        '''Return a list of (probability, value) pairs representing this distribution.'''
        yield from ((p, v) for p, v in zip(self._params, self.values.values()))

    def items(self):
        '''Return a list of (probability, label) pairs representing this distribution.'''
        yield from ((p, self.value2label(v)) for p, v in self._items())

    def kl_divergence(self, other: 'Integer') -> float:
        if type(other) is not type(self):
            raise TypeError(
                'Can only compute KL divergence between '
                'distributions of the same type, got %s' % type(other)
            )
        result = 0
        for v in range(self.n_values):
            result += self._params[v] * abs(self._params[v] - other._params[v])
        return result

    def number_of_parameters(self) -> int:
        return self._params.shape[0]

    def moment(self, order: int = 1, center: float = 0) -> float:
        r"""Calculate the central moment of the r-th order almost everywhere.

        .. math:: \int (x-c)^{r} p(x)

        :param order: The order of the moment to calculate
        :param center: The constant to subtract in the basis of the exponent
        """
        result = 0
        for value, probability in zip(self.labels.values(), self._params):
            result += pow(value - center, order) * probability
        return result

    @staticmethod
    def jaccard_similarity(
            d1: 'Integer',
            d2: 'Integer'
    ) -> float:
        intersect = sum([min(p1, p2) for p1, p2 in zip(d1.probabilities, d2.probabilities)])
        union = sum([max(p1, p2) for p1, p2 in zip(d1.probabilities, d2.probabilities)])
        return intersect/union

    def plot(
            self,
            title: str = None,
            fname: str = None,
            directory: str = '/tmp',
            view: bool = False,
            horizontal: bool = False,
            max_values: int = None,
            alphabet: bool = False,
            **kwargs
    ) -> Figure:
        '''Generates a ``horizontal`` (if set) otherwise `vertical` bar plot representing the variable's distribution.

        :param title:       the name of the variable this distribution represents
        :param fname:       the name of the file to be stored. Available file formats: png, svg, jpeg, webp, html
        :param directory:   the directory to store the generated plot files
        :param view:        whether to display generated plots, default False (only stores files)
        :param horizontal:  whether to plot the bars horizontally, default is False, i.e. vertical bars
        :param max_values:  maximum number of values to plot
        :param alphabet:    whether the bars are sorted in alphabetical order of the variable names. If False, the bars
                            are sorted by probability (descending); default is False
        :return:            plotly.graph_objs.Figure
        '''

        # generate data
        max_values = min(ifnone(max_values, len(self.labels)), len(self.labels))

        # prepare prob-label pairs containing only the first `max_values` highest probability tuples
        pairs = sorted(
            [
                (self._params[idx], lbl) for idx, lbl in enumerate(self.labels.values())
            ],
            key=lambda x: x[0],
            reverse=True
        )[:max_values]

        if alphabet:
            # re-sort remaining values alphabetically
            pairs = sorted(pairs, key=lambda x: x[1])

        probs = project(pairs, 0)
        labels = project(pairs, 1)

        mainfig = go.Figure(
            [
                go.Bar(
                    x=probs if horizontal else labels,  # labels if horizontal else probs,
                    y=labels if horizontal else probs, # probs if horizontal else x,
                    name="Integer Distribution",
                    text=probs,
                    orientation='h' if horizontal else 'v',
                    marker=dict(
                        color='rgba(15,21,110,0.6)',
                        line=dict(color='rgba(15,21,110,1.0)', width=3)
                    )
                )
            ]
        )
        mainfig.update_layout(
            xaxis=dict(
                title='p' if horizontal else 'value',
                range=[0, 1] if horizontal else None
            ),
            yaxis=dict(
                title='value' if horizontal else 'p',
                range=None if horizontal else [0, 1]
            ),
            title=f'{title or f"Distribution of {self._cl}"}'
        )

        if fname is not None:

            if not os.path.exists(directory):
                os.makedirs(directory)

            fpath = os.path.join(directory, fname or self.__class__.__name__)

            if fname.endswith('html'):
                mainfig.write_html(
                    fpath,
                    include_plotlyjs="cdn"
                )
            else:
                mainfig.write_image(
                    fpath,
                    scale=1
                )

        if view:
            mainfig.show(
                config=dict(
                    displaylogo=False,
                    toImageButtonOptions=dict(
                        format='svg',  # one of png, svg, jpeg, webp
                        filename=fname or self.__class__.__name__,
                        scale=1
                   )
                )
            )

        return mainfig

# ----------------------------------------------------------------------------------------------------------------------

# noinspection PyPep8Naming
def IntegerType(name: str, lmin: int, lmax: int) -> Type[Integer]:
    if lmin > lmax:
        raise ValueError('Min label is greater tham max value: %s > %s' % (lmin, lmax))
    t = type(name, (Integer,), {})
    t.values = OrderedDictProxy([(l, v) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.labels = OrderedDictProxy([(v, l) for l, v in zip(range(lmin, lmax + 1), range(lmax - lmin + 1))])
    t.lmin = lmin
    t.lmax = lmax
    t.vmin = 0
    t.vmax = lmax - lmin
    return t