from collections import OrderedDict
from typing import Dict, Any

import numpy as np
from dnutils import ifnot
from numpy import isnan

try:
    from jpt.base.intervals import __module__
except ModuleNotFoundError:
    import pyximport
    pyximport.install()
finally:
    from jpt.base.intervals import R


class DataScaler:
    '''
    A numeric data transformation that represents data points in form of a translation
    by their mean and a scaling by their variance. After the transformation, the transformed
    input data have zero mean and unit variance.
    '''

    def __init__(self, data=None):
        self.mean = None
        self.scale = None
        if data is not None:
            self.fit(data.reshape(-1, 1))

    def fit(self, data):
        self.mean = np.mean(data)
        scale = np.std(data)
        self.scale = ifnot(1 if np.isnan(scale) else scale, 1)

    def inverse_transform(self, x, make_copy=True):
        if type(x) is np.ndarray:
            target = np.array(x) if make_copy else x
            target *= self.scale
            target += self.mean
            return target
        return x * self.scale + self.mean

    def transform(self, x, make_copy=True):
        if type(x) is np.ndarray:
            target = np.array(x) if make_copy else x
            target -= self.mean
            target /= self.scale
            return target
        return (x - self.mean) / self.scale

    def __getitem__(self, x):
        if x in (-np.inf, np.inf):
            return x
        return self.inverse_transform(x)

    def to_json(self):
        return {'mean_': [self.mean],
                'scale_': [self.scale]}

    def __eq__(self, other):
        return self.mean == other.mean and self.scale == other.scale

    @staticmethod
    def from_json(data: Dict[str, Any]):
        scaler = DataScaler()
        scaler.mean = data['mean_'][0]
        scaler.scale = data['scale_'][0]
        return scaler

    def __repr__(self):
        return '<DataScaler mean=%s, scale=%s>' % (self.mean, self.scale)


class Identity:
    '''
    Simple identity mapping that mimics the __getitem__ protocol of dicts.
    '''

    def __getitem__(self, item):
        return item

    __call__ = __getitem__

    def transformer(self):
        return lambda a: self[a]

    def __eq__(self, o):
        return type(o) is Identity

    def __hash__(self):
        return hash(Identity)


class DataScalerProxy:

    def __init__(self, datascaler, inverse=False):
        self.datascaler = datascaler
        self.inverse = inverse
        self.mean = self.datascaler.mean
        self.scale = self.datascaler.scale

    def __hash__(self):
        return hash((self.inverse, self.mean, self.scale))

    def __call__(self, arg):
        return self[arg]

    def __eq__(self, o):
        return (isinstance(o, DataScalerProxy) and
                self.datascaler == o.datascaler and
                self.inverse == o.inverse)

    def transformer(self):
        return lambda a: self[a]

    def __getitem__(self, item):
        if item in (-np.inf, np.inf):
            return item
        if item is None or isnan(item):
            return 0
        return (self.datascaler.transform
                if not self.inverse
                else self.datascaler.inverse_transform)(item)

    def transform(self, x):
        return self.datascaler.transform(x)

    @staticmethod
    def keys():
        return R

    @staticmethod
    def values():
        return R

    def __repr__(self):
        return '<DataScalerProxy, mean=%f, scale=%s>' % (self.datascaler.mean, self.datascaler.scale)


class HashableOrderedDict(OrderedDict):
    '''
    Ordered dict that can be hashed.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, arg):
        return self[arg]

    def __hash__(self):
        return hash((
            HashableOrderedDict,
            tuple(self.items())
        ))


class OrderedDictProxy:
    '''
    This is a proxy class that mimics the interface of a regular dict
    without inheriting from dict.
    '''

    def __init__(self, *args, **kwargs):
        self._dict = HashableOrderedDict(*args, **kwargs)
        self.values = self._dict.values
        self.keys = self._dict.keys

    def get(self, key, default):
        return self._dict.get(key, default)

    def __repr__(self):
        return '<OrderedDictProxy #%d values=[%s]>' % (len(self), ';'.join(map(repr, self.keys())))

    def _mapvalue(self, key: Any) -> int or float:
        try:
            return self._dict[key]
        except KeyError:
            raise ValueError(
                f'Value {key} out of domain (must be in {set(self._dict)})'
            )

    def transformer(self):
        return self._mapvalue

    def __getitem__(self, arg):
        return self._dict[arg]

    def __iter__(self):
        return iter(self._dict)

    def __call__(self, arg):
        return self._dict[arg]

    def __hash__(self):
        return hash((
            OrderedDictProxy,
            self._dict
        ))

    def __len__(self):
        return len(self._dict)

    def __eq__(self, other):
        if not isinstance(other, OrderedDictProxy):
            return False
        for self_, other_ in zip(self._dict.items(), other._dict.items()):
            if self_[0] != other_[0] or self_[1] != other_[1]:
                return False
        return True
