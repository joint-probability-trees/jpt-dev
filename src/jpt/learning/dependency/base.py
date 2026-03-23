"""Abstract base class for dependency discovery.

Defines the contract that all dependency discovery
strategies must satisfy: callable with a fixed
signature and JSON-serializable for model
persistence.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from jpt.variables import Variable


# ------------------------------------------------------


class DependencyDiscovery(ABC):
    """Abstract base class for dependency discovery.

    Subclasses implement a strategy for determining
    which target variables depend on which features,
    given the training data. The result is used by
    the JPT learning algorithm to restrict impurity
    computation to dependent variable pairs.

    Implementations must be serializable via
    ``to_json()``/``from_json()`` so that the
    discovery strategy is preserved when the JPT
    model is saved and loaded.
    """

    _REGISTRY: dict[
        str, type[DependencyDiscovery]
    ] = {}

    def __init_subclass__(
            cls,
            **kwargs: Any
    ) -> None:
        """Auto-register subclasses for
        deserialization.
        """
        super().__init_subclass__(**kwargs)
        DependencyDiscovery._REGISTRY[
            cls.__name__
        ] = cls

    @abstractmethod
    def __call__(
            self,
            data: np.ndarray,
            features: list[Variable],
            targets: list[Variable],
            variables: list[Variable]
    ) -> dict[Variable, list[Variable]]:
        """Discover dependencies from data.

        :param data:      preprocessed data array
                          (n_samples x n_variables)
        :param features:  list of feature Variables
        :param targets:   list of target Variables
        :param variables: list of all Variables
                          (defines column order)
        :returns:         dict mapping each feature
                          Variable to a list of
                          dependent target Variables
        """
        ...

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """Serialize the strategy configuration.

        Must include a ``'type'`` key with the class
        name for deserialization dispatch.

        :returns: JSON-serializable dict
        """
        ...

    @classmethod
    def from_json(
            cls,
            data: dict[str, Any]
    ) -> DependencyDiscovery | None:
        """Deserialize a strategy from JSON.

        Dispatches to the appropriate subclass based
        on the ``'type'`` key.

        :param data: dict from ``to_json()``
        :returns:    DependencyDiscovery instance
        """
        if data is None:
            return None
        type_name: str = data['type']
        subcls = cls._REGISTRY.get(type_name)
        if subcls is None:
            raise ValueError(
                'Unknown DependencyDiscovery '
                f'type: {type_name!r}'
            )
        return subcls.from_json(data)
