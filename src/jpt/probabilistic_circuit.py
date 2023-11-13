import math
from typing import Iterable, Optional, Tuple, Union, Any, List

import numpy as np
import pandas as pd
import portion
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.events import VariableMap
from random_events.variables import Variable, Continuous as REContinuous, Integer, Symbolic


def infer_variables_from_dataframe(data: pd.DataFrame, scale_continuous_types: bool = True) -> List[Variable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :param scale_continuous_types: Whether to scale numeric types.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):

        # handle continuous variables
        if datatype == float:

            minimal_distance_between_values = np.diff(np.sort(data[column].unique())).min()

            if scale_continuous_types:
                mean = data[column].mean()
                std = data[column].std()
                variable = ScaledContinuous(column, mean, std, minimal_distance_between_values)
            else:
                variable = Continuous(column, minimal_distance_between_values)

        # handle discrete variables
        elif datatype in [int, object]:

            unique_values = data[column].unique()

            if datatype == int:
                variable = Integer(column, unique_values)
            elif datatype == object:
                variable = Symbolic(column, unique_values)
            else:
                raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        result.append(variable)

    return result


class Continuous(REContinuous):

    minimal_distance: float
    """
    The minimal distance between two values of the variable.
    """

    def __init__(self, name: str, minimal_distance: float = 1.):
        super().__init__(name)
        self.minimal_distance = minimal_distance



class ScaledContinuous(Continuous):
    """
    A continuous variable that is standardized.
    """

    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.):
        super().__init__(name, minimal_distance)
        self.mean = mean
        self.std = std

    def encode(self, value: Any):
        return (value - self.mean) / self.std

    def decode(self, value: float) -> float:
        return value * self.std + self.mean

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.mean}, {self.std}, {self.minimal_distance})"


class Criterion:
    """
    A criterion that is used to decide which branch to take in a decision node.
    """

    variable: Variable
    value: Union[portion.Interval, Tuple]

    def __init__(self, variable: Variable, value: Union[portion.Interval, Tuple]):
        self.variable = variable
        self.value = value


class DecisionNode(DeterministicSumUnit):
    criterion: Criterion
    """
    The criterion that is used to decide which branch to take.
    """

    def __init__(self, variables: Iterable[Variable], weights: Iterable, criterion: Criterion):
        super().__init__(variables, weights)
        self.criterion = criterion


class JPT(ProbabilisticModel):
    circuit: Optional[DeterministicSumUnit] = None

    targets: Tuple[Variable]
    """
    The variables to optimize for.
    """

    features: Tuple[Variable]
    """
    The variables that are used to craft criteria.
    """

    _min_samples_leaf: Union[int, float]
    """
    The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted
    as fraction w. r. t. the number of samples total.
    """

    min_impurity_improvement: float
    """
    The minimum impurity improvement to create another sum node.
    """

    max_leaves: Union[int, float]
    """
    The maximum number of leaves.
    """

    max_depth: Union[int, float]
    """
    The maximum depth of the tree.
    """

    dependencies: VariableMap
    """
    The dependencies between the variables.
    """

    total_samples: int = 1
    """
    The total amount of samples that were used to fit the model.
    """

    def __init__(self, variables: Iterable[Variable], targets: Optional[Iterable[Variable]] = None,
                 features: Optional[Iterable[Variable]] = None, min_samples_leaf: Union[int, float] = 1,
                 min_impurity_improvement: float = 0.0, max_leaves: Union[int, float] = float("inf"),
                 max_depth: Union[int, float] = float("inf"), dependencies: Optional[VariableMap] = None, ):

        super().__init__(variables)
        self.set_targets_and_features(targets, features)
        self._min_samples_leaf = min_samples_leaf
        self.min_impurity_improvement = min_impurity_improvement
        self.max_leaves = max_leaves
        self.max_depth = max_depth

        if dependencies is None:
            self.dependencies = VariableMap({var: list(self.targets) for var in self.features})
        else:
            self.dependencies = dependencies

    def set_targets_and_features(self, targets: Optional[Iterable[Variable]],
                                 features: Optional[Iterable[Variable]]) -> None:
        """
        Set the targets and features of the model.
        If only one of them is provided, the other is set as the complement of the provided one.
        If none are provided, both of them are set as the variables of the model.
        If both are provided, they are taken as given.

        :param targets: The targets of the model.
        :param features: The features of the model.
        :return: None
        """
        # if targets are not specified
        if targets is None:

            # and features are not specified
            if features is None:
                self.targets = self.variables
                self.features = self.variables

            # and features are specified
            else:
                self.targets = tuple(sorted(set(self.variables) - set(features)))
                self.features = tuple(sorted(features))

        # if targets are specified
        else:
            # and features are not specified
            if features is None:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(self.variables) - set(targets)))

            # and features are specified
            else:
                self.targets = tuple(sorted(set(targets)))
                self.features = tuple(sorted(set(features)))

    @property
    def min_samples_leaf(self):
        """
        The minimum number of samples to create another sum node.
        """
        if self._min_samples_leaf < 1.:
            return math.ceil(self._min_samples_leaf * self.total_samples)
        else:
            return self._min_samples_leaf

    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data to be used in the model.

        :param data: The data to preprocess.
        :return: The preprocessed data.
        """

        result = np.zeros(data.shape)

        for variable_index, variable in enumerate(self.variables):
            column = data[variable.name]
            column = variable.encode_many(column)
            result[:, variable_index] = column

        return result

    def fit(self, data: pd.DataFrame) -> 'JPT':
        """
        Fit the model to the data.

        :param data: The data to fit the model to.
        :return: The fitted model.
        """

        data = self.preprocess_data(data)

        self.total_samples = len(data)

        return self
