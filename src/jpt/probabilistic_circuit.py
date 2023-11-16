import math
from collections import deque
from typing import Iterable, Optional, Tuple, Union, Any, List

import numpy as np
import pandas as pd
import portion
from probabilistic_model.probabilistic_circuit.units import DeterministicSumUnit, DecomposableProductUnit
from probabilistic_model.probabilistic_model import ProbabilisticModel
from random_events.events import VariableMap
from random_events.variables import Variable, Continuous as REContinuous, Integer as REInteger, Symbolic

try:
    from .learning.impurity import Impurity
except ModuleNotFoundError:
    import pyximport

    pyximport.install()
finally:
    from .learning.impurity import Impurity


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
        if datatype in [float]:

            minimal_distance_between_values = np.diff(np.sort(data[column].unique())).min()
            mean = data[column].mean()
            std = data[column].std()

            if scale_continuous_types:
                variable = ScaledContinuous(column, mean, std, minimal_distance_between_values)
            else:
                variable = Continuous(column, mean, std, minimal_distance_between_values)

        # handle discrete variables
        elif datatype in [object, int]:

            unique_values = data[column].unique()

            if datatype == int:
                mean = data[column].mean()
                std = data[column].std()
                variable = Integer(column, unique_values, mean, std)
            elif datatype == object:
                variable = Symbolic(column, unique_values)
            else:
                raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        result.append(variable)

    return result


class Integer(REInteger):
    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    def __init__(self, name: str, domain: Iterable, mean, std):
        super().__init__(name, domain)
        self.mean = mean
        self.std = std


class Continuous(REContinuous):
    """
    Base class for continuous variables in JPTs. This class does not standardize the data,
    but needs to know mean and std anyway.
    """

    minimal_distance: float
    """
    The minimal distance between two values of the variable.
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
        super().__init__(name)
        self.mean = mean
        self.std = std
        self.minimal_distance = minimal_distance


class ScaledContinuous(Continuous):
    """
    A continuous variable that is standardized.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.):
        super().__init__(name, mean, std, minimal_distance)

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

    indices: Optional[np.ndarray] = None
    impurity: Optional[Impurity] = None
    c45queue: deque = deque()

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

    @property
    def numeric_variables(self):
        return [variable for variable in self.variables if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_targets(self):
        return [variable for variable in self.targets if isinstance(variable, (Continuous, Integer))]

    @property
    def numeric_features(self):
        return [variable for variable in self.features if isinstance(variable, (Continuous, Integer))]

    @property
    def symbolic_variables(self):
        return [variable for variable in self.variables if isinstance(variable, Symbolic)]

    @property
    def symbolic_targets(self):
        return [variable for variable in self.targets if isinstance(variable, Symbolic)]

    @property
    def symbolic_features(self):
        return [variable for variable in self.features if isinstance(variable, Symbolic)]

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

        self.impurity = self.construct_impurity()
        self.indices = np.arange(data.shape[0], dtype=np.int64)

        self.c45queue.append((data, 0, len(data)))

        while self.c45queue:
            self.c45(*self.c45queue.popleft())

        return self

    def c45(self, data: np.ndarray, start: int, end: int) -> Optional[Union[DecisionNode, DecomposableProductUnit]]:
        """
        Construct a DecisionNode or DecomposableProductNode from the data.

        :param data: The data to calculate the impurity from.
        :param start: Starting index in the data.
        :param end: Ending index in the data.
        :return: The constructed decision tree node
        """
        number_of_samples = end - start

        # calculate the best gain possible
        max_gain = self.impurity.compute_best_split(start, end)

        if max_gain < self.min_impurity_improvement:
            # create decomposable product node
            return self.create_leaf_node()

        return None

    def create_leaf_node(self) -> DecomposableProductUnit:
        ...

    def construct_impurity(self) -> Impurity:
        min_samples_leaf = self.min_samples_leaf

        numeric_vars = (
            np.array([index for index, variable in enumerate(self.variables) if variable in self.numeric_targets]))
        symbolic_vars = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_targets])

        invert_impurity = np.array([0] * len(self.symbolic_targets))

        n_sym_vars_total = len(self.symbolic_variables)
        n_num_vars_total = len(self.numeric_variables)

        numeric_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.numeric_features])
        symbolic_features = np.array(
            [index for index, variable in enumerate(self.variables) if variable in self.symbolic_features])

        symbols = np.array([len(variable.domain) for variable in self.symbolic_variables])
        max_variances = np.array([variable.std ** 2 for variable in self.numeric_variables])

        dependency_indices = dict()

        for variable, dep_vars in self.dependencies.items():
            # get the index version of the dependent variables and store them
            idx_var = self.variables.index(variable)
            idc_dep = [self.variables.index(var) for var in dep_vars]
            dependency_indices[idx_var] = idc_dep

        return Impurity(min_samples_leaf, numeric_vars, symbolic_vars, invert_impurity, n_sym_vars_total,
                        n_num_vars_total, numeric_features, symbolic_features, symbols, max_variances,
                        dependency_indices)
