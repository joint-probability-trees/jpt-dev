from typing import Optional, Iterable, List

import mlflow
import mlflow.models
import mlflow.types

import jpt.variables
from .trees import JPT


class JPTWrapper(mlflow.pyfunc.PythonModel):
    """Wrapper class to load a JPT from a mlflow server instance."""
    model: Optional[JPT]

    def load_context(self, context):
        """
        This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is
        constructed.

        :param context: MLflow context where the model artifact is stored.
        """

        self.model = JPT.load(context.artifacts["jpt_model_path"])

    def predict(self, context, model_input):
        """Predict the likelihood of samples.

        :param context: MLflow context where the model artifact is stored.
        :param model_input: the input data to fit into the model.

        :return: the loaded model artifact.
        """
        return self.model.likelihood(model_input)


class Schema(mlflow.types.Schema):
    """
    Schema class that create a mlflow schema from a jpt variable definition.
    In the mlflow schema only the inputs are set.
    """
    def __init__(self, variables: Iterable[jpt.variables.Variable]):
        """Create a new schema.
        :param variables: An iterable of jpt variables.
        """

        inputs: List[mlflow.types.ColSpec] = []

        for variable in variables:

            type_ = None

            if variable.numeric:
                type_ = "float"
            elif variable.symbolic:
                type_ = "string"
            elif variable.integer:
                type_ = "int"
            else:
                ValueError(f"mlflow interface does not support variable of type {type(variable)} yet.")

            spec = mlflow.types.ColSpec(type_, variable.name)
            inputs.append(spec)

        super().__init__(inputs)

