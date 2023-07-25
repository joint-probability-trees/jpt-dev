import unittest
import os
import numpy as np

import jpt
import pandas as pd

skiptest = False
try:
    import mlflow
    import requests
    from jpt.mlflow_wrapper import JPTWrapper, Schema
except ModuleNotFoundError as e:
    skiptest = True, 'mlflow or requests not installed. Skipping tests.'
else:
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    if mlflow_uri is None:
        mlflow_uri = "http://127.0.0.1:5000"

    try:
        response = requests.get(mlflow_uri, timeout=2)
    except requests.exceptions.ConnectionError:
        response = None
    skiptest = (
        response is None or response.status_code != requests.codes.ok,
        'mlflow server is not available.'
    )


@unittest.skipIf(*skiptest)
class MLFlowTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = pd.DataFrame(data=np.random.normal(size=(100, 1)), columns=["X"])

    def test_experiment_tracking_and_loading(self):
        """Test the entire mlflow model lifecycle."""
        run = mlflow.start_run(run_name="Unittest for jpt.")
        model = jpt.JPT(jpt.infer_from_dataframe(self.data))
        model.fit(self.data)

        mlflow.log_params(model.get_hyperparameters_dict())
        average_log_likelihood = np.average(np.log(model.likelihood(self.data)))
        mlflow.log_metric("average_log_likelihood", average_log_likelihood)
        mlflow.log_metric("number_of_parameters", model.number_of_parameters())
        model_path = os.path.join("/tmp", "test_mlflow.jpt")
        model.save(model_path)
        mlflow.pyfunc.log_model(
            artifact_path="test_mlflow",
            python_model=JPTWrapper(),
            code_path=[__file__],
            artifacts={"jpt_model_path": model_path},
            signature=mlflow.models.ModelSignature(Schema(model.variables))
        )
        mlflow.end_run()
        loaded_model = mlflow.pyfunc.load_model(model_uri=run.info.artifact_uri + "/test_mlflow")
        loaded_model = loaded_model.unwrap_python_model().model
        self.assertTrue(isinstance(loaded_model, jpt.JPT))


if __name__ == '__main__':
    unittest.main()
