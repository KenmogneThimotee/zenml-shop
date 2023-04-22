
import bentoml
from bentoml.io import NumpyNdarray
import numpy as np


retail_runner = bentoml.sklearn.get("retail:latest").to_runner()

svc = bentoml.Service("retail", runners=[retail_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_series: np.ndarray) -> np.ndarray:
    result = retail_runner.predict.run(input_series)
    return result
