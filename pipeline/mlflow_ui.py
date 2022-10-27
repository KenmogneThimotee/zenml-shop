from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri


def open_mlflow_ui(port=4997):
    print(get_tracking_uri())

open_mlflow_ui()