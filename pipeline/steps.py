from evidently.model_profile.model_profile import Profile
import pandas as pd
import numpy as np
import mlflow
from sklearn.compose import make_column_selector
from zenml.steps import step, Output
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_squared_error
import pickle
from zenml.services import BaseService
from zenml.client import Client
from zenml.steps import step, Output
from zenml.integrations.evidently.steps import (
    EvidentlyProfileParameters,
    evidently_profile_step,
)
from zenml.integrations.evidently.visualizers import EvidentlyVisualizer
from zenml.post_execution import StepView



@step(experiment_tracker="mlflow_tracker")
def data_loader()-> Output(raw_data= pd.DataFrame):
    return  pd.read_csv("./../data/training_data.csv")

@step(experiment_tracker="mlflow_tracker")
def data_loader_inference()-> Output(raw_data= pd.DataFrame):
    data = pd.read_csv("./../data/test_data.csv")
    return  data.drop(['cost'], axis=1)

@step(experiment_tracker="mlflow_tracker")
def data_loader_validator()-> Output(raw_data= pd.DataFrame):
    data = pd.read_csv("./../data/training_data.csv")
    return  data.drop(['cost'], axis=1)


@step(experiment_tracker="mlflow_tracker")
def get_label(data: pd.DataFrame) -> Output(raw_data= pd.DataFrame, label= pd.Series):
    label = data['cost']
    return data.drop(['cost'], axis=1), label



@step(experiment_tracker="mlflow_tracker")
def transform_data(data: pd.DataFrame) -> Output(transformed_data= np.ndarray):
    scaler = StandardScaler()
    encoder = OneHotEncoder()
    
    transformer = make_column_transformer((scaler, make_column_selector(dtype_include=np.float64)),
        (encoder, make_column_selector(dtype_include=object)))

    transformed_data = transformer.fit_transform(data)

    if not isinstance(transformed_data, np.ndarray):
        return transformed_data.toarray()
    return transformed_data

"""
@step(experiment_tracker="mlflow_tracker")
def transform_predict_data(data: pd.DataFrame) -> Output(transformed_data= np.ndarray):

    transformed_data = transformer.transform(data).toarray()
    if not isinstance(transformed_data, np.ndarray):
        return transform_data.toarray()
    return transform_data"""

@step(experiment_tracker="mlflow_tracker")
def split_data(transformed_data: np.ndarray, label: pd.Series) -> Output(X_train= np.ndarray, X_test=np.ndarray, y_train=pd.Series, y_test=pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(transformed_data, label, test_size=0.30, random_state=1998)
    return X_train, X_test, y_train, y_test

@step(enable_cache=False, experiment_tracker="mlflow_tracker")
def train_model(X_train: np.ndarray, y_train: pd.Series) -> Output(model= RegressorMixin):
    mlflow.sklearn.autolog()
    model = GradientBoostingRegressor(n_estimators=500, max_depth=10)
    model.fit(X_train, y_train)
    
    return model

@step(experiment_tracker="mlflow_tracker")
def evaluate(model: RegressorMixin, X_test: np.ndarray, y_test: pd.Series) -> Output(metric= float):
    """ Calculate the mean squared error """
    
    y_pred = model.predict(X_test)
    metric = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("metric", np.sqrt(metric))
    return np.sqrt(metric)

@step(experiment_tracker="mlflow_tracker")
def predict(model: RegressorMixin, transformed_data: np.ndarray) -> Output(predictions= np.ndarray):
    return model.predict(transformed_data)

@step(experiment_tracker="mlflow_tracker")
def save_model(model: RegressorMixin) -> Output(filename= str):
    filename = './../artifact/model/finalized_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    return filename

@step(experiment_tracker="mlflow_tracker")
def load_model() -> Output(model= RegressorMixin):
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    return loaded_model


@step(enable_cache=False)
def prediction_service_loader() -> BaseService:
    """Load the model service of our train_evaluate_deploy_pipeline."""
    client = Client()
    model_deployer = client.active_stack.model_deployer
    services = model_deployer.find_model_server(
        pipeline_name="training_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=True,
    )
    service = services[0]
    print(service)
    return service

@step
def predictor(
    service: BaseService,
    data: np.ndarray,
) -> Output(predictions=list):
    """Run a inference request against a prediction service"""
    service.start(timeout=10)  # should be a NOP if already started
    print(service.prediction_url)
    prediction = service.predict(data)
    #prediction = prediction.argmax(axis=-1)
    #print(f"Prediction is: {[prediction.tolist()]}")
    return [prediction.tolist()]

@step(experiment_tracker="mlflow_tracker")
def deployment_trigger(metric: float) -> bool:
    """Only deploy if mse is below 1"""
    return metric < 1.5

drift_detector = evidently_profile_step(
    step_name="drift_detector",
    params=EvidentlyProfileParameters(
        profile_sections=[
            "datadrift",
        ],
        verbose_level=1,
    ),
)

@step
def visualize_results(drift_report: Profile) -> None:
    """pipeline = get_pipeline(pipeline="inference_pipeline")
    evidently_outputs = pipeline.runs[-1].get_step(step="drift_detector")"""
    EvidentlyVisualizer().visualize(drift_report)
