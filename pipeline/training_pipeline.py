from zenml.integrations.mlflow.steps import (
    MLFlowDeployerParameters,
    mlflow_model_deployer_step
)

from zenml.pipelines import pipeline
from steps import *


@pipeline(enable_cache=False)
def training_pipeline(data_loader=data_loader(),
get_label=get_label(), transform_data=transform_data(), split_data=split_data(),
train_model=train_model(), evaluate=evaluate(), save_model=save_model(),
deployment_trigger=deployment_trigger(),
model_deployer=mlflow_model_deployer_step(MLFlowDeployerParameters(timeout=20))):

    raw_data = data_loader()
    raw_data, label = get_label(raw_data)
    transformed_data = transform_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(transformed_data, label)
    model = train_model(X_train, y_train)
    metric = evaluate(model, X_test, y_test)
    print(metric)
    save_model(model)
    deployment_decision = deployment_trigger(metric)  # new
    model_deployer(deployment_decision, model)  # new


training_pipeline_instance = training_pipeline(data_loader=data_loader(),
get_label=get_label(),
transform_data=transform_data(),
split_data=split_data(),train_model=train_model(),
evaluate=evaluate(), save_model=save_model(),
deployment_trigger=deployment_trigger(),
model_deployer=mlflow_model_deployer_step(MLFlowDeployerParameters(timeout=20)))

training_pipeline_instance.run()