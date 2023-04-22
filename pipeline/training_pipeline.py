from zenml.pipelines import pipeline
from steps import *

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

# Import the step and parameters class
from zenml.integrations.bentoml.steps import BentoMLBuilderParameters, bento_builder_step
from zenml.integrations.bentoml.steps import BentoMLDeployerParameters, bentoml_model_deployer_step
# The name we gave to our deployed model


# retail_runner = bentoml.sklearn.get("retail:latest").to_runner()

# svc = bentoml.Service("retail", runners=[retail_runner])

# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def predict(input_series: np.ndarray) -> np.ndarray:
#     result = retail_runner.predict.run(input_series)
#     return result

MODEL_NAME = "retail"

# Call the step with the parameters
bento_builder = bento_builder_step(
    params=BentoMLBuilderParameters(
        model_name=MODEL_NAME,          # Name of the model
        model_type="sklearn",           # Type of the model (pytorch, tensorflow, sklearn, xgboost..)
        service="service.py:svc",       # Path to the service file within zenml repo
        labels={                        # Labels to be added to the bento bundle
            "framework": "sklearn",
            "dataset": "retail",
            "model_version": "0.21.1",
        },
       exclude=["data"],                # Exclude files from the bento bundle
    )
)

bentoml_model_deployer = bentoml_model_deployer_step(
    params=BentoMLDeployerParameters(
        model_name=MODEL_NAME,          # Name of the model
        port=3001,                      # Port to be used by the http server
        production=False,               # Deploy the model in production mode
    )
)


@pipeline(enable_cache=False)
def training_pipeline(
    data_loader,
    get_label, 
    transform_data, 
    split_data,
    train_model, 
    evaluate, 
    deployment_trigger,
    save_model,
    bento_builder,
    deployer
    ):

    raw_data = data_loader()
    raw_data, label = get_label(raw_data)
    transformed_data, transformer = transform_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(transformed_data, label)
    model = train_model(X_train, y_train)
    metric = evaluate(model, X_test, y_test)
    save_model(model, transformer)
    deployment_decision = deployment_trigger(metric)  # new
    print(deployment_decision)  # new
    bento = bento_builder(model=model)
    deployer(deploy_decision=deployment_decision, bento=bento)


training_pipeline_instance = training_pipeline(
    data_loader=data_loader(),
    get_label=get_label(), 
    transform_data=transform_data(), 
    split_data=split_data(),
    train_model=train_model(), 
    evaluate=evaluate(), 
    deployment_trigger=deployment_trigger(),
    save_model=save_model(),
    bento_builder=bento_builder,
    deployer=bentoml_model_deployer
    
)

training_pipeline_instance.run()