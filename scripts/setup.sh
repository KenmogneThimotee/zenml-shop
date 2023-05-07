# bin/bash

echo Initializing project ...

zenml init

echo Creating the stack ...
zenml stack register mlops -a default -o default

echo "Registering stack component ..."

# -- Data validator  --
echo "-- Data validator  --"

zenml integration install evidently -y
# Register the Evidently data validator
zenml data-validator register evidently_data_validator --flavor=evidently
# Register and set a stack with the new data validator
zenml stack update mlops -dv evidently_data_validator



# -- Experiment tracker --
echo "-- Experiment tracker --"

zenml integration install mlflow -y
# Register the MLflow experiment tracker
zenml experiment-tracker register mlflow_experiment_tracker --flavor=mlflow
# Register and set a stack with the new experiment tracker
zenml stack update mlops -e mlflow_experiment_tracker 



# -- Model Deployer --
echo "-- Model Deployer --"
zenml integration install bentoml -y
zenml model-deployer register bentoml_deployer --flavor=bentoml
zenml stack update mlops -d bentoml_deployer 


# -- Model registery --
echo  "-- Model registery --"

zenml model-registry register mlflow_model_registry --flavor=mlflow
# Register and set a stack with the new model registry as the active stack
zenml stack update mlops -r mlflow_model_registry



# -- Set the stack as active --
echo "-- Set the stack as active --"

# set the imported stack as the active stack
zenml stack set mlops
