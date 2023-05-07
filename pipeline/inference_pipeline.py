from zenml.pipelines import pipeline

from steps import( data_loader_inference,
predictor,
prediction_service_loader)


@pipeline(enable_cache=False)
def inference_pipeline(data_loader,
 load_model, predict):

 raw_data = data_loader()
 model = load_model()
 preditions = predict(model, raw_data)
 print(preditions)


inference_pipeline_instance = inference_pipeline(data_loader=data_loader_inference(),
 load_model=prediction_service_loader(), predict=predictor())

inference_pipeline_instance.run()


