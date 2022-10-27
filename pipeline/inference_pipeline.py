from zenml.pipelines import pipeline

from steps import( data_loader_inference,
predictor, transform_data,
prediction_service_loader,
drift_detector, visualize_results, data_loader_validator)


@pipeline(enable_cache=False)
def inference_pipeline(data_loader,
 transform_predict_data,
 load_model, predict, drift_detector, data_loader_reference):

 raw_data = data_loader()
 transformed_data =transform_predict_data(raw_data)
 model = load_model()
 _ = predict(model, transformed_data)
 reference_data = data_loader_reference()
 drift_report, _ = drift_detector(
        reference_dataset=reference_data,
        comparison_dataset=raw_data,
    )
 print(drift_report)


inference_pipeline_instance = inference_pipeline(data_loader=data_loader_inference(),
 transform_predict_data=transform_data(),
 load_model=prediction_service_loader(), predict=predictor(),
 drift_detector=drift_detector, data_loader_reference=data_loader_validator())

inference_pipeline_instance.run()


