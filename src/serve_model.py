import mlflow.pyfunc

# Path to the MLflow model directory
model_uri = "models:/mobilenetv2_finetuned_model/1"

# Serve the model
mlflow.pyfunc.serve(model_uri=model_uri, host="0.0.0.0", port=5001)
