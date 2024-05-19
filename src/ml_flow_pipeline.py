import mlflow
import mlflow.tensorflow

# Your existing code for data processing, model training, and evaluation
# Make sure to enable MLflow autologging in your training script
# Example:

import train  # Assuming your train.py script is modular enough to be imported

mlflow.tensorflow.autolog()

# Start MLflow run
with mlflow.start_run() as run:
    # Call your training script functions here
    train.main()

# End MLflow run
mlflow.end_run()
