import pandas as pd
import mlflow.pyfunc
import argparse

def predict(model_path, data_path):
    """
    Load a trained MLflow model and make predictions on new data.
    :param model_path: Path to the MLflow model.
    :param data_path: Path to the CSV file containing input data.
    """
    # Load MLflow model
    print("Loading model...")
    model = mlflow.pyfunc.load_model(model_path)
    print("Model loaded successfully.")

    # Load input data
    print("Loading input data...")
    data = pd.read_csv(data_path)
    print("Data loaded successfully:")
    print(data.head())

    # Make predictions
    print("Generating predictions...")
    predictions = model.predict(data)
    print("Predictions:")
    print(predictions)

    # Save predictions to a file
    output_path = "predictions.csv"
    pd.DataFrame(predictions, columns=["Predictions"]).to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using a trained MLflow model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the MLflow model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the input CSV file.")
    
    args = parser.parse_args()
    predict(args.model_path, args.data_path)