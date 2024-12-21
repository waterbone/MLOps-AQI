import mlflow
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from preprocessing import load_and_preprocess_data

def train_model():
    mlflow.start_run()

    # Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/AQI_and_Lat_Long_of_Countries.csv')

    # Train model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Log metrics and parameters
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_param("n_neighbors", 3)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model Accuracy: {accuracy}")

    # Save model
    mlflow.sklearn.log_model(model, "knn_model")
    mlflow.end_run()

if __name__ == "__main__":
    train_model()
