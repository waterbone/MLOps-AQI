import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import numpy as np
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Debug: Print all column names
    print("Columns in CSV:", df.columns)

    # Drop unnecessary columns
    df = df.drop(['Country', 'City', 'NO2 AQI Value', 'NO2 AQI Category'], axis=1, errors='ignore')

    # Use correct column names based on dataset
    final_features = ['CO AQI Category', 'Ozone AQI Category', 'PM2.5 AQI Category', 
                      'CO AQI Value', 'Ozone AQI Value', 'PM2.5 AQI Value']

    # Check for missing columns
    missing_columns = [col for col in final_features if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in the dataset: {missing_columns}")

    # Label encoding
    le = LabelEncoder()
    for col in ['CO AQI Category', 'Ozone AQI Category', 'PM2.5 AQI Category']:
        df[col] = le.fit_transform(df[col])

    # Prepare features and target
    X = df[final_features]
    y = df['AQI Category']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler and feature names
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(scaler, "artifacts/scaler.pkl")
    joblib.dump(final_features, "artifacts/feature_names.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test

def preprocess_single_input(input_data):
    # Load the saved scaler and feature names
    scaler = joblib.load("artifacts/scaler.pkl")
    feature_names = joblib.load("artifacts/feature_names.pkl")

    # Label encoding untuk kategori AQI
    label_encoder = LabelEncoder()
    category_mapping = {"Good": 0, "Moderate": 1, "Unhealthy": 2}

    # Konversi input kategori menjadi numerik
    input_data["CO AQI Category"] = category_mapping[input_data["CO AQI Category"]]
    input_data["Ozone AQI Category"] = category_mapping[input_data["Ozone AQI Category"]]
    input_data["PM2.5 AQI Category"] = category_mapping[input_data["PM2.5 AQI Category"]]

    # Konversi input dictionary ke DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Lakukan scaling menggunakan scaler yang disimpan
    preprocessed_data = scaler.transform(input_df)

    return np.array(preprocessed_data)
