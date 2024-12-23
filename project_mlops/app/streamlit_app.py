import streamlit as st
import pandas as pd
import  mlflow.pyfunc
import joblib
import os
import sys

# Tambahkan path ke folder src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import preprocess_single_input

# Load MLflow model
model = mlflow.pyfunc.load_model("mlruns/0/4d705a5cf9b642c9a1ba33a88684c9e8/artifacts/knn_model")

# Load saved feature names
feature_names = joblib.load("artifacts/feature_names.pkl")

# Streamlit Input
st.title("AQI Prediction")

# Streamlit Input
st.title("AQI Prediction")
CO_AQI_Category = st.selectbox("CO AQI Category", ["Good", "Moderate", "Unhealthy"])
Ozone_AQI_Category = st.selectbox("Ozone AQI Category", ["Good", "Moderate", "Unhealthy"])
PM25_AQI_Category = st.selectbox("PM2.5 AQI Category", ["Good", "Moderate", "Unhealthy"])
CO_AQI_Value = st.number_input("CO AQI Value", min_value=0.0, step=0.1)
Ozone_AQI_Value = st.number_input("Ozone AQI Value", min_value=0.0, step=0.1)
PM25_AQI_Value = st.number_input("PM2.5 AQI Value", min_value=0.0, step=0.1)

# Prepare input dictionary
input_data = {
    "CO AQI Category": CO_AQI_Category,
    "Ozone AQI Category": Ozone_AQI_Category,
    "PM2.5 AQI Category": PM25_AQI_Category,
    "CO AQI Value": CO_AQI_Value,
    "Ozone AQI Value": Ozone_AQI_Value,
    "PM2.5 AQI Value": PM25_AQI_Value,
}

# Preprocess input
try:
    preprocessed_input = preprocess_single_input(input_data)
    # Predict
    predictions = model.predict(preprocessed_input)
    st.write("Predicted AQI Category:", predictions[0])
except Exception as e:
    st.error(f"Error: {str(e)}")
