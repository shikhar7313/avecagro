import json
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np
import tensorflow as tf
import joblib
from fuzzywuzzy import process
import os

# Load dataset once (not every call)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
n = os.path.join(base_path, "Datasets", "crop_soil_weather_dataset.json")
with open(n, "r") as file:
    data = json.load(file)
df = pd.DataFrame(data)

# Utility for closest matching
def suggest_closest_match(user_input, valid_options):
    closest_match = process.extractOne(user_input, valid_options)
    return closest_match[0] if closest_match else None

# Core prediction function
def predict_npk(soil_type, last_crop, residue_left, rainfall_mm, temperature_C, humidity_percent):
    """
    Predict NPK values based on soil type, last crop, residue, and weather inputs.
    Returns dict with estimated N, P, K or an error string.
    """

    # Load preprocessing + model
    model_path = os.path.join(base_path, "trainedmodels", "NPK")
    encoder = joblib.load(os.path.join(model_path, "encoder.pkl"))
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    model = tf.keras.models.load_model(os.path.join(model_path, "npk_model.keras"))

    # Get valid values
    valid_soil_types = df['soil_type'].unique().tolist()
    valid_crop_types = df['last_crop'].unique().tolist()

    # Fuzzy matching
    soil_type = suggest_closest_match(soil_type, valid_soil_types) or soil_type
    last_crop = suggest_closest_match(last_crop, valid_crop_types) or last_crop

    # Validate
    if soil_type not in valid_soil_types or last_crop not in valid_crop_types:
        return {"error": f"The combination of soil type '{soil_type}' and last crop '{last_crop}' is not valid."}

    # Encode categorical inputs
    cat_input = encoder.transform([[soil_type, last_crop]])

    # Residue handling (True/False → int)
    residue_input = int(residue_left)

    # Numerical inputs
    num_input = np.array([[residue_input, rainfall_mm, temperature_C, humidity_percent]])

    # Combine categorical + numerical
    combined_input = np.concatenate([cat_input, num_input], axis=1)

    # Scale
    scaled_input = scaler.transform(combined_input)

    # Predict
    prediction = model.predict(scaled_input)

    return {
        "estimated_N": round(float(prediction[0][0]), 2),
        "estimated_P": round(float(prediction[0][1]), 2),
        "estimated_K": round(float(prediction[0][2]), 2)
    }
