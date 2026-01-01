import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
from difflib import get_close_matches

# =============================
# Paths
# =============================
save_dir = r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\trainedmodels\new_intern"
model_path = os.path.join(save_dir, "compatibility_model.pth")
soil_encoder_path = os.path.join(save_dir, "soil_encoder.pkl")
crop_encoder_path = os.path.join(save_dir, "crop_encoder.pkl")
scaler_X_path = os.path.join(save_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
dataset_json_path = r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\Datasets\intercropping\intercropping_fixed.json"
soil_types_json_path = r"C:\shikhar(D drive)\D drive\avecagro\agri fronti\ansh\crop recommendation\data\predict\intercropping\soil_types.json"

# =============================
# Load Dataset JSON
# =============================
with open(dataset_json_path, "r") as f:
    dataset = json.load(f)

# Load soil types JSON
with open(soil_types_json_path, "r") as f:
    SOIL_TYPES = json.load(f)

# collect all crop names from JSON
ALL_CROPS = set()
for row in dataset:
    ALL_CROPS.add(row["Crop_1"]["Name"])
    ALL_CROPS.add(row["Crop_2"]["Name"])
ALL_CROPS = sorted(list(ALL_CROPS))

# =============================
# Utility Functions
# =============================
def parse_range_midpoint(value):
    value = str(value).replace("–", "-")
    parts = value.split("-")
    try:
        if len(parts) == 2:
            low, high = float(parts[0]), float(parts[1])
            return (low + high) / 2
        else:
            return float(parts[0])
    except:
        return 0.0

def get_npk_value(npk_dict, key):
    if key in npk_dict:
        return parse_range_midpoint(npk_dict[key])
    elif key.upper() in npk_dict:
        return parse_range_midpoint(npk_dict[key.upper()])
    else:
        return 0.0

def is_env_match(env_input, env_dataset, tolerance=1e-3):
    keys = ["Soil_pH","Soil_Moisture_%","Rainfall_mm","Sunlight_Hours_per_day",
            "Cloud_Cover_%","Temperature_C","Humidity_%","Wind_Speed_kmph"]
    for k in keys:
        if abs(env_input.get(k,0) - env_dataset.get(k,0)) > tolerance:
            return False
    for nk in ["N","P","k"]:
        val_input = get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], nk)
        val_ds = get_npk_value(env_dataset["Soil_NPK_Level_kg_per_ha"], nk)
        if abs(val_input - val_ds) > tolerance:
            return False
    return True

def safe_soil_transform(user_input, soil_list):
    """
    Map user input to closest known soil type.
    """
    user_input = user_input.strip()
    match = get_close_matches(user_input, soil_list, n=1, cutoff=0.3)
    if match:
        return match[0]
    else:
        print(f"[WARNING] No close match for '{user_input}'. Using default '{soil_list[0]}'")
        return soil_list[0]

# =============================
# Model Definition
# =============================
class CompatibilityModel(nn.Module):
    def __init__(self, env_dim, soil_vocab, crop_vocab, embed_dim, output_dim):
        super().__init__()
        self.soil_emb = nn.Embedding(soil_vocab, embed_dim)
        self.crop_emb = nn.Embedding(crop_vocab, embed_dim)
        self.fc1 = nn.Linear(env_dim + embed_dim*3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x_num, x_soil, x_crop1, x_crop2):
        soil_vec = self.soil_emb(x_soil)
        crop1_vec = self.crop_emb(x_crop1)
        crop2_vec = self.crop_emb(x_crop2)
        x = torch.cat([x_num, soil_vec, crop1_vec, crop2_vec], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# =============================
# Load model and preprocessors
# =============================
le_soil = joblib.load(soil_encoder_path)
le_crop = joblib.load(crop_encoder_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

model = CompatibilityModel(env_dim=11,
                           soil_vocab=len(le_soil.classes_),
                           crop_vocab=len(le_crop.classes_),
                           embed_dim=16,
                           output_dim=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# =============================
# Preprocessing Function
# =============================
def preprocess_input(env_numeric, soil_type, crop1, crop2):
    soil_type_mapped = safe_soil_transform(soil_type, list(le_soil.classes_))
    soil_id = le_soil.transform([soil_type_mapped])[0]
    crop1_id = le_crop.transform([crop1])[0] if crop1 in le_crop.classes_ else 0
    crop2_id = le_crop.transform([crop2])[0] if crop2 in le_crop.classes_ else 0

    X_num_scaled = scaler_X.transform([env_numeric])
    return (
        torch.tensor(X_num_scaled, dtype=torch.float32),
        torch.tensor([soil_id]),
        torch.tensor([crop1_id]),
        torch.tensor([crop2_id]),
    )

# =============================
# Hybrid prediction (dataset + model)
# =============================
def hybrid_predict(env_input, soil_type, crop1, crop2):
    # Try exact dataset match first
    for entry in dataset:
        if (entry["Crop_1"]["Name"] == crop1 and 
            entry["Crop_2"]["Name"] == crop2 and 
            safe_soil_transform(soil_type, list(le_soil.classes_)) == entry["Shared_Environment"]["Soil_Type"] and 
            is_env_match(env_input, entry["Shared_Environment"])):

            comp = entry["Compatibility_Analysis"]
            hist = entry["Historical_Knowledge"]

            try:
                y = float(hist["Yield_Impact_%"].replace("%","").replace("+",""))
            except:
                y = 0.0

            return comp["Soil_Resource_Sharing_Score"], comp["Space_Utilization_Efficiency_%"], y

    # Otherwise use AI model
    env_numeric = [
        env_input.get("Soil_pH",0),
        env_input.get("Soil_Moisture_%",0),
        env_input.get("Rainfall_mm",0),
        env_input.get("Sunlight_Hours_per_day",0),
        env_input.get("Cloud_Cover_%",0),
        env_input.get("Temperature_C",0),
        env_input.get("Humidity_%",0),
        env_input.get("Wind_Speed_kmph",0),
        get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], "N"),
        get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], "P"),
        get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], "k"),
    ]

    X_num, X_soil, X_c1, X_c2 = preprocess_input(env_numeric, soil_type, crop1, crop2)
    with torch.no_grad():
        pred_scaled = model(X_num, X_soil, X_c1, X_c2).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    return float(pred[0][0]), float(pred[0][1]), float(pred[0][2])

# =============================
# MAIN TERMINAL SCRIPT
# =============================
import time

# =============================
# MAIN TERMINAL SCRIPT (with time complexity info)
# =============================
if __name__ == "__main__":
    print("\n=== Intercropping Recommendation System (Terminal Mode) ===\n")

    crop1 = input("Enter your main crop: ").strip()
    soil_type = input("Soil type: ").strip()

    # Take numeric environmental inputs
    env_input = {
        "Soil_pH": float(input("Soil pH: ")),
        "Soil_Moisture_%": float(input("Soil Moisture %: ")),
        "Rainfall_mm": float(input("Rainfall (mm): ")),
        "Sunlight_Hours_per_day": float(input("Sunlight Hours/day: ")),
        "Cloud_Cover_%": float(input("Cloud Cover %: ")),
        "Temperature_C": float(input("Temperature C: ")),
        "Humidity_%": float(input("Humidity %: ")),
        "Wind_Speed_kmph": float(input("Wind Speed kmph: ")),
        "Soil_NPK_Level_kg_per_ha": {
            "N": input("Soil N (or range): "),
            "P": input("Soil P (or range): "),
            "k": input("Soil K (or range): ")
        }
    }

    results = []

    print("\nCalculating compatibility with all crops...\n")

    start_time = time.time()

    for c2 in ALL_CROPS:
        if c2 == crop1:
            continue
        srs, sue, yi = hybrid_predict(env_input, soil_type, crop1, c2)
        results.append({
            "crop": c2,
            "Soil_Resource_Sharing_Score": srs,
            "Space_Utilization_Efficiency_%": sue,
            "Yield_Impact_%": yi
        })

    end_time = time.time()

    # Sort best 5 by yield impact
    results = sorted(results, key=lambda x: x["Yield_Impact_%"], reverse=True)[:5]

    print("\n=== TOP 5 BEST COMPANION CROPS ===\n")
    for r in results:
        print(f"Crop: {r['crop']}")
        print(f"  Soil Resource Sharing Score: {r['Soil_Resource_Sharing_Score']:.2f}")
        print(f"  Space Utilization Efficiency %: {r['Space_Utilization_Efficiency_%']:.2f}")
        print(f"  Yield Impact %: {r['Yield_Impact_%']:.2f}")
        print("----------------------------------------------------")

    # =============================
    # Time Complexity Info
    # =============================
    n_crops = len(ALL_CROPS) - 1  # excluding main crop
    dataset_size = len(dataset)
    print(f"\n[INFO] Approximate time complexity: O(n × m) = O({n_crops} × {dataset_size})")
    print(f"[INFO] n (number of crops) = {n_crops}")
    print(f"[INFO] m (dataset entries) = {dataset_size}")
    print(f"[INFO] Total runtime: {end_time - start_time:.4f} seconds")
