import os
import json
import joblib
import torch
import torch.nn as nn
import numpy as np

# =============================
# Paths
# =============================
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
save_dir = os.path.join(base_path, "trainedmodels", "new_intern")
model_path = os.path.join(save_dir, "compatibility_model.pth")
soil_encoder_path = os.path.join(save_dir, "soil_encoder.pkl")
crop_encoder_path = os.path.join(save_dir, "crop_encoder.pkl")
scaler_X_path = os.path.join(save_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(save_dir, "scaler_y.pkl")
dataset_json_path = os.path.join(base_path, "Datasets", "intercropping", "intercropping_fixed.json")

# =============================
# Load Dataset JSON
# =============================
with open(dataset_json_path, "r") as f:
    dataset = json.load(f)

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
    for k in ["Soil_pH","Soil_Moisture_%","Rainfall_mm","Sunlight_Hours_per_day",
              "Cloud_Cover_%","Temperature_C","Humidity_%","Wind_Speed_kmph"]:
        if abs(env_input.get(k,0) - env_dataset.get(k,0)) > tolerance:
            return False
    for nk in ["N","P","k"]:
        val_input = get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], nk)
        val_ds = get_npk_value(env_dataset["Soil_NPK_Level_kg_per_ha"], nk)
        if abs(val_input - val_ds) > tolerance:
            return False
    return True

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
# Load model and preprocessors (once)
# =============================
le_soil = joblib.load(soil_encoder_path)
le_crop = joblib.load(crop_encoder_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

model = CompatibilityModel(env_dim=11, soil_vocab=len(le_soil.classes_),
                           crop_vocab=len(le_crop.classes_), embed_dim=16, output_dim=3)
model.load_state_dict(torch.load(model_path))
model.eval()

# =============================
# Preprocessing function
# =============================
def preprocess_input(env_numeric, soil_type, crop1, crop2):
    soil_id = le_soil.transform([soil_type])[0]
    try:
        crop1_id = le_crop.transform([crop1])[0]
    except:
        crop1_id = 0
    try:
        crop2_id = le_crop.transform([crop2])[0]
    except:
        crop2_id = 0
    X_num_scaled = scaler_X.transform([env_numeric])
    return torch.tensor(X_num_scaled, dtype=torch.float32), torch.tensor([soil_id]), torch.tensor([crop1_id]), torch.tensor([crop2_id])

# =============================
# Hybrid prediction function
# =============================
def hybrid_predict_user(env_input, soil_type, crop1, crop2):
    """
    Takes user inputs and returns compatibility prediction for crop intercropping.
    
    env_input: dict with numeric environment values + Soil_NPK_Level_kg_per_ha
    soil_type: str
    crop1, crop2: str
    """
    # 1️⃣ Exact dataset match first
    for entry in dataset:
        if (entry["Crop_1"]["Name"] == crop1 and 
            entry["Crop_2"]["Name"] == crop2 and 
            entry["Shared_Environment"]["Soil_Type"] == soil_type and 
            is_env_match(env_input, entry["Shared_Environment"])):
            
            comp = entry["Compatibility_Analysis"]
            hist = entry["Historical_Knowledge"]
            try:
                yield_val = float(hist["Yield_Impact_%"].replace("%","").replace("+",""))
            except:
                yield_val = 0.0
            return {
                "Soil_Resource_Sharing_Score": comp["Soil_Resource_Sharing_Score"],
                "Space_Utilization_Efficiency_%": comp["Space_Utilization_Efficiency_%"],
                "Yield_Impact_%": yield_val
            }

    # 2️⃣ Generalized model prediction
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
        get_npk_value(env_input["Soil_NPK_Level_kg_per_ha"], "k")
    ]
    X_num, X_soil, X_c1, X_c2 = preprocess_input(env_numeric, soil_type, crop1, crop2)
    with torch.no_grad():
        pred_scaled = model(X_num, X_soil, X_c1, X_c2).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    return {
        "Soil_Resource_Sharing_Score": float(pred[0][0]),
        "Space_Utilization_Efficiency_%": float(pred[0][1]),
        "Yield_Impact_%": float(pred[0][2])
    }
