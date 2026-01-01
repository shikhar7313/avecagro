import torch
import torch.nn as nn
import json
import joblib
import numpy as np
import os
import re
from fuzzywuzzy import fuzz
import random

# -------------------------
# 0. CONFIG
# -------------------------
CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 512,
    "dropout": 0.3,
    "activation": nn.Sigmoid(),
    "json_file": r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\Datasets\multiheight\multiheight_merged.json",
    "model_save_path": r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\trainedmodels\multiheight",
    "scaler_save_path": r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\trainedmodels\multiheight\scalers",
    "USE_HARDCODED_INPUT": True  # Set True to use hardcoded input
}

# -------------------------
# 1. Helpers
# -------------------------
def pick_from_range(value_range):
    value_range = value_range.replace(" ", "")
    value_range = re.sub(r"[^\d.]+", "-", value_range)
    parts = value_range.split("-")
    if len(parts) >= 2:
        low, high = float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        low = high = float(parts[0])
    else:
        raise ValueError(f"Cannot parse range: {value_range}")
    return np.random.uniform(low, high)

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

# -------------------------
# 2. Dataset Metadata
# -------------------------
class CropDataset:
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        crops, soils, regions, seasons = set(), set(), set(), set()
        for item in self.data:
            for c in item["Crops"]:
                crops.add(c["Name"])
            soils.update([s.strip() for s in item["Shared_Environment"]["Soil_Type"].split(",")])
            regions.add(item["Shared_Environment"]["Region"].split("_")[0])
            seasons.add(item["Shared_Environment"]["Season"])
        self.crop_list = sorted(crops)
        self.soil_list = sorted(soils)
        self.region_list = sorted(regions)
        self.season_list = sorted(seasons)
        self.crop_name_map = {name: idx for idx, name in enumerate(self.crop_list)}
        self.soil_type_map = {name: idx for idx, name in enumerate(self.soil_list)}
        self.region_map = {name: idx for idx, name in enumerate(self.region_list)}
        self.season_map = {name: idx for idx, name in enumerate(self.season_list)}
        self.max_crops = max(len(item["Crops"]) for item in self.data)

# -------------------------
# 3. Neural Network
# -------------------------
class SeqMultiHeightNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.max_crops = dataset.max_crops
        self.crop_emb = nn.Embedding(len(dataset.crop_name_map)+1, CONFIG["embedding_dim"], padding_idx=len(dataset.crop_name_map))
        self.region_emb = nn.Embedding(len(dataset.region_map), CONFIG["embedding_dim"])
        self.season_emb = nn.Embedding(len(dataset.season_map), CONFIG["embedding_dim"])
        self.soil_emb = nn.Embedding(len(dataset.soil_type_map), CONFIG["embedding_dim"])
        input_dim = 4 + CONFIG["embedding_dim"] * (3 + self.max_crops)
        self.model = nn.Sequential(
            nn.Linear(input_dim, CONFIG["hidden_dim"]),
            CONFIG["activation"],
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG["hidden_dim"], CONFIG["hidden_dim"]),
            CONFIG["activation"],
            nn.Dropout(CONFIG["dropout"]),
            nn.Linear(CONFIG["hidden_dim"], CONFIG["hidden_dim"]),
            CONFIG["activation"],
            nn.Dropout(CONFIG["dropout"]),
        )
        self.crop_head = nn.Linear(CONFIG["hidden_dim"], self.max_crops * 3)
        self.general_head = nn.Linear(CONFIG["hidden_dim"], 3)

    def forward(self, x_numeric, x_categorical):
        region, season, soil_type = x_categorical[:,0], x_categorical[:,1], x_categorical[:,2]
        crop_ids = x_categorical[:,3:]
        region_e = self.region_emb(region)
        season_e = self.season_emb(season)
        soil_e = self.soil_emb(soil_type)
        crop_e = self.crop_emb(crop_ids).view(crop_ids.size(0), -1)
        x = torch.cat([x_numeric, region_e, season_e, soil_e, crop_e], dim=1)
        h = self.model(x)
        crop_out = self.crop_head(h)
        general_out = self.general_head(h)
        return crop_out, general_out

# -------------------------
# 4. Load Model & Scalers
# -------------------------
def load_model_and_scalers(dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SeqMultiHeightNN(dataset).to(device)
    final_model_path = os.path.join(CONFIG["model_save_path"], "multiheight_final.pth")
    model.load_state_dict(torch.load(final_model_path, map_location=device))
    model.eval()
    x_scaler = joblib.load(os.path.join(CONFIG["scaler_save_path"], "x_scaler.save"))
    y_scaler = joblib.load(os.path.join(CONFIG["scaler_save_path"], "y_scaler.save"))
    with open(os.path.join(CONFIG["scaler_save_path"], "mappings.json"), "r") as f:
        mappings = json.load(f)
    return model, x_scaler, y_scaler, mappings, device

# -------------------------
# 5. Prediction
# -------------------------
def predict(json_input, dataset, model, x_scaler, y_scaler, mappings, device):
    region_map = mappings["region_map"]
    season_map = mappings["season_map"]
    soil_map = mappings["soil_type_map"]
    crop_map = mappings["crop_name_map"]

    # Numeric features
    soil_ph = float(json_input["Shared_Environment"]["Soil_pH_Range"])
    rainfall = float(json_input["Shared_Environment"]["Rainfall_mm_Range"])
    temp = float(json_input["Shared_Environment"]["Temperature_C_Range"])
    humidity = float(json_input["Shared_Environment"]["Humidity_%_Range"])
    x_numeric = torch.tensor(x_scaler.transform([[soil_ph, rainfall, temp, humidity]])[0], dtype=torch.float32).unsqueeze(0).to(device)

    # Categorical features with fuzzy matching
    region_name = normalize_text(json_input["Shared_Environment"]["Region"])
    region = min(region_map.keys(), key=lambda k: fuzz.ratio(normalize_text(k), region_name))
    region_idx = region_map[region]

    season_name = normalize_text(json_input["Shared_Environment"]["Season"])
    season = min(season_map.keys(), key=lambda k: fuzz.ratio(normalize_text(k), season_name))
    season_idx = season_map[season]

    soil_name = normalize_text(json_input["Shared_Environment"]["Soil_Type"])
    soil = min(soil_map.keys(), key=lambda k: fuzz.ratio(normalize_text(k), soil_name))
    soil_idx = soil_map[soil]

    # Crop ids
    crop_ids = [crop_map[min(crop_map.keys(), key=lambda k: fuzz.ratio(normalize_text(k), normalize_text(c["Name"])))] 
                for c in json_input["Crops"]]
    pad_len = dataset.max_crops - len(crop_ids)
    crop_ids += [len(crop_map)] * pad_len
    x_categorical = torch.tensor([[region_idx, season_idx, soil_idx] + crop_ids], dtype=torch.long).to(device)

    # Forward
    with torch.no_grad():
        out_crop, out_gen = model(x_numeric, x_categorical)
        y_pred = torch.cat([out_crop, out_gen], dim=1).cpu().numpy()
    y_pred_inv = y_scaler.inverse_transform(y_pred)[0]

    # Split outputs
    num_crops = dataset.max_crops
    heights = y_pred_inv[:num_crops]
    growth_durations = y_pred_inv[num_crops:2*num_crops]
    water_needs = y_pred_inv[2*num_crops:3*num_crops]
    yield_impact, space_utilization, soil_resource = y_pred_inv[-3:]

    # Human-readable
    readable = {
        "Crops": [
            {
                "Name": c["Name"],
                "Predicted_Height_cm": float(round(h, 2)),
                "Predicted_Growth_Duration_days": float(round(g, 2)),
                "Predicted_Water_Needs_level": float(round(w, 2))
            }
            for c, h, g, w in zip(json_input["Crops"], heights, growth_durations, water_needs)
        ],
        "Yield_Impact_%": float(round(yield_impact*100, 2)),
        "Space_Utilization_%": float(round(space_utilization*100, 2)),
        "Soil_Resource_Score": float(round(soil_resource, 2))
    }

    return y_pred_inv.tolist(), readable

# -------------------------
# 6. Fuzzy Match for Dataset
# -------------------------
def find_best_match(sample_input, dataset):
    region = normalize_text(sample_input["Shared_Environment"]["Region"])
    season = normalize_text(sample_input["Shared_Environment"]["Season"])
    soil_type = normalize_text(sample_input["Shared_Environment"]["Soil_Type"])
    crops = sorted([normalize_text(c["Name"]) for c in sample_input["Crops"]])
    best_score = 0
    best_match = None
    for item in dataset.data:
        item_region = normalize_text(item["Shared_Environment"]["Region"].split("_")[0])
        item_season = normalize_text(item["Shared_Environment"]["Season"])
        item_soil = normalize_text(item["Shared_Environment"]["Soil_Type"].split(",")[0])
        item_crops = sorted([normalize_text(c["Name"]) for c in item["Crops"]])
        if item_region == region and item_season == season and item_soil == soil_type and item_crops == crops:
            return item, 100
        region_score = fuzz.ratio(region, item_region)
        season_score = fuzz.ratio(season, item_season)
        soil_score = fuzz.ratio(soil_type, item_soil)
        crop_score = fuzz.ratio(" ".join(crops), " ".join(item_crops))
        avg_score = (region_score + season_score + soil_score + crop_score) / 4
        if avg_score > best_score:
            best_score = avg_score
            best_match = item
    return best_match, best_score

# -------------------------
# 7. Main
# -------------------------
if __name__ == "__main__":
    dataset = CropDataset(CONFIG["json_file"])
    model, x_scaler, y_scaler, mappings, device = load_model_and_scalers(dataset)

    # --- Input selection ---
    if CONFIG["USE_HARDCODED_INPUT"]:
        sample_input_fixed = {
            "Crops": [{"Name": "Onion"}, {"Name": "Pigeonpea"}, {"Name": "Rice"}],
            "Shared_Environment": {
                "Region": "Andhra Pradesh",
                "Season": "Kharif",
                "Soil_Type": "Red loam",
                "Soil_pH_Range": "6.5",
                "Rainfall_mm_Range": "800",
                "Temperature_C_Range": "30",
                "Humidity_%_Range": "75"
            }
        }
        random_crops = random.sample(dataset.crop_list, k=min(3, len(dataset.crop_list)))
        random_region = random.choice(dataset.region_list)
        random_season = random.choice(dataset.season_list)
        random_soil = random.choice(dataset.soil_list)
        sample_input_random = {
            "Crops": [{"Name": name} for name in random_crops],
            "Shared_Environment": {
                "Region": random_region,
                "Season": random_season,
                "Soil_Type": random_soil,
                "Soil_pH_Range": str(round(random.uniform(5.5, 7.5), 2)),
                "Rainfall_mm_Range": str(round(random.uniform(500, 1200), 2)),
                "Temperature_C_Range": str(round(random.uniform(20, 35), 1)),
                "Humidity_%_Range": str(round(random.uniform(50, 90), 1))
            }
        }
        HARDCODED_VERSION = 2  # 1=fixed, 2=random
        sample_input = sample_input_fixed if HARDCODED_VERSION == 1 else sample_input_random
    else:
        # User input mode
        region = input(f"Enter Region {list(dataset.region_list)}: ").strip()
        season = input(f"Enter Season {list(dataset.season_list)}: ").strip()
        soil_type = input(f"Enter Soil_Type {list(dataset.soil_list)}: ").strip()
        n_crops = int(input("How many crops in the combo? "))
        crops = [{"Name": input(f"Crop {i+1}: ").strip()} for i in range(n_crops)]
        soil_ph_range = input("Soil pH: ").strip()
        rainfall_range = input("Rainfall mm: ").strip()
        temp_range = input("Temperature °C: ").strip()
        humidity_range = input("Humidity %: ").strip()
        sample_input = {
            "Crops": crops,
            "Shared_Environment": {
                "Region": region,
                "Season": season,
                "Soil_Type": soil_type,
                "Soil_pH_Range": soil_ph_range,
                "Rainfall_mm_Range": rainfall_range,
                "Temperature_C_Range": temp_range,
                "Humidity_%_Range": humidity_range
            }
        }

    # --- Match logic ---
    match_found, score = find_best_match(sample_input, dataset)
    if match_found and score >= 85:
        print(f"\n--- Match Found in Dataset (Fuzzy Score: {score:.1f}) ---")
        result, readable = predict(sample_input, dataset, model, x_scaler, y_scaler, mappings, device)
    else:
        print("\n--- No Close Match Found → Using Model Prediction ---")
        result, readable = predict(sample_input, dataset, model, x_scaler, y_scaler, mappings, device)

    print("\n--- Raw Numeric Prediction ---")
    print(json.dumps(result, indent=4))
    print("\n--- Human-Readable Prediction ---")
    print(json.dumps(readable, indent=4))