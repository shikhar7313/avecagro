import torch
import torch.nn as nn
import json
import joblib
import numpy as np
import os
import re
from fuzzywuzzy import fuzz
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------
# CONFIG
# -------------------------
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "embedding_dim": 64,
    "hidden_dim": 512,
    "dropout": 0.3,
    "activation": nn.Sigmoid(),
    "json_file": os.path.join(base_path, "Datasets", "multiheight", "multiheight_merged.json"),
    "model_save_path": os.path.join(base_path, "trainedmodels", "multiheight"),
    "scaler_save_path": os.path.join(base_path, "trainedmodels", "multiheight", "scalers"),
    "height_map": {0: "Underground", 1: "On-ground", 2: "Above-ground"},
    "water_map": {0: "Low", 1: "Medium", 2: "High"}
}

# -------------------------
# HELPERS
# -------------------------
def pick_from_range(value_range):
    value_range = str(value_range).replace(" ", "")
    value_range = re.sub(r"[^\d.]+", "-", value_range)
    parts = value_range.split("-")
    if len(parts) >= 2:
        low, high = float(parts[0]), float(parts[1])
    elif len(parts) == 1:
        low = high = float(parts[0])
    else:
        raise ValueError(f"Cannot parse range: {value_range}")
    return np.random.uniform(low, high)

def map_region_to_dataset(region_name, region_map):
    region_name_norm = region_name.strip().lower()
    best_match, score = max([(r, fuzz.token_set_ratio(region_name_norm, r)) for r in region_map.keys()], key=lambda x:x[1])
    if score < 60:
        raise ValueError(f"No close match found in dataset for region '{region_name}'")
    return best_match

def map_numeric_to_category(value, mapping_dict):
    return mapping_dict.get(int(round(value)), value)

# -------------------------
# DATASET
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
# MODEL
# -------------------------
class SeqMultiHeightNN(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.max_crops = dataset.max_crops
        self.pad_idx = len(dataset.crop_name_map)

        # embeddings
        self.crop_emb = nn.Embedding(len(dataset.crop_name_map) + 1, CONFIG["embedding_dim"], padding_idx=self.pad_idx)
        self.region_emb = nn.Embedding(len(dataset.region_map), CONFIG["embedding_dim"])
        self.season_emb = nn.Embedding(len(dataset.season_map), CONFIG["embedding_dim"])
        self.soil_emb = nn.Embedding(len(dataset.soil_type_map), CONFIG["embedding_dim"])

        input_dim = 4 + CONFIG["embedding_dim"] * 4

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
        region, season, soil_type = x_categorical[:, 0], x_categorical[:, 1], x_categorical[:, 2]
        crop_ids = x_categorical[:, 3:]

        region_e = self.region_emb(region)
        season_e = self.season_emb(season)
        soil_e = self.soil_emb(soil_type)

        crop_embs = self.crop_emb(crop_ids)
        mask = (crop_ids != self.pad_idx).unsqueeze(-1).float()
        sum_emb = (crop_embs * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        crop_e = sum_emb / denom

        x = torch.cat([x_numeric, region_e, season_e, soil_e, crop_e], dim=1)

        h = self.model(x)
        crop_out = self.crop_head(h)
        general_out = self.general_head(h)
        return crop_out, general_out

# -------------------------
# LOADING
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
# MAIN PREDICTION FUNCTION (for Flask)
# -------------------------
def multiheight_prediction(crops, region, season, soil_type,
                           soil_ph, rainfall, temperature, humidity):
    """
    Predict crop parameters. 
    This function can be directly imported and used inside Flask.
    """

    dataset = CropDataset(CONFIG["json_file"])
    model, x_scaler, y_scaler, mappings, device = load_model_and_scalers(dataset)

    # convert numeric inputs
    soil_ph_val = pick_from_range(soil_ph)
    rainfall_val = pick_from_range(rainfall)
    temp_val = pick_from_range(temperature)
    humidity_val = pick_from_range(humidity)

    # scale
    x_numeric = torch.tensor(
        x_scaler.transform([[soil_ph_val, rainfall_val, temp_val, humidity_val]]),
        dtype=torch.float32
    ).to(device)

    # categorical
    dataset_region_key = map_region_to_dataset(region, mappings["region_map"])
    region_id = mappings["region_map"][dataset_region_key]
    season_id = mappings["season_map"][season]
    soil_id = mappings["soil_type_map"][soil_type]

    crop_ids = [mappings["crop_name_map"][c.strip()] for c in crops]
    pad_len = dataset.max_crops - len(crop_ids)
    crop_ids += [len(mappings["crop_name_map"])] * pad_len

    x_categorical = torch.tensor([[region_id, season_id, soil_id] + crop_ids], dtype=torch.long).to(device)

    # predict
    with torch.no_grad():
        out_crop, out_gen = model(x_numeric, x_categorical)
        y_pred = torch.cat([out_crop, out_gen], dim=1).cpu().numpy()
    y_pred_inv = y_scaler.inverse_transform(y_pred)[0]

    num_crops = dataset.max_crops
    heights = y_pred_inv[:num_crops]
    growth = y_pred_inv[num_crops:2*num_crops]
    water = y_pred_inv[2*num_crops:3*num_crops]
    yield_impact, space_utilization, soil_resource = map(float, y_pred_inv[-3:])

    readable_heights = [map_numeric_to_category(h, CONFIG["height_map"]) for h in heights[:len(crops)]]
    readable_water = [map_numeric_to_category(w, CONFIG["water_map"]) for w in water[:len(crops)]]

    result = {
        "multiheight_prediction": {
            "Crops": [
                {
                    "Name": crop,
                    "Predicted_Height": readable_heights[i],
                    "Predicted_Growth_Duration": round(float(growth[i]), 1),
                    "Predicted_Water_Needs": readable_water[i]
                }
                for i, crop in enumerate(crops)
            ],
            "Yield_Impact_%": round(float(yield_impact) * 100, 2),
            "Space_Utilization_%": round(float(space_utilization) * 100, 2),
            "Soil_Resource_Score": round(float(soil_resource), 2)
        }
    }

    return result