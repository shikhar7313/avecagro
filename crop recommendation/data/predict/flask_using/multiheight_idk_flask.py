import torch
import torch.nn as nn
import json
import joblib
import numpy as np
import os
import re
from itertools import combinations
from fuzzywuzzy import fuzz

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
}

# -------------------------
# HELPERS
# -------------------------
def normalize(text):
    return text.strip().title()

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

def safe_lookup(user_value, mapping_dict, value_type="value"):
    user_norm = normalize(user_value)
    lower_map = {k.lower(): k for k in mapping_dict.keys()}
    if user_norm.lower() not in lower_map:
        raise KeyError(f"{value_type} '{user_value}' not found in trained model mappings.")
    real_key = lower_map[user_norm.lower()]
    return mapping_dict[real_key]

def map_region_fuzzy(user_input, region_list, threshold=60):
    user_norm = user_input.strip().upper()
    best_match, best_score = None, 0
    for r in region_list:
        score = fuzz.token_set_ratio(user_norm, r.upper())
        if score > best_score:
            best_match, best_score = r, score
    if best_score < threshold:
        raise ValueError(f"No close match found for region '{user_input}'")
    return best_match

# -------------------------
# DATASET
# -------------------------
class CropDataset:
    def __init__(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        crops, soils, seasons = set(), set(), set()
        for item in self.data:
            for c in item["Crops"]:
                crops.add(c["Name"])
            soils.update([s.strip() for s in item["Shared_Environment"]["Soil_Type"].split(",")])
            seasons.add(item["Shared_Environment"]["Season"])

        self.crop_list = sorted(crops)
        self.soil_list = sorted(soils)
        self.season_list = sorted(seasons)

        self.crop_name_map = {name: idx for idx, name in enumerate(self.crop_list)}
        self.soil_type_map = {name: idx for idx, name in enumerate(self.soil_list)}
        self.season_map = {name: idx for idx, name in enumerate(self.season_list)}

        self.max_crops = max(len(item["Crops"]) for item in self.data)

# -------------------------
# MODEL
# -------------------------
class SeqMultiHeightNN(nn.Module):
    def __init__(self, dataset, region_count):
        super().__init__()
        self.max_crops = dataset.max_crops
        self.pad_idx = len(dataset.crop_name_map)

        self.crop_emb = nn.Embedding(len(dataset.crop_name_map)+1, CONFIG["embedding_dim"], padding_idx=self.pad_idx)
        self.region_emb = nn.Embedding(region_count, CONFIG["embedding_dim"])
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
        crop_e = (crop_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        x = torch.cat([x_numeric, region_e, season_e, soil_e, crop_e], dim=1)
        h = self.model(x)
        return self.crop_head(h), self.general_head(h)

# -------------------------
# LOAD MODEL & SCALERS
# -------------------------
def load_model(dataset, mappings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_regions = list(mappings["region_map"].keys())
    model = SeqMultiHeightNN(dataset, len(all_regions)).to(device)
    model_path = os.path.join(CONFIG["model_save_path"], "multiheight_final.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    x_scaler = joblib.load(os.path.join(CONFIG["scaler_save_path"], "x_scaler.save"))
    y_scaler = joblib.load(os.path.join(CONFIG["scaler_save_path"], "y_scaler.save"))
    return model, x_scaler, y_scaler, device, all_regions

# -------------------------
# FUNCTION: GENERATE TOP COMBINATIONS
# -------------------------
def top_crop_combinations(user_crop, soil_type, region_input, season, soil_ph, rainfall, temperature, humidity):
    dataset = CropDataset(CONFIG["json_file"])
    with open(os.path.join(CONFIG["scaler_save_path"], "mappings.json"), "r") as f:
        mappings = json.load(f)

    model, x_scaler, y_scaler, device, all_regions = load_model(dataset, mappings)

    # Lookup IDs
    user_crop_id = safe_lookup(user_crop, mappings["crop_name_map"], "crop")
    soil_id = safe_lookup(soil_type, mappings["soil_type_map"], "soil type")
    season_id = safe_lookup(season, mappings["season_map"], "season")
    region = map_region_fuzzy(region_input, all_regions)
    region_id = mappings["region_map"][region]

    # Prepare crops for combination
    remaining_crops = [c for c in dataset.crop_list if normalize(c) != normalize(user_crop)]
    results = []

    # Generate combinations from 2 to max_crops
    for r in range(2, dataset.max_crops + 1):
        for combo in combinations(remaining_crops, r - 1):
            full_combo = [user_crop] + list(combo)
            crop_ids = [mappings["crop_name_map"][c] for c in full_combo]
            pad_len = dataset.max_crops - len(crop_ids)
            crop_ids += [len(mappings["crop_name_map"])] * pad_len

            # Numeric features
            soil_ph_val = pick_from_range(soil_ph)
            rainfall_val = pick_from_range(rainfall)
            temp_val = pick_from_range(temperature)
            humidity_val = pick_from_range(humidity)
            x_numeric = torch.tensor(
                x_scaler.transform([[soil_ph_val, rainfall_val, temp_val, humidity_val]]),
                dtype=torch.float32
            ).to(device)

            x_categorical = torch.tensor([[region_id, season_id, soil_id] + crop_ids], dtype=torch.long).to(device)

            # Prediction
            with torch.no_grad():
                out_crop, out_gen = model(x_numeric, x_categorical)
                y_pred = torch.cat([out_crop, out_gen], dim=1).cpu().numpy()
                y_pred_inv = y_scaler.inverse_transform(y_pred)[0]

            yield_impact = float(y_pred_inv[-3])
            results.append({"Crops": full_combo, "Yield_Impact_%": yield_impact})

    # Top 5
    results = sorted(results, key=lambda x: x["Yield_Impact_%"], reverse=True)[:5]
    return results