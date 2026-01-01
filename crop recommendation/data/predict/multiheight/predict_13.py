import torch
import torch.nn as nn
import json
import joblib
import numpy as np
import os
import re
from fuzzywuzzy import fuzz
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    "USE_HARDCODED_INPUT": True,
    "RUN_GENERALIZATION_TEST": True,
    "NUM_GENERALIZATION_SAMPLES": 5,
    "height_map": {0: "Underground", 1: "On-ground", 2: "Above-ground"},
    "water_map": {0: "Low", 1: "Medium", 2: "High"}
}

# -------------------------
# 1. Helpers
# -------------------------
def pick_from_range(value_range, mode="random", noise_std=0.0):
    """
    Parse a numeric range string like '80-120' or single value '6.5'.
    mode = "mid" → midpoint
    mode = "random" → uniform sample
    noise_std → Gaussian noise
    """
    if not isinstance(value_range, str):
        base_value = float(value_range)
    else:
        value_range = value_range.replace(" ", "")
        value_range = re.sub(r"[^\d.]+", "-", value_range)
        parts = value_range.split("-")
        if len(parts) >= 2:
            low, high = float(parts[0]), float(parts[1])
        elif len(parts) == 1:
            low = high = float(parts[0])
        else:
            raise ValueError(f"Cannot parse range: {value_range}")

        if mode == "mid":
            base_value = (low + high) / 2.0
        elif mode == "random":
            base_value = np.random.uniform(low, high)
        else:
            raise ValueError("Invalid mode: choose 'mid' or 'random'")

    if noise_std > 0:
        base_value += np.random.normal(0, noise_std)

    return base_value

def normalize_text(text):
    return re.sub(r'\s+', ' ', text.strip().lower())

def map_region_to_dataset(region_name, region_map):
    region_name_norm = region_name.strip().lower()
    best_match, score = max([(r, fuzz.token_set_ratio(region_name_norm, r)) for r in region_map.keys()], key=lambda x:x[1])
    if score < 60:
        raise ValueError(f"No close match found in dataset for region '{region_name}'")
    return best_match

def map_numeric_to_category(value, mapping_dict):
    return mapping_dict.get(int(round(value)), value)

# -------------------------
# 2. Dataset
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
        self.crop_emb = nn.Embedding(len(dataset.crop_name_map)+1, CONFIG["embedding_dim"],
                                     padding_idx=len(dataset.crop_name_map))
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
# 4. Prediction Functions
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

def predict(sample_input, dataset, model, x_scaler, y_scaler, mappings, device):
    region_map = mappings["region_map"]
    season_map = mappings["season_map"]
    soil_map = mappings["soil_type_map"]
    crop_map = mappings["crop_name_map"]

    # improved random sampling with noise
    soil_ph = pick_from_range(sample_input["Shared_Environment"]["Soil_pH_Range"], mode="random", noise_std=0.05)
    rainfall = pick_from_range(sample_input["Shared_Environment"]["Rainfall_mm_Range"], mode="random", noise_std=5)
    temp = pick_from_range(sample_input["Shared_Environment"]["Temperature_C_Range"], mode="random", noise_std=1)
    humidity = pick_from_range(sample_input["Shared_Environment"]["Humidity_%_Range"], mode="random", noise_std=2)

    x_numeric = torch.tensor(x_scaler.transform([[soil_ph, rainfall, temp, humidity]]), dtype=torch.float32).to(device)
    dataset_region_key = map_region_to_dataset(sample_input["Shared_Environment"]["Region"], region_map)
    region = region_map[dataset_region_key]
    season = season_map[sample_input["Shared_Environment"]["Season"]]
    soil_type = soil_map[sample_input["Shared_Environment"]["Soil_Type"]]

    crop_ids = [crop_map[c["Name"]] for c in sample_input["Crops"]]
    pad_len = dataset.max_crops - len(crop_ids)
    crop_ids += [len(crop_map)] * pad_len
    x_categorical = torch.tensor([[region, season, soil_type] + crop_ids], dtype=torch.long).to(device)

    with torch.no_grad():
        out_crop, out_gen = model(x_numeric, x_categorical)
        y_pred = torch.cat([out_crop, out_gen], dim=1).cpu().numpy()
    y_pred_inv = y_scaler.inverse_transform(y_pred)[0]

    num_crops = dataset.max_crops
    heights = y_pred_inv[:num_crops]
    growth = y_pred_inv[num_crops:2*num_crops]
    water = y_pred_inv[2*num_crops:3*num_crops]
    yield_impact, space_utilization, soil_resource = map(float, y_pred_inv[-3:])

    readable_heights = [map_numeric_to_category(h, CONFIG["height_map"]) for h in heights[:len(sample_input["Crops"])]]
    readable_water = [map_numeric_to_category(w, CONFIG["water_map"]) for w in water[:len(sample_input["Crops"])]]

    readable_result = {
        "Crops": [{"Name": crop["Name"], 
                   "Predicted_Height": readable_heights[i],
                   "Predicted_Growth_Duration": round(growth[i],1),
                   "Predicted_Water_Needs": readable_water[i]} 
                  for i,crop in enumerate(sample_input["Crops"])],
        "Yield_Impact": round(yield_impact*100,2),
        "Space_Utilization": round(space_utilization*100,2),
        "Soil_Resource": round(soil_resource,2)
    }

    return heights, growth, water, readable_result

# -------------------------
# 5. Fuzzy Match
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
        avg_score = (fuzz.ratio(region,item_region)+fuzz.ratio(season,item_season)+
                     fuzz.ratio(soil_type,item_soil)+fuzz.ratio(" ".join(crops)," ".join(item_crops)))/4
        if avg_score > best_score:
            best_score = avg_score
            best_match = item
    return best_match, best_score

# -------------------------
# 6. Generalization Test
# -------------------------
def run_generalization_test(dataset, model, x_scaler, y_scaler, mappings, device, num_samples=5):
    print("\n--- Generalization Test: Random Samples ---")
    for i in range(num_samples):
        random_crops = random.sample(dataset.crop_list, k=min(3,len(dataset.crop_list)))
        random_region = random.choice(dataset.region_list)
        random_season = random.choice(dataset.season_list)
        random_soil = random.choice(dataset.soil_list)
        sample_input = {
            "Crops": [{"Name": c} for c in random_crops],
            "Shared_Environment":{
                "Region": random_region,
                "Season": random_season,
                "Soil_Type": random_soil,
                "Soil_pH_Range": f"{round(random.uniform(5.5,7.5),2)}",
                "Rainfall_mm_Range": f"{round(random.uniform(500,1200),2)}",
                "Temperature_C_Range": f"{round(random.uniform(20,35),1)}",
                "Humidity_%_Range": f"{round(random.uniform(50,90),1)}"
            }
        }

        match_found, score = find_best_match(sample_input, dataset)
        print(f"\n--- Sample {i+1} ---")
        if match_found:
            print(f"Dataset Match Found (Fuzzy Score: {score:.1f})")
        else:
            print("No Close Match → Model Prediction")

        _, _, _, readable = predict(sample_input, dataset, model, x_scaler, y_scaler, mappings, device)

        print("Random Input Used:")
        print(f"  Region: {sample_input['Shared_Environment']['Region']}")
        print(f"  Season: {sample_input['Shared_Environment']['Season']}")
        print(f"  Soil Type: {sample_input['Shared_Environment']['Soil_Type']}")
        print(f"  Crops: {[c['Name'] for c in sample_input['Crops']]}")

        for crop in readable["Crops"]:
            print(f"Crop: {crop['Name']}")
            print(f"  Predicted Height: {crop['Predicted_Height']}")
            print(f"  Predicted Growth Duration: {crop['Predicted_Growth_Duration']}")
            print(f"  Predicted Water Needs: {crop['Predicted_Water_Needs']}")
        print(f"Yield Impact (%): {readable['Yield_Impact']}")
        print(f"Space Utilization (%): {readable['Space_Utilization']}")
        print(f"Soil Resource Score: {readable['Soil_Resource']}")

# -------------------------
# 7. Main
# -------------------------
if __name__ == "__main__":
    dataset = CropDataset(CONFIG["json_file"])
    model, x_scaler, y_scaler, mappings, device = load_model_and_scalers(dataset)

    if CONFIG["USE_HARDCODED_INPUT"]:
        sample_input = {
            "Crops": [{"Name": "Onion"}, {"Name": "Pigeonpea"}, {"Name": "Rice"}],
            "Shared_Environment": {
                "Region": "Andhra Pradesh",
                "Season": "Kharif",
                "Soil_Type": "Coastal alluvium",
                "Soil_pH_Range": "6.5",
                "Rainfall_mm_Range": "900",
                "Temperature_C_Range": "30",
                "Humidity_%_Range": "75"
            }
        }
    else:
        crops = input("Enter crops separated by commas: ").split(",")
        sample_input = {
            "Crops": [{"Name": c.strip()} for c in crops],
            "Shared_Environment": {
                "Region": input("Region: "),
                "Season": input("Season: "),
                "Soil_Type": input("Soil Type: "),
                "Soil_pH_Range": input("Soil pH (e.g., 5.8-7.5): "),
                "Rainfall_mm_Range": input("Rainfall mm (e.g., 500-1200): "),
                "Temperature_C_Range": input("Temperature C (e.g., 20-35): "),
                "Humidity_%_Range": input("Humidity % (e.g., 50-90): ")
            }
        }

    match_found, score = find_best_match(sample_input, dataset)
    if match_found:
        print(f"\nDataset Match Found (Fuzzy Score: {score:.1f})")
    else:
        print("\nNo Close Match Found in Dataset")

    _, _, _, readable = predict(sample_input, dataset, model, x_scaler, y_scaler, mappings, device)
    print("\n--- Model Predictions ---")
    for crop in readable["Crops"]:
        print(f"Crop: {crop['Name']}")
        print(f"  Predicted Height: {crop['Predicted_Height']}")
        print(f"  Predicted Growth Duration: {crop['Predicted_Growth_Duration']}")
        print(f"  Predicted Water Needs: {crop['Predicted_Water_Needs']}")
    print(f"Yield Impact (%): {readable['Yield_Impact']}")
    print(f"Space Utilization (%): {readable['Space_Utilization']}")
    print(f"Soil Resource Score: {readable['Soil_Resource']}")

    if CONFIG["RUN_GENERALIZATION_TEST"]:
        run_generalization_test(dataset, model, x_scaler, y_scaler, mappings, device, CONFIG["NUM_GENERALIZATION_SAMPLES"])
