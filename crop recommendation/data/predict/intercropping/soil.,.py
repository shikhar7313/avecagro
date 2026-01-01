import os
import joblib
import json

# -----------------------------
# Paths
# -----------------------------
save_dir = r"C:\shikhar(D drive)\D drive\agri fronti\ansh\crop recommendation\data\trainedmodels\new_intern"
soil_encoder_path = os.path.join(save_dir, "soil_encoder.pkl")

output_json_path = r"C:\shikhar(D drive)\D drive\avecagro\agri fronti\ansh\crop recommendation\data\predict\intercropping\soil_types.json"

# -----------------------------
# Load soil encoder
# -----------------------------
le_soil = joblib.load(soil_encoder_path)

# -----------------------------
# Get soil names
# -----------------------------
soil_names = list(le_soil.classes_)
print("Soil types in encoder:", soil_names)

# -----------------------------
# Save to JSON
# -----------------------------
with open(output_json_path, "w") as f:
    json.dump(soil_names, f, indent=4)

print(f"\n✅ Soil names saved to {output_json_path}")
