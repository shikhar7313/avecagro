from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from predict.flask_using.intercropping_new_flask import top_companion_crops
from predict.flask_using.multiheight_idk_flask import top_crop_combinations
from predict.flask_using.npk import predict_npk
import random
import time
from difflib import get_close_matches


app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parents[2]

QUESTIONNAIRE_CANDIDATES = [
    BASE_DIR / "agri fronti" / "Agri_nova" / "src" / "Landing page" / "data" / "questionaire.json",
    BASE_DIR / "agri fronti" / "Agri_nova" / "src" / "Landing page" / "New folder" / "Assets" / "questionaire.json",
]

def resolve_questionnaire_path():
    for candidate in QUESTIONNAIRE_CANDIDATES:
        if candidate.exists():
            return candidate
    return QUESTIONNAIRE_CANDIDATES[0]

QUESTIONAIRE_PATH = resolve_questionnaire_path()

last_request_time = 0
request_lock_seconds = 0  # Ignore multiple calls within 2 seconds

def get_current_season():
    now = datetime.now()
    month = now.month
    day = now.day
    print(f"[DEBUG] Today is {now.strftime('%Y-%m-%d')} (month={month}, day={day})")

    if (month == 6 and day >= 1) or (6 < month < 9) or (month == 9 and day <= 30):
        print("[DEBUG] Season detected: Kharif")
        return "Kharif"
    if (month == 10 and day >= 1) or (10 < month <= 12) or (1 <= month < 4) or (month == 3 and day <= 31):
        print("[DEBUG] Season detected: Rabi")
        return "Rabi"
    if (month == 4 and day >= 1) or (4 < month < 6) or (month == 5 and day <= 31):
        print("[DEBUG] Season detected: Summer")
        return "Summer"
    print("[DEBUG] Season fallback: Kharif")
    return "Kharif"

REGION_MAP = {
    "ANDHRA": 0,
    "GUJARAT": 1,
    "KARNATAKA": 2,
    "KERALA": 3,
    "MAHARASHTRA": 4,
    "ODISHA": 5
}

def get_region_from_input(input_region):
    """
    Match user input region to REGION_MAP key.
    - Case insensitive
    - Partial matches allowed
    """
    if not input_region:
        return "ANDHRA"  # default fallback

    input_upper = input_region.strip().upper()
    # Exact match
    if input_upper in REGION_MAP:
        return input_upper
    # Close match
    matches = get_close_matches(input_upper, REGION_MAP.keys(), n=1, cutoff=0.5)
    if matches:
        return matches[0]
    # Fallback
    return "ANDHRA"

@app.route('/questionaire', methods=['GET'])
def get_questionaire():
    try:
        if not os.path.exists(QUESTIONAIRE_PATH):
            return jsonify({"error": "questionaire.json not found"}), 404
        with open(QUESTIONAIRE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("\n=== QUESTIONAIRE DATA RECEIVED ===")
        print(json.dumps(data, indent=4))
        print("===================================\n")
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def is_request_allowed():
    global last_request_time
    now = time.time()
    if now - last_request_time < request_lock_seconds:
        return False
    last_request_time = now
    return True

@app.route("/predictnpk", methods=["POST"])
def predict():
    if not is_request_allowed():
        return jsonify({"error": "Too many requests. Try again shortly."}), 429

    logging.info("Received NPK request: %s", request.json)
    data = request.get_json()
    print(json.dumps(data, indent=4))

    required_fields = ["soil_type", "last_crop", "residue_left", "rainfall_mm", "temperature_C", "humidity_percent"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        soil_type = str(data["soil_type"]).strip().lower()
        last_crop = str(data["last_crop"]).strip().lower()
        residue_left_input = str(data["residue_left"]).strip().lower()
        if residue_left_input not in ['true', 'false']:
            return jsonify({"error": "residue_left must be 'true' or 'false'."}), 400
        residue_left = residue_left_input == 'true'
        rainfall_mm = float(data["rainfall_mm"])
        temperature_C = float(data["temperature_C"])
        humidity_percent = float(data["humidity_percent"])
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input types: {str(e)}"}), 400

    try:
        result = predict_npk(soil_type, last_crop, residue_left, rainfall_mm, temperature_C, humidity_percent)
        logging.info("NPK Prediction result: %s", result)
    except Exception as e:
        logging.error("NPK Prediction error: %s", str(e))
        logging.exception("Full traceback:")
        return jsonify({"error": f"Error during NPK prediction: {str(e)}"}), 500

    return jsonify(result)

@app.route("/predict_intercropping", methods=["POST"])
def predict_intercropping_route():
    if not is_request_allowed():
        return jsonify({"error": "Too many requests. Try again shortly."}), 429

    logging.info("Received Intercropping request: %s", request.json)
    data = request.get_json()
    print(json.dumps(data, indent=4))

    required_fields = ["crop1", "soil_type"]
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        user_crop1 = str(data["crop1"]).strip()
        soil_type = str(data["soil_type"]).strip()

        env_input = {
            "Soil_pH": float(random.randint(1, 7)),
            "Soil_Moisture_%": float(random.randint(0, 60)),
            "Rainfall_mm": float(random.randint(0, 70)),
            "Sunlight_Hours_per_day": float(random.randint(3, 16)),
            "Cloud_Cover_%": float(random.randint(0, 100)),
            "Temperature_C": float(data.get("Temperature_C", 0)),
            "Humidity_%": float(data.get("Humidity_%", 0)),
            "Wind_Speed_kmph": float(data.get("Wind_Speed_kmph", 0)),
            "Soil_NPK_Level_kg_per_ha": {
                "N": float(data.get("Soil_N", 0)),
                "P": float(data.get("Soil_P", 0)),
                "k": float(data.get("Soil_K", 0))
            }
        }

        print("\n=== ENV INPUT FOR INTERCROPPING ===")
        for k, v in env_input.items():
            print(f"{k}: {v}")

        # Predict for crop1
        raw_results = top_companion_crops(user_crop1, soil_type, env_input)

        # Simplify results like multiheight for terminal print
        simplified_results = [
            {"crop2": r["Crops"][1] if len(r["Crops"]) > 1 else "N/A", 
             "Yield_Impact_%": round(r["Yield_Impact_%"], 2)}
            for r in raw_results
        ]

        # Print simplified results in terminal
        print("\n=== SIMPLIFIED RESULTS FOR CROP1 ===")
        for r in simplified_results:
            print(r)

    except Exception as e:
        logging.error("Intercropping Prediction error: %s", str(e))
        return jsonify({"error": "Error during intercropping prediction."}), 500

    # Return both raw and simplified in JSON
    return jsonify({
        "raw_results": raw_results,        # full raw output at root
        "simplified_results": simplified_results  # optional for frontend
    })

@app.route("/predict_multiheight", methods=["POST"])
def predict_multiheight_route():
    if not is_request_allowed():
        return jsonify({"error": "Too many requests. Try again shortly."}), 429

    logging.info("Received Multiheight request: %s", request.json)
    data = request.get_json()
    print(json.dumps(data, indent=4))

    required_fields = ["user_crop", "region", "soil_type", "soil_ph"]  # season handled internally
    missing = [f for f in required_fields if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        # Capitalize the first letter of each word in crop name
        user_crop = str(data["user_crop"]).strip().title()  
        soil_type = str(data["soil_type"]).strip()
        user_region_input = str(data.get("region", "")).strip()

        # Map input region to proper REGION_MAP key
        region = get_region_from_input(user_region_input)
        print(f"[DEBUG] Region after mapping: {region}")

        # Auto-detect season
        season = get_current_season()
        print(f"[DEBUG] Season selected: {season}")

        # Soil pH and random environmental variables
        soil_ph = float(data.get("soil_ph", 0)) or float(random.randint(1,7))
        rainfall = float(random.randint(0,70))
        temperature = float(random.randint(10,40))
        humidity = float(random.randint(20,90))

        print(f"[DEBUG] Soil_pH: {soil_ph}, Rainfall_mm: {rainfall}, Temperature_C: {temperature}, Humidity_%: {humidity}")

    except Exception as e:
        return jsonify({"error": f"Invalid input types: {str(e)}"}), 400

    try:
        # Call the multiheight model with capitalized crop
        results = top_crop_combinations(
            user_crop, soil_type, region, season, soil_ph, rainfall, temperature, humidity
        )

        simplified_results = [
            {"Crops": r["Crops"], "Yield_Impact_%": round(r["Yield_Impact_%"], 2)}
            for r in results
        ]

        for r in simplified_results:
            print(r)

        logging.info("Multiheight Prediction result: %s", simplified_results)

    except Exception as e:
        logging.error("Multiheight Prediction error: %s", str(e))
        return jsonify({"error": f"Error during multiheight prediction: {str(e)}"}), 500

    return jsonify(simplified_results)

if __name__ == "__main__":
    app.run(debug=True, port=5004)
