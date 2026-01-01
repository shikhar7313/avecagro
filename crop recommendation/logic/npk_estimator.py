import json

# Mapping of soil types to their average nutrient-holding capacity (in kg/hectare)
SOIL_NUTRIENT_CAPACITY = {
    "loamy": {"N": 50, "P": 30, "K": 40},  # High N, moderate P/K
    "sandy": {"N": 20, "P": 10, "K": 15},  # Low retention
    "clay": {"N": 60, "P": 35, "K": 50},  # High retention
    "peaty": {"N": 70, "P": 40, "K": 45},  # Very high N
    "chalky": {"N": 30, "P": 20, "K": 25},  # Moderate retention
    "silty": {"N": 55, "P": 35, "K": 45}   # High retention
}

# Dataset of crops with their nutrient consumption and restoration values
CROP_NUTRIENT_DATA = {
    "wheat": {"N": -20, "P": -10, "K": -15, "restores": False},
    "rice": {"N": -25, "P": -12, "K": -18, "restores": False},
    "maize": {"N": -30, "P": -15, "K": -20, "restores": False},
    "soybean": {"N": 10, "P": -5, "K": -5, "restores": True},  # Legume, restores nitrogen
    "groundnut": {"N": 15, "P": -8, "K": -10, "restores": True},  # Legume, restores nitrogen
}

def load_dataset(filepath):
    """
    Load the crop-soil-weather dataset from a JSON file.

    :param filepath: Path to the dataset file.
    :return: List of dataset entries.
    """
    with open(filepath, "r") as file:
        return [json.loads(line) for line in file]

def initialize_npk_from_dataset(soil_type, last_crop, residue_left, rainfall_mm, temperature_C, humidity_percent):
    """
    Initialize NPK values using the dataset and additional parameters.

    :param soil_type: Type of soil.
    :param last_crop: Last crop planted.
    :param residue_left: Whether residue was left.
    :param rainfall_mm: Rainfall in mm.
    :param temperature_C: Temperature in Celsius.
    :param humidity_percent: Humidity percentage.
    :return: Estimated NPK values.
    """
    dataset = load_dataset("d:\\agri fronti\\crop recommendation\\data\\crop_soil_weather_dataset.json")
    for entry in dataset:
        if (
            entry["soil_type"].lower() == soil_type.lower()
            and entry["last_crop"].lower() == last_crop.lower()
            and entry["residue_left"] == residue_left
        ):
            return {
                "N": entry["estimated_N"],
                "P": entry["estimated_P"],
                "K": entry["estimated_K"],
            }
    # Default fallback if no exact match is found
    return {"N": 0, "P": 0, "K": 0}

def initialize_npk_from_soil(soil_type: str):
    """
    Initialize NPK values based on soil type.

    :param soil_type: Type of soil (e.g., loamy, sandy, clayey).
    :return: Initial NPK values in kg/hectare.
    """
    base_values_by_soil = {
        "loamy": {"N": 50, "P": 30, "K": 40},
        "sandy": {"N": 20, "P": 10, "K": 15},
        "clay": {"N": 60, "P": 35, "K": 50},
        "peaty": {"N": 70, "P": 40, "K": 45},
        "chalky": {"N": 30, "P": 20, "K": 25},
        "silty": {"N": 55, "P": 35, "K": 45},
    }
    return base_values_by_soil.get(soil_type.lower(), {"N": 0, "P": 0, "K": 0})

def estimate_npk(soil_type: str, n: float, p: float, k: float, residue_left: bool, fertilizers: dict = None):
    """
    Estimate NPK values in soil based on soil type, last crop planted, crop residue, and fertilizer history.

    :param soil_type: Type of soil (e.g., loamy, sandy, clayey).
    :param n: Nitrogen depletion value of the last crop.
    :param p: Phosphorus depletion value of the last crop.
    :param k: Potassium depletion value of the last crop.
    :param residue_left: Whether crop residue was left on the field.
    :param fertilizers: Dictionary containing fertilizer contributions for N, P, and K.
    :return: Estimated NPK values in kg/hectare.
    """
    # Soil retention factors
    soil_factors = {
        "clay": 0.8,  # Clay retains nutrients better
        "sandy": 1.2,  # Sandy soil loses nutrients faster
        "loamy": 1.0,  # Loamy soil is neutral
    }
    soil_factor = soil_factors.get(soil_type.lower(), 1.0)

    # Adjust depletion based on crop residue
    residue_factor = 0.3 if residue_left else 0.0  # 30% nutrients restored if residue is left
    n_restored = n * residue_factor
    p_restored = p * residue_factor
    k_restored = k * residue_factor

    # Adjust depletion based on fertilizers
    n_fertilizer = fertilizers.get("N", 0.0) if fertilizers else 0.0
    p_fertilizer = fertilizers.get("P", 0.0) if fertilizers else 0.0
    k_fertilizer = fertilizers.get("K", 0.0) if fertilizers else 0.0

    # Final NPK estimation
    estimated_n = (n - n_restored + n_fertilizer) * soil_factor
    estimated_p = (p - p_restored + p_fertilizer) * soil_factor
    estimated_k = (k - k_restored + k_fertilizer) * soil_factor

    return {
        "N": max(estimated_n, 0),  # Ensure values are non-negative
        "P": max(estimated_p, 0),
        "K": max(estimated_k, 0),
    }
