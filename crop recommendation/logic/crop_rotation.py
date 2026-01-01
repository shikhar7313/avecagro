# Crop rotation suggestions based on biodiversity and soil health
CROP_ROTATION_RULES = {
    "wheat": ["soybean", "maize", "groundnut"],
    "rice": ["maize", "wheat", "groundnut"],
    "maize": ["soybean", "wheat", "rice"],
    "soybean": ["wheat", "maize", "rice"],
    "groundnut": ["wheat", "rice", "maize"],
}

def suggest_crop_rotation(crop_history: list):
    """
    Suggest crop rotations based on the farmer's crop history.

    :param crop_history: List of crops grown by the farmer in the past year.
    :return: List of suggested crops for rotation.
    """
    if not crop_history:
        return {"message": "No crop history available. Start with any crop."}

    last_crop = crop_history[-1].lower()
    suggestions = CROP_ROTATION_RULES.get(last_crop, [])
    return {
        "last_crop": last_crop,
        "suggested_rotations": suggestions,
        "message": "Crop rotation suggestions to prevent soil exhaustion and promote biodiversity."
    }
}
