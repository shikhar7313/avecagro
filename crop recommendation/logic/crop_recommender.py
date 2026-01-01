from crop_recommendation.logic.npk_estimator import load_dataset

def recommend_crops_from_dataset(npk, temperature, humidity, crop_database_filepath):
    """
    Recommend crops based on NPK levels, temperature, and humidity using the dataset.

    :param npk: Current NPK values.
    :param temperature: Current temperature in Celsius.
    :param humidity: Current humidity percentage.
    :param crop_database_filepath: Path to the crop database file.
    :return: Top 3 recommended crops with scores.
    """
    dataset = load_dataset(crop_database_filepath)
    scores = {}
    N, P, K = npk["N"], npk["P"], npk["K"]

    for crop in dataset:
        match_score = 100
        match_score -= abs(N - crop["estimated_N"]) * 0.3
        match_score -= abs(P - crop["estimated_P"]) * 0.4
        match_score -= abs(K - crop["estimated_K"]) * 0.3
        if humidity > 75 and temperature > 28:
            match_score -= 10  # High pest risk
        elif humidity > 50:
            match_score -= 5  # Moderate pest risk
        scores[crop["last_crop"]] = match_score

    top_crops = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    return [{"crop": crop, "score": score} for crop, score in top_crops]
