import json
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))

from weather_helper import (
    LocationCoords,
    get_weather_data,
    get_mock_weather,
    prepare_npk_request,
    prepare_intercropping_request,
    prepare_multiheight_request,
    convert_yes_no_to_bool,
    WeatherData,
)

app = Flask(__name__)
CORS(app)

weather_cache = {}


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "weather-logic-server"}), 200


@app.route("/api/get-weather", methods=["POST"])
def get_weather():
    try:
        body = request.get_json() or {}
        use_mock = body.get("use_mock", False)

        # ---------- PRINT LOCATION ----------
        print("\n📍 /api/get-weather — Received Location:")
        print(f"   • Latitude:  {body.get('latitude')}")
        print(f"   • Longitude: {body.get('longitude')}")
        print(f"   • City:      {body.get('city')}")
        print(f"   • Name:      {body.get('name')}")
        print("---------------------------------------------")

        if use_mock:
            weather = get_mock_weather()
        else:
            location = LocationCoords(
                latitude=body.get("latitude"),
                longitude=body.get("longitude"),
                city=body.get("city"),
                name=body.get("name"),
            )
            weather = get_weather_data(location)
            if not weather:
                print("⚠️ Weather fetch failed, falling back to mock data")
                weather = get_mock_weather()

        return jsonify({
            "status": "success",
            "data": {
                "temperature_C": weather.temperature_C,
                "humidity_percent": weather.humidity_percent,
                "rainfall_mm": weather.rainfall_mm,
                "cloud_cover_percent": weather.cloud_cover_percent,
                "wind_speed_kmph": weather.wind_speed_kmph,
                "timezone": weather.timezone,
            }
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/prepare-npk-request", methods=["POST"])
def prepare_npk():
    try:
        body = request.get_json() or {}
        weather_dict = body.get("weather", {})

        # Reconstruct WeatherData from dict
        weather = WeatherData(
            temperature_C=weather_dict.get("temperature_C", 0),
            humidity_percent=weather_dict.get("humidity_percent", 0),
            rainfall_mm=weather_dict.get("rainfall_mm", 0),
            cloud_cover_percent=weather_dict.get("cloud_cover_percent", 0),
            wind_speed_kmph=weather_dict.get("wind_speed_kmph", 0),
            timezone=weather_dict.get("timezone", "UTC"),
        )

        npk_request = prepare_npk_request(
            soil_type=body.get("soil_type"),
            last_crop=body.get("last_crop"),
            residue_left=convert_yes_no_to_bool(body.get("residue_left")),
            weather=weather
        )

        return jsonify({
            "status": "success",
            "data": npk_request
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/prepare-intercropping-request", methods=["POST"])
def prepare_intercropping():
    try:
        body = request.get_json() or {}
        weather_dict = body.get("weather", {})

        weather = WeatherData(
            temperature_C=weather_dict.get("temperature_C", 0),
            humidity_percent=weather_dict.get("humidity_percent", 0),
            rainfall_mm=weather_dict.get("rainfall_mm", 0),
            cloud_cover_percent=weather_dict.get("cloud_cover_percent", 0),
            wind_speed_kmph=weather_dict.get("wind_speed_kmph", 0),
            timezone=weather_dict.get("timezone", "UTC"),
        )

        inter_request = prepare_intercropping_request(
            soil_type=body.get("soil_type"),
            last_crop=body.get("last_crop"),
            soil_n=float(body.get("soil_n", 0)),
            soil_p=float(body.get("soil_p", 0)),
            soil_k=float(body.get("soil_k", 0)),
            soil_moisture=float(body.get("soil_moisture", 0)),
            weather=weather
        )

        return jsonify({
            "status": "success",
            "data": inter_request
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/prepare-multiheight-request", methods=["POST"])
def prepare_multiheight():
    try:
        body = request.get_json() or {}
        weather_dict = body.get("weather", {})

        weather = WeatherData(
            temperature_C=weather_dict.get("temperature_C", 0),
            humidity_percent=weather_dict.get("humidity_percent", 0),
            rainfall_mm=weather_dict.get("rainfall_mm", 0),
            cloud_cover_percent=weather_dict.get("cloud_cover_percent", 0),
            wind_speed_kmph=weather_dict.get("wind_speed_kmph", 0),
            timezone=weather_dict.get("timezone", "UTC"),
        )

        multi_request = prepare_multiheight_request(
            soil_type=body.get("soil_type"),
            last_crop=body.get("last_crop"),
            region=body.get("region"),
            season=body.get("season"),
            soil_ph=float(body.get("soil_ph", 0)),
            weather=weather
        )

        return jsonify({
            "status": "success",
            "data": multi_request
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/full-prediction-workflow", methods=["POST"])
def full_workflow():
    try:
        body = request.get_json() or {}
        loc_dict = body.get("location", {})
        answers = body.get("answers", {})
        npk_result = body.get("npk_result", {})

        # ---------- PRINT LOCATION ----------
        print("\n📍 /api/full-prediction-workflow — Location Received:")
        print(f"   • Latitude:  {loc_dict.get('latitude')}")
        print(f"   • Longitude: {loc_dict.get('longitude')}")
        print(f"   • City:      {loc_dict.get('city')}")
        print(f"   • Name:      {loc_dict.get('name')}")
        print("---------------------------------------------")

        location = LocationCoords(
            latitude=loc_dict.get("latitude"),
            longitude=loc_dict.get("longitude"),
            city=loc_dict.get("city"),
            name=loc_dict.get("name"),
        )

        weather = get_weather_data(location)
        if not weather:
            print("⚠️ Weather fetch failed, using mock data")
            weather = get_mock_weather()

        # Prepare all requests
        npk_request = prepare_npk_request(
            soil_type=answers.get("soil_type"),
            last_crop=answers.get("last_crop"),
            residue_left=convert_yes_no_to_bool(answers.get("residue_left")),
            weather=weather
        )

        intercropping_request = prepare_intercropping_request(
            soil_type=answers.get("soil_type"),
            last_crop=answers.get("last_crop"),
            soil_n=float(npk_result.get("estimated_N", 0)),
            soil_p=float(npk_result.get("estimated_P", 0)),
            soil_k=float(npk_result.get("estimated_K", 0)),
            soil_moisture=float(answers.get("soil_moisture", 0)),
            weather=weather
        )

        multiheight_request = prepare_multiheight_request(
            soil_type=answers.get("soil_type"),
            last_crop=answers.get("last_crop"),
            region=answers.get("region"),
            season=answers.get("season"),
            soil_ph=float(answers.get("soil_ph", 0)),
            weather=weather
        )

        return jsonify({
            "status": "success",
            "weather": {
                "temperature_C": weather.temperature_C,
                "humidity_percent": weather.humidity_percent,
                "rainfall_mm": weather.rainfall_mm,
                "cloud_cover_percent": weather.cloud_cover_percent,
                "wind_speed_kmph": weather.wind_speed_kmph,
                "timezone": weather.timezone,
            },
            "npk_request": npk_request,
            "intercropping_request": intercropping_request,
            "multiheight_request": multiheight_request,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("🌤️ Starting Weather Logic Server on http://localhost:5005")
    app.run(debug=True, port=5005, host="0.0.0.0")