import os
import json
import requests
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# -----------------------------
# Config
# -----------------------------
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "d4e19f2ed5978ad4559762a33f309258")
WEATHER_CACHE: Dict[str, Any] = {}
CACHE_DURATION_SECONDS = 600  # 10 minutes

# -----------------------------
# Data Classes
# -----------------------------
@dataclass
class LocationCoords:
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: Optional[str] = None
    name: Optional[str] = None

    def query(self) -> str:
        if self.latitude is not None and self.longitude is not None:
            return f"?lat={self.latitude}&lon={self.longitude}"
        return f"?city={self.city or self.name or 'Delhi'}"


@dataclass
class WeatherData:
    temperature_C: float
    humidity_percent: float
    rainfall_mm: float
    cloud_cover_percent: float
    wind_speed_kmph: float
    timezone: str

# -----------------------------
# OpenWeather Helpers
# -----------------------------
def get_openweather_data(latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
    """Fetch current weather from free OpenWeather /weather API."""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&units=metric&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ OpenWeather API error: {e}")
        return None


def geocode_city(city: str) -> Optional[tuple[float, float]]:
    """Get latitude & longitude from city name."""
    url = f"https://api.openweathermap.org/geo/1.0/direct?q={city}&limit=1&appid={OPENWEATHER_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data and len(data) > 0:
            return (data[0]["lat"], data[0]["lon"])
    except Exception as e:
        print(f"❌ Geocoding error for '{city}': {e}")
    return None


def parse_weather_response(weather_dict: Dict[str, Any]) -> WeatherData:
    """Parse current weather API response."""
    main = weather_dict.get("main", {})
    wind = weather_dict.get("wind", {})
    clouds = weather_dict.get("clouds", {})
    rain = weather_dict.get("rain", {}).get("1h", 0)  # mm

    wind_speed_kmph = float(wind.get("speed", 0)) * 3.6  # m/s -> km/h
    timezone_offset = weather_dict.get("timezone", 0)  # seconds from UTC
    timezone_str = f"UTC{timezone_offset//3600:+d}"

    return WeatherData(
        temperature_C=float(main.get("temp", 0)),
        humidity_percent=float(main.get("humidity", 0)),
        rainfall_mm=float(rain),
        cloud_cover_percent=float(clouds.get("all", 0)),
        wind_speed_kmph=wind_speed_kmph,
        timezone=timezone_str
    )

# -----------------------------
# Weather Fetching Logic
# -----------------------------
def get_weather_data(location: LocationCoords) -> Optional[WeatherData]:
    """Fetch weather by coordinates or city. Fallback to Delhi mock."""
    if location.latitude is not None and location.longitude is not None:
        weather_dict = get_openweather_data(location.latitude, location.longitude)
        if weather_dict:
            return parse_weather_response(weather_dict)

    city = location.city or location.name or "Delhi"
    coords = geocode_city(city)
    if coords:
        weather_dict = get_openweather_data(coords[0], coords[1])
        if weather_dict:
            return parse_weather_response(weather_dict)

    print("⚠️ Using hardcoded Delhi weather data")
    return get_mock_weather()

def get_mock_weather() -> WeatherData:
    """Return static weather data for fallback."""
    return WeatherData(
        temperature_C=28,
        humidity_percent=65,
        rainfall_mm=2.5,
        cloud_cover_percent=40,
        wind_speed_kmph=18,
        timezone="Asia/Kolkata",
    )

# -----------------------------
# Request Preparers
# -----------------------------
def prepare_npk_request(
    soil_type: str,
    last_crop: str,
    residue_left: bool,
    weather: WeatherData
) -> Dict[str, Any]:
    return {
        "soil_type": soil_type or "unknown",
        "last_crop": last_crop or "unknown",
        "residue_left": bool(residue_left),
        "rainfall_mm": weather.rainfall_mm,
        "temperature_C": weather.temperature_C,
        "humidity_percent": weather.humidity_percent,
        "cloud_cover_percent": weather.cloud_cover_percent,
        "wind_speed_kmph": weather.wind_speed_kmph,
    }

def prepare_intercropping_request(
    soil_type: str,
    last_crop: str,
    soil_n: float,
    soil_p: float,
    soil_k: float,
    soil_moisture: float,
    weather: WeatherData
) -> Dict[str, Any]:
    return {
        "crop1": last_crop,
        "soil_type": soil_type,
        "Soil_N": soil_n,
        "Soil_P": soil_p,
        "Soil_K": soil_k,
        "Rainfall_mm": weather.rainfall_mm,
        "HumidityPercent": weather.humidity_percent,
        "Temperature_C": weather.temperature_C,
        "Cloud_Cover_Percent": weather.cloud_cover_percent,
        "Wind_Speed_kmph": weather.wind_speed_kmph,
        "Soil_Moisture_Percent": float(soil_moisture or 0),
        "Sunlight_Hours_per_day": 8,
    }

def prepare_multiheight_request(
    soil_type: str,
    last_crop: str,
    region: str,
    season: str,
    soil_ph: float,
    weather: WeatherData
) -> Dict[str, Any]:
    return {
        "user_crop": last_crop,
        "region": region or weather.timezone,
        "season": season,
        "soil_type": soil_type,
        "soil_ph": float(soil_ph or 0),
        "rainfall": weather.rainfall_mm,
        "temperature": weather.temperature_C,
        "humidity": weather.humidity_percent,
    }

# -----------------------------
# Utility
# -----------------------------
def convert_yes_no_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in ("yes", "y", "true", "1")

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    loc = LocationCoords(city="Delhi")
    weather = get_weather_data(loc)
    if not weather:
        print("❌ Failed to fetch weather, using mock data")
        weather = get_mock_weather()
    
    print(f"✅ Weather data: {asdict(weather)}")
