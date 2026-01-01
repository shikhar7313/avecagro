import requests

# Your WeatherAPI key
API_KEY = "6551f75d6a534e3b8f4171328252309"

# City name (you can also use lat,long e.g. "28.61,77.23")
city = "Delhi"

# WeatherAPI endpoint for current weather
url = f"https://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no"

try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    location = data["location"]
    current = data["current"]

    print(f"📍 Location: {location['name']}, {location['region']}, {location['country']}")
    print(f"🕒 Local Time: {location['localtime']}")
    print(f"🌡️ Temperature: {current['temp_c']}°C / {current['temp_f']}°F")
    print(f"🌤️ Condition: {current['condition']['text']}")
    print(f"💧 Humidity: {current['humidity']}%")
    print(f"💨 Wind: {current['wind_kph']} kph, direction {current['wind_dir']}")
    print(f"☁️ Cloud Cover: {current['cloud']}%")

except requests.exceptions.RequestException as e:
    print("Error fetching data:", e)
