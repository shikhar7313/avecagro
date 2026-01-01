# Weather Logic Helper & Server

This directory contains Python helpers for weather data fetching and prediction request preparation.

## Files

- **`weather_helper.py`** – Core logic for:
  - Fetching weather from OpenWeather API
  - Geocoding city names to coordinates
  - Preparing request payloads for NPK, intercropping, and multiheight predictions
  - Mock data fallback for offline scenarios

- **`weather_server.py`** – Flask HTTP server that exposes weather_helper functions as REST endpoints
  - Allows the TSX frontend to call weather logic from Python
  - Handles geolocation, API key management, and data transformation

## Setup

### 1. Install Dependencies

```bash
pip install flask flask-cors requests
```

### 2. Set Environment Variables (Optional)

```bash
export OPENWEATHER_API_KEY="your_api_key_here"
```

If not set, defaults to the bundled API key.

## Running the Server

### From Command Line (Windows CMD)

```cmd
cd c:\shikhar(D drive)\D drive\avecagro\agri fronti\ansh\agri fronti\Agri_nova\src\hooks
python weather_server.py
```

Server will start at `http://localhost:5005`

### From Python Programmatically

```python
from weather_server import app

if __name__ == "__main__":
    app.run(debug=True, port=5005, host="0.0.0.0")
```

## API Endpoints

### 1. Health Check

```
GET /health
```

Response:
```json
{
  "status": "ok",
  "service": "weather-logic-server"
}
```

### 2. Get Weather Data

```
POST /api/get-weather

Request body:
{
  "latitude": 28.6139,
  "longitude": 77.2090,
  "city": "Delhi",           // optional fallback
  "name": "Delhi",           // optional fallback
  "use_mock": false          // force mock data
}

Response:
{
  "status": "success",
  "data": {
    "temperature_C": 28,
    "humidity_percent": 65,
    "rainfall_mm": 2.5,
    "cloud_cover_percent": 40,
    "wind_speed_kmph": 18,
    "timezone": "UTC"
  }
}
```

### 3. Prepare NPK Request

```
POST /api/prepare-npk-request

Request body:
{
  "soil_type": "Loamy",
  "last_crop": "Paddy",
  "residue_left": true,
  "weather": {
    "temperature_C": 28,
    "humidity_percent": 65,
    "rainfall_mm": 2.5,
    "cloud_cover_percent": 40,
    "wind_speed_kmph": 18,
    "timezone": "UTC"
  }
}

Response:
{
  "status": "success",
  "data": { /* formatted NPK prediction request */ }
}
```

### 4. Prepare Intercropping Request

```
POST /api/prepare-intercropping-request

Request body:
{
  "soil_type": "Loamy",
  "last_crop": "Paddy",
  "soil_n": 150,
  "soil_p": 25,
  "soil_k": 120,
  "soil_moisture": 35,
  "weather": { /* WeatherData */ }
}

Response:
{
  "status": "success",
  "data": { /* formatted intercropping prediction request */ }
}
```

### 5. Prepare Multiheight Request

```
POST /api/prepare-multiheight-request

Request body:
{
  "soil_type": "Loamy",
  "last_crop": "Paddy",
  "region": "Punjab",
  "season": "Rabi",
  "soil_ph": 6.5,
  "weather": { /* WeatherData */ }
}

Response:
{
  "status": "success",
  "data": { /* formatted multiheight prediction request */ }
}
```

### 6. Full Workflow (All-in-One)

```
POST /api/full-prediction-workflow

Request body:
{
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "city": "Delhi"
  },
  "answers": {
    "soil_type": "Loamy",
    "last_crop": "Paddy",
    "residue_left": true,
    "soil_moisture": 35,
    "region": "Punjab",
    "season": "Rabi",
    "soil_ph": 6.5
  },
  "npk_result": {
    "estimated_N": 150,
    "estimated_P": 25,
    "estimated_K": 120
  }
}

Response:
{
  "status": "success",
  "weather": { /* WeatherData */ },
  "npk_request": { /* NPK prediction request */ },
  "intercropping_request": { /* intercropping prediction request */ },
  "multiheight_request": { /* multiheight prediction request */ }
}
```

## Using from useWeatherLogic.tsx

### Option A: Replace Direct API Calls

Instead of calling OpenWeather and prediction endpoints directly from TSX, call the weather_server:

```typescript
// Before: Direct API calls in TSX
const weatherResponse = await fetch("https://api.openweathermap.org/data/2.5/onecall?...");

// After: Call weather server
const weatherResponse = await fetch("http://localhost:5005/api/get-weather", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    latitude: 28.6139,
    longitude: 77.2090,
    city: "Delhi"
  })
});
```

### Option B: Use Full Workflow Endpoint

When you have all questionnaire answers and NPK results, call the all-in-one endpoint:

```typescript
const response = await fetch("http://localhost:5005/api/full-prediction-workflow", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    location: { city: "Delhi" },
    answers: { 
      soil_type: "Loamy",
      last_crop: "Paddy",
      // ... other fields
    },
    npk_result: { estimated_N: 150, ... }
  })
});
```

## Testing

### Test the Python module directly:

```bash
python weather_helper.py
```

This will run example code that fetches weather for Delhi and prepares all three prediction request types.

### Test the server with curl:

```bash
# Health check
curl http://localhost:5005/health

# Get weather
curl -X POST http://localhost:5005/api/get-weather \
  -H "Content-Type: application/json" \
  -d '{"city": "Delhi"}'

# Prepare NPK request
curl -X POST http://localhost:5005/api/prepare-npk-request \
  -H "Content-Type: application/json" \
  -d '{
    "soil_type": "Loamy",
    "last_crop": "Paddy",
    "residue_left": true,
    "weather": {
      "temperature_C": 28,
      "humidity_percent": 65,
      "rainfall_mm": 2.5,
      "cloud_cover_percent": 40,
      "wind_speed_kmph": 18,
      "timezone": "UTC"
    }
  }'
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"

Install Flask: `pip install flask flask-cors requests`

### "Connection refused on localhost:5005"

Make sure the weather_server is running:
```bash
python weather_server.py
```

### "OpenWeather API error"

- Check your API key (set `OPENWEATHER_API_KEY` env var)
- Check internet connectivity
- The server will fall back to mock data if the API fails
- Enable mock mode: set `"use_mock": true` in request

### "Geocoding failed for city"

- City name may not be recognized by OpenWeather
- Try using explicit coordinates instead
- Server falls back to hardcoded Delhi coordinates if geocoding fails

## Architecture Benefits

✅ **Separation of Concerns** – Python handles weather/data logic, TSX handles UI
✅ **Reusable** – Weather helper can be called from other backends/scripts
✅ **Testable** – Easy to test Python logic independently  
✅ **Centralized API Keys** – Hide API keys on backend, not exposed in frontend
✅ **Fallback Support** – Mock data and multiple data source options
✅ **Type Safety** – Python dataclasses for structure, TypeScript for frontend

## Future Enhancements

- Caching weather responses to reduce API calls
- Support for other weather APIs (e.g., AccuWeather)
- Database logging of predictions and actual vs. predicted outcomes
- Batch processing for multiple locations
- WebSocket real-time weather updates
