const path = require('path');
require('dotenv').config(); // Load .env file
const fetch = require('node-fetch').default; // for external API calls
const express = require('express');
const fs = require('fs');
const cors = require('cors');

const app = express();
// Use SERVER_PORT environment variable for the API server to avoid conflict with CRA dev port
const PORT = process.env.SERVER_PORT || 4000;
const tasksFile = path.join(__dirname, '../src/data/dashboard/tasks.json');
const questionnaireFile = path.join(__dirname, '../src/data/questionaire.json');

app.use(cors());
app.use(express.json());
// Ensure data directory exists for weather caching
const weatherDataDir = path.join(__dirname, 'data');
const weatherDataFile = path.join(weatherDataDir, 'weather.json');
if (!fs.existsSync(weatherDataDir)) {
  fs.mkdirSync(weatherDataDir);
}

// Read tasks
app.get('/api/tasks', (req, res) => {
  fs.readFile(tasksFile, 'utf8', (err, data) => {
    if (err) return res.status(500).json({ error: 'Failed to read tasks' });
    try {
      const tasks = JSON.parse(data);
      res.json(tasks);
    } catch (e) {
      res.status(500).json({ error: 'Invalid JSON' });
    }
  });
});

// Update a specific task by ID
app.put('/api/tasks/:id', (req, res) => {
  const taskId = parseInt(req.params.id, 10);
  const updates = req.body;
  fs.readFile(tasksFile, 'utf8', (err, data) => {
    if (err) return res.status(500).json({ error: 'Failed to read tasks' });
    let tasks;
    try {
      tasks = JSON.parse(data);
    } catch (e) {
      return res.status(500).json({ error: 'Invalid JSON' });
    }
    const index = tasks.findIndex(t => t.id === taskId);
    if (index === -1) return res.status(404).json({ error: 'Task not found' });
    tasks[index] = { ...tasks[index], ...updates };
    fs.writeFile(tasksFile, JSON.stringify(tasks, null, 2), 'utf8', (err2) => {
      if (err2) return res.status(500).json({ error: 'Failed to write tasks' });
      res.json(tasks[index]);
    });
  });
});

// Cached weather endpoint
app.get('/api/weather/cached', (req, res) => {
  try {
    if (fs.existsSync(weatherDataFile)) {
      const raw = fs.readFileSync(weatherDataFile, 'utf8');
      const data = JSON.parse(raw);
      return res.json(data);
    } else {
      return res.status(404).json({ error: 'No cached weather data available' });
    }
  } catch (err) {
    console.error('Error reading cached weather:', err);
    return res.status(500).json({ error: 'Failed to read cached weather' });
  }
});
// Add a new task
app.post('/api/tasks', (req, res) => {
  const newTask = req.body;
  fs.readFile(tasksFile, 'utf8', (err, data) => {
    if (err) return res.status(500).json({ error: 'Failed to read tasks' });
    let tasks;
    try {
      tasks = JSON.parse(data);
    } catch (e) {
      return res.status(500).json({ error: 'Invalid JSON' });
    }
    newTask.id = tasks.length ? Math.max(...tasks.map(t => t.id)) + 1 : 1;
    tasks.push(newTask);
    fs.writeFile(tasksFile, JSON.stringify(tasks, null, 2), 'utf8', (err2) => {
      if (err2) return res.status(500).json({ error: 'Failed to write tasks' });
      res.status(201).json(newTask);
    });
  });
});

// Live weather endpoint (move out of nested routes)
// Live weather endpoint (move out of nested routes)
app.get('/api/weather', async (req, res) => {
  console.log('DEBUG: /api/weather called with query:', req.query);
  // WeatherAPI.com API key from .env file
  const apiKey = process.env.OPENWEATHER_API_KEY || "6551f75d6a534e3b8f4171328252309";

  const { lat, lon, city } = req.query;
  try {
    let current, daily;
    let location_query = '';

    if (lat && lon) {
      location_query = `${lat},${lon}`;
    } else if (city) {
      location_query = city;
    } else {
      return res.status(400).json({ error: 'Latitude/longitude or city required' });
    }

    // WeatherAPI.com endpoints
    const currentUrl = `https://api.weatherapi.com/v1/current.json?key=${apiKey}&q=${location_query}&aqi=yes`;
    const forecastUrl = `https://api.weatherapi.com/v1/forecast.json?key=${apiKey}&q=${location_query}&days=7&aqi=yes&alerts=yes`;

    console.log('DEBUG: Current Weather URL:', currentUrl);
    console.log('DEBUG: Forecast URL:', forecastUrl);

    const [currentResponse, forecastResponse] = await Promise.all([
      fetch(currentUrl),
      fetch(forecastUrl)
    ]);

    console.log('DEBUG: Current response status:', currentResponse.status);
    console.log('DEBUG: Forecast response status:', forecastResponse.status);

    if (currentResponse.status === 401 || forecastResponse.status === 401) {
      return res.status(401).json({ error: 'Unauthorized - Check your WeatherAPI key' });
    }

    if (!currentResponse.ok || !forecastResponse.ok) {
      console.error('DEBUG: Weather API not ok:', currentResponse.status, forecastResponse.status);
      return res.json({ current: {}, daily: [], major: [] });
    }

    const currentData = await currentResponse.json();
    const forecastData = await forecastResponse.json();

    console.log('DEBUG: Current data:', currentData);
    console.log('DEBUG: Forecast data keys:', Object.keys(forecastData));

    // Format current weather data to match expected frontend format
    current = {
      temp: currentData.current.temp_c,
      feels_like: currentData.current.feelslike_c,
      humidity: currentData.current.humidity,
      pressure: currentData.current.pressure_mb,
      wind_speed: currentData.current.wind_kph / 3.6, // Convert kph to m/s
      wind_deg: currentData.current.wind_degree,
      visibility: currentData.current.vis_km,
      uvi: currentData.current.uv,
      weather: [{
        main: currentData.current.condition.text,
        description: currentData.current.condition.text,
        icon: currentData.current.condition.icon
      }],
      location: {
        name: currentData.location.name,
        country: currentData.location.country,
        region: currentData.location.region,
        lat: currentData.location.lat,
        lon: currentData.location.lon
      }
    };

    // Format forecast data to match expected frontend format
    daily = forecastData.forecast.forecastday.map(day => ({
      dt: new Date(day.date).getTime() / 1000,
      temp: {
        min: day.day.mintemp_c,
        max: day.day.maxtemp_c,
        day: day.day.avgtemp_c
      },
      humidity: day.day.avghumidity,
      pressure: currentData.current.pressure_mb, // Daily pressure not available
      wind_speed: day.day.maxwind_kph / 3.6, // Convert kph to m/s
      wind_deg: 0, // Not available in daily data
      weather: [{
        main: day.day.condition.text,
        description: day.day.condition.text,
        icon: day.day.condition.icon
      }],
      rain: day.day.totalprecip_mm,
      clouds: day.day.avgvis_km,
      uvi: day.day.uv
    }));

    console.log('DEBUG: Number of forecast days returned:', daily.length);
    console.log('DEBUG: Forecast days:', daily.map(d => new Date(d.dt * 1000).toDateString()));
    // Major events filter
    const major = Array.isArray(daily)
      ? daily.filter(item => {
        const rainAmt = item.rain || 0;
        const maxTemp = item.temp?.max || 0;
        const minTemp = item.temp?.min || 0;
        const humidity = item.humidity || 0;
        const windSpeed = item.wind_speed || 0;
        const uvi = item.uvi || 0;
        const hasStorm = item.weather.some(w => w.main === 'Thunderstorm');
        const hasHail = item.weather.some(w => w.id === 906 || w.main === 'Hail');
        const isFrost = minTemp < 0;
        const isHeatWave = maxTemp > 35;
        const isColdSnap = maxTemp < 5;
        const isHighWind = windSpeed > 10;
        const isHumid = humidity > 90;
        const isUVIndexHigh = uvi > 8;
        return rainAmt > 20 || hasStorm || hasHail || isHeatWave || isFrost || isColdSnap || isHighWind || isHumid || isUVIndexHigh;
      })
      : [];
    // Cache weather data to file
    const weatherObj = { current, daily, major };
    try {
      fs.writeFileSync(weatherDataFile, JSON.stringify(weatherObj, null, 2));
    } catch (writeErr) {
      console.error('Failed to write weather cache:', writeErr);
    }
    // Return fresh data
    res.json(weatherObj);
  } catch (err) {
    console.error('Error in /api/weather:', err);
    return res.json({ current: {}, daily: [], major: [] });
  }
});

// Reverse geocoding via Nominatim (proxy to avoid CORS)
app.get('/api/reverse-geocode', async (req, res) => {
  const { lat, lon } = req.query;
  if (!lat || !lon) {
    return res.status(400).json({ error: 'Latitude and longitude required' });
  }
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`;
    const response = await fetch(url);
    if (!response.ok) {
      console.error('Reverse geocode failed', response.status);
      return res.status(response.status).json({ error: 'Failed to reverse geocode' });
    }
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Error in reverse-geocode:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Save user data endpoint
app.post('/api/save-user-data', (req, res) => {
  try {
    const userData = req.body;

    // Validate required fields
    if (!userData.username) {
      return res.status(400).json({ message: 'Username is required' });
    }

    // Create users directory if it doesn't exist
    const usersDir = path.join(__dirname, '../src/data/users');
    if (!fs.existsSync(usersDir)) {
      fs.mkdirSync(usersDir, { recursive: true });
    }

    // Save user data to individual file
    const userFile = path.join(usersDir, `${userData.username}.json`);

    // Add timestamp
    userData.timestamp = new Date().toISOString();

    fs.writeFileSync(userFile, JSON.stringify(userData, null, 2));

    console.log(`User data saved for: ${userData.username}`);
    res.json({ message: 'User data saved successfully' });

  } catch (error) {
    console.error('Error saving user data:', error);
    res.status(500).json({ message: 'Failed to save user data' });
  }
});

function readQuestionnaireEntries() {
  try {
    if (!fs.existsSync(questionnaireFile)) {
      fs.writeFileSync(questionnaireFile, '[]', 'utf8');
    }
    const existing = fs.readFileSync(questionnaireFile, 'utf8');
    return JSON.parse(existing);
  } catch (error) {
    console.error('Failed to read questionnaire file:', error);
    return [];
  }
}

app.post('/api/questionnaire', (req, res) => {
  const { username, answers, selectedCrop, recommendations = [], planId } = req.body || {};
  if (!username || !answers || !selectedCrop) {
    return res.status(400).json({ message: 'username, answers, and selectedCrop are required' });
  }

  const entry = {
    id: Date.now(),
    username,
    planId: planId || null,
    answers,
    selectedCrop,
    recommendations,
    savedAt: new Date().toISOString(),
  };

  const existingEntries = readQuestionnaireEntries();
  existingEntries.push(entry);

  try {
    fs.writeFileSync(questionnaireFile, JSON.stringify(existingEntries, null, 2));
    res.status(201).json({ message: 'Questionnaire saved', entry });
  } catch (error) {
    console.error('Failed to save questionnaire:', error);
    res.status(500).json({ message: 'Failed to save questionnaire' });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});

// Reverse geocoding via Nominatim (proxy to avoid CORS)
app.get('/api/reverse-geocode', async (req, res) => {
  const { lat, lon } = req.query;
  if (!lat || !lon) {
    return res.status(400).json({ error: 'Latitude and longitude required' });
  }
  try {
    const url = `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`;
    const response = await fetch(url);
    if (!response.ok) {
      console.error('Reverse geocode failed', response.status);
      return res.status(response.status).json({ error: 'Failed to reverse geocode' });
    }
    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error('Error in reverse-geocode:', err);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Save user data endpoint
app.post('/api/save-user-data', (req, res) => {
  try {
    const userData = req.body;

    // Validate required fields
    if (!userData.username) {
      return res.status(400).json({ message: 'Username is required' });
    }

    // Create users directory if it doesn't exist
    const usersDir = path.join(__dirname, '../src/data/users');
    if (!fs.existsSync(usersDir)) {
      fs.mkdirSync(usersDir, { recursive: true });
    }

    // Save user data to individual file
    const userFile = path.join(usersDir, `${userData.username}.json`);

    // Add timestamp
    userData.timestamp = new Date().toISOString();

    fs.writeFileSync(userFile, JSON.stringify(userData, null, 2));

    console.log(`User data saved for: ${userData.username}`);
    res.json({ message: 'User data saved successfully' });

  } catch (error) {
    console.error('Error saving user data:', error);
    res.status(500).json({ message: 'Failed to save user data' });
  }
});

app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
