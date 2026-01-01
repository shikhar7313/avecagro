import React, { useState, useEffect, useCallback } from 'react';
import { Cloud, Droplets, CheckCircle, Satellite, RotateCcw } from 'lucide-react';
import useText from '../hooks/useText';
import { getLocationForWeather, getShortLocationName } from '../utils/locationUtils';

const API_BASE = process.env.REACT_APP_API_BASE || '';

const Header = ({ data, farmSizeAcres, showIoT = true, showDrone = true }) => {
  const { t } = useText();
  const [time, setTime] = useState(new Date());
  const [location, setLocation] = useState(getShortLocationName());
  const [weatherData, setWeatherData] = useState({
    condition: '',
    temperature: 0,
    humidity: 0,
    forecast: []
  });
  const [lastUpdatedTime, setLastUpdatedTime] = useState(new Date());

  const getRelativeTime = () => {
    const diffMs = Date.now() - lastUpdatedTime.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'just now';
    if (diffMins === 1) return '1 min ago';
    return `${diffMins} min ago`;
  };

  // Fetch weather data using stored location or fallback to Delhi
  const fetchWeather = useCallback(async () => {
    try {
      const locationData = getLocationForWeather();
      const locationName = getShortLocationName();

      console.log('DEBUG: Fetching weather with stored location:', locationData);

      const url = `${API_BASE}/api/weather${locationData.query}`;
      const res = await fetch(url);

      console.log('DEBUG: Fetch response status:', res.status);
      if (!res.ok) throw new Error('Weather fetch failed: ' + res.status);

      const data = await res.json();
      console.log('DEBUG: Weather API returned data:', data);

      const { current, daily } = data;
      const condition = current.weather?.[0]?.description || '';
      const temp = typeof current.temp === 'number' ? Math.round(current.temp) : 0;
      const hum = typeof current.humidity === 'number' ? current.humidity : 0;
      const forecastArr = Array.isArray(daily)
        ? daily.map(d => ({ high: Math.round(d.temp.max), low: Math.round(d.temp.min) }))
        : [];

      setWeatherData({ condition, temperature: temp, humidity: hum, forecast: forecastArr });
      setLocation(locationName);
      setLastUpdatedTime(new Date());

    } catch (e) {
      console.error('DEBUG: fetchWeather error:', e);
    }
  }, []);

  // Live clock update
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Initial weather load on mount
  useEffect(() => {
    fetchWeather();
  }, [fetchWeather]);

  // Listen for user location updates
  useEffect(() => {
    const handleLocationUpdate = (event) => {
      console.log('Header: User location updated', event.detail);
      setLocation(getShortLocationName());
      fetchWeather();
    };

    const handleLocationCleared = () => {
      console.log('Header: User location cleared');
      setLocation(getShortLocationName());
      fetchWeather();
    };

    window.addEventListener('userLocationUpdated', handleLocationUpdate);
    window.addEventListener('userLocationCleared', handleLocationCleared);

    return () => {
      window.removeEventListener('userLocationUpdated', handleLocationUpdate);
      window.removeEventListener('userLocationCleared', handleLocationCleared);
    };
  }, [fetchWeather]);

  // Format date and time
  const formattedDate = time.toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric' });
  const formattedTime = time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });
  const todayForecast = Array.isArray(weatherData.forecast) && weatherData.forecast.length > 0
    ? weatherData.forecast[0]
    : null;

  // System status
  const status = data.systemStatus || { devicesOnline: true, dronesSynced: true, lastUpdated: '3 min ago' };

  return (
    <header className="glass-card m-4 mb-0 px-6 py-2 shadow-lg rounded-b-3xl flex items-center justify-between">
      {/* Location & Weather */}
      <div className="flex items-center space-x-2 text-sm text-gray-100">
        <span>📍 {location}</span>
        <div className="flex items-center space-x-1">
          <Cloud className="w-4 h-4" />
          <span>{weatherData.condition} — {weatherData.temperature}°C</span>
        </div>
        <div className="flex items-center space-x-1">
          <Droplets className="w-3 h-3 text-gray-400" title={`${t('header.humidity')}: ${weatherData.humidity}%`} />
          <span className="text-xs text-gray-400">{weatherData.humidity}%</span>
        </div>
        {typeof farmSizeAcres === 'number' && farmSizeAcres > 0 && (
          <span className="text-xs text-emerald-300">🌾 {farmSizeAcres.toFixed(1)} acres</span>
        )}
      </div>

      {/* Live Date & Time */}
      <div className="text-sm text-gray-100 flex items-center">
        <span>🕒 {formattedDate} — {formattedTime} IST</span>
      </div>

      {/* System Status */}
      <div className="flex items-center space-x-4">
        {showIoT && (
          <span className={status.devicesOnline ? 'text-green-400 flex items-center space-x-1 text-sm' : 'text-red-400 flex items-center space-x-1 text-sm'}>
            <CheckCircle className="w-4 h-4" />
            <span>{t('header.devicesOnline')}</span>
          </span>
        )}
        {showDrone && (
          <span className={status.dronesSynced ? 'text-green-400 flex items-center space-x-1 text-sm' : 'text-yellow-400 flex items-center space-x-1 text-sm'}>
            <Satellite className="w-4 h-4" />
            <span>{t('header.dronesSynced')}</span>
          </span>
        )}
        <button onClick={fetchWeather} className="flex items-center space-x-1 text-sm text-gray-100 hover:text-indigo-300">
          <RotateCcw className="w-5 h-5" />
          <span>⛅ {t('header.updated')} {getRelativeTime()}</span>
        </button>
      </div>
    </header>
  );
};

export default Header;
