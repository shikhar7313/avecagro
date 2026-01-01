import React, { useRef, useState, useEffect } from 'react';
import { Cloud, Droplets, Wind, Eye, MapPin } from 'lucide-react';
import { RotateCcw } from 'lucide-react';
import gsap from 'gsap';
import { addUserLocation, exportLocationsToConsole, getLocationForWeather, getUserLocations } from '../../utils/locationUtils';

// Use your server's API endpoint instead of direct weather API calls
const API_BASE = process.env.REACT_APP_API_BASE || '';

const WeatherCard = ({ data }) => {
  const cardRef = useRef();
  // Real-time weather state
  const [weather, setWeather] = useState({
    temperature: null,
    condition: '',
    humidity: null,
    windSpeed: null,
    forecast: []
  });
  const [lastUpdatedTime, setLastUpdatedTime] = useState(new Date());
  const [locationStatus, setLocationStatus] = useState('idle'); // idle, loading, success, error
  const [showLocationButton, setShowLocationButton] = useState(false);

  // Helper to show relative last-update time
  const getRelativeTime = () => {
    const diffMs = Date.now() - lastUpdatedTime.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'just now';
    if (diffMins === 1) return '1 min ago';
    return `${diffMins} min ago`;
  };

  // Handle location capture
  const handleEnterLocation = async () => {
    setLocationStatus('loading');

    if (!navigator.geolocation) {
      alert('Geolocation is not supported by this browser.');
      setLocationStatus('error');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const { latitude, longitude } = position.coords;

          // Reverse geocode to get address details
          const reverseGeocodeResponse = await fetch(`/api/reverse-geocode?lat=${latitude}&lon=${longitude}`);
          const geocodeData = await reverseGeocodeResponse.json();

          const locationData = {
            coordinates: {
              latitude,
              longitude
            },
            address: geocodeData.address || {},
            displayName: geocodeData.display_name || `${latitude.toFixed(4)}, ${longitude.toFixed(4)}`,
            city: geocodeData.address?.city || geocodeData.address?.town || geocodeData.address?.village || 'Unknown',
            state: geocodeData.address?.state || 'Unknown',
            country: geocodeData.address?.country || 'Unknown',
            accuracy: position.coords.accuracy
          };

          // Store location data using utility function
          const savedLocation = addUserLocation(locationData);

          if (savedLocation) {
            console.log('Location saved:', savedLocation);
            alert(`Location saved: ${locationData.displayName}`);
            setLocationStatus('success');

            // Export all locations to console for debugging
            exportLocationsToConsole();

            // Update weather with new location
            setTimeout(() => {
              fetchWeatherFromStoredLocation();
              setLocationStatus('idle');
            }, 1000);
          } else {
            throw new Error('Failed to save location');
          }

        } catch (error) {
          console.error('Error saving location:', error);
          alert('Failed to save location data');
          setLocationStatus('error');
          setTimeout(() => setLocationStatus('idle'), 2000);
        }
      },
      (error) => {
        console.error('Geolocation error:', error);
        let errorMessage = 'Failed to get location. ';
        switch (error.code) {
          case error.PERMISSION_DENIED:
            errorMessage += 'Please allow location access.';
            break;
          case error.POSITION_UNAVAILABLE:
            errorMessage += 'Location information unavailable.';
            break;
          case error.TIMEOUT:
            errorMessage += 'Location request timed out.';
            break;
          default:
            errorMessage += 'Unknown error occurred.';
            break;
        }
        alert(errorMessage);
        setLocationStatus('error');
        setTimeout(() => setLocationStatus('idle'), 2000);
      }
    );
  };

  // Fetch weather using stored location data
  const fetchWeatherFromStoredLocation = async () => {
    try {
      const locationData = getLocationForWeather();
      console.log('WeatherCard: Using location data:', locationData);

      const response = await fetch(`/api/weather${locationData.query}`);
      const { current, daily } = await response.json();

      console.log('WeatherCard received data:', { current, daily });

      const forecastArr = Array.isArray(daily)
        ? daily.slice(1, 4).map(d => ({
          day: new Date(d.dt * 1000).toLocaleDateString('en-US', { weekday: 'short' }),
          high: Math.round(d.temp.max),
          low: Math.round(d.temp.min),
          condition: d.weather?.[0]?.main || ''
        }))
        : [];

      setWeather({
        temperature: Math.round(current.temp),
        condition: current.weather?.[0]?.description || current.weather?.[0]?.main || '',
        humidity: current.humidity,
        windSpeed: Math.round(current.wind_speed * 3.6), // Convert m/s to km/h
        forecast: forecastArr
      });
      setLastUpdatedTime(new Date());
    } catch (error) {
      console.error('WeatherCard fetch failed:', error);
    }
  };

  // Fetch current weather and forecast using stored location
  const fetchLocalWeather = () => {
    fetchWeatherFromStoredLocation();
  };

  // Check if user has any saved locations
  useEffect(() => {
    const locations = getUserLocations();
    setShowLocationButton(locations.length === 0);
  }, []);

  // initial load
  useEffect(() => {
    fetchWeatherFromStoredLocation();
  }, []);

  // Listen for location updates
  useEffect(() => {
    const handleLocationUpdate = () => {
      fetchWeatherFromStoredLocation();
      // Hide the location button once location is saved
      setShowLocationButton(false);
    };

    window.addEventListener('userLocationUpdated', handleLocationUpdate);
    return () => window.removeEventListener('userLocationUpdated', handleLocationUpdate);
  }, []);

  const handleMouseMove = (e) => {
    const rect = cardRef.current.getBoundingClientRect();
    gsap.to(cardRef.current, { overwrite: 'auto', scale: 1.03, boxShadow: '0 15px 30px rgba(16,185,129,0.7), 0 0 15px rgba(16,185,129,0.5)', transformPerspective: 600, transformOrigin: 'center', ease: 'power3.out', duration: 0.3 });
  };

  const handleMouseLeave = () => {
    gsap.to(cardRef.current, { overwrite: 'auto', boxShadow: '0 4px 10px rgba(0,0,0,0.2)', ease: 'power3.out', duration: 0.6 });
  };

  if (!data) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-xl font-semibold text-white mb-4">Weather</h3>
        <p className="text-gray-400">Weather data unavailable</p>
      </div>
    );
  }

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="glass-card p-6 relative"
      style={{
        transformPerspective: 600, transformOrigin: 'center', ease: 'power3.out', duration: 0.3
      }}
    >
      {/* Location Button - Top Right - Only for first-time users */}
      {showLocationButton && (
        <div className="absolute top-4 right-4">
          <button
            onClick={handleEnterLocation}
            disabled={locationStatus === 'loading'}
            className={`flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-medium transition-all ${locationStatus === 'loading'
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : locationStatus === 'success'
                ? 'bg-green-600 text-white'
                : locationStatus === 'error'
                  ? 'bg-red-600 text-white'
                  : 'bg-indigo-600 hover:bg-indigo-500 text-white'
              }`}
          >
            {locationStatus === 'loading' ? (
              <>
                <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white"></div>
                <span>Getting...</span>
              </>
            ) : locationStatus === 'success' ? (
              <>
                <span>✓</span>
                <span>Saved</span>
              </>
            ) : locationStatus === 'error' ? (
              <>
                <span>✗</span>
                <span>Error</span>
              </>
            ) : (
              <>
                <MapPin className="w-3 h-3" />
                <span>Enter Location</span>
              </>
            )}
          </button>
        </div>
      )}

      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-400 to-blue-600 rounded-lg flex items-center justify-center">
            <Cloud className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-400">Weather Today</h3>
            <div className="flex items-center space-x-2">
              <p className="text-2xl font-bold text-white">
                {weather.temperature != null ? `${weather.temperature}°C` : 'Loading...'}
              </p>
              <button onClick={fetchLocalWeather} className="text-gray-100 hover:text-indigo-300"><RotateCcw /></button>
            </div>
          </div>
        </div>
      </div>

      <div className="mb-4">
        <p className="text-sm text-gray-300 capitalize">
          {weather.condition || 'Loading...'}
        </p>
        <p className="text-xs text-gray-400">Current Conditions</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center space-x-2">
          <Droplets className="w-4 h-4 text-blue-400" />
          <div>
            <p className="text-xs text-gray-400">Humidity</p>
            <p className="text-sm font-semibold text-white">
              {weather.humidity != null ? `${weather.humidity}%` : '–'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Wind className="w-4 h-4 text-gray-400" />
          <div>
            <p className="text-xs text-gray-400">Wind</p>
            <p className="text-sm font-semibold text-white">
              {weather.windSpeed != null ? `${weather.windSpeed} km/h` : '–'}
            </p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Eye className="w-4 h-4 text-orange-400" />
          <div>
            <p className="text-xs text-gray-400">High</p>
            <p className="text-sm font-semibold text-white">{weather.forecast[0]?.high}°</p>
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <Eye className="w-4 h-4 text-blue-400" />
          <div>
            <p className="text-xs text-gray-400">Low</p>
            <p className="text-sm font-semibold text-white">{weather.forecast[0]?.low}°</p>
          </div>
        </div>
      </div>

      {weather.forecast?.length > 0 && (
        <div className="border-t border-gray-700 pt-4">
          <h4 className="text-sm font-medium text-gray-300 mb-3">3-Day Forecast</h4>
          <div className="grid grid-cols-3 gap-2">
            {weather.forecast.map((day, index) => (
              <div key={index} className="text-center p-2 rounded bg-gray-700/30">
                <div className="text-xs text-gray-400">{day.day}</div>
                <div className="text-sm text-white font-medium">{day.high}°/{day.low}°</div>
                <div className="text-xs text-gray-400">{day.condition}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default WeatherCard;
