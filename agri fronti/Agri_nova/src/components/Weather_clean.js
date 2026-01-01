import React, { useState, useEffect, useCallback } from 'react';
import useText from '../hooks/useText';

import {
    Cloud,
    Sun,
    CloudRain,
    Wind,
    Droplets,
    Thermometer,
    Eye,
    Gauge,
    MapPin,
    RefreshCw,
    TrendingUp,
    TrendingDown,
    Compass,
    CloudDrizzle,
    CloudSnow,
    Zap,
    Calendar,
    Clock
} from 'lucide-react';
import { getLocationForWeather, getLocationDisplayName } from '../utils/locationUtils';

const Weather = () => {
    const { t } = useText();
    console.log('Weather component render/mount')
    const [currentWeather, setCurrentWeather] = useState(null);
    const [forecast, setForecast] = useState([]);
    const [loading, setLoading] = useState(true);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [selectedDay, setSelectedDay] = useState(0);
    const [viewMode, setViewMode] = useState('cards'); // 'cards' or 'list'
    const [temperatureUnit, setTemperatureUnit] = useState('C'); // 'C' or 'F'

// ✅ Convert "yes"/"no" (and variants) to true/false
const convertYesNoToBoolean = (value) => {
    if (!value) return false;

    const normalized = String(value).trim().toLowerCase();
    return normalized === "yes" || normalized === "y" || normalized === "true";
};

const fetchWeatherData = useCallback(async () => {
    setLoading(true);

    try {
        // 1️⃣ Fetch weather data
        const locationData = getLocationForWeather();
        console.log('Weather Dashboard: Using location data:', locationData);

        const weatherUrl = `/api/weather${locationData.query}`;
        console.log('Weather: fetching weather from', weatherUrl);
        const response = await fetch(weatherUrl);
        console.log('Weather: weather response status', response.status, response.ok);
        if (!response.ok) {
            const txt = await response.text().catch(() => '');
            throw new Error(`Weather request failed: ${response.status} ${txt}`);
        }
        const data = await response.json();
        console.log('Weather data received:', JSON.stringify(data, null, 2));

        setCurrentWeather(data.current);
        setForecast(data.daily || []);
        setLastUpdated(new Date());

    } catch (error) {
        console.error('Error fetching weather data:', error);
    } finally {
        setLoading(false);
    }
}, []);

    // Listen for questionnaire updates and automatically run weather fetch
    useEffect(() => {
        const handleQuestionnaireUpdate = () => {
            console.log('📋 Questionnaire updated - Fetching weather data');
            fetchWeatherData();
        };

        window.addEventListener('questionnaireUpdated', handleQuestionnaireUpdate);
        return () => window.removeEventListener('questionnaireUpdated', handleQuestionnaireUpdate);
    }, [fetchWeatherData]);

    // Listen for location updates
    useEffect(() => {
        const handleLocationUpdate = () => {
            console.log('Weather Dashboard: Location updated, refreshing weather data');
            fetchWeatherData();
        };

        window.addEventListener('userLocationUpdated', handleLocationUpdate);
        return () => window.removeEventListener('userLocationUpdated', handleLocationUpdate);
    }, [fetchWeatherData]);

    // Initial load - run weather fetch on component mount
    useEffect(() => {
        console.log('⚙️ Weather component mounted - Running initial weather fetch');
        fetchWeatherData();
    }, [fetchWeatherData]);

    const getWeatherIcon = (condition, size = 'w-8 h-8') => {
        const conditionLower = condition?.toLowerCase() || '';
        if (conditionLower.includes('sun') || conditionLower.includes('clear')) {
            return <Sun className={`${size} text-yellow-400`} />;
        } else if (conditionLower.includes('thunder') || conditionLower.includes('storm')) {
            return <Zap className={`${size} text-purple-400`} />;
        } else if (conditionLower.includes('rain') || conditionLower.includes('shower')) {
            return <CloudRain className={`${size} text-blue-400`} />;
        } else if (conditionLower.includes('drizzle')) {
            return <CloudDrizzle className={`${size} text-blue-300`} />;
        } else if (conditionLower.includes('snow') || conditionLower.includes('blizzard')) {
            return <CloudSnow className={`${size} text-blue-200`} />;
        } else if (conditionLower.includes('cloud') || conditionLower.includes('overcast')) {
            return <Cloud className={`${size} text-gray-400`} />;
        } else {
            return <Cloud className={`${size} text-gray-400`} />;
        }
    };

    const convertTemperature = (temp, fromUnit = 'C', toUnit = temperatureUnit) => {
        if (fromUnit === toUnit) return Math.round(temp);
        if (fromUnit === 'C' && toUnit === 'F') {
            return Math.round((temp * 9 / 5) + 32);
        } else if (fromUnit === 'F' && toUnit === 'C') {
            return Math.round((temp - 32) * 5 / 9);
        }
        return Math.round(temp);
    };

    const getWindDirection = (degrees) => {
        const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
        const index = Math.round(degrees / 22.5) % 16;
        return directions[index];
    };

    const getUVIndexColor = (uvi) => {
        if (uvi <= 2) return 'text-green-400';
        if (uvi <= 5) return 'text-yellow-400';
        if (uvi <= 7) return 'text-orange-400';
        if (uvi <= 10) return 'text-red-400';
        return 'text-purple-400';
    };

    const getUVIndexDescription = (uvi) => {
        if (uvi <= 2) return 'Low';
        if (uvi <= 5) return 'Moderate';
        if (uvi <= 7) return 'High';
        if (uvi <= 10) return 'Very High';
        return 'Extreme';
    };

    const formatDate = (timestamp) => {
        return new Date(timestamp * 1000).toLocaleDateString('en-US', {
            weekday: 'short',
            month: 'short',
            day: 'numeric'
        });
    };

    if (loading) {
        return (
            <div className="p-6 space-y-6" data-scroll-section>
                <div className="bg-white/10 backdrop-blur-md rounded-lg p-6 border border-white/20 h-32">
                    <div className="flex items-center justify-center h-full">
                        <div className="text-white">Loading weather data...</div>
                    </div>
                </div>
                <div className="bg-white/10 backdrop-blur-md rounded-lg p-6 border border-white/20 h-96">
                    <div className="flex items-center justify-center h-full">
                        <div className="text-white">Loading forecast...</div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 max-w-7xl mx-auto" data-scroll-section>
            {/* Enhanced Header */}
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 space-y-4 md:space-y-0">
                <div>
                    <h1 className="text-3xl font-bold text-white flex items-center">
                        <Cloud className="w-8 h-8 mr-3 text-blue-400" />
                        {t('weather.dashboard')}
                    </h1>
                    {currentWeather && (
                        <p className="text-gray-400 mt-1 flex items-center">
                            <MapPin className="w-4 h-4 mr-1" />
                            {getLocationDisplayName()}
                        </p>
                    )}
                </div>

                <div className="flex items-center space-x-4">
                    {/* Temperature Unit Toggle */}
                    <div className="flex bg-white/10 rounded-lg p-1">
                        <button
                            onClick={() => setTemperatureUnit('C')}
                            className={`px-3 py-1 rounded text-sm transition-all ${temperatureUnit === 'C'
                                ? 'bg-blue-500 text-white'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            °C
                        </button>
                        <button
                            onClick={() => setTemperatureUnit('F')}
                            className={`px-3 py-1 rounded text-sm transition-all ${temperatureUnit === 'F'
                                ? 'bg-blue-500 text-white'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            °F
                        </button>
                    </div>

                    {/* View Mode Toggle */}
                    <div className="flex bg-white/10 rounded-lg p-1">
                        <button
                            onClick={() => setViewMode('cards')}
                            className={`px-3 py-1 rounded text-sm transition-all ${viewMode === 'cards'
                                ? 'bg-blue-500 text-white'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            {t('weather.cards')}
                        </button>
                        <button
                            onClick={() => setViewMode('list')}
                            className={`px-3 py-1 rounded text-sm transition-all ${viewMode === 'list'
                                ? 'bg-blue-500 text-white'
                                : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            {t('weather.list')}
                        </button>
                    </div>

                    {/* Refresh Button */}
                    <button
                        onClick={fetchWeatherData}
                        className="flex items-center space-x-2 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 px-4 py-2 rounded-lg transition-all duration-200"
                    >
                        <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                        <span>{t('weather.refresh')}</span>
                    </button>
                </div>
            </div>

            {/* Enhanced Current Weather */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl transition-all duration-200">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-white flex items-center">
                        <Thermometer className="w-6 h-6 mr-2 text-blue-400" />
                        {t('weather.currentConditions')}
                        {forecast.length > 0 && selectedDay > 0 && (
                            <span className="ml-3 text-lg text-blue-400">
                                & {formatDate(forecast[selectedDay].dt)}
                            </span>
                        )}
                    </h2>
                    {lastUpdated && (
                        <div className="text-right">
                            <div className="text-gray-400 text-sm flex items-center">
                                <Clock className="w-4 h-4 mr-1" />
                                {t('weather.updated')}: {lastUpdated.toLocaleTimeString()}
                            </div>
                            <div className="text-gray-500 text-xs">
                                {lastUpdated.toLocaleDateString()}
                            </div>
                        </div>
                    )}
                </div>

                {currentWeather && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Main Temperature Display */}
                        <div className="lg:col-span-1">
                            <div className="text-center bg-white/5 rounded-lg p-6">
                                <div className="flex items-center justify-center mb-4">
                                    {selectedDay === 0
                                        ? getWeatherIcon(currentWeather.weather?.[0]?.main, 'w-16 h-16')
                                        : forecast.length > selectedDay
                                            ? getWeatherIcon(forecast[selectedDay].weather?.[0]?.main, 'w-16 h-16')
                                            : getWeatherIcon(currentWeather.weather?.[0]?.main, 'w-16 h-16')
                                    }
                                </div>

                                {selectedDay === 0 ? (
                                    /* Current Weather Display */
                                    <>
                                        <div className="text-5xl font-bold text-white mb-2">
                                            {convertTemperature(currentWeather.temp)}°{temperatureUnit}
                                        </div>
                                        <div className="text-lg text-gray-300 capitalize mb-2">
                                            {currentWeather.weather?.[0]?.description || currentWeather.weather?.[0]?.main}
                                        </div>
                                        <div className="text-sm text-gray-400 mb-2">
                                            Feels like {convertTemperature(currentWeather.feels_like)}°{temperatureUnit}
                                        </div>
                                    </>
                                ) : forecast.length > selectedDay ? (
                                    /* Selected Forecast Day Display */
                                    <>
                                        <div className="text-sm text-blue-400 mb-2 font-semibold">
                                            {selectedDay === 0 ? 'Today' : formatDate(forecast[selectedDay].dt)}
                                        </div>
                                        <div className="flex items-center justify-center space-x-3 mb-2">
                                            <div className="text-center">
                                                <TrendingUp className="w-4 h-4 text-red-400 mx-auto mb-1" />
                                                <div className="text-3xl font-bold text-white">
                                                    {convertTemperature(forecast[selectedDay].temp.max)}°
                                                </div>
                                                <div className="text-xs text-gray-500">High</div>
                                            </div>
                                            <div className="text-gray-500 text-2xl">/</div>
                                            <div className="text-center">
                                                <TrendingDown className="w-4 h-4 text-blue-400 mx-auto mb-1" />
                                                <div className="text-2xl font-bold text-gray-300">
                                                    {convertTemperature(forecast[selectedDay].temp.min)}°
                                                </div>
                                                <div className="text-xs text-gray-500">Low</div>
                                            </div>
                                        </div>
                                        <div className="text-lg text-gray-300 capitalize mb-2">
                                            {forecast[selectedDay].weather?.[0]?.description || forecast[selectedDay].weather?.[0]?.main}
                                        </div>
                                        <div className="text-sm text-gray-400 mb-2">
                                            Average: {convertTemperature(forecast[selectedDay].temp.day)}°{temperatureUnit}
                                        </div>
                                    </>
                                ) : null}

                                {currentWeather && (
                                    <div className="text-sm text-blue-400 mt-3 p-2 bg-blue-500/10 rounded">
                                        📍 {getLocationDisplayName()}
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Enhanced Weather Details */}
                        <div className="lg:col-span-2">
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                    <Droplets className="w-6 h-6 text-blue-400 mx-auto mb-2" />
                                    <div className="text-sm text-gray-400">Humidity</div>
                                    <div className="text-lg font-semibold text-white">
                                        {selectedDay === 0
                                            ? currentWeather.humidity
                                            : forecast.length > selectedDay
                                                ? forecast[selectedDay].humidity
                                                : currentWeather.humidity
                                        }%
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        {((selectedDay === 0 ? currentWeather.humidity : forecast.length > selectedDay ? forecast[selectedDay].humidity : currentWeather.humidity) > 70) ? 'High' :
                                            ((selectedDay === 0 ? currentWeather.humidity : forecast.length > selectedDay ? forecast[selectedDay].humidity : currentWeather.humidity) > 40) ? 'Moderate' : 'Low'}
                                    </div>
                                </div>

                                <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                    <div className="flex items-center justify-center mb-2">
                                        <Wind className="w-6 h-6 text-gray-400" />
                                        <Compass className="w-4 h-4 text-gray-500 ml-1" />
                                    </div>
                                    <div className="text-sm text-gray-400">Wind</div>
                                    <div className="text-lg font-semibold text-white">
                                        {Math.round((selectedDay === 0
                                            ? currentWeather.wind_speed
                                            : forecast.length > selectedDay
                                                ? forecast[selectedDay].wind_speed
                                                : currentWeather.wind_speed
                                        ) * 3.6)} km/h
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        {selectedDay === 0
                                            ? getWindDirection(currentWeather.wind_deg || 0)
                                            : forecast.length > selectedDay
                                                ? getWindDirection(forecast[selectedDay].wind_deg || 0)
                                                : getWindDirection(currentWeather.wind_deg || 0)
                                        }
                                    </div>
                                </div>

                                <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                    <Gauge className="w-6 h-6 text-yellow-400 mx-auto mb-2" />
                                    <div className="text-sm text-gray-400">Pressure</div>
                                    <div className="text-lg font-semibold text-white">
                                        {selectedDay === 0
                                            ? currentWeather.pressure
                                            : forecast.length > selectedDay
                                                ? forecast[selectedDay].pressure
                                                : currentWeather.pressure
                                        } mb
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        {((selectedDay === 0 ? currentWeather.pressure : forecast.length > selectedDay ? forecast[selectedDay].pressure : currentWeather.pressure) > 1013) ? 'High' : 'Low'}
                                    </div>
                                </div>

                                <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                    <Eye className="w-6 h-6 text-green-400 mx-auto mb-2" />
                                    <div className="text-sm text-gray-400">Visibility</div>
                                    <div className="text-lg font-semibold text-white">
                                        {selectedDay === 0
                                            ? currentWeather.visibility
                                            : forecast.length > selectedDay && forecast[selectedDay].clouds
                                                ? forecast[selectedDay].clouds
                                                : currentWeather.visibility
                                        } km
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        {((selectedDay === 0 ? currentWeather.visibility : forecast.length > selectedDay && forecast[selectedDay].clouds ? forecast[selectedDay].clouds : currentWeather.visibility) > 10) ? 'Excellent' :
                                            ((selectedDay === 0 ? currentWeather.visibility : forecast.length > selectedDay && forecast[selectedDay].clouds ? forecast[selectedDay].clouds : currentWeather.visibility) > 5) ? 'Good' : 'Poor'}
                                    </div>
                                </div>

                                <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                    <Sun className={`w-6 h-6 mx-auto mb-2 ${getUVIndexColor((selectedDay === 0 ? currentWeather.uvi : forecast.length > selectedDay ? forecast[selectedDay].uvi : currentWeather.uvi) || 0)}`} />
                                    <div className="text-sm text-gray-400">UV Index</div>
                                    <div className="text-lg font-semibold text-white">
                                        {(selectedDay === 0
                                            ? currentWeather.uvi
                                            : forecast.length > selectedDay
                                                ? forecast[selectedDay].uvi
                                                : currentWeather.uvi
                                        ) || 'N/A'}
                                    </div>
                                    <div className="text-xs text-gray-500 mt-1">
                                        {getUVIndexDescription((selectedDay === 0 ? currentWeather.uvi : forecast.length > selectedDay ? forecast[selectedDay].uvi : currentWeather.uvi) || 0)}
                                    </div>
                                </div>

                                {/* Precipitation Card - Only for forecast days */}
                                {selectedDay > 0 && forecast.length > selectedDay && (
                                    <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                        <CloudRain className="w-6 h-6 text-blue-400 mx-auto mb-2" />
                                        <div className="text-sm text-gray-400">Precipitation</div>
                                        <div className="text-lg font-semibold text-white">
                                            {forecast[selectedDay].rain > 0 ? `${forecast[selectedDay].rain}mm` : '0mm'}
                                        </div>
                                        <div className="text-xs text-gray-500 mt-1">
                                            {forecast[selectedDay].rain > 20 ? 'Heavy' :
                                                forecast[selectedDay].rain > 5 ? 'Moderate' :
                                                    forecast[selectedDay].rain > 0 ? 'Light' : 'None'}
                                        </div>
                                    </div>
                                )}

                                {/* Coordinates Card - Only for current weather */}
                                {selectedDay === 0 && currentWeather && (
                                    <div className="bg-white/5 rounded-lg p-4 text-center hover:bg-white/10 transition-all">
                                        <MapPin className="w-6 h-6 text-purple-400 mx-auto mb-2" />
                                        <div className="text-sm text-gray-400">Location</div>
                                        <div className="text-sm font-semibold text-white">
                                            {getLocationDisplayName()}
                                        </div>
                                        <div className="text-xs text-gray-500 mt-1">
                                            Current Location
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Enhanced 7-Day Forecast */}
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl transition-all duration-200">
                <div className="flex items-center justify-between mb-6">
                    <h2 className="text-2xl font-bold text-white flex items-center">
                        <Calendar className="w-6 h-6 mr-2 text-blue-400" />
                        7-Day Forecast
                    </h2>
                    <div className="text-sm text-gray-400">
                        {forecast.length} day{forecast.length !== 1 ? 's' : ''} available
                    </div>
                </div>

                {forecast.length > 0 ? (
                    viewMode === 'cards' ? (
                        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-7 gap-4">
                            {forecast.slice(0, 7).map((day, index) => (
                                <div
                                    key={index}
                                    className={`bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-all duration-200 cursor-pointer border-2 ${selectedDay === index ? 'border-blue-400' : 'border-transparent'
                                        }`}
                                    onClick={() => setSelectedDay(index)}
                                >
                                    <div className="text-center">
                                        <div className="text-sm font-medium text-gray-300 mb-2">
                                            {index === 0 ? 'Today' : formatDate(day.dt)}
                                        </div>

                                        <div className="flex items-center justify-center mb-3">
                                            {getWeatherIcon(day.weather?.[0]?.main)}
                                        </div>

                                        <div className="flex items-center justify-center space-x-1 mb-2">
                                            <TrendingUp className="w-4 h-4 text-red-400" />
                                            <span className="text-lg font-bold text-white">
                                                {convertTemperature(day.temp.max)}°
                                            </span>
                                        </div>
                                        <div className="flex items-center justify-center space-x-1 mb-3">
                                            <TrendingDown className="w-4 h-4 text-blue-400" />
                                            <span className="text-sm text-gray-400">
                                                {convertTemperature(day.temp.min)}°
                                            </span>
                                        </div>

                                        <div className="text-xs text-gray-400 capitalize mb-2">
                                            {day.weather?.[0]?.description || day.weather?.[0]?.main}
                                        </div>

                                        <div className="text-xs text-blue-400 mb-1">
                                            💧 {day.humidity}%
                                        </div>
                                        <div className="text-xs text-green-400 mb-1">
                                            🌪️ {Math.round(day.wind_speed * 3.6)} km/h
                                        </div>

                                        {day.rain > 0 && (
                                            <div className="text-xs text-blue-300 mt-2 p-1 bg-blue-500/20 rounded">
                                                🌧️ {day.rain}mm
                                            </div>
                                        )}

                                        {day.uvi && (
                                            <div className={`text-xs mt-2 ${getUVIndexColor(day.uvi)}`}>
                                                ☀️ UV: {day.uvi}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    ) : (
                        <div className="space-y-3">
                            {forecast.slice(0, 7).map((day, index) => (
                                <div
                                    key={index}
                                    className={`bg-gray-700 rounded-lg p-4 hover:bg-gray-600 transition-all duration-200 cursor-pointer border-l-4 ${selectedDay === index ? 'border-blue-400' : 'border-transparent'
                                        }`}
                                    onClick={() => setSelectedDay(index)}
                                >
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-4">
                                            <div className="text-sm font-medium text-gray-300 w-20">
                                                {index === 0 ? 'Today' : formatDate(day.dt)}
                                            </div>
                                            <div className="flex items-center space-x-2">
                                                {getWeatherIcon(day.weather?.[0]?.main, 'w-6 h-6')}
                                                <span className="text-gray-400 capitalize w-32 text-sm">
                                                    {day.weather?.[0]?.description}
                                                </span>
                                            </div>
                                        </div>

                                        <div className="flex items-center space-x-6">
                                            <div className="flex items-center space-x-2">
                                                <TrendingUp className="w-4 h-4 text-red-400" />
                                                <span className="text-white font-semibold">
                                                    {convertTemperature(day.temp.max)}°{temperatureUnit}
                                                </span>
                                                <span className="text-gray-500">/</span>
                                                <TrendingDown className="w-4 h-4 text-blue-400" />
                                                <span className="text-gray-400">
                                                    {convertTemperature(day.temp.min)}°{temperatureUnit}
                                                </span>
                                            </div>

                                            <div className="flex items-center space-x-4 text-xs">
                                                <span className="text-blue-400">💧 {day.humidity}%</span>
                                                <span className="text-green-400">🌪️ {Math.round(day.wind_speed * 3.6)}km/h</span>
                                                {day.rain > 0 && (
                                                    <span className="text-blue-300">🌧️ {day.rain}mm</span>
                                                )}
                                                {day.uvi && (
                                                    <span className={getUVIndexColor(day.uvi)}>☀️ {day.uvi}</span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )
                ) : (
                    <div className="text-center text-gray-400 py-12">
                        <Cloud className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                        <div className="text-lg">No forecast data available</div>
                        <div className="text-sm mt-2">Try refreshing or check your connection</div>
                    </div>
                )}
            </div>

        </div>
    );
};

export default Weather;
