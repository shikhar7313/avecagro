import React, { useState, useEffect } from 'react';
import {
    Settings as SettingsIcon,
    Globe,
    Bell,
    Mail,
    MessageSquare,
    Smartphone,
    RefreshCw,
    Save,
    User,
    Clock,
    MapPin,
    AlertTriangle,
    X,
    LogOut
} from 'lucide-react';
import useText from '../hooks/useText';
import GoogleTranslate from '../GoogleTranslate';
import { addUserLocation, clearUserLocations, getLatestUserLocation } from '../utils/locationUtils';
import { getCurrentUserFarmSize, getFarmSizeCategory, shouldShowIoTFeatures, shouldShowDroneFeatures } from '../utils/userDataUtils';

// Backend endpoints (can be overridden via env in future)
const QUESTIONNAIRE_SERVER_URL = 'http://localhost:5004';
const WEATHER_SERVER_URL = 'http://localhost:5005';

const Settings = () => {
    const { t } = useText();
    const [settings, setSettings] = useState(() => ({
        emailNotifications: true,
        smsNotifications: false,
        pushNotifications: true,
        refreshInterval: 5
    }));

    const [unsavedChanges, setUnsavedChanges] = useState(false);
    const [showLocationWarning, setShowLocationWarning] = useState(false);
    const [locationStatus, setLocationStatus] = useState('idle'); // idle, loading, success, error
    const [userLocation, setUserLocation] = useState(null);

    // Farm size information
    const farmSize = getCurrentUserFarmSize();
    const farmCategory = getFarmSizeCategory();
    const showIoT = shouldShowIoTFeatures();
    const showDrone = shouldShowDroneFeatures();

    // Load settings from localStorage on component mount
    useEffect(() => {
        const savedSettings = localStorage.getItem('agrinovaSettings');
        if (savedSettings) {
            const parsed = JSON.parse(savedSettings);
            setSettings(prev => ({ ...prev, ...parsed }));
        }

        // Load current location
        const currentLocation = getLatestUserLocation();
        setUserLocation(currentLocation);
    }, []);
    // Update setting and mark as unsaved
    const updateSetting = (key, value) => {
        setSettings(prev => ({
            ...prev,
            [key]: value
        }));
        setUnsavedChanges(true);
    };

    // Save preferences
    const savePreferences = () => {
        localStorage.setItem('agrinovaSettings', JSON.stringify(settings));
        setUnsavedChanges(false);
        alert(t('settingsPage.preferencesSaved'));
    };

    // Save notification settings
    const saveNotificationSettings = () => {
        localStorage.setItem('agrinovaSettings', JSON.stringify(settings));
        setUnsavedChanges(false);
        alert(t('settingsPage.notificationsSaved'));
    };

    // Save system settings
    const saveSystemSettings = () => {
        localStorage.setItem('agrinovaSettings', JSON.stringify(settings));
        setUnsavedChanges(false);
        alert(t('settingsPage.systemSaved'));
    };

    // Handle location change with warnings
    const handleLocationChange = () => {
        setShowLocationWarning(true);
    };

    // Confirm location change
    const confirmLocationChange = async () => {
        setShowLocationWarning(false);
        setLocationStatus('loading');

        if (!navigator.geolocation) {
            alert(t('settingsPage.geolocationUnsupported'));
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
                        city: geocodeData.address?.city || geocodeData.address?.town || geocodeData.address?.village || t('settingsPage.unknownLocation'),
                        state: geocodeData.address?.state || t('settingsPage.unknownLocation'),
                        country: geocodeData.address?.country || t('settingsPage.unknownLocation'),
                        accuracy: position.coords.accuracy
                    };

                    // Clear existing locations and add new one
                    clearUserLocations();
                    const savedLocation = addUserLocation(locationData);

                    if (savedLocation) {
                        setUserLocation(savedLocation);
                        setLocationStatus('success');
                        alert(t('settingsPage.locationUpdated', { location: locationData.displayName }));

                        // Trigger location update event for other components
                        window.dispatchEvent(new CustomEvent('userLocationUpdated', {
                            detail: savedLocation
                        }));

                        // Ask user whether to refresh plantation-type predictions for the new location
                        try {
                            const msg = t('settingsPage.locationChangedConfirm') || 'You changed your farm location. Update plantation-type predictions for this new location now?';
                            const doUpdate = window.confirm(msg);
                            if (doUpdate) {
                                // Run the prediction update flow (non-blocking)
                                updatePredictionsForNewLocation(savedLocation).catch((err) => {
                                    console.error('Error updating predictions for new location', err);
                                    alert(t('settingsPage.predictionUpdateFailed') || 'Failed to update predictions for the new location.');
                                });
                            }
                        } catch (e) {
                            console.error('Error prompting user for prediction update', e);
                        }

                        setTimeout(() => setLocationStatus('idle'), 2000);
                    } else {
                        throw new Error('Failed to save location');
                    }

                } catch (error) {
                    console.error('Error updating location:', error);
                    alert(t('settingsPage.locationFailed'));
                    setLocationStatus('error');
                    setTimeout(() => setLocationStatus('idle'), 2000);
                }
            },
            (error) => {
                console.error('Geolocation error:', error);
                let errorMessageKey = 'settingsPage.locationFailed';
                switch (error.code) {
                    case error.PERMISSION_DENIED:
                        errorMessageKey = 'settingsPage.locationPermissionDenied';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMessageKey = 'settingsPage.locationUnavailable';
                        break;
                    case error.TIMEOUT:
                        errorMessageKey = 'settingsPage.locationTimeout';
                        break;
                    default:
                        errorMessageKey = 'settingsPage.locationUnknownError';
                        break;
                }
                alert(t(errorMessageKey));
                setLocationStatus('error');
                setTimeout(() => setLocationStatus('idle'), 2000);
            }
        );
    };

    // Cancel location change
    const cancelLocationChange = () => {
        setShowLocationWarning(false);
    };

    // Update backend predictions when location changes and user confirms
    const updatePredictionsForNewLocation = async (savedLocation) => {
        setLocationStatus('loading');
        try {
            // 1) fetch latest questionnaire answers
            const qResp = await fetch(`${QUESTIONNAIRE_SERVER_URL}/questionaire`);
            if (!qResp.ok) throw new Error('Failed to fetch questionnaire');
            const qText = await qResp.text();
            const qData = JSON.parse(qText);
            const latest = Array.isArray(qData) ? qData.sort((a,b)=> new Date(b.savedAt)-new Date(a.savedAt))[0] : qData;
            const answers = latest?.answers || {};

            // 2) request weather for the new location from weather server
            const weatherResp = await fetch(`${WEATHER_SERVER_URL}/api/get-weather`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    latitude: savedLocation.coordinates?.latitude,
                    longitude: savedLocation.coordinates?.longitude,
                    city: savedLocation.city,
                    name: savedLocation.displayName
                })
            });
            if (!weatherResp.ok) throw new Error('Failed to fetch weather from server');
            const weatherJson = await weatherResp.json();
            const weather = weatherJson?.data || {};

            // 3) Prepare NPK request via weather server and call predictnpk
            const npkPrepResp = await fetch(`${WEATHER_SERVER_URL}/api/prepare-npk-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: answers.soilType,
                    last_crop: answers.lastCrop,
                    residue_left: answers.residueLeft,
                    weather
                })
            });
            if (!npkPrepResp.ok) throw new Error('Failed to prepare NPK request');
            const npkPrep = await npkPrepResp.json();
            const npkReq = npkPrep.data;

            const npkResp = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predictnpk`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(npkReq)
            });
            if (!npkResp.ok) throw new Error('NPK prediction failed');
            const npkResultText = await npkResp.text();
            const npkResult = JSON.parse(npkResultText);

            // 4) Prepare and call intercropping prediction
            const interPrepResp = await fetch(`${WEATHER_SERVER_URL}/api/prepare-intercropping-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: answers.soilType,
                    last_crop: answers.lastCrop,
                    soil_n: npkResult?.estimated_N ?? 0,
                    soil_p: npkResult?.estimated_P ?? 0,
                    soil_k: npkResult?.estimated_K ?? 0,
                    soil_moisture: answers.soilMoisture,
                    weather
                })
            });
            if (!interPrepResp.ok) throw new Error('Failed to prepare intercropping request');
            const interPrep = await interPrepResp.json();
            const interResp = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predict_intercropping`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(interPrep.data)
            });
            if (!interResp.ok) throw new Error('Intercropping prediction failed');
            const interText = await interResp.text();
            const interResult = JSON.parse(interText);

            // 5) Prepare and call multiheight prediction
            const multiPrepResp = await fetch(`${WEATHER_SERVER_URL}/api/prepare-multiheight-request`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    soil_type: answers.soilType,
                    last_crop: answers.lastCrop,
                    region: answers.region,
                    season: answers.season,
                    soil_ph: answers.soilPh,
                    weather
                })
            });
            if (!multiPrepResp.ok) throw new Error('Failed to prepare multiheight request');
            const multiPrep = await multiPrepResp.json();
            const multiResp = await fetch(`${QUESTIONNAIRE_SERVER_URL}/predict_multiheight`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(multiPrep.data)
            });
            if (!multiResp.ok) throw new Error('Multiheight prediction failed');
            const multiText = await multiResp.text();
            const multiResult = JSON.parse(multiText);

            // 6) Notify the rest of the app about the new predictions
            window.dispatchEvent(new CustomEvent('predictionsUpdated', {
                detail: {
                    npkPrediction: npkResult,
                    intercroppingPrediction: interResult,
                    multiheightPrediction: multiResult
                }
            }));

            // Also trigger generateRecommendation for components that expect it
            try {
                window.dispatchEvent(new Event('generateRecommendation'));
            } catch (e) {
                // ignore
            }

            alert(t('settingsPage.predictionUpdateSuccess') || 'Predictions updated for the new location.');
            setLocationStatus('success');
            setTimeout(() => setLocationStatus('idle'), 2000);
            return { npkResult, interResult, multiResult };
        } catch (err) {
            console.error('updatePredictionsForNewLocation error', err);
            setLocationStatus('error');
            setTimeout(() => setLocationStatus('idle'), 2000);
            throw err;
        }
    };

    // Handle logout
    const handleLogout = () => {
        if (window.confirm(t('settingsPage.logoutConfirm'))) {
            // Clear user session data
            localStorage.removeItem('username');
            localStorage.removeItem('isLoggedIn');

            // Optionally clear all user data (uncomment if needed)
            // localStorage.clear();

            // Reload the page to reset the app state
            window.location.reload();
        }
    };
    return (
        <div className="p-6 space-y-6" data-scroll-section>
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-emerald-400 to-cyan-400 bg-clip-text text-transparent flex items-center">
                        <SettingsIcon className="w-8 h-8 mr-3 text-emerald-400" />
                        {t('settingsPage.title')}
                    </h1>
                    <p className="text-gray-300 mt-2">
                        {t('settingsPage.subtitle')}
                    </p>
                </div>
                {unsavedChanges && (
                    <div className="bg-gradient-to-r from-amber-500/20 to-orange-500/20 text-amber-300 px-4 py-2 rounded-lg border border-amber-400/30 backdrop-blur-sm">
                        <span className="text-sm font-medium">{t('settingsPage.unsavedNotice')}</span>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* User Preferences */}
                <div className="glass-card p-6 border border-emerald-500/20 hover:border-emerald-400/30 transition-all duration-300">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                        <User className="w-5 h-5 mr-2 text-emerald-400" />
                        {t('settingsPage.userPreferences')}
                    </h2>

                    {/* Language Selection */}
                    <div className="mb-6">
                        <label className="block text-gray-200 font-semibold mb-2 flex items-center">
                            <Globe className="w-4 h-4 mr-2 text-cyan-400" />
                            {t('settingsPage.languageLabel')}
                        </label>
                        <p className="text-gray-400 text-sm mb-3">
                            {t('settingsPage.googleTranslateDescription')}
                        </p>
                        <div className="bg-gray-800/60 border border-gray-600 rounded-lg px-4 py-3">
                            <GoogleTranslate />
                        </div>
                    </div>

                    <button
                        onClick={savePreferences}
                        className="w-full bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white px-4 py-2 rounded-lg transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg hover:shadow-emerald-500/25"
                    >
                        <Save className="w-4 h-4" />
                        <span>{t('settingsPage.savePreferences')}</span>
                    </button>
                </div>

                {/* Location Management */}
                <div className="glass-card p-6 border border-orange-500/20 hover:border-orange-400/30 transition-all duration-300">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                        <MapPin className="w-5 h-5 mr-2 text-orange-400" />
                        {t('settingsPage.locationManagement')}
                    </h2>

                    <div className="mb-4">
                        <label className="block text-gray-200 font-semibold mb-2">
                            {t('settingsPage.currentLocation')}
                        </label>
                        {userLocation ? (
                            <div className="bg-gray-800/60 border border-gray-600 rounded-lg p-3">
                                <p className="text-white font-medium">{userLocation.city}, {userLocation.state}</p>
                                <p className="text-gray-400 text-sm">{userLocation.country}</p>
                                <p className="text-gray-500 text-xs mt-1">
                                    {t('settingsPage.updatedOn', { date: new Date(userLocation.timestamp).toLocaleDateString() })}
                                </p>
                            </div>
                        ) : (
                            <div className="bg-gray-800/60 border border-gray-600 rounded-lg p-3">
                                <p className="text-gray-400">{t('settingsPage.noLocation')}</p>
                            </div>
                        )}
                    </div>

                    <button
                        onClick={handleLocationChange}
                        disabled={locationStatus === 'loading'}
                        className={`w-full transition-all duration-300 flex items-center justify-center space-x-2 px-4 py-2 rounded-lg ${locationStatus === 'loading'
                            ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                            : 'bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 text-white shadow-lg hover:shadow-orange-500/25'
                            }`}
                    >
                        {locationStatus === 'loading' ? (
                            <>
                                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                                <span>{t('settingsPage.updatingLocation')}</span>
                            </>
                        ) : (
                            <>
                                <MapPin className="w-4 h-4" />
                                <span>{t('settingsPage.changeLocation')}</span>
                            </>
                        )}
                    </button>
                </div>
            </div>

            {/* Farm Information Section */}
            <div className="glass-card p-6 border border-emerald-500/20 hover:border-emerald-400/30 transition-all duration-300">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                    <SettingsIcon className="w-5 h-5 mr-2 text-emerald-400" />
                    {t('settingsPage.farmInformation')}
                </h2>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="text-center p-4 bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 rounded-lg border border-emerald-500/20">
                        <div className="text-gray-300 text-sm">{t('settingsPage.farmSize')}</div>
                        <div className="text-white font-bold text-xl text-emerald-400">{t('sidebar.acres', { value: farmSize })}</div>
                        <div className="text-gray-400 text-xs mt-1 capitalize">{t('settingsPage.farmCategory', { value: farmCategory })}</div>
                    </div>

                    <div className="text-center p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg border border-blue-500/20">
                        <div className="text-gray-300 text-sm">{t('settingsPage.iotFeatures')}</div>
                        <div className={`text-white font-bold text-xl ${showIoT ? 'text-green-400' : 'text-red-400'}`}>
                            {showIoT ? t('settingsPage.available') : t('settingsPage.notAvailable')}
                        </div>
                        <div className="text-gray-400 text-xs mt-1">
                            {showIoT ? t('settingsPage.iotAvailableDescription') : t('settingsPage.iotUnavailableDescription')}
                        </div>
                    </div>

                    <div className="text-center p-4 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 rounded-lg border border-purple-500/20">
                        <div className="text-gray-300 text-sm">{t('settingsPage.droneFeatures')}</div>
                        <div className={`text-white font-bold text-xl ${showDrone ? 'text-green-400' : 'text-red-400'}`}>
                            {showDrone ? t('settingsPage.available') : t('settingsPage.notAvailable')}
                        </div>
                        <div className="text-gray-400 text-xs mt-1">
                            {showDrone ? t('settingsPage.droneAvailableDescription') : t('settingsPage.droneUnavailableDescription')}
                        </div>
                    </div>
                </div>

                {!showIoT && (
                    <div className="mt-4 p-3 bg-amber-500/20 border border-amber-500/30 rounded-lg">
                        <p className="text-amber-300 text-sm text-center">
                            {t('settingsPage.iotHint')}
                        </p>
                    </div>
                )}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Notification Settings */}
                <div className="glass-card p-6 border border-blue-500/20 hover:border-blue-400/30 transition-all duration-300">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                        <Bell className="w-5 h-5 mr-2 text-blue-400" />
                        {t('settingsPage.notificationSettings')}
                    </h2>

                    <div className="space-y-4 mb-6">
                        {/* Email Notifications */}
                        <label className="flex items-center space-x-3 text-gray-200 hover:text-white transition-colors cursor-pointer p-2 rounded-lg hover:bg-white/5">
                            <input
                                type="checkbox"
                                checked={settings.emailNotifications}
                                onChange={(e) => updateSetting('emailNotifications', e.target.checked)}
                                className="w-4 h-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-blue-500 focus:ring-2"
                            />
                            <Mail className="w-4 h-4 text-blue-400" />
                            <span>{t('settingsPage.emailNotifications')}</span>
                        </label>

                        {/* SMS Notifications */}
                        <label className="flex items-center space-x-3 text-gray-200 hover:text-white transition-colors cursor-pointer p-2 rounded-lg hover:bg-white/5">
                            <input
                                type="checkbox"
                                checked={settings.smsNotifications}
                                onChange={(e) => updateSetting('smsNotifications', e.target.checked)}
                                className="w-4 h-4 text-emerald-600 bg-gray-800 border-gray-600 rounded focus:ring-emerald-500 focus:ring-2"
                            />
                            <Smartphone className="w-4 h-4 text-emerald-400" />
                            <span>{t('settingsPage.smsNotifications')}</span>
                        </label>

                        {/* Push Notifications */}
                        <label className="flex items-center space-x-3 text-gray-200 hover:text-white transition-colors cursor-pointer p-2 rounded-lg hover:bg-white/5">
                            <input
                                type="checkbox"
                                checked={settings.pushNotifications}
                                onChange={(e) => updateSetting('pushNotifications', e.target.checked)}
                                className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500 focus:ring-2"
                            />
                            <MessageSquare className="w-4 h-4 text-purple-400" />
                            <span>{t('settingsPage.pushNotifications')}</span>
                        </label>
                    </div>

                    <button
                        onClick={saveNotificationSettings}
                        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white px-4 py-2 rounded-lg transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg hover:shadow-blue-500/25"
                    >
                        <Save className="w-4 h-4" />
                        <span>{t('settingsPage.saveNotifications')}</span>
                    </button>
                </div>

                {/* System Settings */}
                <div className="glass-card p-6 lg:col-span-2 border border-purple-500/20 hover:border-purple-400/30 transition-all duration-300">
                    <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                        <RefreshCw className="w-5 h-5 mr-2 text-purple-400" />
                        {t('settingsPage.systemSettings')}
                    </h2>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        {/* Data Refresh Interval */}
                        <div>
                            <label className="block text-gray-200 font-semibold mb-2 flex items-center">
                                <Clock className="w-4 h-4 mr-2 text-cyan-400" />
                                {t('settingsPage.autoRefresh')}
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="60"
                                value={settings.refreshInterval}
                                onChange={(e) => updateSetting('refreshInterval', parseInt(e.target.value))}
                                className="w-full bg-gray-800/60 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 backdrop-blur-sm transition-all"
                            />
                            <p className="text-sm text-gray-300 mt-1">
                                {t('settingsPage.autoRefreshDescription')}
                            </p>
                        </div>

                        {/* Data Storage */}
                        <div>
                            <label className="block text-gray-200 font-semibold mb-2">
                                {t('settingsPage.dataStorage')}
                            </label>
                            <div className="space-y-2">
                                <button
                                    onClick={() => {
                                        localStorage.clear();
                                        alert(t('settingsPage.localDataCleared'));
                                    }}
                                    className="w-full bg-gradient-to-r from-red-600 to-pink-600 hover:from-red-500 hover:to-pink-500 text-white px-4 py-2 rounded-lg transition-all text-sm shadow-lg hover:shadow-red-500/25"
                                >
                                    {t('settingsPage.clearLocalData')}
                                </button>
                                <button
                                    onClick={() => {
                                        const data = localStorage.getItem('agrinovaSettings');
                                        const blob = new Blob([data || '{}'], { type: 'application/json' });
                                        const url = URL.createObjectURL(blob);
                                        const a = document.createElement('a');
                                        a.href = url;
                                        a.download = 'agrinova-settings.json';
                                        a.click();
                                    }}
                                    className="w-full bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white px-4 py-2 rounded-lg transition-all text-sm shadow-lg hover:shadow-indigo-500/25"
                                >
                                    {t('settingsPage.exportSettings')}
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="mt-6 pt-6 border-t border-gray-600/50">
                        <button
                            onClick={saveSystemSettings}
                            className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white px-6 py-2 rounded-lg transition-all duration-300 flex items-center space-x-2 shadow-lg hover:shadow-purple-500/25"
                        >
                            <Save className="w-4 h-4" />
                            <span>{t('settingsPage.saveSystem')}</span>
                        </button>
                    </div>
                </div>
            </div>

            {/* Account Management */}
            <div className="glass-card p-6 border border-red-500/20 hover:border-red-400/30 transition-all duration-300">
                <h2 className="text-xl font-bold text-white mb-4 flex items-center">
                    <User className="w-5 h-5 mr-2 text-red-400" />
                    {t('settingsPage.accountManagement')}
                </h2>

                <div className="mb-4">
                    <label className="block text-gray-200 font-semibold mb-2">
                        {t('settingsPage.currentUser')}
                    </label>
                    <div className="bg-gray-800/60 border border-gray-600 rounded-lg p-3">
                        <p className="text-white font-medium">{localStorage.getItem('username') || t('settingsPage.unknownUser')}</p>
                        <p className="text-gray-400 text-sm">{t('sidebar.profileRole')}</p>
                    </div>
                </div>

                <div className="border-t border-gray-600/50 pt-4">
                    <button
                        onClick={handleLogout}
                        className="w-full bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white px-4 py-2 rounded-lg transition-all duration-300 flex items-center justify-center space-x-2 shadow-lg hover:shadow-red-500/25"
                    >
                        <LogOut className="w-4 h-4" />
                        <span>{t('settingsPage.logout')}</span>
                    </button>
                </div>
            </div>

            {/* Settings Summary */}
            <div className="glass-card p-6 border border-emerald-500/20 hover:border-emerald-400/30 transition-all duration-300">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <SettingsIcon className="w-5 h-5 mr-2 text-emerald-400" />
                    {t('settingsPage.settingsSummary')}
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                    <div className="text-center p-3 bg-gradient-to-br from-emerald-500/10 to-cyan-500/10 rounded-lg border border-emerald-500/20">
                        <div className="text-gray-300">{t('settingsPage.languageLabel')}</div>
                        <div className="text-white font-semibold text-emerald-400">
                            {(settings.language || 'auto').toUpperCase()}
                        </div>
                    </div>
                    <div className="text-center p-3 bg-gradient-to-br from-blue-500/10 to-purple-500/10 rounded-lg border border-blue-500/20">
                        <div className="text-gray-300">{t('settingsPage.notificationsSummary')}</div>
                        <div className="text-white font-semibold text-blue-400">
                            {[settings.emailNotifications, settings.smsNotifications, settings.pushNotifications]
                                .filter(Boolean).length} {t('settingsPage.activeCount')}
                        </div>
                    </div>
                    <div className="text-center p-3 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 rounded-lg border border-purple-500/20">
                        <div className="text-gray-300">{t('settingsPage.refreshRate')}</div>
                        <div className="text-white font-semibold text-purple-400">{t('settingsPage.refreshRateValue', { value: settings.refreshInterval })}</div>
                    </div>
                </div>
            </div>

            {/* Location Change Warning Modal */}
            {showLocationWarning && (
                <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
                    <div className="bg-gray-900 border border-red-500/30 rounded-lg p-6 max-w-md mx-4 shadow-2xl">
                        <div className="flex items-center space-x-3 mb-4">
                            <AlertTriangle className="w-6 h-6 text-red-400" />
                            <h3 className="text-xl font-bold text-white">{t('settingsPage.locationWarningHeading')}</h3>
                        </div>

                        <div className="mb-6 space-y-3 text-gray-300">
                            <p className="font-semibold text-red-300">
                                {t('settingsPage.locationWarningImportant')}
                            </p>
                            <ul className="space-y-2 text-sm">
                                <li>{t('settingsPage.locationWarningBullet1')}</li>
                                <li>{t('settingsPage.locationWarningBullet2')}</li>
                                <li>{t('settingsPage.locationWarningBullet3')}</li>
                                <li>{t('settingsPage.locationWarningBullet4')}</li>
                                <li>{t('settingsPage.locationWarningBullet5')}</li>
                            </ul>
                            <p className="text-yellow-300 font-medium text-sm mt-4">
                                {t('settingsPage.locationWarningFooter')}
                            </p>
                        </div>

                        <div className="flex space-x-3">
                            <button
                                onClick={cancelLocationChange}
                                className="flex-1 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-all flex items-center justify-center space-x-2"
                            >
                                <X className="w-4 h-4" />
                                <span>{t('settingsPage.cancel')}</span>
                            </button>
                            <button
                                onClick={confirmLocationChange}
                                className="flex-1 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-500 hover:to-red-600 text-white px-4 py-2 rounded-lg transition-all flex items-center justify-center space-x-2 shadow-lg"
                            >
                                <MapPin className="w-4 h-4" />
                                <span>{t('settingsPage.locationWarningConfirm')}</span>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Settings;