// Utility functions for managing user locations

export const getUserLocations = () => {
    try {
        const locations = localStorage.getItem('userLocations');
        return locations ? JSON.parse(locations) : [];
    } catch (error) {
        console.error('Error reading user locations:', error);
        return [];
    }
};

export const addUserLocation = (locationData) => {
    try {
        const existingLocations = getUserLocations();
        const newLocation = {
            ...locationData,
            id: Date.now(),
            timestamp: new Date().toISOString()
        };
        existingLocations.push(newLocation);
        localStorage.setItem('userLocations', JSON.stringify(existingLocations));

        // Trigger location update event for other components
        window.dispatchEvent(new CustomEvent('userLocationUpdated', {
            detail: newLocation
        }));

        return newLocation;
    } catch (error) {
        console.error('Error saving user location:', error);
        return null;
    }
};

export const clearUserLocations = () => {
    try {
        localStorage.removeItem('userLocations');
        // Trigger location cleared event
        window.dispatchEvent(new CustomEvent('userLocationCleared'));
        return true;
    } catch (error) {
        console.error('Error clearing user locations:', error);
        return false;
    }
};

export const getLatestUserLocation = () => {
    const locations = getUserLocations();
    if (locations.length === 0) return null;

    // Return the most recent location
    return locations.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
};

// Get location for weather API (stored location or fallback to Delhi)
export const getLocationForWeather = () => {
    const userLocation = getLatestUserLocation();

    if (userLocation && userLocation.coordinates) {
        return {
            type: 'coordinates',
            latitude: userLocation.coordinates.latitude,
            longitude: userLocation.coordinates.longitude,
            query: `?lat=${userLocation.coordinates.latitude}&lon=${userLocation.coordinates.longitude}`
        };
    }

    // Fallback to Delhi, India
    return {
        type: 'city',
        city: 'Delhi',
        country: 'India',
        query: '?city=Delhi'
    };
};

// Get location display name for header and dashboard
export const getLocationDisplayName = () => {
    const userLocation = getLatestUserLocation();

    if (userLocation) {
        // Show City, State format (or City, Country if no state)
        const city = userLocation.city || 'Unknown';
        const state = userLocation.state;
        const country = userLocation.country;

        if (state && state !== city) {
            return `${city}, ${state}`;
        } else if (country) {
            return `${city}, ${country}`;
        } else {
            return city;
        }
    }

    // Fallback to Delhi, India
    return 'Delhi, India';
};

// Get short location name for header (city only)
export const getShortLocationName = () => {
    const userLocation = getLatestUserLocation();

    if (userLocation) {
        return userLocation.city || 'Unknown';
    }

    // Fallback to Delhi
    return 'Delhi';
};

// Get detailed location info for dashboard
export const getLocationDetails = () => {
    const userLocation = getLatestUserLocation();

    if (userLocation) {
        const city = userLocation.city || 'Unknown';
        const state = userLocation.state;
        const country = userLocation.country;

        let compactName;
        if (state && state !== city) {
            compactName = `${city}, ${state}`;
        } else if (country) {
            compactName = `${city}, ${country}`;
        } else {
            compactName = city;
        }

        return {
            displayName: compactName,
            city: userLocation.city,
            state: userLocation.state,
            country: userLocation.country,
            coordinates: userLocation.coordinates,
            timestamp: userLocation.timestamp,
            isUserSet: true
        };
    }

    // Fallback to Delhi, India
    return {
        displayName: 'Delhi, India',
        city: 'Delhi',
        state: 'Delhi',
        country: 'India',
        coordinates: {
            latitude: 28.6139,
            longitude: 77.2090
        },
        isUserSet: false
    };
};

// Check if user has set their location
export const hasUserLocation = () => {
    const userLocation = getLatestUserLocation();
    return userLocation !== null;
};

// Export locations to console for debugging
export const exportLocationsToConsole = () => {
    const locations = getUserLocations();
    console.log('Stored User Locations:', JSON.stringify(locations, null, 2));
    return locations;
};