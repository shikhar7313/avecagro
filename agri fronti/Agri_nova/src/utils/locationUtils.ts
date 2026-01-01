export interface Coordinates { latitude: number; longitude: number }

export interface UserLocation {
  id?: number | string;
  timestamp?: string;
  coordinates?: Coordinates;
  city?: string;
  state?: string;
  country?: string;
  [key: string]: unknown;
}

export function getUserLocations(): UserLocation[] {
  try {
    const locations = localStorage.getItem('userLocations');
    return locations ? (JSON.parse(locations) as UserLocation[]) : [];
  } catch (error) {
    // keep behavior same as JS file: log and return empty
    // eslint-disable-next-line no-console
    console.error('Error reading user locations:', error);
    return [];
  }
}

export function addUserLocation(locationData: Partial<UserLocation>): UserLocation | null {
  try {
    const existingLocations = getUserLocations();
    const newLocation: UserLocation = {
      ...locationData,
      id: Date.now(),
      timestamp: new Date().toISOString(),
    };
    existingLocations.push(newLocation);
    localStorage.setItem('userLocations', JSON.stringify(existingLocations));

    // Trigger location update event for other components
    try {
      window.dispatchEvent(
        new CustomEvent('userLocationUpdated', {
          detail: newLocation,
        })
      );
    } catch (e) {
      // ignore dispatch errors in non-browser environments
    }

    return newLocation;
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('Error saving user location:', error);
    return null;
  }
}

export function clearUserLocations(): boolean {
  try {
    localStorage.removeItem('userLocations');
    try {
      window.dispatchEvent(new CustomEvent('userLocationCleared'));
    } catch (e) {
      /* ignore */
    }
    return true;
  } catch (error) {
    // eslint-disable-next-line no-console
    console.error('Error clearing user locations:', error);
    return false;
  }
}

export function getLatestUserLocation(): UserLocation | null {
  const locations = getUserLocations();
  if (locations.length === 0) return null;
  // Return the most recent location
  return locations.sort((a, b) => new Date(b.timestamp ?? 0).getTime() - new Date(a.timestamp ?? 0).getTime())[0] ?? null;
}

export type LocationForWeather = {
  type: 'coordinates' | 'city';
  latitude?: number;
  longitude?: number;
  query: string;
  name?: string;
  city?: string;
  country?: string;
};

// Get location for weather API (stored location or fallback to Delhi)
export function getLocationForWeather(): LocationForWeather {
  const userLocation = getLatestUserLocation();

  if (userLocation && userLocation.coordinates) {
    return {
      type: 'coordinates',
      latitude: userLocation.coordinates.latitude,
      longitude: userLocation.coordinates.longitude,
      query: `?lat=${userLocation.coordinates.latitude}&lon=${userLocation.coordinates.longitude}`,
      name: userLocation.city ?? undefined,
    };
  }

  // Fallback to Delhi, India (keeps previous behavior)
  return {
    type: 'city',
    city: 'Delhi',
    country: 'India',
    query: '?city=Delhi',
    name: 'Delhi',
  };
}

// Get location display name for header and dashboard
export function getLocationDisplayName(): string {
  const userLocation = getLatestUserLocation();

  if (userLocation) {
    const city = userLocation.city ?? 'Unknown';
    const state = userLocation.state;
    const country = userLocation.country;

    if (state && state !== city) {
      return `${city}, ${state}`;
    } else if (country) {
      return `${city}, ${country}`;
    }

    return city;
  }

  // Fallback to Delhi, India
  return 'Delhi, India';
}

// Get short location name for header (city only)
export function getShortLocationName(): string {
  const userLocation = getLatestUserLocation();
  if (userLocation) return userLocation.city ?? 'Unknown';
  return 'Delhi';
}

// Get detailed location info for dashboard
export function getLocationDetails() {
  const userLocation = getLatestUserLocation();
  if (userLocation) {
    const city = userLocation.city ?? 'Unknown';
    const state = userLocation.state;
    const country = userLocation.country;
    let compactName: string;
    if (state && state !== city) compactName = `${city}, ${state}`;
    else if (country) compactName = `${city}, ${country}`;
    else compactName = city;

    return {
      displayName: compactName,
      city: userLocation.city,
      state: userLocation.state,
      country: userLocation.country,
      coordinates: userLocation.coordinates,
      timestamp: userLocation.timestamp,
      isUserSet: true,
    } as const;
  }

  return {
    displayName: 'Delhi, India',
    city: 'Delhi',
    state: 'Delhi',
    country: 'India',
    coordinates: {
      latitude: 28.6139,
      longitude: 77.209,
    },
    isUserSet: false,
  } as const;
}

// Check if user has set their location
export function hasUserLocation(): boolean {
  return getLatestUserLocation() !== null;
}

// Export locations to console for debugging
export function exportLocationsToConsole(): UserLocation[] {
  const locations = getUserLocations();
  // eslint-disable-next-line no-console
  console.log('Stored User Locations:', JSON.stringify(locations, null, 2));
  return locations;
}
