const DEVICE_STORE_KEY = 'trustedDeviceMap';
const DEVICE_ID_KEY = 'trustedDeviceId';

const safeParse = (raw) => {
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null ? parsed : {};
  } catch (error) {
    console.warn('deviceUtils: failed to parse device map', error);
    return {};
  }
};

const writeStore = (map) => {
  try {
    localStorage.setItem(DEVICE_STORE_KEY, JSON.stringify(map));
  } catch (error) {
    console.error('deviceUtils: failed to persist device map', error);
  }
};

const generateDeviceId = () => {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `device-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
};

export const getOrCreateDeviceId = () => {
  try {
    let id = localStorage.getItem(DEVICE_ID_KEY);
    if (!id) {
      id = generateDeviceId();
      localStorage.setItem(DEVICE_ID_KEY, id);
    }
    return id;
  } catch (error) {
    console.warn('deviceUtils: unable to access device id storage', error);
    return generateDeviceId();
  }
};

export const getRememberedUserForDevice = (deviceId = getOrCreateDeviceId()) => {
  if (!deviceId) return null;
  const map = safeParse(localStorage.getItem(DEVICE_STORE_KEY));
  return map[deviceId]?.username || null;
};

export const rememberDeviceForUser = (username, metadata = {}) => {
  if (!username) return;
  const deviceId = metadata.deviceId || getOrCreateDeviceId();
  const map = safeParse(localStorage.getItem(DEVICE_STORE_KEY));
  map[deviceId] = {
    username,
    rememberedAt: new Date().toISOString(),
    ...metadata,
  };
  writeStore(map);
};

export const clearRememberedDevice = (deviceId = getOrCreateDeviceId()) => {
  if (!deviceId) return;
  const map = safeParse(localStorage.getItem(DEVICE_STORE_KEY));
  if (!map[deviceId]) return;
  delete map[deviceId];
  writeStore(map);
};

export const fetchDeviceIp = async (signal) => {
  try {
    const response = await fetch('https://api.ipify.org?format=json', { signal });
    if (!response.ok) {
      throw new Error('Failed to resolve client IP');
    }
    const data = await response.json();
    return data?.ip || null;
  } catch (error) {
    console.warn('deviceUtils: unable to fetch client IP', error);
    return null;
  }
};
