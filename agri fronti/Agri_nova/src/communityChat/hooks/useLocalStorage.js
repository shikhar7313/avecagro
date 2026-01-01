import { useEffect, useState } from 'react';

export const useLocalStorage = (key, initialValue) => {
  const resolveInitial = () => (typeof initialValue === 'function' ? initialValue() : initialValue);

  const readValue = () => {
    if (typeof window === 'undefined') return resolveInitial();
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : resolveInitial();
    } catch (error) {
      console.warn(`useLocalStorage read error for ${key}`, error);
      return resolveInitial();
    }
  };

  const [storedValue, setStoredValue] = useState(readValue);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const existing = window.localStorage.getItem(key);
    if (existing === null && storedValue !== undefined) {
      try {
        window.localStorage.setItem(key, JSON.stringify(storedValue));
      } catch (error) {
        console.warn(`useLocalStorage init write error for ${key}`, error);
      }
    }
  }, [key, storedValue]);

  const setValue = (value) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(key, JSON.stringify(valueToStore));
      }
    } catch (error) {
      console.warn(`useLocalStorage write error for ${key}`, error);
    }
  };

  return [storedValue, setValue];
};
