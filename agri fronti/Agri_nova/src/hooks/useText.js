import { useCallback, useMemo } from 'react';
import strings from '../strings/en.json';

const getValueFromPath = (path, source) => {
  return path.split('.').reduce((acc, segment) => {
    if (acc && Object.prototype.hasOwnProperty.call(acc, segment)) {
      return acc[segment];
    }
    return undefined;
  }, source);
};

const interpolate = (template, values = {}) => {
  if (typeof template !== 'string') return template;
  return template.replace(/{{\s*([^}]+)\s*}}/g, (_, key) => {
    const trimmed = key.trim();
    if (Object.prototype.hasOwnProperty.call(values, trimmed)) {
      return values[trimmed];
    }
    return '';
  });
};

const useText = () => {
  const t = useCallback((key, values = {}) => {
    const result = getValueFromPath(key, strings);
    if (typeof result === 'string') {
      return interpolate(result, values);
    }
    if (typeof result === 'number') {
      return String(result);
    }
    return key;
  }, []);

  return useMemo(() => ({ t }), [t]);
};

export default useText;
