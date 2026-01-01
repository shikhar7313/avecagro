export const chartColors = {
  primary: '#16a34a',
  primaryAccent: '#4ade80',
  secondary: '#0ea5e9',
  accent: '#f97316',
  warning: '#eab308',
  neutral: '#94a3b8',
  slate: '#1e293b'
};

const hexToRgba = (hex, alpha = 0.2) => {
  const sanitized = hex.replace('#', '');
  const bigint = parseInt(sanitized, 16);
  const r = (bigint >> 16) & 255;
  const g = (bigint >> 8) & 255;
  const b = bigint & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

export const withAlpha = (hex, alpha) => hexToRgba(hex, alpha);

export const tooltipStyles = {
  contentStyle: {
    borderRadius: 12,
    borderColor: 'rgba(15, 23, 42, 0.1)',
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    color: '#f8fafc',
    boxShadow: '0 10px 35px rgba(15,23,42,0.15)'
  },
  itemStyle: { color: '#f8fafc' },
  labelStyle: { color: '#cbd5f5', fontWeight: 600 }
};
