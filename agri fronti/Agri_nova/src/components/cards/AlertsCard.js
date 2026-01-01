import React, { useRef } from 'react';
import gsap from 'gsap';

const AlertsCard = ({ alerts }) => {
  const cardRef = useRef();

  const handleMouseMove = (e) => {
    const rect = cardRef.current.getBoundingClientRect();
    cardRef.current.style.boxShadow = '0 20px 40px rgba(16,185,129,0.7), 0 0 20px rgba(16,185,129,0.5)';
  };

  const handleMouseLeave = () => {
    // animate back to neutral
    gsap.to(cardRef.current, { overwrite: 'auto', boxShadow: '0 4px 10px rgba(0,0,0,0.2)', ease: 'power3.out', duration: 0.6 });
  };

  // Emoji icons for alerts, with dynamic weather icons based on content
  const getAlertEmoji = ({ type, title = '', message = '' }) => {
    if (type === 'warning') return '⚠️';
    if (type === 'error') return '❌';
    if (type === 'success') return '✅';
    if (type === 'weather') {
      const text = (title + ' ' + message).toLowerCase();
      if (text.includes('rain')) return '🌧️';
      if (text.includes('snow')) return '❄️';
      if (text.includes('storm') || text.includes('thunder')) return '⛈️';
      if (text.includes('cloud')) return '☁️';
      if (text.includes('sun') || text.includes('clear')) return '☀️';
      return '🌤️';
    }
    // Task-based alerts: map to farm icons based on title keywords
    if (type === 'info') {
      const text = title.toLowerCase();
      if (text.includes('plant')) return '🌱';
      if (text.includes('fertilize') || text.includes('fertilizer')) return '🌿';
      if (text.includes('pest') || text.includes('aphid') || text.includes('infestation')) return '🐛';
      if (text.includes('water') || text.includes('irrigation')) return '💧';
      if (text.includes('harvest')) return '🌾';
      if (text.includes('ventilation')) return '🌬️';
      if (text.includes('pump') || text.includes('maintenance')) return '🔧';
      // default for other info alerts
      return 'ℹ️';
    }
    return 'ℹ️';
  };

  if (!alerts || !Array.isArray(alerts)) {
    return (
      <div
        ref={cardRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        className="glass-card p-6"
        style={{
          transformStyle: 'preserve-3d',
          willChange: 'transform',
          transition: 'box-shadow 0.3s ease'
        }}
      >
        <h3 className="text-xl font-semibold text-white mb-4">Alerts</h3>
        <p className="text-gray-400">No alerts</p>
      </div>
    );
  }
  
  // Order alerts: weather first
  const alertList = [...alerts].sort((a, b) => {
    if (a.type === 'weather' && b.type !== 'weather') return -1;
    if (b.type === 'weather' && a.type !== 'weather') return 1;
    return 0;
  });

  const getAlertColor = (type) => {
    switch (type) {
      case 'warning':
        return 'border-yellow-500 text-yellow-400';
      case 'error':
        return 'border-red-500 text-red-400';
      case 'success':
        return 'border-green-500 text-green-400';
      default:
        return 'border-blue-500 text-blue-400';
    }
  };

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="glass-card p-6"
      style={{
        transformStyle: 'preserve-3d',
        willChange: 'transform',
        transition: 'box-shadow 0.3s ease'
      }}
    >
      <h3 className="text-xl font-semibold text-white mb-4">Alerts</h3>
      <div className="space-y-3 max-h-72 overflow-y-auto">
        {alertList.map((alert) => (
          <div
            key={alert.id}
            className={`p-3 rounded-lg border-l-4 bg-gray-800/50 hover:bg-gray-700/40 transition-colors duration-200 ${getAlertColor(alert.type)}`}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <h4 className="font-semibold text-white flex items-center gap-2">
                  <span>{getAlertEmoji(alert)}</span>
                  {alert.title}
                </h4>
                <p className="text-gray-300 text-sm mt-1 line-clamp-2">{alert.message}</p>
              </div>
              <span className="text-gray-400 text-xs ml-4">{alert.time}</span>
            </div>
          </div>
        ))}
      </div>
      {/* optional footer or closing tags */}
    </div>
  );
};

export default AlertsCard;
