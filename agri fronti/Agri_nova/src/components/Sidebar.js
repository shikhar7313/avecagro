import React from 'react';
import useText from '../hooks/useText';
import {
  Home,
  Sprout,
  Cloud,
  Droplets,
  Settings,
  Wrench,
  BarChart3,
  MessageCircle,
  Bug,
  DollarSign,
  Users,
  Wifi
} from 'lucide-react';
import { shouldShowIoTFeatures } from '../utils/userDataUtils';

const menuItems = [
  { id: 'dashboard', labelKey: 'sidebar.dashboard', icon: Home },
  { id: 'weather', labelKey: 'sidebar.weather', icon: Cloud },
  { id: 'soil-water', labelKey: 'sidebar.soil', icon: Droplets },
  { id: 'equipment', labelKey: 'sidebar.equipment', icon: Wrench },
  { id: 'reports', labelKey: 'sidebar.reports', icon: BarChart3 },
  { id: 'chatbot', labelKey: 'sidebar.chatbot', icon: MessageCircle },
  { id: 'pest-management', labelKey: 'sidebar.pest', icon: Bug },
  { id: 'loan-management', labelKey: 'sidebar.loans', icon: DollarSign },
  { id: 'community', labelKey: 'sidebar.community', icon: Users },
  { id: 'iot-devices', labelKey: 'sidebar.iotDevices', icon: Wifi },
  { id: 'settings', labelKey: 'sidebar.settings', icon: Settings },
];

const Sidebar = ({ activeTab, setActiveTab, farmSizeAcres }) => {
  const { t } = useText();
  // Filter menu items based on farm size
  const getFilteredMenuItems = () => {
    const computedSize = typeof farmSizeAcres === 'number' && !Number.isNaN(farmSizeAcres)
      ? farmSizeAcres
      : null;
    const showIoT = computedSize !== null ? computedSize >= 2 : shouldShowIoTFeatures();

    return menuItems.filter(item => {
      if (item.id === 'iot-devices') {
        return showIoT; // Only show IoT if farm is 2+ acres
      }
      return true; // Show all other items
    });
  };

  // Get current user information
  const getCurrentUser = () => {
    const username = localStorage.getItem('username') || 'Unknown User';
    // Generate initials from username
    const initials = username.split(' ')
      .map(word => word.charAt(0).toUpperCase())
      .join('')
      .slice(0, 2); // Take only first 2 initials

    return {
      name: username,
      initials: initials || 'UN' // Default to 'UN' for Unknown
    };
  };

  const filteredMenuItems = getFilteredMenuItems();
  const currentUser = getCurrentUser();
  return (
    <div className="w-64 glass-sidebar shadow-xl h-full flex flex-col">
      {/* Logo Section */}
      <div className="p-6 border-b border-white/20 flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-green-600 rounded-lg flex items-center justify-center">
            <Sprout className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold gradient-text">AvecAgro</h1>
              <p className="text-sm text-gray-400">{t('sidebar.smartFarming')}</p>
              {typeof farmSizeAcres === 'number' && farmSizeAcres > 0 && (
                <p className="text-xs text-emerald-300">🌾 {farmSizeAcres.toFixed(1)} acres</p>
              )}
          </div>
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1 p-4 space-y-2 overflow-y-auto">
        {filteredMenuItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;

          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-all duration-200 ${isActive
                ? 'bg-gradient-to-r from-emerald-500 to-green-600 text-white shadow-lg'
                : 'text-gray-300 hover:bg-gray-700/60 hover:shadow-md hover:text-white'
                }`}
            >
              <Icon className={`w-5 h-5 ${isActive ? 'text-white' : 'text-gray-400'}`} />
              <span className="font-medium">{t(item.labelKey)}</span>
            </button>
          );
        })}
      </nav>

      {/* User Profile Section */}
      <div className="p-4 border-t border-white/20 flex-shrink-0">
        <div className="flex items-center space-x-3 p-3 rounded-lg bg-gray-700/60 hover:bg-gray-600/60 transition-all duration-200 cursor-pointer">
          <div className="w-10 h-10 bg-gradient-to-br from-emerald-400 to-green-500 rounded-full flex items-center justify-center">
            <span className="text-white font-semibold">{currentUser.initials}</span>
          </div>
          <div className="flex-1">
            <p className="font-semibold text-gray-100">{currentUser.name}</p>
            <p className="text-sm text-gray-400">{t('sidebar.profileRole')}</p>
            {typeof farmSizeAcres === 'number' && farmSizeAcres > 0 && (
              <p className="text-xs text-gray-400">
                {t('sidebar.landSize')}: {t('sidebar.acres', { value: farmSizeAcres.toFixed(1) })}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
