import React, { useState, useEffect } from 'react';
import {
  Wifi,
  WifiOff,
  Camera,
  Plane,
  Droplets,
  Power,
  MapPin,
  Clock,
  Activity,
  Gauge,
  Thermometer,
  Eye,
  Settings,
  PlayCircle,
  PauseCircle,
  RotateCcw
} from 'lucide-react';
import { shouldShowDroneFeatures, getFarmSizeCategory } from '../utils/userDataUtils';

const IotDevices = () => {
  const [currentTime, setCurrentTime] = useState(new Date());

  // Check farm size for conditional rendering
  const showDrone = shouldShowDroneFeatures();
  const farmCategory = getFarmSizeCategory();

  // Land Rover IoT Data
  const [landRoverData, setLandRoverData] = useState({
    deviceId: 'LR-001',
    status: 'online',
    location: { lat: 28.7041, lng: 77.1025, address: 'Field A, Sector 12' },
    speed: 15.2, // km/h
    fuelLevel: 78, // percentage
    engineTemp: 85, // celsius
    operatingHours: 342.5,
    todayHours: 4.2,
    currentTask: 'Soil Analysis',
    taskStartTime: new Date(Date.now() - 2.5 * 60 * 60 * 1000), // 2.5 hours ago
    lastUpdate: new Date(),
    batteryLevel: 92,
    totalDistance: 1250.8 // km
  });

  // Drone IoT Data
  const [droneData, setDroneData] = useState({
    deviceId: 'DR-002',
    status: 'online',
    isFlying: true,
    altitude: 150, // meters
    speed: 25.8, // km/h
    batteryLevel: 67,
    flightTime: 45, // minutes today
    totalFlightTime: 1680, // total minutes
    currentMission: 'Crop Monitoring',
    missionStartTime: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
    location: { lat: 28.7055, lng: 77.1035, address: 'Field B, Sector 15' },
    imagesCaptered: 127,
    todayImages: 67,
    lastUpdate: new Date(),
    weatherCondition: 'Clear',
    windSpeed: 12.5 // km/h
  });

  // Smart Irrigation Data
  const [irrigationData, setIrrigationData] = useState({
    deviceId: 'SI-003',
    status: 'online',
    isActive: true,
    waterFlow: 45.8, // liters per minute
    soilMoisture: 35, // percentage
    temperature: 24, // celsius
    humidity: 68, // percentage
    valve1Status: 'open',
    valve2Status: 'closed',
    valve3Status: 'open',
    dailyWaterUsage: 2340, // liters
    totalWaterUsage: 45600, // liters this month
    scheduleActive: true,
    nextSchedule: new Date(Date.now() + 2 * 60 * 60 * 1000), // 2 hours from now
    lastUpdate: new Date(),
    pressure: 2.8, // bar
    systemEfficiency: 94 // percentage
  });

  // Update timestamps every second
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());

      // Update last update times
      setLandRoverData(prev => ({ ...prev, lastUpdate: new Date() }));
      setDroneData(prev => ({ ...prev, lastUpdate: new Date() }));
      setIrrigationData(prev => ({ ...prev, lastUpdate: new Date() }));
    }, 5000); // Update every 5 seconds

    return () => clearInterval(timer);
  }, []);

  // Calculate time differences
  const getTimeDifference = (startTime) => {
    const diff = currentTime - startTime;
    const hours = Math.floor(diff / (1000 * 60 * 60));
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return `${hours}h ${minutes}m`;
  };

  const getLastUpdateTime = (lastUpdate) => {
    const diff = currentTime - lastUpdate;
    const seconds = Math.floor(diff / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ago`;
  };

  const StatusIndicator = ({ status, label }) => (
    <div className="flex items-center space-x-2">
      <div className={`w-3 h-3 rounded-full ${status === 'online' ? 'bg-green-400' : 'bg-red-400'} animate-pulse`} />
      <span className={`text-sm font-medium ${status === 'online' ? 'text-green-400' : 'text-red-400'}`}>
        {status === 'online' ? 'Online' : 'Offline'}
      </span>
    </div>
  );

  const DataPoint = ({ icon: Icon, label, value, unit = '', color = 'text-blue-400' }) => (
    <div className="flex items-center justify-between py-2">
      <div className="flex items-center space-x-2">
        <Icon className={`w-4 h-4 ${color}`} />
        <span className="text-gray-300 text-sm">{label}</span>
      </div>
      <span className="text-white font-medium">{value}{unit}</span>
    </div>
  );

  return (
    <div className="p-6 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
          <Wifi className="w-8 h-8 mr-3 text-blue-400" />
          IoT Devices Dashboard
        </h1>
        <p className="text-gray-300 text-lg">Monitor and control your smart farm devices</p>

        {/* Farm Size Information */}
        {!showDrone && (
          <div className="mt-4 p-4 bg-amber-500/20 border border-amber-500/30 rounded-lg">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-amber-500/20 rounded-full flex items-center justify-center">
                <span className="text-amber-400 font-bold text-sm">!</span>
              </div>
              <div>
                <h3 className="text-amber-400 font-semibold">Limited IoT Features</h3>
                <p className="text-gray-300 text-sm">
                  {farmCategory === 'small'
                    ? 'IoT devices are recommended for farms 2+ acres. Upgrade your farm size to access advanced equipment monitoring.'
                    : 'Drone features are available for farms 7+ acres. Current farm size qualifies for basic IoT monitoring only.'
                  }
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* IoT Devices Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Land Rover IoT Device */}
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-orange-500/20 rounded-full">
                <Gauge className="w-6 h-6 text-orange-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Land Rover</h2>
                <p className="text-gray-400 text-sm">{landRoverData.deviceId}</p>
              </div>
            </div>
            <StatusIndicator status={landRoverData.status} />
          </div>

          <div className="space-y-3">
            <DataPoint
              icon={MapPin}
              label="Location"
              value={landRoverData.location.address}
              color="text-orange-400"
            />
            <DataPoint
              icon={Activity}
              label="Current Task"
              value={landRoverData.currentTask}
              color="text-orange-400"
            />
            <DataPoint
              icon={Clock}
              label="Task Duration"
              value={getTimeDifference(landRoverData.taskStartTime)}
              color="text-orange-400"
            />
            <DataPoint
              icon={Gauge}
              label="Speed"
              value={landRoverData.speed}
              unit=" km/h"
              color="text-orange-400"
            />
            <DataPoint
              icon={Droplets}
              label="Fuel Level"
              value={landRoverData.fuelLevel}
              unit="%"
              color="text-orange-400"
            />
            <DataPoint
              icon={Thermometer}
              label="Engine Temp"
              value={landRoverData.engineTemp}
              unit="°C"
              color="text-orange-400"
            />
            <DataPoint
              icon={Clock}
              label="Today Hours"
              value={landRoverData.todayHours}
              unit="h"
              color="text-orange-400"
            />
            <DataPoint
              icon={RotateCcw}
              label="Total Hours"
              value={landRoverData.operatingHours}
              unit="h"
              color="text-orange-400"
            />
          </div>

          <div className="mt-4 pt-4 border-t border-white/20">
            <p className="text-xs text-gray-400">
              Last Update: {getLastUpdateTime(landRoverData.lastUpdate)}
            </p>
          </div>
        </div>

        {/* Drone IoT Device - Only for farms 7+ acres */}
        {showDrone && (
          <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center space-x-3">
                <div className="p-3 bg-blue-500/20 rounded-full">
                  <Plane className="w-6 h-6 text-blue-400" />
                </div>
                <div>
                  <h2 className="text-xl font-bold text-white">Agricultural Drone</h2>
                  <p className="text-gray-400 text-sm">{droneData.deviceId}</p>
                </div>
              </div>
              <StatusIndicator status={droneData.status} />
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between py-2">
                <div className="flex items-center space-x-2">
                  <div className={`w-4 h-4 rounded-full ${droneData.isFlying ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                  <span className="text-gray-300 text-sm">Flight Status</span>
                </div>
                <span className={`font-medium ${droneData.isFlying ? 'text-green-400' : 'text-gray-400'}`}>
                  {droneData.isFlying ? 'Flying' : 'Landed'}
                </span>
              </div>

              <DataPoint
                icon={Activity}
                label="Mission"
                value={droneData.currentMission}
                color="text-blue-400"
              />
              <DataPoint
                icon={Clock}
                label="Flight Time"
                value={getTimeDifference(droneData.missionStartTime)}
                color="text-blue-400"
              />
              <DataPoint
                icon={MapPin}
                label="Altitude"
                value={droneData.altitude}
                unit="m"
                color="text-blue-400"
              />
              <DataPoint
                icon={Gauge}
                label="Speed"
                value={droneData.speed}
                unit=" km/h"
                color="text-blue-400"
              />
              <DataPoint
                icon={Camera}
                label="Images Today"
                value={droneData.todayImages}
                color="text-blue-400"
              />
              <DataPoint
                icon={Eye}
                label="Total Images"
                value={droneData.imagesCaptered}
                color="text-blue-400"
              />
              <DataPoint
                icon={Clock}
                label="Total Flight Time"
                value={Math.floor(droneData.totalFlightTime / 60)}
                unit="h"
                color="text-blue-400"
              />
              <DataPoint
                icon={Power}
                label="Battery"
                value={droneData.batteryLevel}
                unit="%"
                color="text-blue-400"
              />
            </div>

            <div className="mt-4 pt-4 border-t border-white/20">
              <p className="text-xs text-gray-400">
                Last Update: {getLastUpdateTime(droneData.lastUpdate)}
              </p>
            </div>
          </div>
        )}

        {/* Smart Irrigation System */}
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <div className="p-3 bg-green-500/20 rounded-full">
                <Droplets className="w-6 h-6 text-green-400" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white">Smart Irrigation</h2>
                <p className="text-gray-400 text-sm">{irrigationData.deviceId}</p>
              </div>
            </div>
            <StatusIndicator status={irrigationData.status} />
          </div>

          <div className="space-y-3">
            <div className="flex items-center justify-between py-2">
              <div className="flex items-center space-x-2">
                <div className={`w-4 h-4 rounded-full ${irrigationData.isActive ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`} />
                <span className="text-gray-300 text-sm">System Status</span>
              </div>
              <span className={`font-medium ${irrigationData.isActive ? 'text-green-400' : 'text-gray-400'}`}>
                {irrigationData.isActive ? 'Active' : 'Inactive'}
              </span>
            </div>

            <DataPoint
              icon={Droplets}
              label="Water Flow"
              value={irrigationData.waterFlow}
              unit=" L/min"
              color="text-green-400"
            />
            <DataPoint
              icon={Gauge}
              label="Soil Moisture"
              value={irrigationData.soilMoisture}
              unit="%"
              color="text-green-400"
            />
            <DataPoint
              icon={Thermometer}
              label="Temperature"
              value={irrigationData.temperature}
              unit="°C"
              color="text-green-400"
            />
            <DataPoint
              icon={Activity}
              label="Humidity"
              value={irrigationData.humidity}
              unit="%"
              color="text-green-400"
            />

            {/* Valve Status */}
            <div className="py-2">
              <span className="text-gray-300 text-sm mb-2 block">Valve Status</span>
              <div className="grid grid-cols-3 gap-2">
                {[1, 2, 3].map(valve => {
                  const isOpen = irrigationData[`valve${valve}Status`] === 'open';
                  return (
                    <div key={valve} className="text-center">
                      <div className={`w-6 h-6 mx-auto rounded-full ${isOpen ? 'bg-green-400' : 'bg-gray-400'} mb-1`} />
                      <span className="text-xs text-gray-400">V{valve}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            <DataPoint
              icon={Droplets}
              label="Daily Usage"
              value={irrigationData.dailyWaterUsage}
              unit=" L"
              color="text-green-400"
            />
            <DataPoint
              icon={Settings}
              label="Efficiency"
              value={irrigationData.systemEfficiency}
              unit="%"
              color="text-green-400"
            />
          </div>

          <div className="mt-4 pt-4 border-t border-white/20">
            <p className="text-xs text-gray-400">
              Last Update: {getLastUpdateTime(irrigationData.lastUpdate)}
            </p>
          </div>
        </div>
      </div>

      {/* System Overview */}
      <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6">
        <h2 className="text-xl font-bold text-white mb-4 flex items-center">
          <Activity className="w-5 h-5 mr-2 text-blue-400" />
          System Overview
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">3/3</div>
            <div className="text-gray-300 text-sm">Devices Online</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {droneData.todayImages + Math.floor(landRoverData.todayHours * 10)}
            </div>
            <div className="text-gray-300 text-sm">Data Points Collected</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {irrigationData.dailyWaterUsage + Math.floor(landRoverData.fuelLevel * 10)}L
            </div>
            <div className="text-gray-300 text-sm">Resources Used Today</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default IotDevices;