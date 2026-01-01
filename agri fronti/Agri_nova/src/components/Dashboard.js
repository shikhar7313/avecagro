import React from 'react';
import WeatherCard from './cards/WeatherCard';
import AlertsCard from './cards/AlertsCard';
import RecentActivities from './cards/RecentActivities';
import TasksCalendar from './cards/TasksCalendar';
import GrowthMeter from './cards/GrowthMeter';
import WaterUsageChart from './charts/WaterUsageChart';
import EquipmentAnalyticsChart from './charts/EquipmentAnalyticsChart';
import ProductionOverviewChart from './charts/ProductionOverviewChart';
import RevenueTrendsChart from './charts/RevenueTrendsChart';





const Dashboard = ({
  alerts, setAlerts,
  recentActivities, setRecentActivities,
  tasks, setTasks,
  weather, setWeather
}) => {
  // Ensure data slices are loaded
  if (!tasks || !alerts || !recentActivities || !weather) return null;

  return (
    <div className="p-6 space-y-6" data-scroll-section>
      {/* Top Row - Weather and Alerts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" data-scroll data-scroll-speed="1">
        <WeatherCard data={weather} setWeather={setWeather} />
        <AlertsCard alerts={alerts} setAlerts={setAlerts} />
      </div>

      {/* Tasks Calendar */}
      <div className="w-full" data-scroll data-scroll-speed="2">
        <TasksCalendar
          tasks={tasks}
          setTasks={setTasks}
          setRecentActivities={setRecentActivities}
          setAlerts={setAlerts}
        />
      </div>

      {/* Recent Activities Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6" data-scroll data-scroll-speed="3">
        {/* Recent Activities - 67% width (2 columns out of 3) */}
        <div className="lg:col-span-2">
          <RecentActivities
            activities={recentActivities}
            tasks={tasks}
            setRecentActivities={setRecentActivities}
          />
        </div>

        {/* GROWTH METER */}
        <div className="lg:col-span-1">
          <GrowthMeter tasks={tasks} />
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" data-scroll data-scroll-speed="4">
        <WaterUsageChart />
        <EquipmentAnalyticsChart />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6" data-scroll data-scroll-speed="5">
        <ProductionOverviewChart />
        <RevenueTrendsChart />
      </div>
    </div>
  );
};

export default Dashboard;

