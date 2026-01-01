// Generates alerts based on current tasks and weather data
import alertsConfig from '../data/dashboard/alerts.json';
// config: taskAlertThresholdHours, weatherRainConditions, heatAlertTemperature
import recentConfig from '../data/dashboard/recentActivities.json';

export function generateAlerts(tasks, weather) {
  const alerts = [];
  const now = new Date();
  const threshold = alertsConfig.taskAlertThresholdHours || 24;

  // Task-based alerts: upcoming tasks within threshold hours
  // Task-based alerts: show all pending (upcoming) or in-progress tasks
  tasks.forEach((task) => {
    if (['pending', 'in_progress'].includes(task.status)) {
      alerts.push({
        id: task.id,
        type: 'info',
        title: `Upcoming: ${task.title}`,
        message: task.description,
        time: task.time
      });
    }
  });

  // Weather-based alerts: server 'major' filtered events
  if (weather && Array.isArray(weather.major)) {
    weather.major.forEach((f, idx) => {
      const mainCond = f.weather[0].main;
      const desc = f.weather[0].description;
      // Determine alert type
      const type = mainCond === 'Thunderstorm' || mainCond === 'Tornado' ? 'error' : 'weather';
      alerts.push({
        id: 1000 + idx,
        type,
        title: `Weather Alert: ${mainCond}`,
        message: desc,
        time: new Date(f.dt * 1000).toLocaleDateString(),
      });
    });
  }

  return alerts;
}

// Generates recent activity entries based on task status
export function generateRecentActivities(tasks) {
  const activities = [];
  const now = Date.now();
  const windowMs = (recentConfig.windowHours || 24) * 3600 * 1000;
  // Include only completed tasks within the time window
  tasks.forEach((task) => {
    if (task.status === 'complete') {
      // determine timestamp: use completedAt or derive from date/time
      let ts = task.completedAt;
      if (!ts) {
        // parse date and time into timestamp
        const [time, meridiem] = task.time.split(' ');
        let [h, m] = time.split(':').map(Number);
        if (meridiem === 'PM' && h < 12) h += 12;
        if (meridiem === 'AM' && h === 12) h = 0;
        const [Y, Mo, D] = task.date.split('-').map(Number);
        ts = new Date(Y, Mo - 1, D, h, m).getTime();
      }
      const age = now - ts;
      // only include if completion timestamp is not in the future and within window
      if (ts <= now && age <= windowMs) {
        const diffMins = Math.round(age / 60000);
        activities.push({
          id: ts,
          action: `Completed: ${task.title}`,
          description: task.description,
          time: `${diffMins} min ago`,
          user: (recentConfig.defaultUser || 'System')
        });
      }
    }
  });
  // sort descending by timestamp and limit
  return activities
    .sort((a, b) => b.id - a.id)
    .slice(0, recentConfig.maxActivities || 10);
}
