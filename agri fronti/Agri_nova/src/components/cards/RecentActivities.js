import React, { useRef } from 'react';
import { Clock, CheckCircle, AlertCircle, Calendar } from 'lucide-react';
import gsap from 'gsap';

// Return an emoji based on activity action
const getActivityEmoji = (action) => {
  const act = action.toLowerCase();
  if (act.includes('completed')) return '✅';
  if (act.includes('review')) return '📝';
  if (act.includes('irrigation') || act.includes('water')) return '💧';
  if (act.includes('plant') || act.includes('seed')) return '🌱';
  if (act.includes('harvest')) return '🌾';
  if (act.includes('soil') || act.includes('sampling')) return '🌍';
  if (act.includes('fertilizer') || act.includes('compost')) return '🌿';
  if (act.includes('pest') || act.includes('spray')) return '🐛';
  if (act.includes('weeding')) return '🌾';
  return '📋';
};
const RecentActivities = ({ activities, tasks }) => {
  // Get completed tasks and convert them to activity format
  const completedTasks = (tasks || []).filter(task => task.status === 'complete');

  // Convert completed tasks to activity format
  const taskActivities = completedTasks.map(task => ({
    id: `task-${task.id}`,
    action: `Completed: ${task.title}`,
    description: task.actions ? task.actions.join(', ') : '',
    time: task.completedAt ? new Date(task.completedAt).toLocaleString() :
      `${task.date} ${task.time}`,
    user: task.assigned_to || 'Farm Worker',
    type: 'task_completion'
  }));

  // Combine task activities with regular activities and sort by most recent first
  const allActivities = [...(activities || []), ...taskActivities];
  const sortedActivities = allActivities.sort((a, b) => {
    // Sort by completedAt timestamp if available, otherwise by id
    const aTime = a.completedAt || a.id;
    const bTime = b.completedAt || b.id;
    return bTime - aTime;
  });

  const cardRef = useRef();
  const frame = useRef(null);

  // Tilt effects disabled - only keeping basic hover shadow effect
  const handleMouseMove = (e) => {
    // Basic shadow effect without any transform/tilt
    const rect = cardRef.current.getBoundingClientRect();
    cardRef.current.style.boxShadow = '0 15px 30px rgba(16,185,129,0.7), 0 0 15px rgba(16,185,129,0.5)';
  };
  const handleMouseLeave = () => {
    gsap.to(cardRef.current, { overwrite: 'auto', boxShadow: '0 4px 10px rgba(0,0,0,0.2)', ease: 'power3.out', duration: 0.6 });
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'in_progress':
        return <Clock className="w-5 h-5 text-blue-500" />;
      case 'pending':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      default:
        return <Calendar className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'text-green-600 bg-green-50';
      case 'in_progress':
        return 'text-blue-600 bg-blue-50';
      case 'pending':
        return 'text-yellow-600 bg-yellow-50';
      default:
        return 'text-gray-600 bg-gray-50';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  if (!activities && (!tasks || !Array.isArray(tasks))) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-xl font-semibold text-white mb-4">Recent Activities</h3>
        <p className="text-gray-400">No activities available</p>
      </div>
    );
  }

  return (
    <div
      ref={cardRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      className="glass-card p-6"
      style={{
        // Removed transform style properties to disable tilt
        willChange: 'box-shadow',
        transition: 'box-shadow 0.3s ease'
      }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-white">Recent Activities</h3>
        <span className="text-sm text-gray-400">
          {sortedActivities.length} activities
        </span>
      </div>

      <div className="space-y-4 max-h-60 overflow-y-auto">
        {sortedActivities.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-400 text-sm">No recent activities</p>
            <p className="text-gray-500 text-xs mt-1">Complete some tasks to see them here!</p>
          </div>
        ) : (
          sortedActivities.map((activity, index) => (
            <div key={activity.id || index} className="flex items-start space-x-4 p-3 rounded-lg hover:bg-gray-700/30 transition-all duration-200">
              <div className="flex-shrink-0 mt-1 text-xl">
                <span>{getActivityEmoji(activity.action)}</span>
              </div>

              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white mb-1">
                  {activity.action}
                </p>
                {activity.description && (
                  <p className="text-xs text-gray-300 mb-1">
                    {activity.description}
                  </p>
                )}
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">
                    {activity.time}
                  </span>
                  {activity.user && (
                    <span className="text-xs text-gray-500">
                      {activity.user}
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="mt-6 pt-4 border-t border-gray-600">
        <button className="w-full text-center text-sm text-blue-400 hover:text-blue-300 font-medium transition-colors duration-200">
          View All Activities
        </button>
      </div>
    </div>
  );
};

export default RecentActivities;
