import React, { useState, useRef, useEffect } from 'react';
import { Calendar, Clock, MapPin, User, AlertCircle, CheckCircle, Play, Archive, Info, Wrench } from 'lucide-react';
import gsap from 'gsap';

const TasksCalendar = ({ tasks, setTasks, setRecentActivities, setAlerts }) => {
  // Helper to parse task date and 12h time strings into Date
  const parseTaskDate = (dateStr, timeStr) => {
    const [time, meridiem] = timeStr.split(' ');
    const [hourStr, minStr] = time.split(':');
    let hour = parseInt(hourStr, 10);
    const minute = parseInt(minStr, 10);
    if (meridiem.toUpperCase() === 'PM' && hour < 12) hour += 12;
    if (meridiem.toUpperCase() === 'AM' && hour === 12) hour = 0;
    // parse YYYY-MM-DD into local date
    const [y, m, d] = dateStr.split('-').map(Number);
    const dt = new Date(y, m - 1, d, hour, minute, 0, 0);
    return dt;
  };
  const [selectedTask, setSelectedTask] = useState(null);
  const [filterStatus, setFilterStatus] = useState('all');
  const cardRef = useRef();

  const handleMouseMove = (e) => {
    const rect = cardRef.current.getBoundingClientRect();

    gsap.to(cardRef.current, { boxShadow: '0 8px 16px rgba(16,185,129,0.7), 0 0 8px rgba(16,185,129,0.5)', transformPerspective: 600, transformOrigin: 'center', ease: 'power3.out', duration: 0.3 });
  };

  const handleMouseLeave = () => {
    gsap.to(cardRef.current, { rotationX: 0, rotationY: 0, scale: 1, boxShadow: '0 4px 10px rgba(0,0,0,0.2)', ease: 'power3.out', duration: 0.6 });
  };

  const acceptTask = (task) => {
    const updated = { status: 'in_progress', acceptedAt: Date.now() };
    fetch(`/api/tasks/${task.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    })
      .then(res => res.json())
      .then((saved) => {
        setTasks(prev => prev.map(t => t.id === saved.id ? saved : t));
        setSelectedTask(saved);
      })
      .catch(err => console.error('Accept failed', err));
  };

  const completeTask = (task) => {
    const updated = { status: 'complete', completedAt: Date.now() };
    fetch(`/api/tasks/${task.id}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(updated)
    })
      .then(res => res.json())
      .then((saved) => {
        setTasks(prev => prev.map(t => t.id === saved.id ? saved : t));
        setRecentActivities(prev => [{ id: Date.now(), action: saved.title, time: new Date().toLocaleString(), user: saved.assigned_to }, ...prev]);
        setSelectedTask(null);
      })
      .catch(err => console.error('Complete failed', err));
  };

  useEffect(() => {
    const now = Date.now();
    tasks.forEach(task => {
      const due = parseTaskDate(task.date, task.time).getTime();
      const diff = due - now;
      if (task.status === 'pending' && diff > 0 && diff < 24 * 3600 * 1000) {
        setAlerts(prev => {
          const exists = prev.some(a => a.title.includes(task.title));
          if (exists) return prev;
          return [...prev, { id: Date.now() + task.id, type: 'info', title: `Upcoming: ${task.title}`, message: `Due ${Math.round(diff / 3600000)}h`, time: `${Math.round(diff / 3600000)}h` }];
        });
      }
      if (task.status === 'pending' && due < now) {
        setTasks(prev => prev.map(t => t.id === task.id ? { ...t, status: 'backlogged' } : t));
      }
    });
  }, [tasks]);

  // Helper to check if task is due today
  const isToday = (dateString) => {
    const today = new Date();
    const taskDate = new Date(dateString);
    return taskDate.getFullYear() === today.getFullYear()
      && taskDate.getMonth() === today.getMonth()
      && taskDate.getDate() === today.getDate();
  };

  if (!tasks || !Array.isArray(tasks)) {
    return (
      <div className="glass-card p-6">
        <h3 className="text-xl font-semibold text-white mb-4">Tasks Calendar</h3>
        <p className="text-gray-400">No tasks available</p>
      </div>
    );
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'complete':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'in_progress':
        return <Play className="w-5 h-5 text-blue-500" />;
      case 'pending':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      case 'backlogged':
        return <Archive className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'complete':
        return 'border-green-500 bg-green-900/20 text-green-400';
      case 'in_progress':
        return 'border-blue-500 bg-blue-900/20 text-blue-400';
      case 'pending':
        return 'border-yellow-500 bg-yellow-900/20 text-yellow-400';
      case 'backlogged':
        return 'border-red-500 bg-red-900/20 text-red-400';
      default:
        return 'border-gray-500 bg-gray-900/20 text-gray-400';
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'bg-red-600 text-white';
      case 'medium':
        return 'bg-yellow-600 text-white';
      case 'low':
        return 'bg-green-600 text-white';
      default:
        return 'bg-gray-600 text-white';
    }
  };

  const getCategoryIcon = (category) => {
    switch (category) {
      case 'irrigation':
        return '💧';
      case 'fertilization':
        return '🌱';
      case 'pest_control':
        return '🐛';
      case 'soil_management':
        return '🌍';
      case 'harvest':
        return '🌾';
      case 'maintenance':
        return '🔧';
      default:
        return '📋';
    }
  };

  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const nowTime = Date.now();
  // define today range
  const todayStart = new Date(); todayStart.setHours(0, 0, 0, 0);
  const todayEnd = new Date(); todayEnd.setHours(23, 59, 59, 999);
  // Filter tasks based on status or upcoming
  let filteredTasks = [];
  if (filterStatus === 'all') {
    filteredTasks = tasks.filter(task => parseTaskDate(task.date, task.time).getTime() <= nowTime);
  } else if (filterStatus === 'upcoming') {
    // show pending tasks scheduled for future (due > now)
    filteredTasks = tasks.filter(task => task.status === 'pending' && parseTaskDate(task.date, task.time).getTime() > nowTime);
  } else if (filterStatus === 'pending') {
    // show pending tasks due now or overdue (due <= now)
    filteredTasks = tasks.filter(task => task.status === 'pending' && parseTaskDate(task.date, task.time).getTime() <= nowTime);
  } else if (filterStatus === 'backlogged') {
    // explicitly backlogged
    filteredTasks = tasks.filter(task => task.status === 'backlogged');
  } else {
    // in_progress or complete
    filteredTasks = tasks.filter(task => task.status === filterStatus);
  }

  // Counts for each status tab
  const statusCounts = {
    all: tasks.length,
    upcoming: tasks.filter(task => task.status === 'pending' && parseTaskDate(task.date, task.time).getTime() > todayEnd.getTime()).length,
    pending: tasks.filter(task => task.status === 'pending' && (() => {
      const due = parseTaskDate(task.date, task.time).getTime();
      return due >= todayStart.getTime() && due <= todayEnd.getTime();
    })()).length,
    in_progress: tasks.filter(task => task.status === 'in_progress').length,
    complete: tasks.filter(task => task.status === 'complete' && parseTaskDate(task.date, task.time).getTime() <= nowTime).length,
    backlogged: tasks.filter(task => task.status === 'backlogged').length,
  };

  return (
    <div ref={cardRef} onMouseMove={handleMouseMove} onMouseLeave={handleMouseLeave} className="glass-card p-6" style={{ transformStyle: 'preserve-3d', willChange: 'transform' }}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-white flex items-center gap-2">
          <Calendar className="w-6 h-6" />
          Tasks Calendar
        </h3>
        <span className="text-sm text-gray-400">{filteredTasks.length} tasks</span>
      </div>

      {/* Status Filter */}
      <div className="mb-6">
        <div className="flex flex-wrap gap-2">
          {[
            { key: 'all', label: 'All', count: statusCounts.all },
            { key: 'upcoming', label: 'Upcoming', count: statusCounts.upcoming },
            { key: 'pending', label: 'Pending', count: statusCounts.pending },
            { key: 'in_progress', label: 'In Progress', count: statusCounts.in_progress },
            { key: 'complete', label: 'Complete', count: statusCounts.complete },
            { key: 'backlogged', label: 'Backlogged', count: statusCounts.backlogged },
          ].map((filter) => (
            <button
              key={filter.key}
              onClick={() => setFilterStatus(filter.key)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${filterStatus === filter.key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
            >
              {filter.label} ({filter.count})
            </button>
          ))}
        </div>
      </div>

      {/* Tasks List */}
      <div className="space-y-4 max-h-96 overflow-y-auto">
        {filteredTasks.map((task) => (
          <div
            key={task.id}
            className={`p-4 rounded-lg border-l-4 cursor-pointer transition-all duration-200 hover:bg-gray-700/30 ${getStatusColor(task.status)}`}
            onClick={() => setSelectedTask(selectedTask?.id === task.id ? null : task)}
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-2">
                  {getStatusIcon(task.status)}
                  <span className="text-lg">{getCategoryIcon(task.category)}</span>
                  <h4 className="font-medium text-white">{task.title}</h4>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(task.priority)}`}>
                    {task.priority}
                  </span>
                </div>

                <p className="text-gray-300 text-sm mb-2">{task.description}</p>
                {/* Show action steps in collapsed card */}
                {task.actions && task.actions.length > 0 && (
                  <ul className="text-xs text-gray-400 space-y-1 mb-2 list-disc list-inside">
                    {task.actions.map((act, idx) => (
                      <li key={idx}>{act}</li>
                    ))}
                  </ul>
                )}

                <div className="flex items-center gap-4 text-xs text-gray-400">
                  <span className="flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    {formatDate(task.date)}
                  </span>
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {task.time} ({task.duration})
                  </span>
                  <span className="flex items-center gap-1">
                    <MapPin className="w-3 h-3" />
                    {task.location}
                  </span>
                  <span className="flex items-center gap-1">
                    <User className="w-3 h-3" />
                    {task.assigned_to}
                  </span>
                </div>
              </div>
              {/* Complete button in collapsed view */}
              <div className="flex-shrink-0 self-start ml-4">
                {task.status === 'in_progress' && (
                  <button
                    onClick={(e) => { e.stopPropagation(); completeTask(task); }}
                    className="px-2 py-1 bg-green-600 hover:bg-green-500 text-xs text-white rounded"
                  >
                    Complete
                  </button>
                )}
              </div>
            </div>

            {/* Expanded Task Details */}
            {selectedTask?.id === task.id && (
              <div className="mt-4 pt-4 border-t border-gray-600 space-y-4">
                {/* Action Buttons */}
                <div className="flex space-x-2">
                  {task.status === 'pending' && (
                    <button
                      onClick={() => acceptTask(task)}
                      className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-sm text-white rounded"
                    >
                      Accept
                    </button>
                  )}
                  {task.status === 'in_progress' && (
                    <button
                      onClick={() => completeTask(task)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-500 text-sm text-white rounded"
                    >
                      Complete
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* How To Section */}
                  {/* Actions Section */}
                  <div>
                    <h5 className="font-medium text-white mb-2 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      Actions
                    </h5>
                    <ul className="text-sm text-gray-300 space-y-1 list-disc list-inside">
                      {task.actions.map((act, idx) => (
                        <li key={idx}>{act}</li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h5 className="font-medium text-white mb-2 flex items-center gap-2">
                      <Info className="w-4 h-4" />
                      How To Complete
                    </h5>
                    <ol className="text-sm text-gray-300 space-y-1">
                      {task.how_to.map((step, index) => (
                        <li key={index} className="flex gap-2">
                          <span className="text-blue-400 font-medium">{index + 1}.</span>
                          {step}
                        </li>
                      ))}
                    </ol>
                  </div>

                  {/* Tools Needed Section */}
                  <div>
                    <h5 className="font-medium text-white mb-2 flex items-center gap-2">
                      <Wrench className="w-4 h-4" />
                      Tools Needed
                    </h5>
                    <ul className="text-sm text-gray-300 space-y-1">
                      {task.tools.map((tool, index) => (
                        <li key={index} className="flex items-center gap-2">
                          <span className="w-1.5 h-1.5 bg-blue-400 rounded-full"></span>
                          {tool}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>

      {filteredTasks.length === 0 && (
        <div className="text-center py-8 text-gray-400">
          <Calendar className="w-12 h-12 mx-auto mb-2 opacity-50" />
          <p>No tasks found for the selected filter</p>
        </div>
      )}
    </div>
  );
};

export default TasksCalendar;