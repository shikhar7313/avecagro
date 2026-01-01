import React, { useState, useEffect } from 'react';
import useText from '../hooks/useText';
import {
    Wrench,
    Calendar,
    BarChart3,
    CheckCircle,
    Clock,
    AlertTriangle,
    Settings
} from 'lucide-react';
import EquipmentAnalyticsChart from './charts/EquipmentAnalyticsChart';

// Import data files
import chartsDemoData from '../data/dashboard/chartsDemoData.json';
import tasksData from '../data/dashboard/tasks.json';

const EquipmentManagement = () => {
    const { t } = useText();
    const [equipmentData, setEquipmentData] = useState({
        totalEquipment: 0,
        operationalEquipment: 0,
        maintenanceEquipment: 0,
        idleEquipment: 0
    });

    const [maintenanceSchedule, setMaintenanceSchedule] = useState([]);
    const [maintenanceHistory, setMaintenanceHistory] = useState([]);
    const [recommendations, setRecommendations] = useState([]);
    const [equipmentUsage, setEquipmentUsage] = useState([]);

    // Generate dynamic equipment data from existing JSON files
    const generateEquipmentStats = () => {
        try {
            // Calculate stats based on tasks and production data
            const equipmentTypes = ['Tractor', 'Harvester', 'Seeder', 'Plow', 'Sprayer', 'Cultivator'];
            const totalEquipment = equipmentTypes.length * 3; // Assume 3 of each type

            // Calculate operational equipment based on task completion rates
            const taskCompletionRate = tasksData.filter(task => task.status === 'completed').length / tasksData.length;
            const operationalEquipment = Math.floor(totalEquipment * taskCompletionRate);

            // Calculate maintenance equipment based on overdue tasks
            const overdueTasks = tasksData.filter(task =>
                task.status === 'overdue' || task.priority === 'high'
            ).length;
            const maintenanceEquipment = Math.min(overdueTasks, totalEquipment - operationalEquipment);

            const idleEquipment = totalEquipment - operationalEquipment - maintenanceEquipment;

            return {
                totalEquipment,
                operationalEquipment,
                maintenanceEquipment,
                idleEquipment
            };
        } catch (error) {
            console.error('Error generating equipment stats:', error);
            return {
                totalEquipment: 15,
                operationalEquipment: 12,
                maintenanceEquipment: 3,
                idleEquipment: 2
            };
        }
    };

    // Generate maintenance schedule from tasks data
    const generateMaintenanceSchedule = () => {
        try {
            const equipmentTypes = ['Tractor A', 'Harvester B', 'Seeder C', 'Plow D', 'Sprayer E'];

            return equipmentTypes.map((equipment, index) => {
                // Base date calculation on current date and index
                const baseDate = new Date();
                baseDate.setDate(baseDate.getDate() + (index * 7) + Math.floor(Math.random() * 14));

                // Determine status based on tasks priority
                const highPriorityTasks = tasksData.filter(task => task.priority === 'high').length;
                let status;
                if (index < highPriorityTasks) {
                    status = Math.random() > 0.5 ? 'Overdue' : 'Pending';
                } else {
                    status = 'Scheduled';
                }

                return {
                    equipment,
                    nextMaintenance: baseDate.toLocaleDateString('en-GB'),
                    status
                };
            });
        } catch (error) {
            console.error('Error generating maintenance schedule:', error);
            return [];
        }
    };

    // Generate maintenance history from production data
    const generateMaintenanceHistory = () => {
        try {
            const equipmentTypes = ['Tractor A', 'Harvester B', 'Seeder C', 'Plow D'];
            const maintenanceTypes = [
                'Oil Change, Tire Replacement',
                'Blade Sharpening',
                'Hydraulic System Check',
                'Blade Replacement',
                'Engine Tune-up',
                'Filter Replacement'
            ];

            return equipmentTypes.map((equipment, index) => {
                const baseDate = new Date();
                baseDate.setDate(baseDate.getDate() - (30 + index * 15 + Math.floor(Math.random() * 30)));

                return {
                    equipment,
                    lastMaintenance: baseDate.toLocaleDateString('en-GB'),
                    details: maintenanceTypes[index % maintenanceTypes.length]
                };
            });
        } catch (error) {
            console.error('Error generating maintenance history:', error);
            return [];
        }
    };

    // Generate equipment usage from consolidated dashboard data
    const generateEquipmentUsage = () => {
        try {
            const machines = chartsDemoData.equipmentUtilization?.machines;
            if (Array.isArray(machines)) {
                return machines.map(machine => ({
                    equipment: machine.name,
                    hours: machine.hours || 0
                }));
            }

            // Fallback usage data
            return [
                { equipment: 'Tractor A', hours: 80 },
                { equipment: 'Harvester B', hours: 60 },
                { equipment: 'Seeder C', hours: 40 },
                { equipment: 'Plow D', hours: 55 }
            ];
        } catch (error) {
            console.error('Error generating equipment usage:', error);
            return [];
        }
    };

    // Generate smart recommendations based on data
    const generateRecommendations = (schedule, history, usage) => {
        try {
            const recommendations = [];

            // Check for overdue maintenance
            const overdue = schedule.filter(item => item.status === 'Overdue');
            if (overdue.length > 0) {
                recommendations.push(`Urgent: ${overdue.length} equipment(s) overdue for maintenance - ${overdue[0].equipment}`);
            }

            // Check for high usage equipment
            const highUsage = usage.filter(item => item.hours > 70);
            if (highUsage.length > 0) {
                recommendations.push(`Monitor high-usage equipment: ${highUsage[0].equipment} (${highUsage[0].hours} hours)`);
            }

            // Check for old maintenance
            const oldMaintenance = history.filter(item => {
                const maintDate = new Date(item.lastMaintenance.split('-').reverse().join('-'));
                const daysSince = (new Date() - maintDate) / (1000 * 60 * 60 * 24);
                return daysSince > 90;
            });

            if (oldMaintenance.length > 0) {
                recommendations.push(`Schedule preventive maintenance for ${oldMaintenance[0].equipment} (last maintained ${oldMaintenance[0].lastMaintenance})`);
            }

            // Add general recommendations
            recommendations.push('Consider purchasing backup equipment for peak seasons');

            // Ensure we have at least 4 recommendations
            const defaultRecs = [
                'Implement IoT sensors for real-time equipment monitoring',
                'Upgrade to precision agriculture technology',
                'Schedule quarterly equipment performance reviews',
                'Invest in operator training programs'
            ];

            while (recommendations.length < 4) {
                const randomRec = defaultRecs[Math.floor(Math.random() * defaultRecs.length)];
                if (!recommendations.includes(randomRec)) {
                    recommendations.push(randomRec);
                }
            }

            return recommendations.slice(0, 4);
        } catch (error) {
            console.error('Error generating recommendations:', error);
            return [
                'Schedule maintenance for Seeder C to avoid downtime.',
                'Upgrade Harvester B for better efficiency.',
                'Consider purchasing backup equipment for peak seasons.',
                'Implement preventive maintenance schedule for all tractors.'
            ];
        }
    };

    useEffect(() => {
        // Generate all dynamic data
        const stats = generateEquipmentStats();
        const schedule = generateMaintenanceSchedule();
        const history = generateMaintenanceHistory();
        const usage = generateEquipmentUsage();
        const recs = generateRecommendations(schedule, history, usage);

        setEquipmentData(stats);
        setMaintenanceSchedule(schedule);
        setMaintenanceHistory(history);
        setEquipmentUsage(usage);
        setRecommendations(recs);
    }, []);

    const getStatusColor = (status) => {
        switch (status.toLowerCase()) {
            case 'scheduled': return 'text-green-400 bg-green-400/20';
            case 'pending': return 'text-yellow-400 bg-yellow-400/20';
            case 'overdue': return 'text-red-400 bg-red-400/20';
            default: return 'text-gray-400 bg-gray-400/20';
        }
    };

    const getStatusIcon = (status) => {
        switch (status.toLowerCase()) {
            case 'scheduled': return <CheckCircle className="w-4 h-4" />;
            case 'pending': return <Clock className="w-4 h-4" />;
            case 'overdue': return <AlertTriangle className="w-4 h-4" />;
            default: return <Clock className="w-4 h-4" />;
        }
    };

    const StatCard = ({ title, value, icon: Icon, color }) => (
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
            <div className="flex items-center justify-between">
                <div>
                    <p className="text-gray-300 text-sm font-medium">{title}</p>
                    <p className="text-white text-2xl font-bold mt-1">{value}</p>
                </div>
                <div className={`p-3 rounded-full ${color}`}>
                    <Icon className="w-6 h-6 text-white" />
                </div>
            </div>
        </div>
    );

    return (
        <div className="p-6 space-y-8 max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center">
                    <Wrench className="w-8 h-8 mr-3 text-blue-400" />
                    Equipment Management
                </h1>
                <p className="text-gray-300 text-lg">Monitor and manage your farm equipment</p>
            </div>

            {/* Equipment Overview Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                <StatCard
                    title="Total Equipment"
                    value={equipmentData.totalEquipment}
                    icon={Wrench}
                    color="bg-blue-500/20"
                />
                <StatCard
                    title="Operational"
                    value={equipmentData.operationalEquipment}
                    icon={CheckCircle}
                    color="bg-green-500/20"
                />
                <StatCard
                    title="Under Maintenance"
                    value={equipmentData.maintenanceEquipment}
                    icon={Settings}
                    color="bg-yellow-500/20"
                />
                <StatCard
                    title="Idle"
                    value={equipmentData.idleEquipment}
                    icon={Clock}
                    color="bg-gray-500/20"
                />
            </div>

            {/* Main Content Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Maintenance Schedule */}
                <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
                    <h2 className="text-xl font-bold mb-6 text-white flex items-center">
                        <Calendar className="w-5 h-5 mr-2 text-blue-400" />
                        Maintenance Schedule
                    </h2>
                    <div className="space-y-3">
                        {maintenanceSchedule.map((item, index) => (
                            <div key={index} className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all duration-200">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <h3 className="text-white font-semibold">{item.equipment}</h3>
                                        <p className="text-gray-300 text-sm">{item.nextMaintenance}</p>
                                    </div>
                                    <div className={`flex items-center space-x-1 px-3 py-1 rounded-full text-xs font-medium ${getStatusColor(item.status)}`}>
                                        {getStatusIcon(item.status)}
                                        <span>{item.status}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Equipment Usage Analytics */}
                <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
                    <h2 className="text-xl font-bold mb-6 text-white flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-blue-400" />
                        Equipment Usage Analytics
                    </h2>
                    <div className="bg-white/5 rounded-lg p-4">
                        <EquipmentAnalyticsChart />
                    </div>
                </div>
            </div>

            {/* Maintenance History */}
            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
                <h2 className="text-xl font-bold mb-6 text-white flex items-center">
                    <Settings className="w-5 h-5 mr-2 text-blue-400" />
                    Maintenance History
                </h2>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-white/20">
                                <th className="text-left py-3 px-4 text-gray-200 font-medium">Equipment</th>
                                <th className="text-left py-3 px-4 text-gray-200 font-medium">Last Maintenance</th>
                                <th className="text-left py-3 px-4 text-gray-200 font-medium">Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            {maintenanceHistory.map((item, index) => (
                                <tr key={index} className="border-b border-white/10 last:border-b-0 hover:bg-white/5 transition-all duration-200">
                                    <td className="py-3 px-4 text-white font-medium">{item.equipment}</td>
                                    <td className="py-3 px-4 text-gray-300">{item.lastMaintenance}</td>
                                    <td className="py-3 px-4 text-gray-300">{item.details}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Recommendations */}
            <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-200">
                <h2 className="text-xl font-bold mb-6 text-white flex items-center">
                    <AlertTriangle className="w-5 h-5 mr-2 text-blue-400" />
                    Recommendations
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {recommendations.map((rec, index) => (
                        <div key={index} className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all duration-200">
                            <div className="flex items-start space-x-3">
                                <div className="w-2 h-2 bg-blue-400 rounded-full mt-2 flex-shrink-0"></div>
                                <span className="text-gray-300">{rec}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default EquipmentManagement;