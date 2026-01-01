import React, { useMemo } from 'react';
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    CartesianGrid,
    XAxis,
    YAxis,
    Tooltip,
    Legend
} from 'recharts';

import chartsData from '../../data/dashboard/chartsDemoData.json';
import { chartColors, tooltipStyles } from '../../utils/chartTheme';

const EquipmentAnalyticsChart = () => {
    const { dailyHours, utilizationPct } = chartsData.equipmentUsage;
    const { machines } = chartsData.equipmentUtilization;

    const chartData = useMemo(() => dailyHours, [dailyHours]);

    const topMachines = machines
        .slice()
        .sort((a, b) => b.hours - a.hours)
        .slice(0, 3);

    return (
        <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300 border border-gray-100">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-1">Equipment Usage Analytics</h3>
                    <p className="text-gray-600 text-sm">Weekly run hours by machine</p>
                </div>
                <div className="flex items-center space-x-2 text-green-600 bg-green-50 px-3 py-1 rounded-full">
                    <span className="text-sm font-medium">{utilizationPct}% utilization</span>
                </div>
            </div>

            <div className="relative h-80">
                <ResponsiveContainer>
                    <BarChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.3)" vertical={false} />
                        <XAxis dataKey="day" stroke="#94a3b8" tickLine={false} />
                        <YAxis stroke="#94a3b8" tickFormatter={(value) => `${value} hrs`} tickLine={false} />
                        <Tooltip
                            {...tooltipStyles}
                            formatter={(value, key) => [`${value} hrs`, key.charAt(0).toUpperCase() + key.slice(1)]}
                        />
                        <Legend verticalAlign="top" align="right" height={36} iconType="circle" />
                        <Bar dataKey="tractor" fill={chartColors.primary} radius={[8, 8, 0, 0]} />
                        <Bar dataKey="sprayer" fill={chartColors.secondary} radius={[8, 8, 0, 0]} />
                        <Bar dataKey="harvester" fill={chartColors.accent} radius={[8, 8, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                {topMachines.map((machine) => (
                    <div key={machine.name} className="p-3 rounded-lg bg-slate-50">
                        <p className="font-semibold text-slate-700">{machine.name}</p>
                        <p className="text-slate-500">{machine.category}</p>
                        <p className="text-slate-600 mt-1">
                            {machine.hours} hrs • {machine.availabilityPct}% uptime
                        </p>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default EquipmentAnalyticsChart;