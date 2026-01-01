import React, { useMemo } from 'react';
import {
    ResponsiveContainer,
    ComposedChart,
    Area,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend
} from 'recharts';

import chartsData from '../../data/dashboard/chartsDemoData.json';
import { chartColors, tooltipStyles } from '../../utils/chartTheme';

const RevenueTrendsChart = () => {
    const { quarters, actualUSDk, targetUSDk, channels } = chartsData.revenue;

    const chartData = useMemo(
        () =>
            quarters.map((quarter, index) => ({
                quarter,
                actual: actualUSDk[index],
                target: targetUSDk[index]
            })),
        [quarters, actualUSDk, targetUSDk]
    );

    const latestActual = actualUSDk[actualUSDk.length - 1];

    return (
        <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-1">Revenue Trends</h3>
                    <p className="text-gray-600 text-sm">Quarterly revenue analysis</p>
                </div>
                <div className="flex items-center space-x-2 text-green-600">
                    <span className="text-sm font-medium">${latestActual}K Q4</span>
                </div>
            </div>

            <div className="relative h-72">
                <ResponsiveContainer>
                    <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="revenueArea" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={chartColors.primary} stopOpacity={0.3} />
                                <stop offset="95%" stopColor={chartColors.primary} stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.3)" vertical={false} />
                        <XAxis dataKey="quarter" stroke="#94a3b8" tickLine={false} />
                        <YAxis stroke="#94a3b8" tickFormatter={(value) => `$${value}K`} tickLine={false} />
                        <Tooltip
                            {...tooltipStyles}
                            formatter={(value, key) => [`$${value}K`, key === 'actual' ? 'Actual Revenue' : 'Target Revenue']}
                        />
                        <Legend verticalAlign="top" align="right" height={32} iconType="circle" />
                        <Area
                            type="monotone"
                            dataKey="actual"
                            stroke={chartColors.primary}
                            fill="url(#revenueArea)"
                            strokeWidth={3}
                            dot={{ r: 4, strokeWidth: 2, stroke: '#fff' }}
                            activeDot={{ r: 6 }}
                            name="Actual Revenue"
                        />
                        <Line
                            type="monotone"
                            dataKey="target"
                            stroke={chartColors.secondary}
                            strokeDasharray="6 6"
                            strokeWidth={2.5}
                            dot={{ r: 4 }}
                            name="Target Revenue"
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-6">
                <h4 className="text-sm font-semibold text-gray-600 mb-3">Channel Contribution</h4>
                <div className="grid grid-cols-2 gap-3 text-sm">
                    {channels.map((channel) => (
                        <div key={channel.channel} className="flex items-center justify-between bg-slate-50 rounded-lg px-3 py-2">
                            <span className="font-medium text-slate-700">{channel.channel}</span>
                            <span className="text-slate-600">{channel.sharePct}%</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default RevenueTrendsChart;