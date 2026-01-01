import React, { useMemo } from 'react';
import {
    ResponsiveContainer,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip
} from 'recharts';

import chartsData from '../../data/dashboard/chartsDemoData.json';
import { chartColors, tooltipStyles } from '../../utils/chartTheme';

const ProductionOverviewChart = () => {
    const { crops, harvestSchedule } = chartsData.productionOverview;

    const chartData = useMemo(
        () =>
            crops.map((crop) => ({
                name: crop.name,
                yield: crop.yieldTonnes,
                area: crop.areaHa,
                change: crop.changePct
            })),
        [crops]
    );

    const avgGrowth = useMemo(
        () => (crops.reduce((sum, crop) => sum + crop.changePct, 0) / crops.length).toFixed(1),
        [crops]
    );

    return (
        <div className="bg-white rounded-xl shadow-lg p-6 hover:shadow-xl transition-shadow duration-300">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-1">Production Overview</h3>
                    <p className="text-gray-600 text-sm">Crop production by type</p>
                </div>
                <div className="flex items-center space-x-2 text-green-600">
                    <span className="text-sm font-medium">{avgGrowth}% avg growth</span>
                </div>
            </div>

            <div className="relative h-72">
                <ResponsiveContainer>
                    <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.3)" vertical={false} />
                        <XAxis dataKey="name" stroke="#94a3b8" tickLine={false} />
                        <YAxis stroke="#94a3b8" tickFormatter={(value) => `${value} t`} tickLine={false} />
                        <Tooltip
                            {...tooltipStyles}
                            formatter={(value, key) => {
                                if (key === 'area') return [`${value} ha`, 'Cultivated Area'];
                                if (key === 'change') return [`${value}%`, 'YoY Change'];
                                return [`${value} t`, 'Yield'];
                            }}
                        />
                        <Bar dataKey="yield" fill={chartColors.primary} radius={[12, 12, 0, 0]} barSize={32} />
                    </BarChart>
                </ResponsiveContainer>
            </div>

            <div className="mt-6">
                <h4 className="text-sm font-semibold text-gray-600 mb-3">Upcoming Harvest Windows</h4>
                <div className="space-y-2">
                    {harvestSchedule.map((slot) => (
                        <div key={slot.crop} className="flex items-center justify-between text-sm bg-slate-50 rounded-lg px-3 py-2">
                            <div>
                                <p className="font-medium text-slate-700">{slot.crop}</p>
                                <p className="text-slate-500">{slot.status}</p>
                            </div>
                            <span className="text-slate-600 font-semibold">{slot.window}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default ProductionOverviewChart;