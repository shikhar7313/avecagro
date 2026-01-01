import React, { useMemo } from 'react';
import {
    ResponsiveContainer,
    ComposedChart,
    Area,
    Line,
    XAxis,
    YAxis,
    Tooltip,
    CartesianGrid
} from 'recharts';

import chartsData from '../../data/dashboard/chartsDemoData.json';
import { chartColors, tooltipStyles } from '../../utils/chartTheme';

const WaterUsageChart = () => {
    const { months, actualKL, targetKL, recirculationRatePct, highlights } = chartsData.waterUsage;

    const chartData = useMemo(
        () =>
            months.map((month, index) => ({
                month,
                actual: actualKL[index],
                target: targetKL[index],
                recirc: recirculationRatePct[index]
            })),
        [months, actualKL, targetKL, recirculationRatePct]
    );

    return (
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-xl font-semibold text-gray-800">Water Usage Trends</h3>
                    <p className="text-gray-600 text-sm">Monthly water consumption analysis</p>
                </div>
                <div className="bg-green-100 px-3 py-1 rounded-full">
                    <span className="text-green-800 text-sm font-medium">
                        Recirculation {recirculationRatePct[recirculationRatePct.length - 1]}%
                    </span>
                </div>
            </div>

            <div className="relative h-80">
                <ResponsiveContainer>
                    <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                        <defs>
                            <linearGradient id="waterArea" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor={chartColors.primary} stopOpacity={0.35} />
                                <stop offset="95%" stopColor={chartColors.primary} stopOpacity={0.05} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.3)" vertical={false} />
                        <XAxis dataKey="month" stroke="#94a3b8" tickLine={false} />
                        <YAxis
                            yAxisId="left"
                            stroke="#94a3b8"
                            tickFormatter={(value) => `${value} kL`}
                            tickLine={false}
                        />
                        <YAxis
                            yAxisId="right"
                            orientation="right"
                            stroke="#94a3b8"
                            tickFormatter={(value) => `${value}%`}
                            tickLine={false}
                        />
                        <Tooltip
                            {...tooltipStyles}
                            formatter={(value, key) => {
                                if (key === 'recirc') return [`${value}%`, 'Recirculation Rate'];
                                return [`${value} kL`, key === 'actual' ? 'Actual Usage' : 'Target Plan'];
                            }}
                        />
                        <Area
                            yAxisId="left"
                            type="monotone"
                            dataKey="actual"
                            stroke={chartColors.primary}
                            fill="url(#waterArea)"
                            strokeWidth={3}
                            dot={{ r: 4, strokeWidth: 2, stroke: '#fff' }}
                            activeDot={{ r: 6 }}
                        />
                        <Line
                            yAxisId="left"
                            type="monotone"
                            dataKey="target"
                            stroke={chartColors.secondary}
                            strokeDasharray="6 6"
                            strokeWidth={2.5}
                            dot={{ r: 4, strokeWidth: 0 }}
                        />
                        <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="recirc"
                            stroke={chartColors.accent}
                            strokeWidth={2}
                            dot={false}
                        />
                    </ComposedChart>
                </ResponsiveContainer>
            </div>

            <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-gray-100">
                <div className="text-center">
                    <div className="text-lg font-bold text-blue-600">{highlights.peakKL} kL</div>
                    <div className="text-xs text-gray-600">Peak Usage ({highlights.peakMonth})</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-green-600">{highlights.efficiencyPct}%</div>
                    <div className="text-xs text-gray-600">Avg Efficiency</div>
                </div>
                <div className="text-center">
                    <div className="text-lg font-bold text-orange-600">
                        +{highlights.deltaVsPlanPct.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-600">vs Plan</div>
                </div>
            </div>
        </div>
    );
};

export default WaterUsageChart;