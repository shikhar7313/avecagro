import React, { useState, useEffect } from 'react';
import useText from '../hooks/useText';
import { Leaf, Droplets, TrendingUp, Calendar, AlertCircle } from 'lucide-react';
import WaterUsageChart from './charts/WaterUsageChart';

const SoilWater = () => {
    const { t } = useText();
    // Mock data - you can replace with real API calls
    const [soilData, setSoilData] = useState({
        pH: 6.5,
        moisture: 45,
        nutrients: 'Optimal',
        temperature: 22,
        health: 'Healthy'
    });

    const [waterData, setWaterData] = useState({
        dailyUsage: 1200,
        monthlyUsage: 36000,
        efficiency: 85,
        quality: 'Good',
        source: 'Well'
    });

    const [irrigationSchedule, setIrrigationSchedule] = useState([
        { field: 'Field A', crop: 'Corn', nextIrrigation: 'Tomorrow', waterRequired: '500 liters' },
        { field: 'Field B', crop: 'Wheat', nextIrrigation: 'In 2 days', waterRequired: '800 liters' },
        { field: 'Field C', crop: 'Rice', nextIrrigation: 'In 3 days', waterRequired: '600 liters' }
    ]);

    const [recommendations, setRecommendations] = useState([
        'Increase irrigation for Field A due to low soil moisture.',
        'Reduce water usage for Field B to improve efficiency.',
        'Consider adding organic matter to improve soil structure in Field C.'
    ]);

    return (
        <div className="p-6 space-y-6" data-scroll-section>
            {/* Header */}
            <div className="mb-6">
                    <h1 className="text-3xl font-bold text-gray-100 hover:text-indigo-300 transition-colors duration-300 mb-2">{t('soilWater.title')}</h1>
                    <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">{t('soilWater.subtitle')}</p>
            </div>

            {/* Soil Quality and Water Usage Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Soil Quality Card */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">{t('soilWater.soilQuality')}</h2>
                    <div className="flex items-start space-x-4">
                        <Leaf className="text-green-400 text-3xl mt-1 flex-shrink-0" size={32} />
                        <div className="space-y-2">
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                PH Level: <span className="font-bold text-green-400">{soilData.pH}</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Moisture: <span className="font-bold text-green-400">{soilData.moisture}%</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Nutrients: <span className="font-bold text-green-400">{soilData.nutrients}</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Temperature: <span className="font-bold text-green-400">{soilData.temperature}°C</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Health Status: <span className="font-bold text-green-400">{soilData.health}</span>
                            </p>
                        </div>
                    </div>
                </div>

                {/* Water Usage Card */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">Water Usage</h2>
                    <div className="flex items-start space-x-4">
                        <Droplets className="text-blue-400 text-3xl mt-1 flex-shrink-0" size={32} />
                        <div className="space-y-2">
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Daily Usage: <span className="font-bold text-blue-400">{waterData.dailyUsage.toLocaleString()} liters</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Monthly Usage: <span className="font-bold text-blue-400">{waterData.monthlyUsage.toLocaleString()} liters</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Irrigation Efficiency: <span className="font-bold text-blue-400">{waterData.efficiency}%</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Water Quality: <span className="font-bold text-blue-400">{waterData.quality}</span>
                            </p>
                            <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                Source: <span className="font-bold text-blue-400">{waterData.source}</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>

            {/* Water Usage Trends Chart */}
            <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                <h2 className="text-2xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                    <TrendingUp className="mr-3 text-blue-400" size={24} />
                    Water Usage Trends
                </h2>
                <div className="bg-white/5 rounded-lg p-4">
                    <WaterUsageChart />
                </div>
            </div>

            {/* Irrigation Schedule and Recommendations - Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Irrigation Schedule */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                        <Calendar className="mr-3 text-green-400" size={24} />
                        Irrigation Schedule
                    </h2>
                    <div className="overflow-x-auto">
                        <table className="min-w-full bg-white/5 rounded-lg overflow-hidden">
                            <thead>
                                <tr className="bg-white/10 text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                    <th className="py-3 px-4 text-left font-semibold">Field</th>
                                    <th className="py-3 px-4 text-left font-semibold">Crop</th>
                                    <th className="py-3 px-4 text-left font-semibold">Next Irrigation</th>
                                    <th className="py-3 px-4 text-left font-semibold">Water Required</th>
                                </tr>
                            </thead>
                            <tbody>
                                {irrigationSchedule.map((item, index) => (
                                    <tr key={index} className="hover:bg-white/5 transition-all duration-200 border-b border-white/10">
                                        <td className="py-3 px-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">{item.field}</td>
                                        <td className="py-3 px-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">{item.crop}</td>
                                        <td className="py-3 px-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">{item.nextIrrigation}</td>
                                        <td className="py-3 px-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300">{item.waterRequired}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Irrigation Recommendations */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-2xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                        <AlertCircle className="mr-3 text-yellow-400" size={24} />
                        Irrigation Recommendations
                    </h2>
                    <ul className="space-y-3">
                        {recommendations.map((recommendation, index) => (
                            <li key={index} className="flex items-start text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                <span className="text-green-400 mr-2 mt-1">•</span>
                                {recommendation}
                            </li>
                        ))}
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default SoilWater;