import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Sprout, Calendar, TrendingUp, RefreshCw } from 'lucide-react';
import gsap from 'gsap';

const REFRESH_INTERVAL = 60 * 1000; // refresh growth calculations every minute

const GrowthMeter = ({ tasks = [] }) => {
    const cardRef = useRef();
    const intervalRef = useRef();
    const [plantingData, setPlantingData] = useState([]);
    const [lastUpdated, setLastUpdated] = useState(Date.now());

    const handleMouseMove = () => {
        if (!cardRef.current) return;
        // Basic shadow effect without any transform/tilt
        cardRef.current.style.boxShadow = '0 15px 30px rgba(16,185,129,0.7), 0 0 15px rgba(16,185,129,0.5)';
    };

    const handleMouseLeave = () => {
        if (!cardRef.current) return;
        gsap.to(cardRef.current, {
            overwrite: 'auto',
            boxShadow: '0 4px 10px rgba(0,0,0,0.2)',
            ease: 'power3.out',
            duration: 0.6
        });
    };

    const computePlantingTasks = useCallback(() => {
        if (!Array.isArray(tasks) || tasks.length === 0) return [];
        const plantingTasks = tasks.filter(task =>
            task.title &&
            task.title.toLowerCase().includes('plant') &&
            task.title.toLowerCase().includes('seed')
        );

        return plantingTasks.map(task => {
            const cropName = task.title.replace(/plant\s+/i, '').replace(/\s+seeds?/i, '').trim();
            const plantingDate = new Date(task.date);
            const today = new Date();
            const daysElapsed = Math.floor((today - plantingDate) / (1000 * 60 * 60 * 24));

            const growthPeriods = {
                'corn': 120,
                'wheat': 90,
                'rice': 150,
                'tomato': 80,
                'potato': 100,
                'soybean': 110,
                'cotton': 180,
                'sunflower': 100,
                'default': 90
            };

            const cropKey = cropName.toLowerCase();
            const totalGrowthDays = growthPeriods[cropKey] || growthPeriods.default;
            const growthPercentage = Math.min(Math.max((daysElapsed / totalGrowthDays) * 100, 0), 100);

            return {
                cropName,
                plantingDate: task.date,
                daysElapsed: Math.max(daysElapsed, 0),
                growthPercentage,
                status: task.status,
                totalGrowthDays
            };
        }).filter(crop => crop.status === 'complete');
    }, [tasks]);

    useEffect(() => {
        const data = computePlantingTasks();
        setPlantingData(data);
        setLastUpdated(Date.now());
    }, [computePlantingTasks]);

    useEffect(() => {
        intervalRef.current = setInterval(() => {
            const data = computePlantingTasks();
            setPlantingData(data);
            setLastUpdated(Date.now());
        }, REFRESH_INTERVAL);

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [computePlantingTasks]);

    const getGrowthStage = (percentage) => {
        if (percentage < 25) return { stage: 'Seedling', emoji: '🌱', color: 'text-green-300' };
        if (percentage < 50) return { stage: 'Growing', emoji: '🌿', color: 'text-green-400' };
        if (percentage < 75) return { stage: 'Maturing', emoji: '🌾', color: 'text-yellow-400' };
        if (percentage < 100) return { stage: 'Pre-Harvest', emoji: '🌽', color: 'text-orange-400' };
        return { stage: 'Ready to Harvest', emoji: '🌾', color: 'text-yellow-300' };
    };

    const formatLastUpdated = () => {
        const diffMs = Date.now() - lastUpdated;
        if (diffMs < 60 * 1000) return 'a few seconds ago';
        if (diffMs < 60 * 60 * 1000) return `${Math.round(diffMs / 60000)} min ago`;
        return `${Math.round(diffMs / 3600000)} hr ago`;
    };

    if (!tasks || !Array.isArray(tasks)) {
        return (
            <div className="glass-card p-6 h-full flex items-center justify-center">
                <p className="text-gray-400 text-center">No crop data available</p>
            </div>
        );
    }

    return (
        <div
            ref={cardRef}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className="glass-card p-6 h-full"
            style={{
                // Removed transform style properties to disable tilt
                willChange: 'box-shadow',
                transition: 'box-shadow 0.3s ease'
            }}
        >
            <div className="flex items-center justify-between mb-6">
                <h3 className="text-xl font-semibold text-white flex items-center">
                    <Sprout className="w-6 h-6 mr-2 text-green-400" />
                    Crop Growth
                </h3>
                <div className="flex items-center gap-3 text-xs text-gray-400">
                    <div className="flex items-center gap-1">
                        <RefreshCw className="w-3 h-3" />
                        <span>Updated {formatLastUpdated()}</span>
                    </div>
                    <TrendingUp className="w-5 h-5 text-green-400" />
                </div>
            </div>

            {plantingData.length === 0 ? (
                <div className="flex flex-col items-center justify-center h-full text-center">
                    <Sprout className="w-12 h-12 text-gray-500 mb-4" />
                    <p className="text-gray-400 text-sm mb-2">No planted crops found</p>
                    <p className="text-gray-500 text-xs">
                        Complete planting tasks to track growth progress
                    </p>
                </div>
            ) : (
                <div className="space-y-4 max-h-80 overflow-y-auto">
                    {plantingData.map((crop, index) => {
                        const growthStage = getGrowthStage(crop.growthPercentage);

                        return (
                            <div key={index} className="bg-gray-700/30 rounded-lg p-4 hover:bg-gray-700/50 transition-all duration-200">
                                <div className="flex items-center justify-between mb-3">
                                    <div className="flex items-center">
                                        <span className="text-lg mr-3">{growthStage.emoji}</span>
                                        <div>
                                            <h4 className="text-sm font-medium text-white capitalize">
                                                {crop.cropName}
                                            </h4>
                                            <p className={`text-xs ${growthStage.color}`}>
                                                {growthStage.stage}
                                            </p>
                                        </div>
                                    </div>
                                    
                                    {/* Circular Progress Bar */}
                                    <div className="relative w-16 h-16">
                                        <svg className="w-16 h-16 transform -rotate-90" viewBox="0 0 64 64">
                                            {/* Background circle */}
                                            <circle
                                                cx="32"
                                                cy="32"
                                                r="28"
                                                stroke="rgb(75, 85, 99)"
                                                strokeWidth="4"
                                                fill="none"
                                            />
                                            {/* Progress circle */}
                                            <circle
                                                cx="32"
                                                cy="32"
                                                r="28"
                                                stroke="url(#gradient)"
                                                strokeWidth="4"
                                                fill="none"
                                                strokeLinecap="round"
                                                strokeDasharray={`${2 * Math.PI * 28}`}
                                                strokeDashoffset={`${2 * Math.PI * 28 * (1 - crop.growthPercentage / 100)}`}
                                                className="transition-all duration-700 ease-out"
                                            />
                                            {/* Gradient definition */}
                                            <defs>
                                                <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                                                    <stop offset="0%" stopColor="rgb(74, 222, 128)" />
                                                    <stop offset="100%" stopColor="rgb(34, 197, 94)" />
                                                </linearGradient>
                                            </defs>
                                        </svg>
                                        {/* Percentage text in center */}
                                        <div className="absolute inset-0 flex items-center justify-center">
                                            <span className="text-sm font-bold text-green-400">
                                                {Math.round(crop.growthPercentage)}%
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                <div className="flex items-center justify-between text-xs text-gray-400 mb-2">
                                    <span>{crop.daysElapsed} days elapsed</span>
                                    <span>Est. {crop.totalGrowthDays} days total</span>
                                </div>

                                <div className="flex items-center justify-between text-xs text-gray-400">
                                    <div className="flex items-center">
                                        <Calendar className="w-3 h-3 mr-1" />
                                        <span>Planted: {new Date(crop.plantingDate).toLocaleDateString()}</span>
                                    </div>
                                    <span>Est. {crop.totalGrowthDays} days total</span>
                                </div>

                                {crop.growthPercentage >= 100 && (
                                    <div className="mt-3 bg-yellow-400/20 rounded-lg p-2">
                                        <div className="flex items-center text-yellow-300 text-xs">
                                            <span className="mr-1">🎉</span>
                                            <span className="font-medium">Ready for harvest!</span>
                                        </div>
                                    </div>
                                )}

                                {crop.growthPercentage >= 75 && crop.growthPercentage < 100 && (
                                    <div className="mt-3 bg-orange-400/20 rounded-lg p-2">
                                        <div className="flex items-center text-orange-300 text-xs">
                                            <span className="mr-1">⚠️</span>
                                            <span>Monitor closely - harvest approaching</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </div>
            )}

            {plantingData.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-600">
                    <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>{plantingData.length} crops tracked</span>
                        <span>Auto-refreshing every minute</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default GrowthMeter;