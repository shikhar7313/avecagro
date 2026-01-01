import React, { useState, useEffect } from 'react';
import useText from '../hooks/useText';
import {
    Sprout,
    Scale,
    TrendingUp,
    Calendar,
    BarChart3,
    DollarSign,
    Droplets,
    FileText,
    PieChart,
    Activity,
    Target,
    Users,

} from 'lucide-react';
import ProductionOverviewChart from './charts/ProductionOverviewChart';
import RevenueTrendsChart from './charts/RevenueTrendsChart';

// Consolidated dashboard data
import chartsDemoData from '../data/dashboard/chartsDemoData.json';

const ReportsAnalytics = () => {
    const { t } = useText();
    const [stats, setStats] = useState({
        totalCrops: 0,
        currentYield: 0,
        growthStatus: 'Loading...',
        upcomingHarvests: []
    });

    const [cropPerformance, setCropPerformance] = useState([]);
    const [financialData, setFinancialData] = useState([]);

    // Function to calculate dynamic statistics
    const calculateStats = () => {
        try {
            const crops = chartsDemoData.productionOverview?.crops || [];
            const growthStages = chartsDemoData.cropGrowth?.stages || [];

            const totalCrops = crops.length;

            const currentYield = crops.reduce((total, crop) => total + (crop.yieldTonnes || 0), 0);

            const avgProgress = growthStages.length
                ? growthStages.reduce((sum, crop) => sum + (crop.progressPct || 0), 0) / growthStages.length
                : 0;
            const growthStatus = avgProgress >= 75 ? 'Excellent' : avgProgress >= 55 ? 'Good' : 'Healthy';

            const upcomingHarvests = chartsDemoData.productionOverview?.harvestSchedule || [];

            return {
                totalCrops,
                currentYield,
                growthStatus,
                upcomingHarvests
            };
        } catch (error) {
            console.error('Error calculating stats:', error);
            return {
                totalCrops: 0,
                currentYield: 0,
                growthStatus: 'Error',
                upcomingHarvests: []
            };
        }
    };

    // Function to process crop performance data
    const processCropPerformance = () => {
        try {
            const crops = chartsDemoData.productionOverview?.crops || [];
            return crops.map(crop => ({
                name: crop.name,
                value: crop.yieldTonnes || 0,
                trend: Math.round(crop.changePct || 0)
            })).slice(0, 4);
        } catch (error) {
            console.error('Error processing crop performance:', error);
            return [];
        }
    };

    // Function to calculate financial data
    const calculateFinancialData = () => {
        try {
            const crops = chartsDemoData.productionOverview?.crops || [];
            const totalYield = crops.reduce((total, crop) => total + (crop.yieldTonnes || 0), 0);
            const totalRevenue = totalYield * 850; // Price per ton
            const operatingCosts = Math.floor(totalRevenue * 0.65); // 65% of revenue as costs
            const netProfit = totalRevenue - operatingCosts;
            const roi = totalRevenue > 0 ? ((netProfit / operatingCosts) * 100).toFixed(1) : 0;

            return [
                { label: 'Total Revenue', value: Math.floor(totalRevenue).toLocaleString(), unit: '₹', trend: 12 },
                { label: 'Operating Costs', value: operatingCosts.toLocaleString(), unit: '₹', trend: -5 },
                { label: 'Net Profit', value: netProfit.toLocaleString(), unit: '₹', trend: 18 },
                { label: 'ROI', value: roi, unit: '%', trend: 8 }
            ];
        } catch (error) {
            console.error('Error calculating financial data:', error);
            return [
                { label: 'Total Revenue', value: '0', unit: '₹', trend: 0 },
                { label: 'Operating Costs', value: '0', unit: '₹', trend: 0 },
                { label: 'Net Profit', value: '0', unit: '₹', trend: 0 },
                { label: 'ROI', value: '0', unit: '%', trend: 0 }
            ];
        }
    };

    // Function to fetch and process all data
    const fetchAndProcessData = () => {
        try {
            // Calculate statistics
            const calculatedStats = calculateStats();
            setStats(calculatedStats);

            // Process crop performance
            const processedCropPerformance = processCropPerformance();
            setCropPerformance(processedCropPerformance);

            // Calculate financial data
            const calculatedFinancialData = calculateFinancialData();
            setFinancialData(calculatedFinancialData);

            console.log('Data processed successfully:', {
                stats: calculatedStats,
                cropPerformance: processedCropPerformance,
                financialData: calculatedFinancialData
            });
        } catch (error) {
            console.error('Error fetching and processing data:', error);
        }
    };

    useEffect(() => {
        fetchAndProcessData();
    }, []);

    const StatCard = ({ title, value, subtitle, icon: Icon, iconColor, trend = null }) => (
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 hover:bg-white/15 transition-all duration-300">
            <div className="flex justify-between items-center mb-4">
                <h2 className="text-lg font-semibold text-gray-100">{title}</h2>
                <Icon className={`w-6 h-6 ${iconColor}`} />
            </div>
            <p className="text-2xl font-semibold text-white mb-2">{value}</p>
            {subtitle && (
                <p className="text-sm text-green-400">{subtitle}</p>
            )}
            {trend && (
                <p className="text-sm text-green-400 mt-1">{trend}</p>
            )}
        </div>
    );

    const ChartPlaceholder = ({ title, chartType = "chart" }) => (
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 h-64">
            <h2 className="text-xl font-bold mb-4 text-gray-100">{title}</h2>
            <div className="flex items-center justify-center h-full bg-white/5 rounded-lg border-2 border-dashed border-white/20">
                <div className="text-center">
                    <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-400 text-sm">{title} Visualization</p>
                </div>
            </div>
        </div>
    );

    const ReportSection = ({ title, children }) => (
        <div className="bg-white/10 backdrop-blur-md border border-white/20 rounded-xl p-6 mb-6">
            <h3 className="text-lg font-semibold text-gray-100 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                {title}
            </h3>
            {children}
        </div>
    );

    const MetricRow = ({ label, value, unit = "", trend = null }) => (
        <div className="flex justify-between items-center py-2 border-b border-white/10 last:border-b-0">
            <span className="text-gray-300">{label}</span>
            <div className="text-right">
                <span className="text-white font-medium">{value} {unit}</span>
                {trend && (
                    <span className={`text-sm ml-2 ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {trend > 0 ? '+' : ''}{trend}%
                    </span>
                )}
            </div>
        </div>
    );

    return (
        <div className="p-6 space-y-8 max-w-7xl mx-auto">
            {/* Header */}
            <div className="mb-8">
                <h1 className="text-3xl font-bold text-white mb-2">Reports & Analytics</h1>
                <p className="text-gray-300">Farm performance insights</p>
            </div>

            {/* Statistics Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatCard
                    title="Total Crops"
                    value={`${stats.totalCrops} Types`}
                    subtitle="+2% this month"
                    icon={Sprout}
                    iconColor="text-green-400"
                />
                <StatCard
                    title="Current Yield"
                    value={`${stats.currentYield} tons`}
                    subtitle="+5.67% from last month"
                    icon={Scale}
                    iconColor="text-blue-400"
                />
                <StatCard
                    title="Growth Status"
                    value={stats.growthStatus}
                    icon={TrendingUp}
                    iconColor="text-purple-400"
                />
                <StatCard
                    title="Upcoming Harvests"
                    value={`${stats.upcomingHarvests.length} Crops`}
                    subtitle="Next 30 days"
                    icon={Calendar}
                    iconColor="text-yellow-400"
                />
            </div>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <div className="bg-white/5 rounded-lg p-4">
                    <ProductionOverviewChart />
                </div>
                <div className="bg-white/5 rounded-lg p-4">
                    <RevenueTrendsChart />
                </div>
            </div>

            {/* Simplified Reports */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Crop Performance */}
                <ReportSection title="Crop Performance">
                    <div className="space-y-3">
                        {cropPerformance.map((crop, index) => (
                            <MetricRow
                                key={index}
                                label={crop.name}
                                value={crop.value}
                                unit="tons"
                                trend={crop.trend}
                            />
                        ))}
                    </div>
                </ReportSection>

                {/* Financial Overview */}
                <ReportSection title="Financial Overview">
                    <div className="space-y-3">
                        {financialData.map((item, index) => (
                            <MetricRow
                                key={index}
                                label={item.label}
                                value={item.value}
                                unit={item.unit}
                                trend={item.trend}
                            />
                        ))}
                    </div>
                </ReportSection>
            </div>
        </div>
    );
};

export default ReportsAnalytics;