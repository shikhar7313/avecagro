const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require('fs');
const path = require('path');

class ChartGenerator {
    constructor(options = {}) {
        this.chartJSNodeCanvas = new ChartJSNodeCanvas({
            width: 800,
            height: 600,
            backgroundColour: 'white'
        });
        this.dataFile = options.dataFile || path.join(__dirname, '../data/dashboard/chartsDemoData.json');
        this.chartData = null;
    }

    async loadChartData() {
        try {
            const raw = fs.readFileSync(this.dataFile, 'utf8');
            this.chartData = JSON.parse(raw);
        } catch (error) {
            console.warn('Failed to load chart data, using defaults', error.message);
            this.chartData = null;
        }
    }

    getSection(key, fallback = {}) {
        if (!this.chartData) return fallback;
        return this.chartData[key] || fallback;
    }

    // Generate Water Usage Trends Chart
    async generateWaterUsageChart(section = {}) {
        const months = section.months || ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
        const actual = section.actualKL || [15, 18, 22, 28, 35, 42];
        const target = section.targetKL || [14, 17, 20, 26, 32, 38];

        const data = {
            labels: months,
            datasets: [
                {
                    label: 'Actual (kL)',
                    data: actual,
                    borderColor: 'rgb(34,197,94)',
                    backgroundColor: 'rgba(34,197,94,0.15)',
                    fill: true,
                    tension: 0.35
                },
                {
                    label: 'Plan (kL)',
                    data: target,
                    borderColor: 'rgb(59,130,246)',
                    borderDash: [6, 4],
                    fill: false,
                    tension: 0.2
                }
            ]
        };

        const configuration = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Water Usage Trends'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Month'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Water Usage (Liters)'
                        }
                    }
                }
            }
        };

        const imageBuffer = await this.chartJSNodeCanvas.renderToBuffer(configuration);
        return imageBuffer;
    }

    // Generate Equipment Usage Analytics Chart
    async generateEquipmentUsageChart(section = {}) {
        const machines = Array.isArray(section.machines)
            ? section.machines
            : [
                { name: 'Tractors', hours: 450 },
                { name: 'Harvesters', hours: 320 }
            ];

        const data = {
            labels: machines.map(item => item.name),
            datasets: [{
                label: 'Usage Hours',
                data: machines.map(item => item.hours),
                backgroundColor: machines.map((_, idx) => {
                    const palette = ['#22c55e', '#0ea5e9', '#f97316', '#8b5cf6', '#14b8a6', '#facc15'];
                    return palette[idx % palette.length] + 'CC';
                })
            }]
        };

        const configuration = {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Equipment Usage Analytics'
                    },
                    legend: {
                        position: 'bottom',
                    }
                }
            }
        };

        const imageBuffer = await this.chartJSNodeCanvas.renderToBuffer(configuration);
        return imageBuffer;
    }

    // Generate Production Overview Chart
    async generateProductionOverviewChart(section = {}) {
        const crops = Array.isArray(section.crops)
            ? section.crops
            : [
                { name: 'Wheat', yieldTonnes: 1200, previousTonnes: 1100 },
                { name: 'Rice', yieldTonnes: 980, previousTonnes: 900 }
            ];

        const data = {
            labels: crops.map(crop => crop.name),
            datasets: [
                {
                    label: 'Current Year (tons)',
                    data: crops.map(crop => Number(crop.yieldTonnes ?? crop.tonnes ?? 0)),
                    backgroundColor: 'rgba(59,130,246,0.8)'
                },
                {
                    label: 'Previous Year (tons)',
                    data: crops.map(crop => {
                        if (typeof crop.previousTonnes === 'number') return crop.previousTonnes;
                        const current = Number(crop.yieldTonnes ?? crop.tonnes ?? 0);
                        return current ? Number((current * 0.92).toFixed(1)) : 0;
                    }),
                    backgroundColor: 'rgba(248,113,113,0.8)'
                }
            ]
        };

        const configuration = {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Production Overview by Crop Type'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Production (tons)'
                        }
                    }
                }
            }
        };

        const imageBuffer = await this.chartJSNodeCanvas.renderToBuffer(configuration);
        return imageBuffer;
    }

    // Generate Revenue Trends Chart
    async generateRevenueTrendsChart(section = {}) {
        const quarters = section.quarters || ['Q1', 'Q2', 'Q3', 'Q4'];
        const actual = section.actualUSDk || [200, 230, 260, 310];
        const target = section.targetUSDk || actual.map(v => v * 0.95);

        const data = {
            labels: quarters,
            datasets: [
                {
                    label: 'Actual ($K)',
                    data: actual,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16,185,129,0.2)',
                    fill: true,
                    tension: 0.35
                },
                {
                    label: 'Target ($K)',
                    data: target,
                    borderColor: '#f97316',
                    borderDash: [4, 4],
                    fill: false,
                    tension: 0.2
                }
            ]
        };

        const configuration = {
            type: 'line',
            data: data,
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Revenue Trends'
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Quarter'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Amount ($)'
                        }
                    }
                }
            }
        };

        const imageBuffer = await this.chartJSNodeCanvas.renderToBuffer(configuration);
        return imageBuffer;
    }

    // Save chart to file
    async saveChart(chartBuffer, filename) {
        const chartsDir = path.join(__dirname, '../../public/charts');

        if (!fs.existsSync(chartsDir)) {
            fs.mkdirSync(chartsDir, { recursive: true });
        }

        const filePath = path.join(chartsDir, filename);
        fs.writeFileSync(filePath, chartBuffer);
        console.log(`Chart saved to: ${filePath}`);
        return filePath;
    }

    // Generate all charts
    async generateAllCharts() {
        try {
            console.log('Generating charts...');
            await this.loadChartData();
            const waterSection = this.getSection('waterUsage');
            const equipmentSection = this.getSection('equipmentUtilization');
            const productionSection = this.getSection('productionOverview');
            const revenueSection = this.getSection('revenue');

            const waterUsageChart = await this.generateWaterUsageChart(waterSection);
            await this.saveChart(waterUsageChart, 'waterUsageChart.png');

            const equipmentChart = await this.generateEquipmentUsageChart(equipmentSection);
            await this.saveChart(equipmentChart, 'equipmentAnalyticsChart.png');

            const productionChart = await this.generateProductionOverviewChart(productionSection);
            await this.saveChart(productionChart, 'productionOverviewChart.png');

            const revenueChart = await this.generateRevenueTrendsChart(revenueSection);
            await this.saveChart(revenueChart, 'revenueTrendsChart.png');

            console.log('All charts generated successfully!');
            return true;
        } catch (error) {
            console.error('Error generating charts:', error);
            return false;
        }
    }
}

module.exports = ChartGenerator;

