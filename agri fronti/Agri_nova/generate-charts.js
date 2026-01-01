const ChartGenerator = require('./src/utils/chartGenerator');

// Function to generate all required charts
async function generateCharts() {
    const generator = new ChartGenerator();

    console.log('Starting chart generation process...');
    console.log('Charts to generate:');
    console.log('1. Water Usage Trends Chart');
    console.log('2. Equipment Usage Analytics Chart');
    console.log('3. Production Overview Chart');
    console.log('4. Revenue Trends Chart');
    console.log('');

    const success = await generator.generateAllCharts();

    if (success) {
        console.log('✅ All charts generated successfully!');
        console.log('Charts saved in: public/charts/');
        console.log('- waterUsageChart.png');
        console.log('- equipmentAnalyticsChart.png');
        console.log('- productionOverviewChart.png');
        console.log('- revenueTrendsChart.png');
    } else {
        console.log('❌ Error generating charts. Please check the logs above.');
    }
}

// Run the chart generation
if (require.main === module) {
    generateCharts().catch(console.error);
}

module.exports = { generateCharts };