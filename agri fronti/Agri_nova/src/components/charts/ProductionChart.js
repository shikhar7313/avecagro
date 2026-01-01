import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp } from 'lucide-react';

const ProductionChart = ({ data }) => {
  if (!data) {
    return (
      <div className="chart-container">
        <h3 className="text-xl font-semibold text-white mb-4">Production Overview</h3>
        <p className="text-gray-400">Production data unavailable</p>
      </div>
    );
  }

  return (
    <div className="chart-container hover-lift">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-semibold text-gray-800 mb-1">Production Overview</h3>
          <p className="text-gray-600">Monthly production trends</p>
        </div>
        <div className="flex items-center space-x-2 text-primary-600">
          <TrendingUp className="w-5 h-5" />
          <span className="text-sm font-medium">+15.2% vs last year</span>
        </div>
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
            <XAxis 
              dataKey="month" 
              tick={{ fontSize: 12 }}
              axisLine={{ stroke: '#e0e7ff' }}
            />
            <YAxis 
              tick={{ fontSize: 12 }}
              axisLine={{ stroke: '#e0e7ff' }}
            />
            <Tooltip 
              contentStyle={{
                backgroundColor: 'rgba(255, 255, 255, 0.9)',
                border: '1px solid rgba(16, 185, 129, 0.2)',
                borderRadius: '8px',
                backdropFilter: 'blur(10px)'
              }}
            />
            <Line 
              type="monotone" 
              dataKey="production" 
              stroke="#10b981" 
              strokeWidth={3}
              dot={{ fill: '#10b981', strokeWidth: 2, r: 6 }}
              activeDot={{ r: 8, stroke: '#10b981', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
        <span>Peak Season: Aug - Dec</span>
        <span>Avg Monthly: 185 tons</span>
      </div>

      {/* Monthly Production */}
      {data.monthly && (
        <div className="mb-8">
          <h4 className="text-lg font-medium text-gray-300 mb-4">Monthly Production</h4>
          <div className="grid grid-cols-6 gap-4">
            {data.monthly.map((month, index) => (
              <div key={index} className="text-center">
                <div 
                  className="bg-emerald-500 rounded-t mb-2 mx-auto transition-all duration-300 hover:bg-emerald-400"
                  style={{ 
                    height: `${(month.value / Math.max(...data.monthly.map(m => m.value))) * 100}px`,
                    width: '30px'
                  }}
                ></div>
                <div className="text-sm text-gray-400">{month.month}</div>
                <div className="text-xs text-white">{month.value}T</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Crop Distribution */}
      {data.crops && (
        <div>
          <h4 className="text-lg font-medium text-gray-300 mb-4">Crop Distribution</h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {data.crops.map((crop, index) => (
              <div key={index} className="text-center p-4 rounded-lg bg-gray-700/30">
                <div className="text-lg font-bold text-white">{crop.value}T</div>
                <div className="text-sm text-gray-400">{crop.name}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ProductionChart;
