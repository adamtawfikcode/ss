'use client'

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { motion } from 'framer-motion';
import { TrendingUp, Database, Eye, Users, Activity } from 'lucide-react';

// Mock data for charts
const generateMockData = () => {
  const data = [];
  for (let i = 0; i < 20; i++) {
    data.push({
      name: `Item ${i+1}`,
      value: Math.floor(Math.random() * 1000) + 500,
      uv: Math.floor(Math.random() * 100) + 50,
      pv: Math.floor(Math.random() * 1000) + 1000,
      amt: Math.floor(Math.random() * 2000) + 2000,
    });
  }
  return data;
};

const pieData = [
  { name: 'Group A', value: 400 },
  { name: 'Group B', value: 300 },
  { name: 'Group C', value: 300 },
  { name: 'Group D', value: 200 },
];

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const StatCard = ({ title, value, icon, trend }: { title: string, value: string, icon: React.ReactNode, trend?: string }) => (
  <motion.div 
    className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 hover:border-blue-500 transition-all duration-300 glow-pulse"
    whileHover={{ y: -5 }}
    initial={{ opacity: 0, scale: 0.9 }}
    animate={{ opacity: 1, scale: 1 }}
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-400 text-sm">{title}</p>
        <p className="text-2xl font-bold mt-1">{value}</p>
        {trend && <p className="text-green-500 text-sm mt-1 flex items-center"><TrendingUp size={16} className="mr-1" /> {trend}</p>}
      </div>
      <div className="text-blue-400 p-3 bg-blue-400/10 rounded-lg">
        {icon}
      </div>
    </div>
  </motion.div>
);

export default function GraphPage() {
  const [chartData, setChartData] = useState([]);
  const [timeRange, setTimeRange] = useState('7d');

  useEffect(() => {
    // Simulate fetching data
    setChartData(generateMockData());
  }, [timeRange]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white p-6">
      <div className="container mx-auto">
        <motion.h1 
          className="text-4xl font-bold mb-2 text-center bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          Interactive Data Visualization
        </motion.h1>
        <p className="text-gray-400 text-center mb-8">Real-time analytics and interactive graphs</p>

        {/* Time Range Selector */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex rounded-md bg-gray-800 p-1 border border-gray-700">
            {['1d', '7d', '30d', '90d'].map((range) => (
              <button
                key={range}
                className={`px-4 py-2 text-sm font-medium rounded-md ${
                  timeRange === range
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
                    : 'text-gray-300 hover:text-white'
                }`}
                onClick={() => setTimeRange(range)}
              >
                {range}
              </button>
            ))}
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <StatCard 
            title="Total Requests" 
            value="24.8K" 
            icon={<Database size={24} />} 
            trend="+12.4%"
          />
          <StatCard 
            title="Active Users" 
            value="1.2K" 
            icon={<Users size={24} />} 
            trend="+5.2%"
          />
          <StatCard 
            title="Avg Response" 
            value="42ms" 
            icon={<Activity size={24} />} 
            trend="-2.1%"
          />
          <StatCard 
            title="Success Rate" 
            value="99.8%" 
            icon={<Eye size={24} />} 
            trend="+0.3%"
          />
        </div>

        {/* Charts Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Line Chart */}
          <motion.div 
            className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 glow-pulse"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '0.5rem' }} 
                  itemStyle={{ color: 'white' }}
                  labelStyle={{ color: '#93C5FD' }}
                />
                <Line 
                  type="monotone" 
                  dataKey="uv" 
                  stroke="#60A5FA" 
                  strokeWidth={2} 
                  dot={{ r: 4, fill: '#3B82F6' }} 
                  activeDot={{ r: 6, fill: '#2563EB' }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="pv" 
                  stroke="#34D399" 
                  strokeWidth={2} 
                  dot={{ r: 4, fill: '#10B981' }} 
                  activeDot={{ r: 6, fill: '#059669' }} 
                />
              </LineChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Area Chart */}
          <motion.div 
            className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 glow-pulse"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h3 className="text-lg font-semibold mb-4">Data Volume</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorUv" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0.1}/>
                  </linearGradient>
                </defs>
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '0.5rem' }} 
                  itemStyle={{ color: 'white' }}
                  labelStyle={{ color: '#93C5FD' }}
                />
                <Area 
                  type="monotone" 
                  dataKey="amt" 
                  stroke="#3B82F6" 
                  fillOpacity={1} 
                  fill="url(#colorUv)" 
                />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Additional Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Bar Chart */}
          <motion.div 
            className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 glow-pulse"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
          >
            <h3 className="text-lg font-semibold mb-4">Resource Usage</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={chartData.slice(0, 10)}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="name" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '0.5rem' }} 
                  itemStyle={{ color: 'white' }}
                  labelStyle={{ color: '#93C5FD' }}
                />
                <Bar dataKey="value" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Pie Chart */}
          <motion.div 
            className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 glow-pulse"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
          >
            <h3 className="text-lg font-semibold mb-4">Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '0.5rem' }} 
                  itemStyle={{ color: 'white' }}
                  labelStyle={{ color: '#93C5FD' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </motion.div>
        </div>
      </div>
    </div>
  );
}