'use client'

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { BarChart3, Globe, Zap, Activity, Network, Settings } from 'lucide-react';

const GlowButton = ({ children, href = "#", onClick, variant = "primary" }) => {
  const baseClasses = "px-6 py-3 rounded-lg font-medium transition-all duration-300 transform hover:scale-105 focus:outline-none";
  const variants = {
    primary: "bg-blue-600 text-white shadow-lg shadow-blue-500/30 hover:shadow-blue-500/50",
    secondary: "bg-gray-800 text-white border border-gray-700 shadow-lg shadow-gray-500/20 hover:shadow-gray-500/30"
  };

  return (
    <motion.div
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      className="relative inline-block"
    >
      <div className={`absolute -inset-0.5 bg-blue-500 rounded-lg blur opacity-75 animate-pulse ${variant === 'primary' ? 'group-hover:opacity-100' : ''}`}></div>
      <Link href={href}>
        <button className={`${baseClasses} ${variants[variant]}`}>
          {children}
        </button>
      </Link>
    </motion.div>
  );
};

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <motion.h1 
            className="text-5xl md:text-7xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-600"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            Nexum Analytics
          </motion.h1>
          <motion.p 
            className="text-xl text-gray-300 max-w-2xl mx-auto mb-10"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Advanced analytics platform with real-time data visualization and interactive insights
          </motion.p>
          
          <motion.div 
            className="flex flex-col sm:flex-row justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <GlowButton href="/dashboard">Get Started</GlowButton>
            <GlowButton variant="secondary" href="/graph">View Graph</GlowButton>
          </motion.div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-16">
          {[{
            icon: <BarChart3 size={40} />,
            title: "Real-time Analytics",
            description: "Monitor your data streams in real-time with advanced visualization tools"
          }, {
            icon: <Globe size={40} />,
            title: "Global Insights",
            description: "Access insights from distributed data sources worldwide"
          }, {
            icon: <Zap size={40} />,
            title: "Lightning Fast",
            description: "Optimized algorithms for rapid data processing and analysis"
          }, {
            icon: <Activity size={40} />,
            title: "Activity Tracking",
            description: "Track and analyze user behavior patterns effectively"
          }, {
            icon: <Network size={40} />,
            title: "Network Analysis",
            description: "Analyze complex network topologies and relationships"
          }, {
            icon: <Settings size={40} />,
            title: "Customizable UI",
            description: "Fully customizable interface to suit your specific needs"
          }].map((feature, index) => (
            <motion.div 
              key={index}
              className="bg-gray-800/50 backdrop-blur-sm p-6 rounded-xl border border-gray-700 hover:border-blue-500 transition-all duration-300"
              whileHover={{ y: -10 }}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 * index }}
            >
              <div className="text-blue-400 mb-4 flex justify-center">{feature.icon}</div>
              <h3 className="text-xl font-semibold mb-2 text-center">{feature.title}</h3>
              <p className="text-gray-400 text-center">{feature.description}</p>
            </motion.div>
          ))}
        </div>

        {/* Stats Section */}
        <div className="bg-gray-800/30 backdrop-blur-sm rounded-2xl p-8 border border-gray-700">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
            {[
              { value: "99.9%", label: "Uptime" },
              { value: "1M+", label: "Data Points" },
              { value: "50ms", label: "Response Time" },
              { value: "24/7", label: "Monitoring" }
            ].map((stat, index) => (
              <div key={index} className="p-4">
                <div className="text-3xl font-bold text-blue-400 mb-2">{stat.value}</div>
                <div className="text-gray-400">{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}