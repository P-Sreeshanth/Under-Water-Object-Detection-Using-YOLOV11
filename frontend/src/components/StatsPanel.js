import React from 'react';
import { motion } from 'framer-motion';
import { BarChart3, TrendingUp, Clock, Target } from 'lucide-react';
import './StatsPanel.css';

const StatsPanel = ({ stats, detections }) => {
  // Group detections by model
  const seaclearCount = detections.filter(d => d.model === 'seaclear').length;
  const aquariumCount = detections.filter(d => d.model === 'aquarium').length;

  return (
    <motion.div 
      className="stats-panel"
      whileHover={{ boxShadow: '0 0 20px rgba(0, 212, 255, 0.3)' }}
    >
      <div className="panel-header">
        <BarChart3 className="panel-icon" />
        <h3 className="panel-title">STATISTICS</h3>
      </div>

      <div className="panel-content">
        {/* Main Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-icon-wrapper">
              <Target className="stat-icon" />
            </div>
            <div className="stat-info">
              <span className="stat-label">Total Detected</span>
              <span className="stat-number">{stats.totalDetections}</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon-wrapper">
              <TrendingUp className="stat-icon" />
            </div>
            <div className="stat-info">
              <span className="stat-label">Avg Confidence</span>
              <span className="stat-number">{stats.confidence}%</span>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-icon-wrapper">
              <Clock className="stat-icon" />
            </div>
            <div className="stat-info">
              <span className="stat-label">Process Time</span>
              <span className="stat-number">{stats.processingTime}s</span>
            </div>
          </div>
        </div>

        {/* Model Distribution */}
        {detections.length > 0 && (
          <div className="model-distribution">
            <h4 className="distribution-title">Model Distribution</h4>
            <div className="distribution-bars">
              <div className="bar-item">
                <div className="bar-label">
                  <span className="bar-name">Seaclear</span>
                  <span className="bar-value">{seaclearCount}</span>
                </div>
                <div className="bar-track">
                  <motion.div 
                    className="bar-fill seaclear"
                    initial={{ width: 0 }}
                    animate={{ width: `${(seaclearCount / stats.totalDetections) * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                </div>
              </div>

              <div className="bar-item">
                <div className="bar-label">
                  <span className="bar-name">Aquarium</span>
                  <span className="bar-value">{aquariumCount}</span>
                </div>
                <div className="bar-track">
                  <motion.div 
                    className="bar-fill aquarium"
                    initial={{ width: 0 }}
                    animate={{ width: `${(aquariumCount / stats.totalDetections) * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Model Status Indicators */}
        <div className="model-status">
          <h4 className="status-title">Active Models</h4>
          <div className="status-list">
            <div className={`status-item ${stats.models.seaclear ? 'active' : 'inactive'}`}>
              <div className="status-dot"></div>
              <span className="status-name">Seaclear Marine (40 classes)</span>
            </div>
            <div className={`status-item ${stats.models.aquarium ? 'active' : 'inactive'}`}>
              <div className="status-dot"></div>
              <span className="status-name">Aquarium Life (7 classes)</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default StatsPanel;
