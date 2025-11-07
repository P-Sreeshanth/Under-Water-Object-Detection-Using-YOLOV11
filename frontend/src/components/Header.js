import React from 'react';
import { motion } from 'framer-motion';
import { Activity, Zap, Waves } from 'lucide-react';
import './Header.css';

const Header = ({ stats }) => {
  return (
    <motion.header 
      className="header"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5, type: "spring" }}
    >
      <div className="header-content">
        <div className="header-left">
          <div className="logo-container">
            <Waves className="logo-icon" />
            <div className="logo-text">
              <h1 className="logo-title glow">AQUA VISION</h1>
              <p className="logo-subtitle">Underwater Detection System</p>
            </div>
          </div>
        </div>

        <div className="header-center">
          <div className="system-status">
            <div className="status-indicator pulse-dot"></div>
            <span className="status-text">SYSTEM OPERATIONAL</span>
          </div>
        </div>

        <div className="header-right">
          <div className="stat-badge">
            <Activity className="stat-icon" />
            <div className="stat-content">
              <span className="stat-label">DETECTIONS</span>
              <span className="stat-value">{stats.totalDetections}</span>
            </div>
          </div>
          
          <div className="stat-badge">
            <Zap className="stat-icon" />
            <div className="stat-content">
              <span className="stat-label">MODELS</span>
              <span className="stat-value">
                {(stats.models.seaclear && stats.models.aquarium) ? '2' : '0'}
              </span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="header-line"></div>
    </motion.header>
  );
};

export default Header;
