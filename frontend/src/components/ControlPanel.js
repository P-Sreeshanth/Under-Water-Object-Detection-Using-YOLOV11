import React from 'react';
import { motion } from 'framer-motion';
import { Play, RefreshCw, Upload, Sliders } from 'lucide-react';
import './ControlPanel.css';

const ControlPanel = ({ settings, setSettings, onAnalyze, onClear, isAnalyzing, hasImage, fileInputRef, onFileChange }) => {
  return (
    <motion.div 
      className="control-panel"
      whileHover={{ boxShadow: '0 0 20px rgba(0, 212, 255, 0.3)' }}
    >
      <div className="panel-header">
        <Sliders className="panel-icon" />
        <h3 className="panel-title">CONTROL PANEL</h3>
      </div>

      <div className="panel-content">
        {/* Upload Button */}
        <button 
          className="control-button upload-btn"
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload size={20} />
          <span>Upload Image</span>
        </button>

        {/* Confidence Threshold */}
        <div className="control-group">
          <label className="control-label">
            <span>Confidence Threshold</span>
            <span className="control-value">{(settings.confidence * 100).toFixed(0)}%</span>
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={settings.confidence}
            onChange={(e) => setSettings({ ...settings, confidence: parseFloat(e.target.value) })}
            className="control-slider"
          />
          <div className="slider-marks">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>

        {/* Image Enhancement Toggle */}
        <div className="control-group">
          <label className="toggle-label">
            <span>Image Enhancement</span>
            <div className="toggle-switch">
              <input
                type="checkbox"
                checked={settings.enhanceImage}
                onChange={(e) => setSettings({ ...settings, enhanceImage: e.target.checked })}
                className="toggle-input"
              />
              <span className="toggle-slider"></span>
            </div>
          </label>
        </div>

        {/* Action Buttons */}
        <div className="action-buttons">
          <button 
            className="control-button analyze-btn"
            onClick={onAnalyze}
            disabled={!hasImage || isAnalyzing}
          >
            {isAnalyzing ? (
              <>
                <div className="spinner"></div>
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Play size={20} />
                <span>Analyze</span>
              </>
            )}
          </button>

          <button 
            className="control-button clear-btn"
            onClick={onClear}
            disabled={!hasImage}
          >
            <RefreshCw size={20} />
            <span>Clear</span>
          </button>
        </div>

        {/* System Info */}
        <div className="system-info">
          <div className="info-item">
            <span className="info-label">Models Active</span>
            <span className="info-value">Seaclear + Aquarium</span>
          </div>
          <div className="info-item">
            <span className="info-label">Total Classes</span>
            <span className="info-value">47</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ControlPanel;
