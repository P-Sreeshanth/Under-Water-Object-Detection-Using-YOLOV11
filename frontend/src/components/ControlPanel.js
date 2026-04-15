import React from 'react';
import { motion } from 'framer-motion';
import { Play, RefreshCw, Upload, Sliders, Radio, Link, Link2Off, Image as ImageIcon } from 'lucide-react';
import './ControlPanel.css';

const ControlPanel = ({
  settings,
  setSettings,
  mode,
  setMode,
  streamSource,
  setStreamSource,
  onConnectStream,
  onDisconnectStream,
  isStreamConnected,
  onAnalyze,
  onClear,
  isAnalyzing,
  hasImage,
  fileInputRef,
}) => {
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
        <div className="mode-switcher">
          <button
            className={`mode-btn ${mode === 'image' ? 'active' : ''}`}
            onClick={() => setMode('image')}
            type="button"
          >
            <ImageIcon size={16} />
            <span>Image Mode</span>
          </button>
          <button
            className={`mode-btn ${mode === 'stream' ? 'active' : ''}`}
            onClick={() => setMode('stream')}
            type="button"
          >
            <Radio size={16} />
            <span>Live Stream</span>
          </button>
        </div>

        {/* Upload Button */}
        <button 
          className="control-button upload-btn"
          onClick={() => fileInputRef.current?.click()}
          disabled={mode !== 'image'}
        >
          <Upload size={20} />
          <span>Upload Image</span>
        </button>

        {mode === 'stream' && (
          <div className="stream-controls">
            <label className="stream-label">Stream Source (camera index, file, or RTSP URL)</label>
            <input
              type="text"
              value={streamSource}
              onChange={(e) => setStreamSource(e.target.value)}
              className="stream-input"
              placeholder="0 or rtsp://..."
            />
            <div className="stream-buttons">
              <button
                className="control-button analyze-btn"
                onClick={onConnectStream}
                disabled={isStreamConnected}
                type="button"
              >
                <Link size={18} />
                <span>Connect</span>
              </button>
              <button
                className="control-button clear-btn"
                onClick={onDisconnectStream}
                disabled={!isStreamConnected}
                type="button"
              >
                <Link2Off size={18} />
                <span>Disconnect</span>
              </button>
            </div>
          </div>
        )}

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
            disabled={!hasImage || isAnalyzing || mode !== 'image'}
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
            disabled={!hasImage && !isStreamConnected}
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
            <span className="info-label">Stream Status</span>
            <span className="info-value">{isStreamConnected ? 'Connected' : 'Idle'}</span>
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
