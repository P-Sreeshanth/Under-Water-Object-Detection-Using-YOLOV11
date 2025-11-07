import React from 'react';
import { motion } from 'framer-motion';
import { Upload, Loader, ScanLine } from 'lucide-react';
import './VideoCanvas.css';

const VideoCanvas = ({ image, annotatedImage, detections, isAnalyzing, onImageUpload }) => {
  return (
    <div className="video-canvas-container">
      <div className="canvas-wrapper glow-border">
        {!image && !annotatedImage && (
          <motion.div 
            className="upload-zone"
            onClick={onImageUpload}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <Upload className="upload-icon" size={64} />
            <h3 className="upload-title">Deploy Detection System</h3>
            <p className="upload-subtitle">Drop underwater image or click to select</p>
            <div className="upload-formats">
              <span className="format-badge">JPG</span>
              <span className="format-badge">PNG</span>
              <span className="format-badge">JPEG</span>
            </div>
          </motion.div>
        )}

        {image && !annotatedImage && !isAnalyzing && (
          <motion.div 
            className="image-preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <img src={image} alt="Original" className="preview-image" />
            <div className="preview-overlay">
              <div className="ready-indicator">
                <ScanLine className="scan-icon" />
                <span>Ready for Analysis</span>
              </div>
            </div>
          </motion.div>
        )}

        {isAnalyzing && (
          <motion.div 
            className="analyzing-state"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <img src={image} alt="Analyzing" className="preview-image analyzing" />
            <div className="analyzing-overlay">
              <div className="scan-line"></div>
              <div className="analyzing-content">
                <Loader className="loader-icon" />
                <h3 className="analyzing-title">AI PROCESSING</h3>
                <p className="analyzing-subtitle">Scanning for underwater objects...</p>
                <div className="progress-bar">
                  <div className="progress-fill"></div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {annotatedImage && !isAnalyzing && (
          <motion.div 
            className="result-view"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <img src={annotatedImage} alt="Detection Results" className="result-image" />
            <div className="detection-overlay">
              <div className="corner-frame top-left"></div>
              <div className="corner-frame top-right"></div>
              <div className="corner-frame bottom-left"></div>
              <div className="corner-frame bottom-right"></div>
              
              <div className="detection-count-badge">
                <span className="count-number">{detections.length}</span>
                <span className="count-label">Objects Detected</span>
              </div>
            </div>
          </motion.div>
        )}

        {/* HUD Grid Overlay */}
        {(image || annotatedImage) && (
          <div className="hud-grid">
            <svg className="grid-svg" width="100%" height="100%">
              <defs>
                <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                  <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(0,212,255,0.1)" strokeWidth="1"/>
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
            </svg>
          </div>
        )}
      </div>

      {/* Detection Info Panel */}
      {annotatedImage && detections.length > 0 && (
        <motion.div 
          className="detection-info-panel"
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <h4 className="panel-title">Detected Objects</h4>
          <div className="detection-list">
            {detections.slice(0, 5).map((det, index) => (
              <motion.div
                key={index}
                className="detection-item"
                initial={{ x: -20, opacity: 0 }}
                animate={{ x: 0, opacity: 1 }}
                transition={{ delay: 0.4 + index * 0.1 }}
              >
                <div className="detection-marker" style={{
                  background: det.model === 'seaclear' ? '#ffaa00' : '#00d4ff'
                }}></div>
                <span className="detection-name">{det.class_name}</span>
                <span className="detection-confidence">{(det.confidence * 100).toFixed(1)}%</span>
              </motion.div>
            ))}
            {detections.length > 5 && (
              <div className="detection-more">
                +{detections.length - 5} more objects
              </div>
            )}
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default VideoCanvas;
