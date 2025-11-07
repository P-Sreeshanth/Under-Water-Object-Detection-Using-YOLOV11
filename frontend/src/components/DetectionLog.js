import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ScrollText, Clock, X } from 'lucide-react';
import './DetectionLog.css';

const DetectionLog = ({ history }) => {
  const [selectedEntry, setSelectedEntry] = React.useState(null);

  return (
    <motion.div 
      className="detection-log"
      whileHover={{ boxShadow: '0 0 20px rgba(0, 212, 255, 0.3)' }}
    >
      <div className="panel-header">
        <ScrollText className="panel-icon" />
        <h3 className="panel-title">DETECTION LOG</h3>
        <span className="log-count">{history.length}</span>
      </div>

      <div className="panel-content">
        {history.length === 0 ? (
          <div className="empty-state">
            <ScrollText size={48} className="empty-icon" />
            <p className="empty-text">No detections yet</p>
            <p className="empty-subtext">Upload an image to start detecting</p>
          </div>
        ) : (
          <div className="log-list">
            <AnimatePresence>
              {history.map((entry, index) => (
                <motion.div
                  key={entry.id}
                  className="log-entry"
                  initial={{ x: 50, opacity: 0 }}
                  animate={{ x: 0, opacity: 1 }}
                  exit={{ x: -50, opacity: 0 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => setSelectedEntry(entry)}
                >
                  <div className="entry-header">
                    <Clock size={14} className="entry-icon" />
                    <span className="entry-time">{entry.timestamp}</span>
                    <span className="entry-badge">{entry.count} objects</span>
                  </div>

                  <div className="entry-thumbnail">
                    <img src={entry.image} alt="Detection" />
                    <div className="thumbnail-overlay">
                      <span>View Details</span>
                    </div>
                  </div>

                  <div className="entry-details">
                    {entry.detections.slice(0, 3).map((det, idx) => (
                      <div key={idx} className="detail-item">
                        <div 
                          className="detail-dot"
                          style={{
                            background: det.model === 'seaclear' ? '#ffaa00' : '#00d4ff'
                          }}
                        ></div>
                        <span className="detail-name">{det.class_name}</span>
                        <span className="detail-confidence">
                          {(det.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                    {entry.detections.length > 3 && (
                      <div className="detail-more">
                        +{entry.detections.length - 3} more
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      <AnimatePresence>
        {selectedEntry && (
          <motion.div 
            className="detail-modal-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedEntry(null)}
          >
            <motion.div 
              className="detail-modal"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="modal-header">
                <h3>Detection Details</h3>
                <button 
                  className="modal-close"
                  onClick={() => setSelectedEntry(null)}
                >
                  <X size={20} />
                </button>
              </div>

              <div className="modal-content">
                <img src={selectedEntry.image} alt="Full size" className="modal-image" />
                
                <div className="modal-info">
                  <div className="modal-stat">
                    <span className="modal-label">Time</span>
                    <span className="modal-value">{selectedEntry.timestamp}</span>
                  </div>
                  <div className="modal-stat">
                    <span className="modal-label">Total Detections</span>
                    <span className="modal-value">{selectedEntry.count}</span>
                  </div>
                </div>

                <div className="modal-detections">
                  <h4>All Detections</h4>
                  <div className="modal-detection-list">
                    {selectedEntry.detections.map((det, idx) => (
                      <div key={idx} className="modal-detection-item">
                        <div 
                          className="modal-det-marker"
                          style={{
                            background: det.model === 'seaclear' ? '#ffaa00' : '#00d4ff'
                          }}
                        ></div>
                        <span className="modal-det-name">{det.class_name}</span>
                        <span className="modal-det-model">{det.model}</span>
                        <span className="modal-det-conf">
                          {(det.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default DetectionLog;
