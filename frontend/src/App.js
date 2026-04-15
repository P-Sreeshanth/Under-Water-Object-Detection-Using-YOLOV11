import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import VideoCanvas from './components/VideoCanvas';
import ControlPanel from './components/ControlPanel';
import DetectionLog from './components/DetectionLog';
import StatsPanel from './components/StatsPanel';
import Header from './components/Header';
import './App.css';

function App() {
  const wsRef = useRef(null);
  const frameTimeRef = useRef(Date.now());
  const logThrottleRef = useRef(0);
  const [image, setImage] = useState(null);
  const [streamFrame, setStreamFrame] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [mode, setMode] = useState('image');
  const [streamSource, setStreamSource] = useState('0');
  const [isStreamConnected, setIsStreamConnected] = useState(false);
  const [streamError, setStreamError] = useState('');
  const [stats, setStats] = useState({
    totalDetections: 0,
    confidence: 0,
    processingTime: 0,
    models: { seaclear: false, aquarium: false }
  });
  const [settings, setSettings] = useState({
    confidence: 0.25,
    enhanceImage: false
  });
  const [detectionHistory, setDetectionHistory] = useState([]);

  const fileInputRef = useRef(null);

  const normalizeDetections = (rawDetections = []) => {
    return rawDetections.map((det) => ({
      class_name: det.class_name || det.label || 'object',
      confidence: typeof det.confidence === 'number' ? det.confidence : 0,
      model: det.model || (String(det.class_name || '').startsWith('seaclear_') ? 'seaclear' : 'aquarium'),
      bbox: det.bbox || []
    }));
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setImage(event.target.result);
        setDetections([]);
        setAnnotatedImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  const analyzeImage = async () => {
    if (!image || mode !== 'image') return;

    setIsAnalyzing(true);
    const startTime = Date.now();

    try {
      // Convert base64 to blob
      const response = await fetch(image);
      const blob = await response.blob();
      
      const formData = new FormData();
      formData.append('file', blob, 'image.jpg');
      formData.append('confidence_threshold', settings.confidence);
      formData.append('enhance', settings.enhanceImage);

      const apiResponse = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      const data = await apiResponse.json();
      
      const normalized = normalizeDetections(data.detections || []);
      setDetections(normalized);
      setAnnotatedImage(`http://localhost:8000${data.annotated_image_url}`);
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      setStats({
        totalDetections: normalized.length || 0,
        confidence: normalized.length > 0 
          ? (normalized.reduce((sum, d) => sum + d.confidence, 0) / normalized.length * 100).toFixed(1)
          : 0,
        processingTime: processingTime.toFixed(2),
        models: { seaclear: true, aquarium: true }
      });

      // Add to history
      if (normalized.length > 0) {
        const newEntry = {
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          detections: normalized,
          count: normalized.length,
          image: image
        };
        setDetectionHistory(prev => [newEntry, ...prev].slice(0, 20));
      }

    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Failed to analyze image. Make sure the backend is running on port 8000.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const disconnectStream = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsStreamConnected(false);
  };

  const connectStream = () => {
    if (wsRef.current || mode !== 'stream') return;

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const wsUrl = `${protocol}://localhost:8000/ws/stream?source=${encodeURIComponent(streamSource)}&confidence_threshold=${settings.confidence}&enhance=${settings.enhanceImage}`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    setStreamError('');

    ws.onopen = () => {
      setIsStreamConnected(true);
      frameTimeRef.current = Date.now();
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.type === 'error') {
          setStreamError(payload.message || 'Stream error');
          disconnectStream();
          return;
        }

        if (payload.type !== 'frame') return;

        const now = Date.now();
        const detectionsFrame = normalizeDetections(payload.detections || []);
        const dt = Math.max(1, now - frameTimeRef.current);
        frameTimeRef.current = now;

        setStreamFrame(`data:image/jpeg;base64,${payload.image}`);
        setDetections(detectionsFrame);
        setAnnotatedImage(null);

        const avgConfidence = detectionsFrame.length > 0
          ? (detectionsFrame.reduce((sum, d) => sum + d.confidence, 0) / detectionsFrame.length * 100).toFixed(1)
          : 0;

        setStats({
          totalDetections: detectionsFrame.length,
          confidence: avgConfidence,
          processingTime: (dt / 1000).toFixed(2),
          models: { seaclear: true, aquarium: true }
        });

        if (detectionsFrame.length > 0 && now - logThrottleRef.current > 2000) {
          logThrottleRef.current = now;
          const entry = {
            id: now,
            timestamp: new Date().toLocaleTimeString(),
            detections: detectionsFrame,
            count: detectionsFrame.length,
            image: `data:image/jpeg;base64,${payload.image}`
          };
          setDetectionHistory(prev => [entry, ...prev].slice(0, 20));
        }
      } catch (err) {
        console.error('Invalid stream payload:', err);
      }
    };

    ws.onerror = () => {
      setStreamError('WebSocket connection failed. Check backend /ws/stream.');
    };

    ws.onclose = () => {
      setIsStreamConnected(false);
      wsRef.current = null;
    };
  };

  useEffect(() => {
    return () => disconnectStream();
  }, []);

  useEffect(() => {
    if (mode === 'image') {
      disconnectStream();
      setStreamFrame(null);
      setStreamError('');
    }
  }, [mode]);

  const clearAnalysis = () => {
    setImage(null);
    setStreamFrame(null);
    setDetections([]);
    setAnnotatedImage(null);
    setStreamError('');
    setStats({
      totalDetections: 0,
      confidence: 0,
      processingTime: 0,
      models: { seaclear: false, aquarium: false }
    });
  };

  return (
    <div className="app">
      <Header stats={stats} />
      
      <div className="main-container">
        {/* Left Panel - Controls */}
        <motion.div 
          className="left-panel"
          initial={{ x: -300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5 }}
        >
          <ControlPanel
            settings={settings}
            setSettings={setSettings}
            mode={mode}
            setMode={setMode}
            streamSource={streamSource}
            setStreamSource={setStreamSource}
            onConnectStream={connectStream}
            onDisconnectStream={disconnectStream}
            isStreamConnected={isStreamConnected}
            onAnalyze={analyzeImage}
            onClear={clearAnalysis}
            isAnalyzing={isAnalyzing}
            hasImage={!!image || !!streamFrame}
            fileInputRef={fileInputRef}
            onFileChange={handleImageUpload}
          />
          <StatsPanel stats={stats} detections={detections} />
        </motion.div>

        {/* Center Panel - Video/Canvas */}
        <motion.div 
          className="center-panel"
          initial={{ y: 50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <VideoCanvas
            image={image}
            annotatedImage={annotatedImage}
            streamFrame={streamFrame}
            isStreamMode={mode === 'stream'}
            isStreamConnected={isStreamConnected}
            detections={detections}
            isAnalyzing={isAnalyzing}
            onImageUpload={() => fileInputRef.current?.click()}
            streamError={streamError}
          />
        </motion.div>

        {/* Right Panel - Detection Log */}
        <motion.div 
          className="right-panel"
          initial={{ x: 300, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
        >
          <DetectionLog history={detectionHistory} />
        </motion.div>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        style={{ display: 'none' }}
      />
    </div>
  );
}

export default App;
