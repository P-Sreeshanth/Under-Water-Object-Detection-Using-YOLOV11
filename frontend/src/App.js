import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import VideoCanvas from './components/VideoCanvas';
import ControlPanel from './components/ControlPanel';
import DetectionLog from './components/DetectionLog';
import StatsPanel from './components/StatsPanel';
import Header from './components/Header';
import './App.css';

function App() {
  const [image, setImage] = useState(null);
  const [detections, setDetections] = useState([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [annotatedImage, setAnnotatedImage] = useState(null);
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
    if (!image) return;

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
      
      setDetections(data.detections || []);
      setAnnotatedImage(`http://localhost:8000${data.annotated_image_url}`);
      
      const processingTime = (Date.now() - startTime) / 1000;
      
      setStats({
        totalDetections: data.detections?.length || 0,
        confidence: data.detections?.length > 0 
          ? (data.detections.reduce((sum, d) => sum + d.confidence, 0) / data.detections.length * 100).toFixed(1)
          : 0,
        processingTime: processingTime.toFixed(2),
        models: { seaclear: true, aquarium: true }
      });

      // Add to history
      if (data.detections?.length > 0) {
        const newEntry = {
          id: Date.now(),
          timestamp: new Date().toLocaleTimeString(),
          detections: data.detections,
          count: data.detections.length,
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

  const clearAnalysis = () => {
    setImage(null);
    setDetections([]);
    setAnnotatedImage(null);
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
            onAnalyze={analyzeImage}
            onClear={clearAnalysis}
            isAnalyzing={isAnalyzing}
            hasImage={!!image}
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
            detections={detections}
            isAnalyzing={isAnalyzing}
            onImageUpload={() => fileInputRef.current?.click()}
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
