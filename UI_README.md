# 🌊 Underwater Object Detection System

## Full-Stack AI Application with Stunning UI

A professional-grade underwater object detection system featuring:
- 🤖 **Dual YOLOv11 Models**: Seaclear Marine Debris (40 classes) + Aquarium Life (7 classes)
- 🎨 **Cyberpunk UI**: Dark theme with neon accents and smooth animations
- ⚡ **Real-time Detection**: Fast inference with GPU acceleration
- 📊 **Live Statistics**: Model performance tracking and visualization
- 🖼️ **HUD-Style Display**: Professional canvas-based detection overlay

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with CUDA support
- Node.js 16+
- NVIDIA GPU (recommended)

### Installation

1. **Clone and Setup Backend**:
```bash
# Install Python dependencies
pip install -r requirements.txt

# Download YOLOv11 models
python download_models.py
```

2. **Setup Frontend**:
```bash
# Install React dependencies
cd frontend
npm install
cd ..
```

### Launch Application

**Windows**:
```bash
# Use the convenient launcher
launch.bat
# Select option 3 (Full System)
```

**Manual Launch**:
```bash
# Terminal 1 - Backend
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm start
```

### Access the Application

- **Frontend UI**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📸 Screenshots

### Main Interface
- Upload underwater images via drag-and-drop
- Real-time detection with bounding boxes
- Color-coded model indicators (Orange=Seaclear, Cyan=Aquarium)

### Detection Log
- Historical tracking of all detections
- Detailed view with confidence scores
- Model attribution for each detection

### Statistics Dashboard
- Total detections counter
- Average confidence metric
- Processing time tracking
- Model distribution visualization

---

## 🎨 UI/UX Features

### Design System
- **Dark Cyberpunk Theme**: Professional underwater aesthetic
- **Gradient Backgrounds**: Smooth color transitions
- **Neon Accents**: #00d4ff (Cyan), #ffaa00 (Orange)
- **Typography**: Orbitron + Rajdhani fonts
- **Animations**: Framer Motion for fluid interactions

### Key Components
1. **Header**: System status and live metrics
2. **Control Panel**: Upload, settings, and actions
3. **Video Canvas**: Main detection display with HUD overlay
4. **Stats Panel**: Real-time performance monitoring
5. **Detection Log**: Historical review with modal details

### Responsive Design
- Desktop-first layout (1920x1080 optimized)
- Tablet support (768px - 1199px)
- Mobile adaptation (< 768px)

---

## 🔧 Architecture

### Backend (FastAPI)
```
app/
├── main.py          # API endpoints and routing
├── models.py        # ModelManager for dual YOLOv11
├── utils.py         # Settings and utilities
└── schemas.py       # Pydantic models
```

### Frontend (React)
```
frontend/src/
├── components/
│   ├── Header.js           # Top navigation
│   ├── VideoCanvas.js      # Detection display
│   ├── ControlPanel.js     # Controls and settings
│   ├── StatsPanel.js       # Statistics dashboard
│   └── DetectionLog.js     # History tracking
├── App.js                  # Main container
└── index.css               # Global styles
```

### Models
- **Seaclear Marine**: `runs/seaclear/yolov11n_seaclear/weights/best.pt`
- **Aquarium Life**: `runs/detect/aquarium_yolov11/weights/best.pt`

---

## 📊 Performance

### Seaclear Model
- **Dataset**: 8,610 images (6,888 train / 1,722 val)
- **Classes**: 40 (marine debris, animals, equipment)
- **mAP@50**: 85.41%
- **Precision**: 87.81%
- **Recall**: 80.06%

### Aquarium Model
- **Classes**: 7 (fish, jellyfish, penguin, puffin, shark, starfish, stingray)
- **Pre-trained**: Optimized for aquarium scenarios

### Inference Speed
- **GPU**: ~0.7s per image (RTX 4080)
- **Dual-model**: Simultaneous execution

---

## 🎯 API Endpoints

### POST /analyze
Upload image for detection
```json
{
  "file": "image.jpg",
  "confidence_threshold": 0.25,
  "enhance": false
}
```

### GET /health
System health check
```json
{
  "status": "healthy",
  "models_loaded": {
    "seaclear": true,
    "aquarium": true,
    "enhancer": false
  }
}
```

### GET /config
Model configuration
```json
{
  "seaclear_classes": 40,
  "aquarium_classes": 7,
  "confidence_threshold": 0.25
}
```

---

## 🛠️ Configuration

### Backend Settings (`app/utils.py`)
```python
# Model paths
SEACLEAR_MODEL_PATH = "runs/seaclear/.../best.pt"
AQUARIUM_MODEL_PATH = "runs/detect/.../best.pt"

# Detection settings
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
USE_MULTI_MODEL = True  # Enable dual-model detection
```

### Frontend Settings
- API endpoint: `http://localhost:8000`
- Default confidence: 25%
- Image enhancement: Optional

---

## 🎓 Training Your Own Models

### Seaclear Dataset
```bash
# Prepare dataset
python prepare_seaclear_dataset.py

# Train model
python train_seaclear_yolov11.py
```

### Custom Dataset
```bash
# Train on your own data
python train_yolov11.py --data your_data.yaml --epochs 100
```

---

## 📦 Dependencies

### Backend
- ultralytics (YOLOv11)
- FastAPI + Uvicorn
- OpenCV
- PyTorch (CUDA)
- Pillow

### Frontend
- React 18
- Framer Motion
- Lucide Icons
- Axios

---

## 🤝 Contributing

This project demonstrates:
- Full-stack AI application development
- Modern React UI/UX design
- Professional API architecture
- Multi-model ensemble detection
- Real-time visualization techniques

Feel free to fork and customize!

---

## 📄 License

MIT License - Free to use for educational and commercial projects

---

## 🌟 Acknowledgments

- **YOLOv11**: Ultralytics team
- **Seaclear Dataset**: Marine debris detection research
- **UI Design**: Inspired by professional underwater HUD systems

---

**Built with ❤️ using Python, React, and cutting-edge AI technologies**

For questions or issues, please open a GitHub issue.
