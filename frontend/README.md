# 🌊 Underwater Object Detection - Frontend

## Professional UI/UX Design

This is a stunning, modern React frontend for the underwater object detection system featuring:

### ✨ Features

- **🎨 Dark Theme with Cyberpunk Aesthetics**: Beautiful gradient backgrounds with neon accents
- **📊 Real-time Statistics Dashboard**: Live monitoring of detection metrics
- **🎯 Dual-Model Support**: Seamless integration of Seaclear + Aquarium models
- **📱 Responsive Design**: Works perfectly on all screen sizes
- **⚡ Smooth Animations**: Powered by Framer Motion for fluid transitions
- **🖼️ Canvas-based Detection Visualization**: HUD-style overlays and corner frames
- **📝 Detection History Log**: Track and review all past detections
- **🎛️ Advanced Controls**: Adjustable confidence threshold and enhancement options

### 🚀 Getting Started

#### Prerequisites

- Node.js 16+ and npm
- Backend API running on `http://localhost:8000`

#### Installation

```bash
cd frontend
npm install
```

#### Run Development Server

```bash
npm start
```

The app will open at `http://localhost:3000`

#### Build for Production

```bash
npm run build
```

### 🎨 Design System

#### Color Palette
- **Primary**: `#00d4ff` (Cyan Blue)
- **Secondary**: `#0099ff` (Deep Blue)
- **Accent**: `#ffaa00` (Orange - for Seaclear detections)
- **Success**: `#00ff88` (Neon Green)
- **Background**: `#0a0e27` → `#1a1f3a` (Dark gradient)

#### Typography
- **Headings**: Orbitron (Futuristic sans-serif)
- **Body**: Rajdhani (Modern technical font)

### 📦 Components Architecture

```
src/
├── components/
│   ├── Header.js           # Top navigation bar with system status
│   ├── VideoCanvas.js      # Main detection display with HUD overlay
│   ├── ControlPanel.js     # Upload, settings, and action controls
│   ├── StatsPanel.js       # Real-time statistics and model info
│   └── DetectionLog.js     # Historical detection log with modal view
├── App.js                  # Main application container
├── App.css                 # Global styles and grid layout
└── index.css               # Base styles and animations
```

### 🔌 API Integration

The frontend connects to the FastAPI backend at `http://localhost:8000`:

- `POST /analyze` - Upload image for detection
- `GET /health` - System health check
- `GET /config` - Get model configuration

### 🎯 Key UI Elements

1. **Upload Zone**: Drag-and-drop or click to upload underwater images
2. **Scanning Animation**: Futuristic scanning line effect during analysis
3. **Corner Frames**: HUD-style targeting corners on detection results
4. **Detection Badges**: Color-coded badges (Orange=Seaclear, Cyan=Aquarium)
5. **Progress Bars**: Animated model distribution visualization
6. **Glow Effects**: Neon glow on text and borders for cyberpunk aesthetic

### 📱 Responsive Breakpoints

- Desktop: 1200px+
- Tablet: 768px - 1199px
- Mobile: < 768px

### 🌟 Animation System

- **Framer Motion**: Smooth component transitions and list animations
- **CSS Keyframes**: Pulse, scan, spin, and glow effects
- **Hover States**: Interactive feedback on all clickable elements

### 🛠️ Customization

To customize the theme, edit the CSS custom properties in `src/index.css`:

```css
:root {
  --primary-color: #00d4ff;
  --secondary-color: #0099ff;
  --accent-color: #ffaa00;
  --bg-dark: #0a0e27;
}
```

### 📄 License

MIT License - Feel free to use in your projects!

---

**Built with ❤️ using React, Framer Motion, and modern web technologies**
