# 🚀 Complete Installation & Launch Guide

## Step-by-Step Setup

### 📋 Prerequisites Check

Before starting, ensure you have:

- ✅ **Python 3.8+** - Check: `python --version`
- ✅ **Node.js 16+** - Check: `node --version`
- ✅ **npm** - Check: `npm --version`
- ✅ **CUDA Toolkit** (Optional but recommended for GPU)
- ✅ **Git** - For cloning the repository

---

## 🔧 Installation

### 1. Install Node.js (if not installed)

**Download**: https://nodejs.org/ (Choose LTS version)

**Verify installation**:
```bash
node --version
npm --version
```

### 2. Setup Backend

```bash
# Navigate to project root
cd Under-Water-Object-Detection-Using-YOLOV11

# Create virtual environment (if not exists)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 3. Setup Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install React dependencies
npm install

# Return to project root
cd ..
```

---

## 🎯 Quick Launch

### Option 1: Automated Launcher (Windows)

```bash
# Double-click or run:
launch.bat

# Then select:
# [3] Launch Full System (Backend + Frontend)
```

This will:
1. Start FastAPI backend on port 8000
2. Start React frontend on port 3000
3. Open both in separate terminal windows

### Option 2: Manual Launch

**Terminal 1 - Backend**:
```bash
# Windows
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Linux/Mac
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm start
```

---

## 🌐 Access Points

After launching, access:

- **🎨 Frontend UI**: http://localhost:3000
- **⚡ Backend API**: http://localhost:8000
- **📚 API Documentation**: http://localhost:8000/docs
- **📖 ReDoc**: http://localhost:8000/redoc

---

## 🎮 Using the Application

### 1. Upload an Image

**Method A - Drag & Drop**:
- Drag an underwater image to the upload zone
- Supported formats: JPG, PNG, JPEG

**Method B - Click to Browse**:
- Click the upload zone or "Upload Image" button
- Select image from your computer

### 2. Configure Settings

**Confidence Threshold** (default: 25%):
- Lower = More detections (may include false positives)
- Higher = Fewer, more confident detections
- Recommended: 20-30% for underwater images

**Image Enhancement**:
- Toggle ON for low-light/murky images
- Currently optional (enhancer model not loaded)

### 3. Analyze Image

- Click **"Analyze"** button
- Watch the scanning animation
- Wait ~0.7 seconds for results

### 4. View Results

**Main Canvas**:
- Annotated image with bounding boxes
- Orange boxes = Seaclear model detections
- Cyan boxes = Aquarium model detections

**Statistics Panel**:
- Total objects detected
- Average confidence score
- Processing time
- Model distribution chart

**Detection Info**:
- List of detected objects
- Confidence percentages
- Model attribution

**Detection Log**:
- Historical record of all analyses
- Click entries for detailed view
- Timestamps and thumbnails

### 5. Clear & Repeat

- Click **"Clear"** to reset
- Upload new image
- Results are saved in detection log

---

## 🔍 Troubleshooting

### Backend Issues

**Problem**: "Port 8000 already in use"
```bash
# Find and kill the process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :8000
kill -9 <PID>
```

**Problem**: "Model file not found"
```bash
# Check model paths in app/utils.py:
SEACLEAR_MODEL_PATH = "runs/seaclear/yolov11n_seaclear/weights/best.pt"
AQUARIUM_MODEL_PATH = "runs/detect/aquarium_yolov11/weights/best.pt"

# Verify files exist:
dir runs\seaclear\yolov11n_seaclear\weights\best.pt
dir runs\detect\aquarium_yolov11\weights\best.pt
```

**Problem**: "CUDA not available"
```bash
# Check PyTorch CUDA:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# If False, reinstall PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Frontend Issues

**Problem**: "npm command not found"
- Install Node.js from https://nodejs.org/
- Restart terminal after installation
- Verify: `npm --version`

**Problem**: "Port 3000 already in use"
```bash
# Frontend will auto-select port 3001
# Or manually kill process on port 3000

# Windows:
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Problem**: "Cannot connect to backend"
- Ensure backend is running on port 8000
- Check CORS settings in app/main.py
- Verify backend URL in frontend/src/App.js

### CORS Errors

If you see CORS errors in browser console:

1. Check backend CORS settings in `app/main.py`
2. Ensure backend allows `http://localhost:3000`
3. Restart backend server

---

## 🎨 Customization

### Change Theme Colors

Edit `frontend/src/index.css`:
```css
/* Primary color */
--primary: #00d4ff;  /* Change to your color */

/* Background */
background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
```

### Adjust Detection Threshold

Edit `frontend/src/App.js`:
```javascript
const [settings, setSettings] = useState({
  confidence: 0.25,  // Change default (0.0 - 1.0)
  enhanceImage: false
});
```

### Change API Endpoint

Edit `frontend/src/App.js`:
```javascript
const apiResponse = await fetch('http://localhost:8000/analyze', {
  // Change to your backend URL
```

---

## 📊 Model Information

### Seaclear Marine Debris Model
- **Classes**: 40
- **Dataset**: 8,610 images
- **mAP@50**: 85.41%
- **Detects**: Marine litter, animals, ROV equipment

### Aquarium Life Model
- **Classes**: 7
- **Detects**: fish, jellyfish, penguin, puffin, shark, starfish, stingray

---

## 🔄 Updating

### Update Backend Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update Frontend Dependencies
```bash
cd frontend
npm update
cd ..
```

### Update Models
```bash
# Re-train or download new model weights
# Update paths in app/utils.py
```

---

## 📦 Building for Production

### Backend
```bash
# Use production ASGI server
pip install gunicorn

# Run with:
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
cd frontend
npm run build

# Serve build folder with nginx or any static server
# Build output: frontend/build/
```

---

## 🆘 Getting Help

### Common Questions

**Q: Can I use CPU only?**
A: Yes, but detection will be slower (~3-5s per image)

**Q: Can I add my own models?**
A: Yes, train YOLOv11 and update paths in app/utils.py

**Q: Can I deploy this online?**
A: Yes, use services like Heroku, AWS, or DigitalOcean

**Q: Can I change the UI colors?**
A: Yes, all colors are in CSS files (see Customization)

### Support Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check README files in each directory
- **API Docs**: http://localhost:8000/docs for API reference

---

## ✅ Verification Checklist

Before reporting issues, verify:

- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] All pip packages installed successfully
- [ ] All npm packages installed successfully
- [ ] Model files exist at specified paths
- [ ] Backend starts without errors on port 8000
- [ ] Frontend starts without errors on port 3000
- [ ] CORS is properly configured
- [ ] Firewall allows local connections
- [ ] Browser is up to date

---

## 🎉 Success!

If everything is working, you should see:

1. **Backend Terminal**: "Application startup complete"
2. **Frontend Browser**: Beautiful underwater detection UI
3. **Upload**: Drag and drop works
4. **Analyze**: Detections appear with bounding boxes
5. **Log**: History updates automatically

**Congratulations! Your system is ready to detect underwater objects!** 🌊🤖

---

## 📝 Next Steps

- Upload test images from the datasets
- Adjust confidence threshold for best results
- Review detection log for insights
- Train models on custom data
- Customize UI to your preference
- Deploy to production server

**Happy detecting!** 🚀
