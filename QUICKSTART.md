# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup Environment

```bash
# Navigate to project directory
cd underwater_analysis

# Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Download YOLOv11 model
python download_models.py

# Or manually download a specific size:
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').save('models/best.pt')"
```

### 3. Configure

```bash
# Copy environment file
cp .env.example .env

# Edit if needed (optional)
nano .env
```

### 4. Start Server

```bash
# Start the API server
python app/main.py

# Or using uvicorn directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test API

Open your browser to:
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

Or use curl:
```bash
# Check health
curl http://localhost:8000/health

# Analyze an image
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@your_image.jpg"
```

## 📝 Quick Examples

### Python Client Example

```python
import requests

# Analyze an image
with open('underwater_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f}
    )
    
result = response.json()
print(f"Found {len(result['detections'])} objects")
print(f"Annotated image: {result['annotated_image_url']}")
```

### Using the Example Script

```bash
# Interactive mode
python example_usage.py

# Analyze specific image
python example_usage.py path/to/image.jpg

# With custom confidence threshold
python example_usage.py path/to/image.jpg 0.7
```

## 🐳 Docker Quick Start

```bash
# Build and run with Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f

# Stop
docker-compose down
```

## 🔧 Troubleshooting

### Models Not Loading
```bash
# Check model files exist
ls -lh models/

# Download YOLOv11 if missing
python download_models.py
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### Port Already in Use
```bash
# Change port in .env file
PORT=8001

# Or specify when running
uvicorn app.main:app --port 8001
```

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Run tests: `pytest test_api.py -v`
- Check API docs: http://localhost:8000/docs
- Try batch processing with multiple images

## 🎯 Key Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/analyze` | POST | Analyze single image |
| `/analyze-batch` | POST | Analyze multiple images |
| `/classes` | GET | List detectable classes |
| `/config` | GET | Get API configuration |

## 💡 Tips

1. **Use appropriate model size**: 
   - Development: yolo11n (nano) - fastest
   - Production: yolo11m or yolo11l - better accuracy

2. **Adjust thresholds**:
   - Lower confidence (0.3-0.4) for more detections
   - Higher confidence (0.6-0.7) for fewer, more certain detections

3. **GPU acceleration**:
   - Install PyTorch with CUDA for faster inference
   - Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

4. **Monitor performance**:
   - Check logs: `tail -f underwater_analysis.log`
   - Monitor processing times in responses

## 🆘 Need Help?

- Check the [README.md](README.md) for detailed documentation
- Review API docs at http://localhost:8000/docs
- Test with example script: `python example_usage.py`
