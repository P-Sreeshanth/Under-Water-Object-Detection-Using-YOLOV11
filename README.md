# Underwater Image Analysis API

A production-ready FastAPI application for underwater image enhancement and obstruction detection using U-Net and YOLOv11 models.

## 🌊 Features

- **Image Enhancement**: Enhance underwater images using a U-Net deep learning model
- **Object Detection**: Detect underwater obstructions and objects using YOLOv11
- **RESTful API**: Clean, well-documented API endpoints
- **Async Processing**: Efficient asynchronous request handling
- **Rate Limiting**: Built-in rate limiting to prevent abuse
- **CORS Support**: Cross-origin resource sharing for frontend integration
- **Comprehensive Logging**: Structured logging for monitoring and debugging
- **Error Handling**: Robust error handling with detailed error responses
- **Health Monitoring**: Health check endpoint for service monitoring
- **Batch Processing**: Support for analyzing multiple images

## 📋 Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Model Setup](#model-setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- CUDA-capable GPU (optional, but recommended for faster inference)

### Step 1: Clone or Navigate to Project Directory

```bash
cd underwater_analysis
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download YOLOv11

If you don't have a trained YOLOv11 model, you can download a pretrained one:

```python
from ultralytics import YOLO

# Download YOLOv11 nano model (smallest)
model = YOLO('yolo11n.pt')
model.save('models/best.pt')

# Or for better accuracy, use larger models:
# yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (extra large)
```

## ⚙️ Configuration

### Environment Variables

Copy the example environment file and configure it:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Model Paths
ENHANCER_MODEL_PATH=models/enhancer_model.pth
DETECTOR_MODEL_PATH=models/best.pt

# Detection Settings
CONFIDENCE_THRESHOLD=0.5
NMS_THRESHOLD=0.45

# File Upload Settings
MAX_FILE_SIZE_MB=10.0

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
```

### Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `HOST` | Server host address | 0.0.0.0 |
| `PORT` | Server port number | 8000 |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `ENHANCER_MODEL_PATH` | Path to U-Net enhancement model | models/enhancer_model.pth |
| `DETECTOR_MODEL_PATH` | Path to YOLOv11 detection model | models/best.pt |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for detections | 0.5 |
| `NMS_THRESHOLD` | IoU threshold for Non-Maximum Suppression | 0.45 |
| `MAX_FILE_SIZE_MB` | Maximum upload file size in MB | 10.0 |
| `RATE_LIMIT_PER_MINUTE` | API rate limit per minute | 60 |

## 🎯 Model Setup

### Directory Structure

Place your model files in the `models/` directory:

```
underwater_analysis/
├── models/
│   ├── enhancer_model.pth    # U-Net enhancement model
│   └── best.pt                # YOLOv11 detection model
```

### U-Net Enhancement Model

If you have a trained U-Net model for underwater image enhancement:

1. Save your model weights to `models/enhancer_model.pth`
2. Ensure the model architecture in `app/models.py` matches your trained model
3. If needed, modify the `UNet` class to match your architecture

**Note**: The enhancement model is optional. If not provided, the API will skip enhancement and only perform detection.

### YOLOv11 Detection Model

You can use either:

1. **Pretrained YOLOv11** (for general object detection):
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11n.pt')  # or yolo11s, yolo11m, yolo11l, yolo11x
   model.save('models/best.pt')
   ```

2. **Custom Trained YOLOv11** (for underwater-specific detection):
   - Train YOLOv11 on your underwater dataset
   - Save the best weights to `models/best.pt`

## 💻 Usage

### Starting the Server

#### Development Mode

```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Or simply:

```bash
python app/main.py
```

#### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Accessing the API

Once the server is running:

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### 1. Root Endpoint

**GET** `/`

Get API information and available endpoints.

```bash
curl http://localhost:8000/
```

### 2. Health Check

**GET** `/health`

Check API status and model availability.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": {
    "enhancer": true,
    "detector": true
  },
  "timestamp": "2025-10-30T12:00:00Z"
}
```

### 3. Get Configuration

**GET** `/config`

Get current API configuration settings.

```bash
curl http://localhost:8000/config
```

### 4. Analyze Image

**POST** `/analyze`

Analyze an underwater image with enhancement and detection.

**Parameters:**
- `file` (required): Image file (JPEG, PNG, BMP)
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0)
- `nms_threshold` (optional): NMS IoU threshold (0.0-1.0)

**Example using curl:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@underwater_image.jpg" \
  -F "confidence_threshold=0.5"
```

**Example using Python:**

```python
import requests

url = "http://localhost:8000/analyze"
files = {"file": open("underwater_image.jpg", "rb")}
data = {"confidence_threshold": 0.5, "nms_threshold": 0.45}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Detections: {len(result['detections'])}")
print(f"Image URL: {result['annotated_image_url']}")
```

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully. Found 3 object(s).",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "detections": [
    {
      "class_name": "fish",
      "confidence": 0.95,
      "bbox": [100, 150, 300, 400]
    }
  ],
  "annotated_image_url": "/static/annotated_550e8400-e29b-41d4-a716-446655440000.jpg",
  "processing_time": 2.34,
  "image_dimensions": {
    "original": {"width": 1920, "height": 1080},
    "enhanced": {"width": 1920, "height": 1080}
  }
}
```

### 5. Batch Analysis

**POST** `/analyze-batch`

Analyze multiple images at once (max 10 images).

```bash
curl -X POST "http://localhost:8000/analyze-batch" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### 6. Get Detectable Classes

**GET** `/classes`

Get list of classes that can be detected.

```bash
curl http://localhost:8000/classes
```

### 7. Cleanup Old Images

**DELETE** `/cleanup?max_age_hours=24`

Remove old annotated images from the static directory.

```bash
curl -X DELETE "http://localhost:8000/cleanup?max_age_hours=24"
```

## 🧪 Testing

### Manual Testing

1. Start the server:
   ```bash
   python app/main.py
   ```

2. Open your browser to http://localhost:8000/docs

3. Use the interactive Swagger UI to test endpoints

### Automated Testing

Run the test suite:

```bash
pytest test_api.py -v
```

### Example Test Script

```python
import requests
import json

# Test health endpoint
response = requests.get("http://localhost:8000/health")
print("Health:", response.json())

# Test image analysis
with open("test_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/analyze", files=files)
    result = response.json()
    
    print(f"\nAnalysis Result:")
    print(f"Success: {result['success']}")
    print(f"Detections: {len(result['detections'])}")
    print(f"Processing Time: {result['processing_time']}s")
    
    for i, det in enumerate(result['detections'], 1):
        print(f"\nDetection {i}:")
        print(f"  Class: {det['class_name']}")
        print(f"  Confidence: {det['confidence']:.2f}")
        print(f"  BBox: {det['bbox']}")
```

## 🐳 Deployment

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p static models

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t underwater-analysis .
docker run -p 8000:8000 -v $(pwd)/models:/app/models underwater-analysis
```

### Using Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./static:/app/static
    environment:
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
    restart: unless-stopped
```

Run:

```bash
docker-compose up -d
```

## 📊 Performance Optimization

### GPU Acceleration

The API automatically uses CUDA if available. To verify:

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Batch Processing

For processing multiple images, use the `/analyze-batch` endpoint to improve throughput.

### Model Optimization

1. **Use TorchScript** for faster inference:
   ```python
   traced_model = torch.jit.trace(model, example_input)
   torch.jit.save(traced_model, "model_traced.pt")
   ```

2. **Quantization** for CPU deployment:
   ```python
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   ```

## 🔒 Security Considerations

1. **File Upload Validation**: The API validates file types and sizes
2. **Rate Limiting**: Built-in rate limiting prevents abuse
3. **CORS**: Configure `allow_origins` in production for specific domains
4. **Environment Variables**: Use `.env` for sensitive configuration
5. **HTTPS**: Deploy behind a reverse proxy with SSL/TLS

## 📝 Logging

Logs are written to:
- Console (stdout)
- `underwater_analysis.log` file

Log format:
```
2025-10-30 12:00:00 - app.main - INFO - [request-id] Message
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Troubleshooting

### Models Not Loading

**Issue**: "Models not loaded. Service unavailable."

**Solution**:
- Ensure model files exist in the `models/` directory
- Check file paths in `.env`
- Verify model file integrity

### Out of Memory

**Issue**: CUDA out of memory errors

**Solution**:
- Reduce batch size
- Use smaller YOLOv11 model (yolo11n instead of yolo11x)
- Process images on CPU by setting CUDA_VISIBLE_DEVICES=""

### Slow Inference

**Issue**: Analysis takes too long

**Solution**:
- Use GPU if available
- Use smaller models
- Reduce input image resolution
- Enable batch processing

### Import Errors

**Issue**: Module not found errors

**Solution**:
```bash
pip install -r requirements.txt --upgrade
```

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Contact the development team

---

**Built with ❤️ for underwater research and conservation**
