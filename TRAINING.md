# Training Guide - Underwater Image Analysis Models

This guide walks you through training both the U-Net enhancement model and YOLOv11 detection model on the Aquarium Dataset.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Download Dataset](#download-dataset)
3. [Prepare Dataset](#prepare-dataset)
4. [Train YOLOv11](#train-yolov11-detection-model)
5. [Train U-Net](#train-u-net-enhancement-model)
6. [Evaluate Models](#evaluate-models)
7. [Use Trained Models](#use-trained-models)

## 🔧 Prerequisites

### Install Training Dependencies

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install all dependencies including training tools
pip install -r requirements.txt
```

### GPU Setup (Recommended)

For faster training, ensure you have:
- NVIDIA GPU with CUDA support
- CUDA toolkit installed
- PyTorch with CUDA support

Check GPU availability:
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 📥 Download Dataset

### Option 1: Using Python Script

Create `download_dataset.py`:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("slavkoprytula/aquarium-data-cots")
print("Path to dataset files:", path)
```

Run:
```bash
python download_dataset.py
```

### Option 2: Manual Download

1. Go to https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots
2. Download the dataset
3. Extract to a known location

## 🗂️ Prepare Dataset

The Aquarium Dataset comes in COCO format. We need to convert it to YOLO format.

### Automatic Preparation

```bash
# Automatic detection and preparation
python prepare_aquarium_dataset.py

# Or specify dataset path
python prepare_aquarium_dataset.py --dataset_path /path/to/aquarium-data-cots

# Prepare and train immediately
python prepare_aquarium_dataset.py --train --epochs 100 --model_size n
```

### What This Does

1. **Finds the dataset** in common locations
2. **Explores structure** to understand the format
3. **Converts annotations** from COCO to YOLO format
4. **Organizes images** into train/val splits
5. **Creates dataset.yaml** for YOLO training
6. **Shows statistics** about the prepared dataset

### Expected Output Structure

```
aquarium_yolo/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
└── dataset.yaml
```

## 🎯 Train YOLOv11 Detection Model

### Quick Start

```bash
# Train with default settings (YOLOv11-nano, 100 epochs)
python train_yolov11.py --data aquarium_yolo/dataset.yaml --epochs 100

# Copy best model to models directory
python train_yolov11.py --data aquarium_yolo/dataset.yaml --epochs 100 --copy-to-models
```

### Training Options

#### Model Sizes

Choose based on your needs:

| Size | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| `n` (nano) | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Development, edge devices |
| `s` (small) | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | Balanced performance |
| `m` (medium) | ⚡⚡⚡ | ⭐⭐⭐⭐⭐ | Production, good hardware |
| `l` (large) | ⚡⚡ | ⭐⭐⭐⭐⭐⭐ | High accuracy needs |
| `x` (extra-large) | ⚡ | ⭐⭐⭐⭐⭐⭐⭐ | Research, maximum accuracy |

```bash
# Train different model sizes
python train_yolov11.py --data aquarium_yolo/dataset.yaml --model_size n --epochs 100
python train_yolov11.py --data aquarium_yolo/dataset.yaml --model_size s --epochs 150
python train_yolov11.py --data aquarium_yolo/dataset.yaml --model_size m --epochs 200
```

#### Advanced Training

```bash
# Full custom training
python train_yolov11.py \
  --data aquarium_yolo/dataset.yaml \
  --model_size m \
  --epochs 200 \
  --batch 16 \
  --img_size 640 \
  --patience 50 \
  --device 0 \
  --project aquarium_detection \
  --name yolov11m_v1 \
  --pretrained \
  --validate \
  --copy-to-models
```

#### Training on CPU

```bash
python train_yolov11.py --data aquarium_yolo/dataset.yaml --device cpu --batch 4
```

#### Training on Multiple GPUs

```bash
python train_yolov11.py --data aquarium_yolo/dataset.yaml --device 0,1
```

### Monitor Training

Training progress is saved in the project directory:

```bash
# View with TensorBoard
tensorboard --logdir aquarium_detection/yolov11_training
```

Or check the results directory:
```
aquarium_detection/yolov11_training/
├── weights/
│   ├── best.pt          # Best model (lowest validation loss)
│   └── last.pt          # Last epoch model
├── results.png          # Training metrics plots
├── confusion_matrix.png # Confusion matrix
├── F1_curve.png        # F1 score curve
├── P_curve.png         # Precision curve
├── R_curve.png         # Recall curve
└── PR_curve.png        # Precision-Recall curve
```

### Expected Training Time

| Model Size | GPU (V100) | GPU (RTX 3090) | CPU |
|------------|-----------|----------------|-----|
| Nano (n) | ~30 min | ~20 min | ~4 hours |
| Small (s) | ~45 min | ~30 min | ~6 hours |
| Medium (m) | ~90 min | ~60 min | ~12 hours |
| Large (l) | ~3 hours | ~2 hours | ~24 hours |

*Times are approximate for 100 epochs with the aquarium dataset*

## 🎨 Train U-Net Enhancement Model

The U-Net model enhances underwater image quality before detection.

### Basic Training

```bash
# Train U-Net (self-supervised enhancement)
python train_unet.py --train_dir aquarium_yolo/images/train --epochs 100
```

### With Validation Set

```bash
python train_unet.py \
  --train_dir aquarium_yolo/images/train \
  --val_dir aquarium_yolo/images/val \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --image_size 256
```

### With Paired Images (if available)

If you have paired degraded/clean images:

```bash
python train_unet.py \
  --train_dir path/to/degraded/train \
  --target_train_dir path/to/clean/train \
  --val_dir path/to/degraded/val \
  --target_val_dir path/to/clean/val \
  --epochs 100 \
  --save_dir unet_training_output
```

### Training Options

```bash
# Full custom training
python train_unet.py \
  --train_dir aquarium_yolo/images/train \
  --val_dir aquarium_yolo/images/val \
  --batch_size 32 \
  --epochs 150 \
  --lr 0.0001 \
  --image_size 256 \
  --save_dir unet_output
```

### Monitor U-Net Training

Check the output directory:
```
unet_output/
├── best_model.pth           # Best model checkpoint
├── enhancer_model.pth       # Final model
├── checkpoint_epoch_10.pth  # Periodic checkpoints
├── checkpoint_epoch_20.pth
├── samples/                 # Sample enhanced images per epoch
│   ├── samples_epoch_10.png
│   ├── samples_epoch_20.png
│   └── ...
└── training_history.png     # Loss curves
```

## 📊 Evaluate Models

### Evaluate YOLOv11

```bash
# Validate on test set
python train_yolov11.py \
  --data aquarium_yolo/dataset.yaml \
  --validate
```

Or use the validation function directly:

```python
from train_yolov11 import validate_model

results = validate_model(
    'models/best.pt',
    'aquarium_yolo/dataset.yaml',
    img_size=640
)
```

### Test on Sample Images

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/best.pt')

# Predict on test images
results = model.predict(
    'aquarium_yolo/images/val',
    save=True,
    conf=0.5
)
```

### Evaluate U-Net

Visual evaluation by checking sample outputs:

```bash
# Check samples directory
ls unet_output/samples/
```

Or test on new images:

```python
import torch
from app.models import UNet
import cv2

# Load model
model = UNet()
model.load_state_dict(torch.load('models/enhancer_model.pth'))
model.eval()

# Test on image
image = cv2.imread('test_image.jpg')
# ... preprocess and enhance ...
```

## 🚀 Use Trained Models

### Copy Models to API Directory

```bash
# YOLOv11 model
cp aquarium_detection/yolov11_training/weights/best.pt models/best.pt

# U-Net model
cp unet_output/enhancer_model.pth models/enhancer_model.pth
```

### Update Configuration

Edit `.env`:
```bash
ENHANCER_MODEL_PATH=models/enhancer_model.pth
DETECTOR_MODEL_PATH=models/best.pt
```

### Start API with Trained Models

```bash
python app/main.py
```

### Test the API

```bash
# Test with example script
python example_usage.py path/to/test/image.jpg

# Or use curl
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@test_image.jpg" \
  -F "confidence_threshold=0.5"
```

## 🎯 Training Tips & Best Practices

### For YOLOv11

1. **Start Small**: Begin with `yolov11n` for quick iteration
2. **Adjust Batch Size**: Reduce if you get OOM errors
3. **Use Pretrained Weights**: Almost always better than training from scratch
4. **Early Stopping**: Use `--patience 50` to stop if not improving
5. **Image Size**: 640 is default, use 1280 for better accuracy (slower)
6. **Augmentation**: Default settings are good, but can be tuned

### For U-Net

1. **Self-Supervised**: Works even without paired images
2. **Batch Size**: 16-32 works well for 256x256 images
3. **Learning Rate**: Start with 1e-4, reduce if unstable
4. **Image Size**: 256x256 is good balance, use 512 for better quality
5. **Monitor Samples**: Check visual quality in samples directory
6. **Perceptual Loss**: Helps maintain content while enhancing

### General Tips

1. **GPU Memory**: Reduce batch size if OOM errors
2. **Checkpoints**: Save regularly to resume if interrupted
3. **Validation**: Always use a validation set
4. **Patience**: Training takes time, don't stop too early
5. **Monitoring**: Use TensorBoard or check plots regularly

## 🔧 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train_yolov11.py --data dataset.yaml --batch 8

# Or use smaller model
python train_yolov11.py --data dataset.yaml --model_size n
```

### Training Too Slow

```bash
# Use smaller image size
python train_yolov11.py --data dataset.yaml --img_size 416

# Reduce workers
python train_yolov11.py --data dataset.yaml  # Edit workers in script
```

### Low Accuracy

1. Train longer: increase `--epochs`
2. Use larger model: try `--model_size m` or `l`
3. Increase image size: `--img_size 1280`
4. Check data quality and labels
5. Add more data or augmentation

### Model Not Converging

1. Reduce learning rate
2. Check dataset labels
3. Ensure class balance
4. Try different optimizer settings

## 📈 Expected Results

### YOLOv11 on Aquarium Dataset

After 100 epochs:

| Metric | Expected Value |
|--------|---------------|
| mAP50 | 0.70 - 0.85 |
| mAP50-95 | 0.45 - 0.65 |
| Precision | 0.75 - 0.90 |
| Recall | 0.70 - 0.85 |

*Results vary based on model size and training settings*

### U-Net Enhancement

Subjective improvement:
- Better color correction
- Improved contrast
- Reduced haze/blur
- Better detection performance

## 📚 Additional Resources

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Aquarium Dataset](https://www.kaggle.com/datasets/slavkoprytula/aquarium-data-cots)

## 🎉 Next Steps

After training:

1. ✅ Copy models to `models/` directory
2. ✅ Update `.env` configuration
3. ✅ Start the API: `python app/main.py`
4. ✅ Test with examples: `python example_usage.py test.jpg`
5. ✅ Deploy to production (see README.md)

---

**Happy Training! 🚀**
