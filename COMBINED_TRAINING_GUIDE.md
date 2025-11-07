# 🌊🐠 Combined Seaclear + Aquarium Dataset Training Guide

## Overview

This guide explains how to train YOLOv11 on a **combined dataset** that includes both:
- **Seaclear Marine Debris Dataset** (40 classes - marine litter, animals, ROV equipment)
- **Aquarium Dataset** (7 classes - fish, jellyfish, sharks, etc.)

The combined model can detect **47 total classes**!

## Dataset Breakdown

### Seaclear Dataset (Classes 0-39)
- **40 classes** of underwater objects
- Marine debris (plastics, metals, rubber, etc.)
- Marine life (fish, starfish, urchins, sponges, etc.)
- ROV equipment
- **8,610 images** (6,888 train / 1,722 val)

### Aquarium Dataset (Classes 40-46)
- **7 classes** of aquatic animals
- Classes: fish, jellyfish, penguin, puffin, shark, starfish, stingray
- Aquarium/controlled environment images

### Combined Dataset (47 Classes Total)
- Classes 0-39: Seaclear objects (prefixed with `seaclear_`)
- Classes 40-46: Aquarium animals (prefixed with `aquarium_`)

## Prerequisites

1. **Seaclear dataset prepared:**
   ```bash
   python prepare_seaclear_dataset.py
   ```
   This should create `seaclear_dataset/` folder

2. **Aquarium dataset** should be available at the path specified in `aquarium_data.yaml`

## Training Steps

### Option 1: Automated Combined Training

Run the combined training script directly:

```powershell
.venv\Scripts\python.exe train_combined_datasets.py
```

This will:
1. Combine both datasets into `combined_dataset/`
2. Create a unified YAML config with 47 classes
3. Train YOLOv11n for 150 epochs

### Option 2: Custom Training

With different model sizes and parameters:

```powershell
# Nano model (fast)
.venv\Scripts\python.exe train_combined_datasets.py --model n --epochs 150 --batch 16

# Small model (balanced)
.venv\Scripts\python.exe train_combined_datasets.py --model s --epochs 150 --batch 12

# Medium model (better accuracy)
.venv\Scripts\python.exe train_combined_datasets.py --model m --epochs 200 --batch 8

# Large model (best accuracy, requires more GPU memory)
.venv\Scripts\python.exe train_combined_datasets.py --model l --epochs 200 --batch 4
```

### Option 3: Skip Dataset Combination (If Already Combined)

If you've already combined the datasets:

```powershell
.venv\Scripts\python.exe train_combined_datasets.py --skip-combine
```

## Training Parameters

```powershell
.venv\Scripts\python.exe train_combined_datasets.py \
    --seaclear-yaml seaclear_dataset/seaclear_data.yaml \
    --aquarium-yaml aquarium_data.yaml \
    --output-dir combined_dataset \
    --model m \
    --epochs 200 \
    --batch 16 \
    --imgsz 640 \
    --device 0 \
    --project runs/combined \
    --name yolov11m_combined_200ep
```

## Expected Results

### Training Time (RTX 4080)
- **Nano (n)**: ~4-5 hours for 150 epochs
- **Small (s)**: ~7-9 hours for 150 epochs
- **Medium (m)**: ~12-15 hours for 200 epochs
- **Large (l)**: ~18-24 hours for 200 epochs

### Expected Performance
With the combined dataset, you should expect:
- **Overall mAP@50**: 70-80%
- **Seaclear classes**: 75-85% mAP@50
- **Aquarium classes**: 80-90% mAP@50

## Using the Trained Model

### Update the Application

After training, update the FastAPI app to use the combined model:

1. Edit `app/utils.py`:
   ```python
   DETECTOR_MODEL_PATH: str = "runs/combined/yolov11_seaclear_aquarium/weights/best.pt"
   ```

2. Restart the application:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Validation

Validate the combined model:

```powershell
.venv\Scripts\python.exe -m ultralytics val \
    model=runs/combined/yolov11_seaclear_aquarium/weights/best.pt \
    data=combined_dataset/combined_data.yaml
```

### Inference

Test on individual images:

```powershell
# Seaclear (marine debris) image
.venv\Scripts\python.exe inference_seaclear.py \
    --model runs/combined/yolov11_seaclear_aquarium/weights/best.pt \
    --source "seaclear_dataset/images/val/1.jpg" \
    --conf 0.25

# Aquarium (fish/animals) image
.venv\Scripts\python.exe inference_seaclear.py \
    --model runs/combined/yolov11_seaclear_aquarium/weights/best.pt \
    --source "path/to/aquarium/image.jpg" \
    --conf 0.25
```

## Class Names

The combined model will detect:

**Seaclear Classes (0-39):**
- seaclear_can_metal
- seaclear_tarp_plastic
- seaclear_container_plastic
- seaclear_bottle_plastic
- seaclear_plant
- seaclear_animal_fish
- seaclear_animal_starfish
- ... (40 total)

**Aquarium Classes (40-46):**
- aquarium_fish
- aquarium_jellyfish
- aquarium_penguin
- aquarium_puffin
- aquarium_shark
- aquarium_starfish
- aquarium_stingray

## Advantages of Combined Training

1. **Unified Model**: One model for all underwater object detection
2. **Better Generalization**: Training on diverse datasets improves robustness
3. **Shared Features**: Model learns common underwater image features
4. **Cost Efficient**: One inference instead of running multiple models

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch 8` or `--batch 4`
- Use smaller model: `--model n`
- Reduce image size: `--imgsz 416`

### Class Imbalance
The Seaclear dataset (8,610 images) is much larger than typical aquarium datasets. Consider:
- Using class weights during training
- Augmenting aquarium data
- Adjusting learning rate

### Path Issues
Make sure to update the paths in `aquarium_data.yaml` to match your local setup:
```yaml
path: C:/path/to/your/aquarium_pretrain  # Use absolute path
```

## Monitoring Training

Training outputs will be in `runs/combined/yolov11_seaclear_aquarium/`:
- `weights/best.pt` - Best model
- `weights/last.pt` - Last epoch
- `results.csv` - Metrics
- `confusion_matrix.png` - Confusion matrix showing all 47 classes

## Next Steps

After successful training:
1. ✅ Validate on both datasets
2. ✅ Test inference on sample images
3. ✅ Update the web application
4. ✅ Export model for deployment (ONNX, TensorRT, etc.)
5. ✅ Create demo videos showing detection on both marine debris and aquarium scenes
