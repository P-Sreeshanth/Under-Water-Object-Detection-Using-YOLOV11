# 🌊 Seaclear Marine Debris Dataset - YOLOv11 Training Guide

## Dataset Overview
- **Total Images**: 8,610
- **Categories**: 40 classes (marine debris, animals, plants, ROV parts)
- **Locations**: Bistrina, Jakljan, Lokrum, Slano (Croatia), Marseille (France)
- **Resolution**: 1920x1080
- **Format**: COCO annotations → Converted to YOLO format

## 40 Object Categories

### Marine Litter
1. can_metal
2. tarp_plastic
3. container_plastic
4. bottle_plastic
5. tube_cement
6. container_middle_size_metal
7. bottle_glass
8. wreckage_metal
9. pipe_plastic
10. net_plastic
11. rope_fiber
12. cup_plastic
13. brick_clay
14. bag_plastic
15. sanitaries_plastic
16. clothing_fiber
17. cup_ceramic
18. boot_rubber
19. tire_rubber
20. jar_glass
21. branch_wood
22. furniture_wood
23. snack_wrapper_plastic
24. lid_plastic
25. cardboard_paper
26. rope_plastic
27. cable_metal
28. snack_wrapper_paper

### Marine Life & Environment
29. plant
30. animal_etc
31. animal_sponge
32. animal_shells
33. animal_urchin
34. animal_fish
35. animal_starfish

### ROV Equipment
36. rov_cable
37. rov_tortuga
38. rov_vehicle_leg
39. rov_bluerov

### Other
40. unknown_instance

## Quick Start

### Option 1: Automated Pipeline (Recommended)
```powershell
# Run the complete pipeline (prepare + train)
.\train_seaclear_pipeline.ps1
```

### Option 2: Manual Step-by-Step

#### Step 1: Prepare Dataset
```bash
python prepare_seaclear_dataset.py
```

This will:
- Convert COCO format to YOLO format
- Create train/val split (80/20)
- Generate `seaclear_dataset/` with proper structure:
  ```
  seaclear_dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  ├── labels/
  │   ├── train/
  │   └── val/
  └── seaclear_data.yaml
  ```

#### Step 2: Train YOLOv11

**Default Training (YOLOv11n - Nano model)**
```bash
python train_seaclear_yolov11.py
```

**Custom Training Options**

Train with different model sizes:
```bash
# Nano (fastest, smallest)
python train_seaclear_yolov11.py --model n --epochs 100 --batch 16

# Small
python train_seaclear_yolov11.py --model s --epochs 100 --batch 12

# Medium
python train_seaclear_yolov11.py --model m --epochs 100 --batch 8

# Large
python train_seaclear_yolov11.py --model l --epochs 100 --batch 4

# Extra Large
python train_seaclear_yolov11.py --model x --epochs 100 --batch 2
```

**Advanced Training Options**
```bash
python train_seaclear_yolov11.py \
    --model m \
    --epochs 200 \
    --batch 16 \
    --imgsz 1024 \
    --device 0 \
    --workers 8 \
    --lr0 0.01 \
    --patience 100 \
    --cache \
    --amp \
    --project runs/seaclear \
    --name yolov11m_1024_200ep
```

## Training Parameters Explained

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `n` | Model size: n, s, m, l, x |
| `--epochs` | `100` | Number of training epochs |
| `--batch` | `16` | Batch size (reduce if OOM) |
| `--imgsz` | `640` | Input image size |
| `--device` | `auto` | GPU device (0, 1, cpu) |
| `--workers` | `8` | Data loader workers |
| `--lr0` | `0.01` | Initial learning rate |
| `--patience` | `50` | Early stopping patience |
| `--cache` | `False` | Cache images in RAM |
| `--amp` | `True` | Automatic Mixed Precision |
| `--pretrained` | `True` | Use pretrained weights |
| `--optimizer` | `auto` | SGD, Adam, AdamW, auto |

## Model Sizes Comparison

| Model | Parameters | FLOPs | Speed (ms) | mAP@50 (COCO) |
|-------|-----------|-------|------------|---------------|
| YOLOv11n | 2.6M | 6.5G | 1.5 | 39.5 |
| YOLOv11s | 9.4M | 21.5G | 2.5 | 47.0 |
| YOLOv11m | 20.1M | 68.0G | 4.7 | 51.5 |
| YOLOv11l | 25.3M | 86.9G | 6.2 | 53.4 |
| YOLOv11x | 56.9M | 194.9G | 11.3 | 54.7 |

## Recommended Configurations

### For Fast Prototyping
```bash
python train_seaclear_yolov11.py --model n --epochs 50 --batch 32
```

### For Best Accuracy (with powerful GPU)
```bash
python train_seaclear_yolov11.py --model l --epochs 300 --batch 8 --imgsz 1024 --cache
```

### For Balanced Performance
```bash
python train_seaclear_yolov11.py --model m --epochs 150 --batch 16 --imgsz 640
```

### For Limited GPU Memory
```bash
python train_seaclear_yolov11.py --model n --epochs 100 --batch 8 --imgsz 416
```

## Post-Training

### Validate Model
```bash
python -m ultralytics val \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    data=seaclear_dataset/seaclear_data.yaml
```

### Test Inference
```bash
# Single image
python -m ultralytics predict \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    source=path/to/test/image.jpg \
    save=True

# Folder of images
python -m ultralytics predict \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    source=path/to/test/folder/ \
    save=True

# Video
python -m ultralytics predict \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    source=path/to/video.mp4 \
    save=True
```

### Export Model
```bash
# Export to ONNX
python -m ultralytics export \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    format=onnx

# Export to TensorRT
python -m ultralytics export \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    format=engine

# Export to CoreML
python -m ultralytics export \
    model=runs/seaclear/yolov11n_100ep/weights/best.pt \
    format=coreml
```

## Monitoring Training

Training outputs will be saved to `runs/seaclear/yolov11n_100ep/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.csv` - Training metrics
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `val_batch*.jpg` - Validation predictions

## Tips for Better Results

1. **Data Augmentation**: Already configured with optimal settings
2. **Learning Rate**: Try `--lr0 0.001` for fine-tuning
3. **Image Size**: Increase to 1024 for better accuracy (if GPU allows)
4. **Epochs**: Increase to 200-300 for better convergence
5. **Model Size**: Use larger models (m, l, x) for better accuracy
6. **Cache**: Enable `--cache` if you have enough RAM (speeds up training)
7. **Mixed Precision**: Keep `--amp` enabled for faster training

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch` size
- Reduce `--imgsz` (e.g., 416 or 512)
- Use smaller model (`--model n`)
- Disable cache (`--no-cache`)

### Slow Training
- Enable cache (`--cache`)
- Increase `--workers` (if CPU allows)
- Enable AMP (`--amp`)
- Use smaller image size initially

### Poor Results
- Increase epochs (`--epochs 200`)
- Use larger model (`--model m` or `--model l`)
- Increase image size (`--imgsz 1024`)
- Check data quality and annotations

## Expected Training Time

Approximate training times for 100 epochs (on RTX 3090):

| Model | Image Size | Batch | Time |
|-------|-----------|-------|------|
| n | 640 | 32 | ~2-3 hours |
| s | 640 | 16 | ~4-5 hours |
| m | 640 | 16 | ~6-8 hours |
| l | 640 | 8 | ~10-12 hours |
| x | 640 | 4 | ~18-24 hours |

## Requirements

```bash
pip install ultralytics opencv-python pillow pyyaml tqdm
```

Python 3.8+ recommended
CUDA 11.8+ for GPU acceleration

## Support

For issues or questions:
1. Check training logs in `runs/seaclear/`
2. Review YOLOv11 documentation: https://docs.ultralytics.com/
3. Check dataset preparation output
