#!/usr/bin/env python3
"""
Quick training script for Aquarium Dataset with YOLOv11.
Trains on the pre-prepared dataset at /home/campus/Downloads/archive/aquarium_pretrain
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

def main():
    print("=" * 70)
    print("TRAINING YOLOV11 ON AQUARIUM DATASET")
    print("=" * 70)
    
    # Configuration
    data_yaml = "aquarium_data.yaml"
    model_size = "n"  # Start with nano for quick training
    epochs = 100
    batch_size = 16
    img_size = 640
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Model: YOLOv11-{model_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    
    # Check if data.yaml exists
    if not Path(data_yaml).exists():
        print(f"\n✗ Error: {data_yaml} not found!")
        print("  Please ensure the aquarium_data.yaml file is in the current directory.")
        return
    
    # Initialize YOLOv11 model with pretrained weights
    print(f"\nInitializing YOLOv11-{model_size} with pretrained weights...")
    model = YOLO(f'yolo11{model_size}.pt')
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print("\nThis may take a while. Training progress will be shown below.")
    print("You can monitor with TensorBoard: tensorboard --logdir runs/detect")
    print("=" * 70 + "\n")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=50,
        save=True,
        device='',  # Auto-detect GPU/CPU
        workers=8,
        project='runs/detect',
        name='aquarium_yolov11',
        exist_ok=True,
        
        # Augmentation (optimized for underwater images)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        
        # Optimization
        optimizer='auto',
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        
        # Validation
        val=True,
        plots=True,
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    # Get paths
    best_model = Path('runs/detect/aquarium_yolov11/weights/best.pt')
    last_model = Path('runs/detect/aquarium_yolov11/weights/last.pt')
    
    print(f"\nModel saved to:")
    print(f"  Best: {best_model}")
    print(f"  Last: {last_model}")
    
    # Print metrics
    print("\nTraining Results:")
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
    
    # Copy best model to models directory
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    destination = models_dir / 'best.pt'
    
    if best_model.exists():
        shutil.copy(best_model, destination)
        print(f"\n✓ Best model copied to: {destination}")
        print("\nYou can now use this model with the API:")
        print("  python app/main.py")
    
    # Validation
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION")
    print("=" * 70)
    
    val_results = model.val()
    print(f"\nValidation Metrics:")
    print(f"  mAP50: {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")
    print(f"  Precision: {val_results.box.mp:.4f}")
    print(f"  Recall: {val_results.box.mr:.4f}")
    
    print("\n" + "=" * 70)
    print("ALL DONE! 🎉")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check training results: runs/detect/aquarium_yolov11/")
    print("  2. View plots: results.png, confusion_matrix.png, etc.")
    print("  3. Start API: python app/main.py")
    print("  4. Test API: python example_usage.py <image_path>")
    print("=" * 70)

if __name__ == "__main__":
    main()
