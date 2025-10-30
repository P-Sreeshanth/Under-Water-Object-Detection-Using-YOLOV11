"""
Training script for YOLOv11 object detection model.

This script trains YOLOv11 on the aquarium dataset for underwater
obstruction detection.
"""

import os
from pathlib import Path
from ultralytics import YOLO
import yaml
import shutil
from datetime import datetime


def prepare_dataset_structure(dataset_path, output_path='dataset_prepared'):
    """
    Prepare dataset in YOLO format.
    
    Expected structure:
    dataset_prepared/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    print("Preparing dataset structure...")
    
    # Create directories
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Dataset structure created at: {output_path}")
    
    return output_path


def create_dataset_yaml(dataset_path, class_names, output_file='dataset.yaml'):
    """
    Create dataset configuration YAML file for YOLO.
    
    Args:
        dataset_path: Path to dataset root
        class_names: List of class names
        output_file: Output YAML file path
    """
    
    dataset_path = Path(dataset_path).absolute()
    
    # Create YAML configuration
    config = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/val',  # Use val for test if no separate test set
        
        # Number of classes
        'nc': len(class_names),
        
        # Class names
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Save YAML file
    output_path = dataset_path / output_file
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Dataset YAML created: {output_path}")
    print(f"  Classes: {class_names}")
    
    return output_path


def train_yolov11(
    data_yaml,
    model_size='n',
    epochs=100,
    batch_size=16,
    img_size=640,
    patience=50,
    save_dir='runs/detect/train',
    pretrained=True,
    device='',
    project='underwater_detection',
    name='yolov11_training'
):
    """
    Train YOLOv11 model.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        model_size: Model size ('n', 's', 'm', 'l', 'x')
                   n=nano, s=small, m=medium, l=large, x=extra-large
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        patience: Early stopping patience
        save_dir: Directory to save results
        pretrained: Use pretrained weights
        device: Device to use ('', '0', '0,1', 'cpu')
        project: Project name
        name: Run name
    """
    
    print("=" * 60)
    print("YOLOV11 TRAINING")
    print("=" * 60)
    
    # Initialize model
    model_name = f'yolo11{model_size}.pt'
    print(f"\nInitializing YOLOv11-{model_size.upper()} model...")
    
    if pretrained:
        print(f"Loading pretrained weights: {model_name}")
        model = YOLO(model_name)
    else:
        print(f"Training from scratch")
        model = YOLO(f'yolo11{model_size}.yaml')
    
    # Training arguments
    print("\nTraining Configuration:")
    print(f"  Dataset: {data_yaml}")
    print(f"  Model: YOLOv11-{model_size.upper()}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {device if device else 'auto'}")
    print(f"  Patience: {patience}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        save=True,
        device=device,
        workers=8,
        project=project,
        name=name,
        exist_ok=True,
        
        # Augmentation settings
        hsv_h=0.015,  # HSV-Hue augmentation
        hsv_s=0.7,    # HSV-Saturation augmentation
        hsv_v=0.4,    # HSV-Value augmentation
        degrees=0.0,  # Rotation augmentation
        translate=0.1,  # Translation augmentation
        scale=0.5,    # Scale augmentation
        shear=0.0,    # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,   # Vertical flip probability
        fliplr=0.5,   # Horizontal flip probability
        mosaic=1.0,   # Mosaic augmentation
        mixup=0.0,    # Mixup augmentation
        
        # Optimization settings
        optimizer='auto',
        lr0=0.01,     # Initial learning rate
        lrf=0.01,     # Final learning rate factor
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        
        # Loss weights
        box=7.5,      # Box loss weight
        cls=0.5,      # Classification loss weight
        dfl=1.5,      # Distribution focal loss weight
        
        # Validation
        val=True,
        plots=True,
        save_period=-1,  # Save checkpoint every N epochs (-1 to disable)
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Print results
    print(f"\nBest model saved to: {model.trainer.best}")
    print(f"Last model saved to: {model.trainer.last}")
    
    # Validation results
    print("\nValidation Results:")
    metrics = results.results_dict
    if metrics:
        print(f"  mAP50: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")
    
    return model, results


def validate_model(model_path, data_yaml, img_size=640, device=''):
    """
    Validate trained model on validation set.
    
    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        img_size: Image size for validation
        device: Device to use
    """
    
    print("\n" + "=" * 60)
    print("MODEL VALIDATION")
    print("=" * 60)
    
    # Load model
    model = YOLO(model_path)
    
    # Validate
    results = model.val(
        data=str(data_yaml),
        imgsz=img_size,
        device=device,
        plots=True
    )
    
    # Print metrics
    print("\nValidation Metrics:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(model_path, export_format='onnx', img_size=640):
    """
    Export trained model to different formats.
    
    Args:
        model_path: Path to trained model
        export_format: Export format ('onnx', 'torchscript', 'tflite', etc.)
        img_size: Image size
    """
    
    print(f"\nExporting model to {export_format}...")
    
    model = YOLO(model_path)
    model.export(format=export_format, imgsz=img_size)
    
    print(f"✓ Model exported successfully")


def copy_best_model_to_models_dir(training_results_path, destination='models/best.pt'):
    """
    Copy the best trained model to the models directory.
    
    Args:
        training_results_path: Path to training results directory
        destination: Destination path for the model
    """
    
    training_results_path = Path(training_results_path)
    best_model = training_results_path / 'weights' / 'best.pt'
    
    if best_model.exists():
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_model, destination)
        print(f"\n✓ Best model copied to: {destination}")
        return destination
    else:
        print(f"\n✗ Best model not found at: {best_model}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv11 for underwater object detection')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--model_size', type=str, default='n',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=extra-large)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='',
                        help='Device to use (empty=auto, 0=cuda:0, cpu=cpu)')
    
    # Project arguments
    parser.add_argument('--project', type=str, default='underwater_detection',
                        help='Project name')
    parser.add_argument('--name', type=str, default='yolov11_training',
                        help='Run name')
    
    # Additional options
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                        help='Train from scratch')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation after training')
    parser.add_argument('--export', type=str, default=None,
                        help='Export format (onnx, torchscript, etc.)')
    parser.add_argument('--copy-to-models', action='store_true',
                        help='Copy best model to models/ directory')
    
    args = parser.parse_args()
    
    # Train model
    model, results = train_yolov11(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained
    )
    
    # Get the best model path
    best_model_path = model.trainer.best
    
    # Validate if requested
    if args.validate:
        validate_model(best_model_path, args.data, args.img_size, args.device)
    
    # Export if requested
    if args.export:
        export_model(best_model_path, args.export, args.img_size)
    
    # Copy to models directory if requested
    if args.copy_to_models:
        copy_best_model_to_models_dir(
            Path(args.project) / args.name,
            'models/best.pt'
        )
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nTo use this model in the API:")
    print(f"  1. Copy {best_model_path} to models/best.pt")
    print(f"  2. Update DETECTOR_MODEL_PATH in .env if needed")
    print(f"  3. Start the API: python app/main.py")
