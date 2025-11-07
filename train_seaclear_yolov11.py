"""
Train YOLOv11 on Seaclear Marine Debris Dataset
Supports multiple YOLOv11 model sizes (n, s, m, l, x)
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import argparse

def train_yolov11(
    data_yaml,
    model_size='n',
    epochs=100,
    imgsz=640,
    batch_size=16,
    device='',
    project='runs/seaclear',
    name='yolov11',
    pretrained=True,
    workers=8,
    optimizer='auto',
    lr0=0.01,
    patience=50,
    save_period=10,
    cache=False,
    amp=True
):
    """
    Train YOLOv11 model on Seaclear dataset
    
    Args:
        data_yaml: Path to data.yaml configuration file
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        batch_size: Batch size
        device: Device to use ('' for auto, '0' for GPU 0, 'cpu' for CPU)
        project: Project directory
        name: Experiment name
        pretrained: Use pretrained weights
        workers: Number of data loader workers
        optimizer: Optimizer type
        lr0: Initial learning rate
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        cache: Cache images for faster training
        amp: Use Automatic Mixed Precision
    """
    
    print("=" * 80)
    print("🌊 TRAINING YOLOV11 ON SEACLEAR MARINE DEBRIS DATASET 🌊")
    print("=" * 80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        device = device if device else '0'
    else:
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    
    # Load data configuration
    print(f"\n📂 Loading dataset configuration from: {data_yaml}")
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = len(data_config['names'])
    print(f"✓ Dataset has {num_classes} classes")
    
    # Initialize model
    model_name = f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml'
    print(f"\n🤖 Initializing model: {model_name}")
    print(f"   Pretrained: {pretrained}")
    print(f"   Model size: {model_size}")
    
    model = YOLO(model_name)
    
    # Print training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Workers: {workers}")
    print(f"   Optimizer: {optimizer}")
    print(f"   Learning rate: {lr0}")
    print(f"   Patience: {patience}")
    print(f"   AMP: {amp}")
    print(f"   Cache: {cache}")
    
    # Start training
    print(f"\n🚀 Starting training...")
    print("=" * 80)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        workers=workers,
        optimizer=optimizer,
        lr0=lr0,
        patience=patience,
        save_period=save_period,
        cache=cache,
        amp=amp,
        # Data augmentation
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
        # Additional settings
        verbose=True,
        seed=42,
        deterministic=False,
        single_cls=False,
        rect=False,
        cos_lr=False,
        close_mosaic=10,
        resume=False,
        overlap_mask=True,
        mask_ratio=4,
        dropout=0.0,
        val=True,
        plots=True,
    )
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    
    # Print results
    print("\n📊 Training Results:")
    print(f"   Best weights: {Path(project) / name / 'weights' / 'best.pt'}")
    print(f"   Last weights: {Path(project) / name / 'weights' / 'last.pt'}")
    
    # Validate the model
    print("\n🔍 Running validation on best model...")
    metrics = model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP@50: {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    return results, metrics

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Seaclear Marine Debris Dataset')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, 
                       default='seaclear_dataset/seaclear_data.yaml',
                       help='Path to data.yaml file')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size (n=nano, s=small, m=medium, l=large, x=extra large)')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from scratch')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (0, 1, 2, cpu, or empty for auto)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of data loader workers')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='auto',
                       choices=['SGD', 'Adam', 'AdamW', 'auto'],
                       help='Optimizer type')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/seaclear',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11',
                       help='Experiment name')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # Performance arguments
    parser.add_argument('--cache', action='store_true', default=False,
                       help='Cache images for faster training')
    parser.add_argument('--amp', action='store_true', default=True,
                       help='Use Automatic Mixed Precision')
    parser.add_argument('--no-amp', dest='amp', action='store_false',
                       help='Disable Automatic Mixed Precision')
    
    args = parser.parse_args()
    
    # Train the model
    train_yolov11(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save_period=args.save_period,
        cache=args.cache,
        amp=args.amp
    )

if __name__ == "__main__":
    main()
