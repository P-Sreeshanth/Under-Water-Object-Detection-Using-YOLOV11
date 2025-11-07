"""
Train YOLOv11 on Combined Seaclear + Aquarium Datasets
This creates a unified model that can detect both marine debris and aquatic animals
"""

import torch
from ultralytics import YOLO
from pathlib import Path
import yaml
import shutil
import argparse
from datetime import datetime

def create_combined_dataset(seaclear_yaml, aquarium_yaml, output_dir):
    """
    Create a combined dataset from Seaclear and Aquarium datasets.
    
    Args:
        seaclear_yaml: Path to seaclear_data.yaml
        aquarium_yaml: Path to aquarium_data.yaml
        output_dir: Output directory for combined dataset
    """
    print("=" * 80)
    print("📦 CREATING COMBINED SEACLEAR + AQUARIUM DATASET")
    print("=" * 80)
    
    # Load both dataset configs
    with open(seaclear_yaml, 'r') as f:
        seaclear_config = yaml.safe_load(f)
    
    with open(aquarium_yaml, 'r') as f:
        aquarium_config = yaml.safe_load(f)
    
    print(f"\n📊 Seaclear Dataset:")
    print(f"   Classes: {len(seaclear_config['names'])}")
    print(f"   Path: {seaclear_config['path']}")
    
    print(f"\n🐠 Aquarium Dataset:")
    print(f"   Classes: {len(aquarium_config['names'])}")
    print(f"   Path: {aquarium_config['path']}")
    
    # Create output directory structure
    output_path = Path(output_dir)
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Combine class names
    # Seaclear classes first (0-39), then Aquarium classes (40-46)
    combined_names = {}
    offset = 0
    
    # Add Seaclear classes
    for class_id, class_name in seaclear_config['names'].items():
        combined_names[class_id] = f"seaclear_{class_name}"
    
    # Add Aquarium classes with offset
    offset = len(seaclear_config['names'])
    for class_id, class_name in aquarium_config['names'].items():
        combined_names[offset + class_id] = f"aquarium_{class_name}"
    
    print(f"\n🔗 Combined Dataset:")
    print(f"   Total Classes: {len(combined_names)}")
    print(f"   Seaclear classes: 0-{len(seaclear_config['names'])-1}")
    print(f"   Aquarium classes: {offset}-{offset + len(aquarium_config['names'])-1}")
    
    # Copy and relabel Seaclear data
    print(f"\n📋 Processing Seaclear dataset...")
    seaclear_path = Path(seaclear_config['path'])
    for split in ['train', 'val']:
        img_src = seaclear_path / 'images' / split
        lbl_src = seaclear_path / 'labels' / split
        
        if img_src.exists():
            img_dst = output_path / 'images' / split
            lbl_dst = output_path / 'labels' / split
            
            # Copy images
            for img_file in img_src.glob('*.jpg'):
                shutil.copy2(img_file, img_dst / f"seaclear_{img_file.name}")
            
            # Copy labels (no class ID modification needed, they're already 0-39)
            for lbl_file in lbl_src.glob('*.txt'):
                shutil.copy2(lbl_file, lbl_dst / f"seaclear_{lbl_file.name}")
    
    # Copy and relabel Aquarium data
    print(f"\n🐠 Processing Aquarium dataset...")
    aquarium_path = Path(aquarium_config['path'])
    for split in ['train', 'val']:
        img_src = aquarium_path / split / 'images'
        lbl_src = aquarium_path / split / 'labels'
        
        if img_src.exists():
            img_dst = output_path / 'images' / split
            lbl_dst = output_path / 'labels' / split
            
            # Copy images
            for img_file in img_src.glob('*.jpg'):
                shutil.copy2(img_file, img_dst / f"aquarium_{img_file.name}")
            
            # Copy and update labels (add offset to class IDs)
            for lbl_file in lbl_src.glob('*.txt'):
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                
                # Update class IDs by adding offset
                updated_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        new_class_id = class_id + offset
                        updated_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                        updated_lines.append(updated_line)
                
                # Write updated labels
                with open(lbl_dst / f"aquarium_{lbl_file.name}", 'w') as f:
                    f.writelines(updated_lines)
    
    # Create combined YAML config
    combined_config = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': combined_names
    }
    
    yaml_path = output_path / 'combined_data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(combined_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Combined dataset created!")
    print(f"📄 Config saved to: {yaml_path}")
    
    return yaml_path, combined_names


def train_combined_model(
    data_yaml,
    model_size='n',
    epochs=150,
    imgsz=640,
    batch_size=16,
    device='',
    project='runs/combined',
    name='yolov11_seaclear_aquarium',
    pretrained=True
):
    """
    Train YOLOv11 on combined dataset.
    """
    print("\n" + "=" * 80)
    print("🌊🐠 TRAINING YOLOV11 ON COMBINED SEACLEAR + AQUARIUM DATASET 🐠🌊")
    print("=" * 80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        device = device if device else '0'
    else:
        print("⚠ CUDA not available, using CPU")
        device = 'cpu'
    
    # Load data configuration
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    num_classes = len(data_config['names'])
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total classes: {num_classes}")
    print(f"   Training images path: {Path(data_config['path']) / data_config['train']}")
    print(f"   Validation images path: {Path(data_config['path']) / data_config['val']}")
    
    # Initialize model
    model_name = f'yolo11{model_size}.pt' if pretrained else f'yolo11{model_size}.yaml'
    print(f"\n🤖 Initializing model: {model_name}")
    
    model = YOLO(model_name)
    
    # Training configuration
    print(f"\n⚙️  Training Configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Model: YOLOv11{model_size}")
    
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
        # Optimization
        optimizer='auto',
        lr0=0.01,
        patience=50,
        save_period=10,
        amp=True,
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
        val=True,
        plots=True,
    )
    
    print("\n" + "=" * 80)
    print("✅ Training complete!")
    print("=" * 80)
    
    # Validate the model
    print("\n🔍 Running validation...")
    metrics = model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP@50: {metrics.box.map50:.4f}")
    print(f"   mAP@50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    return results, metrics


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 on Combined Seaclear + Aquarium Dataset')
    
    # Dataset arguments
    parser.add_argument('--seaclear-yaml', type=str,
                       default='seaclear_dataset/seaclear_data.yaml',
                       help='Path to seaclear_data.yaml')
    parser.add_argument('--aquarium-yaml', type=str,
                       default='aquarium_data.yaml',
                       help='Path to aquarium_data.yaml')
    parser.add_argument('--output-dir', type=str,
                       default='combined_dataset',
                       help='Output directory for combined dataset')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='Model size')
    parser.add_argument('--pretrained', action='store_true', default=True)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/combined',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='yolov11_seaclear_aquarium',
                       help='Experiment name')
    
    # Options
    parser.add_argument('--skip-combine', action='store_true',
                       help='Skip dataset combination (use existing)')
    
    args = parser.parse_args()
    
    # Step 1: Create combined dataset
    if not args.skip_combine:
        combined_yaml, combined_names = create_combined_dataset(
            args.seaclear_yaml,
            args.aquarium_yaml,
            args.output_dir
        )
    else:
        combined_yaml = Path(args.output_dir) / 'combined_data.yaml'
        print(f"Using existing combined dataset: {combined_yaml}")
    
    # Step 2: Train model
    train_combined_model(
        data_yaml=str(combined_yaml),
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained
    )


if __name__ == "__main__":
    main()
