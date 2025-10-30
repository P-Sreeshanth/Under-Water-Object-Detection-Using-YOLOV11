"""
Prepare and train on the Aquarium Dataset (COTS).

This script:
1. Prepares the aquarium dataset downloaded from Kaggle
2. Converts it to YOLO format
3. Trains YOLOv11 on the dataset
"""

import os
import shutil
from pathlib import Path
import yaml
import json
from tqdm import tqdm
import cv2


def find_dataset_path():
    """Find the aquarium dataset path."""
    
    possible_paths = [
        Path.home() / '.cache' / 'kagglehub' / 'datasets' / 'slavkoprytula' / 'aquarium-data-cots',
        Path('./aquarium-data-cots'),
        Path('../aquarium-data-cots'),
        Path('./dataset'),
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found dataset at: {path}")
            return path
    
    print("✗ Dataset not found. Please specify the path.")
    return None


def explore_dataset_structure(dataset_path):
    """Explore the dataset structure to understand the format."""
    
    dataset_path = Path(dataset_path)
    print("\n" + "=" * 60)
    print("EXPLORING DATASET STRUCTURE")
    print("=" * 60)
    
    # List all directories
    print("\nDirectory structure:")
    for item in sorted(dataset_path.rglob('*')):
        if item.is_dir():
            num_files = len(list(item.glob('*')))
            print(f"  {item.relative_to(dataset_path)}/  ({num_files} items)")
    
    # List file types
    print("\nFile types found:")
    extensions = {}
    for item in dataset_path.rglob('*'):
        if item.is_file():
            ext = item.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
    
    for ext, count in sorted(extensions.items()):
        print(f"  {ext or '(no extension)'}: {count} files")
    
    return extensions


def convert_coco_to_yolo(coco_annotation_file, output_labels_dir, images_dir):
    """
    Convert COCO format annotations to YOLO format.
    
    COCO format: [x, y, width, height] (absolute)
    YOLO format: [class_id, x_center, y_center, width, height] (normalized)
    """
    
    print(f"\nConverting COCO annotations to YOLO format...")
    
    # Load COCO annotations
    with open(coco_annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to filename mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Create class mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    class_names = list(categories.values())
    
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert each image's annotations
    converted_count = 0
    for img_id, annotations in tqdm(annotations_by_image.items(), desc="Converting"):
        img_info = images_dict[img_id]
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Create YOLO label file
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get COCO bbox [x, y, width, height]
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format [x_center, y_center, width, height] (normalized)
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # Get class ID (COCO IDs start from 1, YOLO from 0)
                class_id = ann['category_id'] - 1
                
                # Write YOLO format
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        converted_count += 1
    
    print(f"✓ Converted {converted_count} images")
    return class_names


def prepare_aquarium_dataset(dataset_path, output_path='aquarium_yolo'):
    """
    Prepare the aquarium dataset in YOLO format.
    
    Expected input structure:
    - train/ (images and annotations)
    - valid/ (images and annotations)
    - test/ (images and annotations)
    - _annotations.coco.json files
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    print("\n" + "=" * 60)
    print("PREPARING AQUARIUM DATASET")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    class_names = None
    
    # Process train and valid splits
    splits_mapping = {
        'train': 'train',
        'valid': 'val',
        'test': 'val'  # Merge test into val
    }
    
    for source_split, target_split in splits_mapping.items():
        source_dir = dataset_path / source_split
        
        if not source_dir.exists():
            print(f"⚠ {source_split} directory not found, skipping...")
            continue
        
        print(f"\n Processing {source_split} split...")
        
        # Find annotation file
        anno_file = source_dir / '_annotations.coco.json'
        if not anno_file.exists():
            print(f"⚠ Annotation file not found: {anno_file}")
            continue
        
        # Convert annotations
        split_class_names = convert_coco_to_yolo(
            anno_file,
            output_path / 'labels' / target_split,
            source_dir
        )
        
        if class_names is None:
            class_names = split_class_names
        
        # Copy images
        print(f"Copying images...")
        image_files = list(source_dir.glob('*.jpg')) + list(source_dir.glob('*.png'))
        
        for img_file in tqdm(image_files, desc=f"Copying {source_split} images"):
            dest_file = output_path / 'images' / target_split / img_file.name
            shutil.copy(img_file, dest_file)
        
        print(f"✓ Processed {len(image_files)} images from {source_split}")
    
    # Create dataset YAML
    if class_names:
        yaml_path = create_dataset_yaml_file(output_path, class_names)
        print(f"\n✓ Dataset prepared successfully!")
        print(f"  Output: {output_path}")
        print(f"  YAML: {yaml_path}")
        return output_path, yaml_path, class_names
    else:
        print("\n✗ Failed to prepare dataset")
        return None, None, None


def create_dataset_yaml_file(dataset_path, class_names):
    """Create dataset.yaml file for YOLO training."""
    
    dataset_path = Path(dataset_path).absolute()
    
    config = {
        'path': str(dataset_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = dataset_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def show_dataset_statistics(dataset_path):
    """Show statistics about the prepared dataset."""
    
    dataset_path = Path(dataset_path)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    for split in ['train', 'val']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png')))
        num_labels = len(list(labels_dir.glob('*.txt')))
        
        # Count total annotations
        total_annotations = 0
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                total_annotations += len(f.readlines())
        
        print(f"\n{split.upper()} Set:")
        print(f"  Images: {num_images}")
        print(f"  Label files: {num_labels}")
        print(f"  Total annotations: {total_annotations}")
        if num_images > 0:
            print(f"  Avg annotations per image: {total_annotations/num_images:.2f}")


def main():
    """Main function to prepare and optionally train on aquarium dataset."""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare and train on Aquarium Dataset'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=None,
        help='Path to aquarium dataset (auto-detect if not specified)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='aquarium_yolo',
        help='Output directory for prepared dataset'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train YOLOv11 after preparing dataset'
    )
    parser.add_argument(
        '--model_size',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='YOLOv11 model size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=640,
        help='Image size'
    )
    
    args = parser.parse_args()
    
    # Find dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = find_dataset_path()
    
    if not dataset_path or not dataset_path.exists():
        print("\n✗ Dataset not found!")
        print("\nPlease download the dataset first:")
        print("  python download_dataset.py")
        print("\nOr specify the path:")
        print(f"  python {__file__} --dataset_path /path/to/aquarium-data-cots")
        return
    
    # Explore dataset
    explore_dataset_structure(dataset_path)
    
    # Prepare dataset
    output_path, yaml_path, class_names = prepare_aquarium_dataset(
        dataset_path,
        args.output
    )
    
    if not output_path:
        print("\n✗ Failed to prepare dataset")
        return
    
    # Show statistics
    show_dataset_statistics(output_path)
    
    # Train if requested
    if args.train:
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        from train_yolov11 import train_yolov11, copy_best_model_to_models_dir
        
        model, results = train_yolov11(
            data_yaml=str(yaml_path),
            model_size=args.model_size,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            project='aquarium_training',
            name=f'yolov11{args.model_size}_aquarium'
        )
        
        # Copy best model to models directory
        training_dir = Path('aquarium_training') / f'yolov11{args.model_size}_aquarium'
        copy_best_model_to_models_dir(training_dir, 'models/best.pt')
        
        print("\n✓ Training complete!")
        print("\nTo use the trained model:")
        print("  1. The model is already copied to models/best.pt")
        print("  2. Start the API: python app/main.py")
        print("  3. Test: python example_usage.py")
    else:
        print("\n" + "=" * 60)
        print("DATASET READY FOR TRAINING")
        print("=" * 60)
        print(f"\nTo train YOLOv11 on this dataset:")
        print(f"  python train_yolov11.py --data {yaml_path} --epochs 100 --model_size n")
        print("\nOr run this script with --train flag:")
        print(f"  python prepare_aquarium_dataset.py --train --epochs 100")


if __name__ == "__main__":
    main()
