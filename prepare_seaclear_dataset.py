"""
Prepare Seaclear Marine Debris Dataset for YOLOv11 Training
Converts COCO format annotations to YOLO format
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm
import random

def create_directories(base_path):
    """Create the YOLOv11 dataset directory structure"""
    dirs = [
        'images/train',
        'images/val',
        'labels/train',
        'labels/val'
    ]
    for dir_path in dirs:
        Path(base_path / dir_path).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created directory structure at {base_path}")

def convert_coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height]
    All normalized to [0, 1]
    """
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return [x_center, y_center, width, height]

def process_dataset(coco_json_path, dataset_root, output_path, train_split=0.8):
    """
    Process COCO format dataset and convert to YOLO format
    
    Args:
        coco_json_path: Path to dataset.json
        dataset_root: Root directory containing all location folders
        output_path: Output directory for YOLO format dataset
        train_split: Percentage of data for training (default: 0.8)
    """
    print(f"\n📂 Loading COCO annotations from {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    create_directories(output_path)
    
    # Parse categories
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"\n📋 Found {len(categories)} categories:")
    for cat_id, cat_name in sorted(categories.items())[:10]:
        print(f"   {cat_id}: {cat_name}")
    if len(categories) > 10:
        print(f"   ... and {len(categories) - 10} more")
    
    # Create category mapping (COCO IDs to YOLO IDs starting from 0)
    cat_id_to_yolo_id = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}
    
    # Parse images - create a mapping from file_name to image info
    images_dict = {img['file_name']: img for img in coco_data['images']}
    print(f"\n🖼️  Total images in dataset: {len(images_dict)}")
    
    # Parse annotations - group by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"📊 Total annotations: {len(coco_data['annotations'])}")
    
    # Find all image files in the dataset
    print(f"\n🔍 Searching for images in {dataset_root}")
    all_image_files = []
    locations = ['Bistrina', 'Jakljan', 'Lokrum', 'Marseille', 'Slano']
    
    for location in locations:
        location_path = dataset_root / location
        if location_path.exists():
            # Find all JPG files recursively
            for img_file in location_path.rglob('*.jpg'):
                all_image_files.append(img_file)
    
    print(f"✓ Found {len(all_image_files)} image files on disk")
    
    # Split dataset into train and val
    random.seed(42)
    random.shuffle(all_image_files)
    split_idx = int(len(all_image_files) * train_split)
    train_files = all_image_files[:split_idx]
    val_files = all_image_files[split_idx:]
    
    print(f"\n📊 Dataset split:")
    print(f"   Training: {len(train_files)} images ({train_split*100:.0f}%)")
    print(f"   Validation: {len(val_files)} images ({(1-train_split)*100:.0f}%)")
    
    # Process train and val sets
    for split_name, file_list in [('train', train_files), ('val', val_files)]:
        print(f"\n🔄 Processing {split_name} set...")
        
        images_copied = 0
        labels_created = 0
        
        for img_path in tqdm(file_list, desc=f"Converting {split_name}"):
            img_filename = img_path.name
            
            # Find image info in COCO data
            if img_filename not in images_dict:
                continue
            
            img_info = images_dict[img_filename]
            img_id = img_info['id']
            img_width = img_info['width']
            img_height = img_info['height']
            
            # Copy image
            dst_img_path = output_path / 'images' / split_name / img_filename
            shutil.copy2(img_path, dst_img_path)
            images_copied += 1
            
            # Create YOLO format label file
            label_filename = img_filename.replace('.jpg', '.txt')
            label_path = output_path / 'labels' / split_name / label_filename
            
            # Get annotations for this image
            if img_id in annotations_by_image:
                with open(label_path, 'w') as f:
                    for ann in annotations_by_image[img_id]:
                        cat_id = ann['category_id']
                        yolo_cat_id = cat_id_to_yolo_id[cat_id]
                        bbox = ann['bbox']
                        
                        # Convert to YOLO format
                        yolo_bbox = convert_coco_to_yolo_bbox(bbox, img_width, img_height)
                        
                        # Write to file: class_id x_center y_center width height
                        f.write(f"{yolo_cat_id} {' '.join([f'{v:.6f}' for v in yolo_bbox])}\n")
                labels_created += 1
            else:
                # Create empty label file for images without annotations
                label_path.touch()
        
        print(f"   ✓ Copied {images_copied} images")
        print(f"   ✓ Created {labels_created} label files")
    
    # Create data.yaml file
    yaml_content = f"""# Seaclear Marine Debris Dataset Configuration for YOLOv11
path: {output_path.absolute()}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes ({len(categories)} total)
names:
"""
    
    # Add class names in order
    for yolo_id in range(len(categories)):
        # Find the original category ID
        original_cat_id = [cid for cid, yid in cat_id_to_yolo_id.items() if yid == yolo_id][0]
        cat_name = categories[original_cat_id]
        yaml_content += f"  {yolo_id}: {cat_name}\n"
    
    yaml_path = output_path / 'seaclear_data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📄 Configuration file saved to: {yaml_path}")
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total classes: {len(categories)}")
    print(f"   Training images: {len(train_files)}")
    print(f"   Validation images: {len(val_files)}")
    print(f"   Total images: {len(all_image_files)}")
    
    return yaml_path

if __name__ == "__main__":
    # Define paths
    COCO_JSON = Path(r"c:\Users\KBhagyaRekha\Downloads\Compressed\archive\Seaclear Marine Debris Dataset\dataset.json")
    DATASET_ROOT = Path(r"c:\Users\KBhagyaRekha\Downloads\Compressed\archive\Seaclear Marine Debris Dataset")
    OUTPUT_PATH = Path(r"c:\Users\KBhagyaRekha\Under-Water-Object-Detection-Using-YOLOV11\seaclear_dataset")
    
    print("=" * 70)
    print("🌊 SEACLEAR MARINE DEBRIS DATASET PREPARATION FOR YOLOV11 🌊")
    print("=" * 70)
    
    # Process the dataset
    yaml_file = process_dataset(COCO_JSON, DATASET_ROOT, OUTPUT_PATH, train_split=0.8)
    
    print("\n" + "=" * 70)
    print("🎯 Next steps:")
    print(f"   Run: python train_seaclear_yolov11.py")
    print("=" * 70)
