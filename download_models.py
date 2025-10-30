"""
Model download helper script.

This script helps download YOLOv11 models for the API.
"""

import os
from pathlib import Path
from ultralytics import YOLO


def download_yolo_model(model_size='n', save_path='models/best.pt'):
    """
    Download a pretrained YOLOv11 model.
    
    Args:
        model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        save_path: Path to save the model
    """
    print("=" * 60)
    print("YOLOv11 Model Downloader")
    print("=" * 60)
    print()
    
    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)
    
    # Map size to model name
    model_map = {
        'n': 'yolo11n.pt',
        's': 'yolo11s.pt',
        'm': 'yolo11m.pt',
        'l': 'yolo11l.pt',
        'x': 'yolo11x.pt'
    }
    
    if model_size not in model_map:
        print(f"Error: Invalid model size '{model_size}'")
        print(f"Valid sizes: {', '.join(model_map.keys())}")
        return False
    
    model_name = model_map[model_size]
    
    print(f"Downloading YOLOv11 model: {model_name}")
    print(f"This may take a few minutes depending on your internet connection...")
    print()
    
    try:
        # Download model
        model = YOLO(model_name)
        
        # Save to specified path
        print(f"\nSaving model to: {save_path}")
        
        # Copy the model file
        import shutil
        source_path = model_name
        shutil.copy(source_path, save_path)
        
        print(f"✓ Model downloaded and saved successfully!")
        print(f"  Path: {save_path}")
        
        # Show model info
        print(f"\nModel Information:")
        print(f"  Name: YOLOv11{model_size.upper()}")
        print(f"  Size: {model_size.upper()}")
        
        size_info = {
            'n': 'Nano - Fastest, smallest, good for edge devices',
            's': 'Small - Good balance of speed and accuracy',
            'm': 'Medium - Better accuracy, moderate speed',
            'l': 'Large - High accuracy, slower inference',
            'x': 'Extra Large - Highest accuracy, slowest inference'
        }
        print(f"  Description: {size_info[model_size]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        return False


def check_existing_models():
    """Check for existing model files."""
    print("=" * 60)
    print("Checking Existing Models")
    print("=" * 60)
    print()
    
    models_dir = Path('models')
    
    if not models_dir.exists():
        print("⚠ Models directory does not exist")
        models_dir.mkdir()
        print("✓ Created models directory")
        return
    
    # Check for YOLOv11 detector
    detector_path = models_dir / 'best.pt'
    if detector_path.exists():
        size_mb = detector_path.stat().st_size / (1024 * 1024)
        print(f"✓ YOLOv11 detector found: {detector_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ YOLOv11 detector not found: {detector_path}")
    
    # Check for enhancement model
    enhancer_path = models_dir / 'enhancer_model.pth'
    if enhancer_path.exists():
        size_mb = enhancer_path.stat().st_size / (1024 * 1024)
        print(f"✓ Enhancement model found: {enhancer_path}")
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"✗ Enhancement model not found: {enhancer_path}")
        print(f"  Note: Enhancement is optional. API will work without it.")


def main():
    """Main function."""
    print()
    print("=" * 60)
    print("UNDERWATER IMAGE ANALYSIS API")
    print("Model Download Helper")
    print("=" * 60)
    print()
    
    # Check existing models
    check_existing_models()
    
    print()
    print("=" * 60)
    print()
    
    # Ask user what to do
    print("Options:")
    print("  1. Download YOLOv11 Nano (fastest, ~6MB)")
    print("  2. Download YOLOv11 Small (~22MB)")
    print("  3. Download YOLOv11 Medium (~52MB)")
    print("  4. Download YOLOv11 Large (~110MB)")
    print("  5. Download YOLOv11 Extra Large (~140MB)")
    print("  6. Exit")
    print()
    
    choice = input("Enter choice (1-6): ").strip()
    
    size_map = {
        '1': 'n',
        '2': 's',
        '3': 'm',
        '4': 'l',
        '5': 'x'
    }
    
    if choice in size_map:
        size = size_map[choice]
        
        # Confirm before downloading
        print()
        confirm = input(f"Download YOLOv11{size.upper()} model? (y/n): ").lower()
        
        if confirm == 'y':
            print()
            success = download_yolo_model(size)
            
            if success:
                print()
                print("=" * 60)
                print("✓ Download Complete!")
                print("=" * 60)
                print()
                print("You can now start the API server:")
                print("  python app/main.py")
                print()
        else:
            print("Download cancelled.")
    
    elif choice == '6':
        print("Exiting...")
    
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
