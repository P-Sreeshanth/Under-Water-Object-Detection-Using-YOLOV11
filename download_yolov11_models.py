"""
Download YOLOv11 pretrained models
"""

from ultralytics import YOLO
from pathlib import Path
import os

def download_yolov11_models():
    """Download all YOLOv11 model variants"""
    
    print("=" * 80)
    print("📥 DOWNLOADING YOLOV11 PRETRAINED MODELS")
    print("=" * 80)
    
    models = ['n', 's', 'm', 'l', 'x']
    
    for model_size in models:
        model_name = f'yolo11{model_size}.pt'
        print(f"\n📦 Downloading YOLO11{model_size}...")
        
        try:
            # This will automatically download the model if it doesn't exist
            model = YOLO(model_name)
            if os.path.exists(model_name):
                print(f"✓ YOLO11{model_size} downloaded successfully to: {os.path.abspath(model_name)}")
            else:
                print(f"✓ YOLO11{model_size} loaded successfully")
        except Exception as e:
            print(f"✗ Failed to download YOLO11{model_size}: {e}")
    
    print("\n" + "=" * 80)
    print("✅ Model download complete!")
    print("=" * 80)

if __name__ == "__main__":
    download_yolov11_models()
