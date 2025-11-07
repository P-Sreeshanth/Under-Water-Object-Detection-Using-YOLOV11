"""
Inference script for YOLOv11 trained on Seaclear Marine Debris Dataset
Supports image, video, and folder inference with visualization
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import torch

def run_inference(
    model_path,
    source,
    save_dir='runs/detect',
    conf_threshold=0.25,
    iou_threshold=0.45,
    imgsz=640,
    device='',
    save=True,
    show=False,
    save_txt=False,
    save_conf=False,
    line_thickness=2,
    hide_labels=False,
    hide_conf=False,
    max_det=300
):
    """
    Run inference on images, videos, or folders
    
    Args:
        model_path: Path to trained model weights
        source: Path to image, video, or folder
        save_dir: Directory to save results
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        imgsz: Input image size
        device: Device to use
        save: Save results
        show: Show results
        save_txt: Save results to txt
        save_conf: Save confidence scores
        line_thickness: Bounding box line thickness
        hide_labels: Hide labels
        hide_conf: Hide confidence scores
        max_det: Maximum detections per image
    """
    
    print("=" * 80)
    print("🌊 SEACLEAR MARINE DEBRIS DETECTION - INFERENCE 🌊")
    print("=" * 80)
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
        device = device if device else '0'
    else:
        print("⚠ Using CPU")
        device = 'cpu'
    
    # Load model
    print(f"\n📦 Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Print model info
    print(f"✓ Model loaded successfully")
    print(f"   Classes: {len(model.names)}")
    print(f"   Device: {device}")
    
    # Run inference
    print(f"\n🔍 Running inference on: {source}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IoU threshold: {iou_threshold}")
    print(f"   Image size: {imgsz}")
    
    results = model.predict(
        source=source,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        show=show,
        line_width=line_thickness,
        show_labels=not hide_labels,
        show_conf=not hide_conf,
        max_det=max_det,
        project=save_dir,
        exist_ok=True,
        verbose=True
    )
    
    # Print detection statistics
    print("\n" + "=" * 80)
    print("📊 Detection Statistics")
    print("=" * 80)
    
    total_detections = 0
    class_counts = {}
    
    for result in results:
        boxes = result.boxes
        total_detections += len(boxes)
        
        # Count detections per class
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print(f"\n📈 Total Detections: {total_detections}")
    
    if class_counts:
        print(f"\n🏷️  Detections by Class:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {cls_name}: {count}")
    else:
        print("\n⚠ No detections found")
    
    if save:
        print(f"\n💾 Results saved to: {save_dir}")
    
    print("\n" + "=" * 80)
    print("✅ Inference Complete!")
    print("=" * 80)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='YOLOv11 Seaclear Marine Debris Detection - Inference')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, video, or folder')
    
    # Detection arguments
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (0-1)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS (0-1)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--max-det', type=int, default=300,
                       help='Maximum detections per image')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (0, 1, cpu, or empty for auto)')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='runs/detect',
                       help='Directory to save results')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save results')
    parser.add_argument('--no-save', dest='save', action='store_false',
                       help='Do not save results')
    parser.add_argument('--show', action='store_true', default=False,
                       help='Show results')
    parser.add_argument('--save-txt', action='store_true', default=False,
                       help='Save results to txt file')
    parser.add_argument('--save-conf', action='store_true', default=False,
                       help='Save confidence scores in txt file')
    
    # Visualization arguments
    parser.add_argument('--line-thickness', type=int, default=2,
                       help='Bounding box line thickness')
    parser.add_argument('--hide-labels', action='store_true', default=False,
                       help='Hide labels')
    parser.add_argument('--hide-conf', action='store_true', default=False,
                       help='Hide confidence scores')
    
    args = parser.parse_args()
    
    # Run inference
    run_inference(
        model_path=args.model,
        source=args.source,
        save_dir=args.save_dir,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        show=args.show,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        line_thickness=args.line_thickness,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
        max_det=args.max_det
    )

if __name__ == "__main__":
    main()
