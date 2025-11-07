# YOLOv11 Training Pipeline for Seaclear Marine Debris Dataset
# PowerShell Script

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🌊 YOLOv11 SEACLEAR MARINE DEBRIS DETECTION TRAINING PIPELINE 🌊" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

# Step 1: Check Python and install dependencies
Write-Host "`n📦 Step 1: Installing dependencies..." -ForegroundColor Yellow
pip install -q ultralytics opencv-python pillow pyyaml tqdm

# Step 2: Prepare the dataset
Write-Host "`n📂 Step 2: Preparing Seaclear dataset (COCO to YOLO format)..." -ForegroundColor Yellow
python prepare_seaclear_dataset.py

# Check if dataset preparation was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Dataset preparation completed successfully!" -ForegroundColor Green
} else {
    Write-Host "✗ Dataset preparation failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Train YOLOv11
Write-Host "`n🚀 Step 3: Starting YOLOv11 training..." -ForegroundColor Yellow
Write-Host "   Model: YOLOv11n (nano - fastest)" -ForegroundColor Cyan
Write-Host "   Epochs: 100" -ForegroundColor Cyan
Write-Host "   Image size: 640x640" -ForegroundColor Cyan
Write-Host "   Batch size: 16" -ForegroundColor Cyan

# Train with default settings (YOLOv11n, 100 epochs)
python train_seaclear_yolov11.py `
    --model n `
    --epochs 100 `
    --batch 16 `
    --imgsz 640 `
    --device "" `
    --workers 8 `
    --pretrained `
    --amp `
    --project runs/seaclear `
    --name yolov11n_100ep

# Check if training was successful
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Training completed successfully!" -ForegroundColor Green
    Write-Host "`n📊 Results saved to: runs/seaclear/yolov11n_100ep" -ForegroundColor Cyan
    Write-Host "   - Best weights: runs/seaclear/yolov11n_100ep/weights/best.pt" -ForegroundColor Cyan
    Write-Host "   - Last weights: runs/seaclear/yolov11n_100ep/weights/last.pt" -ForegroundColor Cyan
    Write-Host "   - Training plots: runs/seaclear/yolov11n_100ep/" -ForegroundColor Cyan
} else {
    Write-Host "`n✗ Training failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan
Write-Host "🎉 PIPELINE COMPLETED! 🎉" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 79) -ForegroundColor Cyan

Write-Host "`n📝 Next Steps:" -ForegroundColor Yellow
Write-Host "   1. Check training results in runs/seaclear/yolov11n_100ep/" -ForegroundColor Cyan
Write-Host "   2. Validate: python -m ultralytics val model=runs/seaclear/yolov11n_100ep/weights/best.pt data=seaclear_dataset/seaclear_data.yaml" -ForegroundColor Cyan
Write-Host "   3. Test inference: python -m ultralytics predict model=runs/seaclear/yolov11n_100ep/weights/best.pt source=path/to/test/images" -ForegroundColor Cyan
