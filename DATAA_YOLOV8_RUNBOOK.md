# dataa YOLOv8 Runbook

This runbook is for a resumable YOLOv8 training flow on the `dataa` folder.

The pipeline now supports automatic pseudo-class discovery for DATAA tracking boxes.
Use `--class-mode auto --num-classes 10..15` to build multi-class labels from full-dataset bbox patterns.

## What the pipeline saves

Each run writes to:

- `runs/dataa_yolov8/<run_name>/progress.json`
- `runs/dataa_yolov8/<run_name>/integrity_report.json`
- `runs/dataa_yolov8/<run_name>/prepare_report.json`
- `runs/dataa_yolov8/<run_name>/dataset/data.yaml`
- `runs/dataa_yolov8/<run_name>/train/weights/best.pt`
- `runs/dataa_yolov8/<run_name>/validation_metrics.json`
- `runs/dataa_yolov8/<run_name>/pipeline_summary.json`

`progress.json` always includes:

- completed steps
- pending steps
- failed step (if any)
- next resume command

## Stages

1. integrity_scan
2. prepare_dataset
3. train
4. validate
5. inference_smoke

## Quick start (Windows)

Fast baseline:

```powershell
.\run_dataa_yolov8.ps1 -Profile fast
```

Higher accuracy:

```powershell
.\run_dataa_yolov8.ps1 -Profile accurate
```

Stable run name (recommended):

```powershell
.\run_dataa_yolov8.ps1 -RunName dataa_fast_v1 -Profile fast
```

12-class auto-discovery (recommended for DATAA):

```powershell
.\run_dataa_yolov8.ps1 -RunName dataa_auto12_v1 -Profile fast -ClassMode auto -NumClasses 12
```

15-class auto-discovery:

```powershell
.\run_dataa_yolov8.ps1 -RunName dataa_auto15_v1 -Profile accurate -ClassMode auto -NumClasses 15
```

Resume interrupted run:

```powershell
.\run_dataa_yolov8.ps1 -RunName dataa_fast_v1 -Profile fast -Resume
```

## Direct Python usage

```powershell
.\.venv\Scripts\python.exe yolov8_dataa_pipeline.py --repo-root . --data-root dataa --output-root runs/dataa_yolov8 --run-name dataa_fast_v1 --profile fast
```

With class auto-discovery:

```powershell
.\.venv\Scripts\python.exe yolov8_dataa_pipeline.py --repo-root . --data-root dataa --output-root runs/dataa_yolov8 --run-name dataa_auto12_v1 --profile fast --class-mode auto --num-classes 12
```

Resume:

```powershell
.\.venv\Scripts\python.exe yolov8_dataa_pipeline.py --repo-root . --data-root dataa --output-root runs/dataa_yolov8 --run-name dataa_fast_v1 --profile fast --resume
```

## Recommended defaults for RTX 4080 16GB

- Fast profile: `yolov8n`, 30 epochs, 640 image size
- Accurate profile: `yolov8s`, 80 epochs, 832 image size

## Smoke validation commands

Validate best checkpoint:

```powershell
.\.venv\Scripts\python.exe -m ultralytics val model=runs/dataa_yolov8/dataa_fast_v1/train/weights/best.pt data=runs/dataa_yolov8/dataa_fast_v1/dataset/data.yaml device=0
```

Inference smoke test:

```powershell
.\.venv\Scripts\python.exe -m ultralytics predict model=runs/dataa_yolov8/dataa_fast_v1/train/weights/best.pt source=runs/dataa_yolov8/dataa_fast_v1/dataset/images/val imgsz=640 conf=0.25 device=0 save=True
```

## Notes

- The dataset converter assumes `groundtruth_rect.txt` has one bbox per frame line and frame files are named numerically.
- Missing frames are handled safely by skipping unavailable frame indices.
- The pipeline uses hardlinks first, then falls back to file copy.
- DATAA does not provide semantic class IDs in `groundtruth_rect.txt`; `class-mode auto` creates pseudo-classes by clustering bbox geometry and location statistics.
