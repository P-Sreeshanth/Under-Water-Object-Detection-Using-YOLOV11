"""
Resumable YOLOv8 pipeline for the dataa tracking dataset.

Stages:
1) Integrity scan
2) Dataset preparation (tracking bboxes -> YOLO labels)
3) Training
4) Validation
5) Inference smoke test

The pipeline writes progress and artifacts after every stage so interrupted runs
can resume safely.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import torch
import numpy as np
from ultralytics import YOLO


DEFAULT_STAGES = [
    "integrity_scan",
    "prepare_dataset",
    "train",
    "validate",
    "inference_smoke",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_lines(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def atomic_write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def parse_bbox(line: str) -> Optional[Tuple[float, float, float, float]]:
    parts = line.replace(",", " ").split()
    if len(parts) < 4:
        return None
    try:
        x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None
    return x, y, w, h


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def normalize_xywh(
    x: float,
    y: float,
    w: float,
    h: float,
    img_w: int,
    img_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    if img_w <= 0 or img_h <= 0:
        return None

    x = max(0.0, min(x, img_w - 1.0))
    y = max(0.0, min(y, img_h - 1.0))
    w = max(0.0, min(w, img_w - x))
    h = max(0.0, min(h, img_h - y))

    if w <= 1.0 or h <= 1.0:
        return None

    x_center = (x + w / 2.0) / img_w
    y_center = (y + h / 2.0) / img_h
    width = w / img_w
    height = h / img_h
    return x_center, y_center, width, height


def kmeans_numpy(features: np.ndarray, k: int, seed: int = 42, max_iters: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """Simple, dependency-free k-means clustering for small feature vectors."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    n_samples = features.shape[0]
    if n_samples == 0:
        raise ValueError("Cannot run k-means with zero samples")

    k = max(1, min(k, n_samples))
    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n_samples, size=k, replace=False)
    centers = features[init_idx].copy()
    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iters):
        # Squared Euclidean distance for assignment.
        distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        for cluster_id in range(k):
            mask = labels == cluster_id
            if not np.any(mask):
                # Re-seed empty clusters with a random data point.
                centers[cluster_id] = features[rng.integers(0, n_samples)]
                continue
            centers[cluster_id] = features[mask].mean(axis=0)

    return labels, centers


@dataclass
class Profile:
    name: str
    model: str
    epochs: int
    batch: int
    imgsz: int
    patience: int


PROFILES = {
    "fast": Profile("fast", "yolov8n.pt", 30, 24, 640, 12),
    "accurate": Profile("accurate", "yolov8s.pt", 80, 16, 832, 20),
}


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo_root = Path(args.repo_root).resolve()
        self.data_root = Path(args.data_root).resolve()
        self.output_root = Path(args.output_root).resolve()

        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_root / run_name
        self.dataset_dir = self.run_dir / "dataset"
        self.progress_file = self.run_dir / "progress.json"
        self.integrity_report = self.run_dir / "integrity_report.json"
        self.prepare_report = self.run_dir / "prepare_report.json"
        self.metrics_file = self.run_dir / "validation_metrics.json"
        self.summary_file = self.run_dir / "pipeline_summary.json"

        self.state = self._load_or_init_state()

    def _load_or_init_state(self) -> Dict:
        if self.args.resume and self.progress_file.exists():
            with open(self.progress_file, "r", encoding="utf-8") as f:
                return json.load(f)

        profile = PROFILES[self.args.profile]
        model = self.args.model or profile.model
        epochs = self.args.epochs or profile.epochs
        batch = self.args.batch or profile.batch
        imgsz = self.args.imgsz or profile.imgsz
        patience = self.args.patience or profile.patience

        state = {
            "run_name": self.run_dir.name,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "initialized",
            "completed_steps": [],
            "pending_steps": list(DEFAULT_STAGES),
            "failed_step": None,
            "next_command": self._next_command_hint(),
            "config": {
                "repo_root": str(self.repo_root),
                "data_root": str(self.data_root),
                "output_root": str(self.output_root),
                "run_dir": str(self.run_dir),
                "profile": self.args.profile,
                "model": model,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "patience": patience,
                "device": self.args.device,
                "workers": self.args.workers,
                "train_split": self.args.train_split,
                "seed": self.args.seed,
                "max_videos": self.args.max_videos,
            },
            "stages": {},
            "artifacts": {
                "progress_file": str(self.progress_file),
                "integrity_report": str(self.integrity_report),
                "prepare_report": str(self.prepare_report),
                "data_yaml": str(self.dataset_dir / "data.yaml"),
                "train_dir": str(self.run_dir / "train"),
                "validation_metrics": str(self.metrics_file),
                "summary": str(self.summary_file),
            },
            "runtime": {
                "expected_minutes": self._estimate_runtime_minutes(epochs, self.args.profile),
                "actual_seconds_by_stage": {},
            },
        }

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._save_state(state)
        return state

    def _estimate_runtime_minutes(self, epochs: int, profile: str) -> int:
        # Rough estimate tuned for RTX 4080 on this dataset scale.
        base = 180 if profile == "fast" else 420
        return int(base * max(1.0, epochs / (30 if profile == "fast" else 80)))

    def _resolve_device(self, requested: str) -> str:
        request = (requested or "").strip().lower()
        if request in {"", "auto"}:
            return "0" if torch.cuda.is_available() else "cpu"
        if request.startswith("cpu"):
            return "cpu"
        if not torch.cuda.is_available():
            print("CUDA requested but not available in current environment. Falling back to CPU.")
            return "cpu"
        return requested

    def _next_command_hint(self) -> str:
        return (
            f"python yolov8_dataa_pipeline.py --resume --run-name {self.run_dir.name} "
            f"--repo-root {self.repo_root} --data-root {self.data_root}"
        )

    def _save_state(self, state: Optional[Dict] = None) -> None:
        if state is not None:
            self.state = state
        self.state["updated_at"] = utc_now()
        atomic_write_json(self.progress_file, self.state)

    def _mark_stage_start(self, stage: str) -> float:
        if stage not in self.state["stages"]:
            self.state["stages"][stage] = {}
        self.state["stages"][stage]["started_at"] = utc_now()
        self.state["stages"][stage]["status"] = "running"
        self.state["status"] = f"running:{stage}"
        self.state["failed_step"] = None
        self._save_state()
        return time.perf_counter()

    def _mark_stage_done(self, stage: str, start_perf: float, meta: Optional[Dict] = None) -> None:
        elapsed = time.perf_counter() - start_perf
        stage_state = self.state["stages"].setdefault(stage, {})
        stage_state["finished_at"] = utc_now()
        stage_state["status"] = "completed"
        stage_state["elapsed_seconds"] = elapsed
        if meta:
            stage_state["meta"] = meta

        self.state["runtime"]["actual_seconds_by_stage"][stage] = elapsed
        if stage not in self.state["completed_steps"]:
            self.state["completed_steps"].append(stage)
        self.state["pending_steps"] = [s for s in DEFAULT_STAGES if s not in self.state["completed_steps"]]
        self.state["status"] = "running"
        self.state["next_command"] = self._next_command_hint()
        self._save_state()

    def _mark_stage_failed(self, stage: str, exc: Exception, start_perf: float) -> None:
        elapsed = time.perf_counter() - start_perf
        stage_state = self.state["stages"].setdefault(stage, {})
        stage_state["finished_at"] = utc_now()
        stage_state["status"] = "failed"
        stage_state["elapsed_seconds"] = elapsed
        stage_state["error"] = str(exc)

        self.state["status"] = "failed"
        self.state["failed_step"] = stage
        self.state["next_command"] = self._next_command_hint()
        self._save_state()

    def run(self) -> None:
        stages = [
            ("integrity_scan", self.stage_integrity_scan),
            ("prepare_dataset", self.stage_prepare_dataset),
            ("train", self.stage_train),
            ("validate", self.stage_validate),
            ("inference_smoke", self.stage_inference_smoke),
        ]

        for stage, fn in stages:
            if stage in self.state["completed_steps"]:
                print(f"[SKIP] {stage} already completed")
                continue
            if stage == "train" and self.args.skip_train:
                print("[SKIP] train stage skipped by flag")
                self.state["completed_steps"].append(stage)
                self.state["pending_steps"] = [s for s in DEFAULT_STAGES if s not in self.state["completed_steps"]]
                self._save_state()
                continue

            start_perf = self._mark_stage_start(stage)
            try:
                meta = fn()
                self._mark_stage_done(stage, start_perf, meta)
            except Exception as exc:  # pylint: disable=broad-except
                self._mark_stage_failed(stage, exc, start_perf)
                raise

        self.state["status"] = "completed"
        self.state["finished_at"] = utc_now()
        self.state["next_command"] = "completed"
        self._save_state()

        summary = {
            "status": self.state["status"],
            "run_name": self.run_dir.name,
            "completed_steps": self.state["completed_steps"],
            "actual_seconds_by_stage": self.state["runtime"]["actual_seconds_by_stage"],
            "expected_minutes": self.state["runtime"]["expected_minutes"],
            "artifacts": self.state["artifacts"],
        }
        atomic_write_json(self.summary_file, summary)

    def _video_dirs(self) -> List[Path]:
        items = [p for p in self.data_root.rglob("Video_*") if p.is_dir()]
        items.sort()
        if self.args.max_videos > 0:
            items = items[: self.args.max_videos]
        return items

    def _make_feature_vector(self, bbox: Tuple[float, float, float, float], img_w: int, img_h: int) -> Optional[np.ndarray]:
        normalized = normalize_xywh(
            x=bbox[0],
            y=bbox[1],
            w=bbox[2],
            h=bbox[3],
            img_w=img_w,
            img_h=img_h,
        )
        if normalized is None:
            return None

        cx, cy, bw, bh = normalized
        area = bw * bh
        aspect = bw / max(bh, 1e-6)

        # Feature set designed to separate track-size/shape/location patterns.
        return np.array([cx, cy, bw, bh, area, aspect], dtype=np.float32)

    def _build_class_name(self, center: np.ndarray, class_id: int) -> str:
        area = float(center[4])
        aspect = float(center[5])
        cy = float(center[1])

        if area < 0.01:
            size = "tiny"
        elif area < 0.03:
            size = "small"
        elif area < 0.07:
            size = "medium"
        elif area < 0.12:
            size = "large"
        else:
            size = "xlarge"

        if aspect < 0.75:
            shape = "tall"
        elif aspect <= 1.35:
            shape = "balanced"
        else:
            shape = "wide"

        if cy < 0.33:
            zone = "top"
        elif cy < 0.66:
            zone = "mid"
        else:
            zone = "bottom"

        return f"underwater_{size}_{shape}_{zone}_{class_id:02d}"

    def _discover_classes(self, videos: List[Path]) -> Dict:
        class_mode = self.args.class_mode
        if class_mode == "single":
            return {
                "class_mode": "single",
                "num_classes": 1,
                "names": {0: "underwater_object"},
                "assignment": {},
                "samples_used": 0,
            }

        entries: List[Tuple[str, int]] = []
        features: List[np.ndarray] = []
        invalid_bboxes = 0
        skipped_videos = 0

        for video_dir in videos:
            imgs_dir = video_dir / "imgs"
            gt_path = video_dir / "groundtruth_rect.txt"
            if not imgs_dir.exists() or not gt_path.exists():
                skipped_videos += 1
                continue

            image_files = sorted([*imgs_dir.glob("*.jpg"), *imgs_dir.glob("*.jpeg"), *imgs_dir.glob("*.png")])
            if not image_files:
                skipped_videos += 1
                continue

            first_img = cv2.imread(str(image_files[0]))
            if first_img is None:
                skipped_videos += 1
                continue

            img_h, img_w = first_img.shape[:2]
            gt_rows = [parse_bbox(line) for line in read_lines(gt_path)]

            for gt_idx, bbox in enumerate(gt_rows):
                if bbox is None:
                    invalid_bboxes += 1
                    continue

                feature = self._make_feature_vector(bbox, img_w, img_h)
                if feature is None:
                    invalid_bboxes += 1
                    continue

                entries.append((video_dir.name, gt_idx))
                features.append(feature)

        if not features:
            raise RuntimeError("No valid bounding boxes found to discover classes")

        feature_matrix = np.vstack(features)

        # Standardize so one dimension does not dominate clustering.
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        std[std == 0] = 1.0
        z_features = (feature_matrix - mean) / std

        requested_k = max(1, int(self.args.num_classes))
        k = min(requested_k, z_features.shape[0])
        labels, z_centers = kmeans_numpy(z_features, k=k, seed=int(self.args.seed))
        centers = (z_centers * std) + mean

        names = {class_id: self._build_class_name(centers[class_id], class_id) for class_id in range(k)}

        assignment = {f"{video}:{gt_idx}": int(labels[idx]) for idx, (video, gt_idx) in enumerate(entries)}
        counts = {class_id: int((labels == class_id).sum()) for class_id in range(k)}

        return {
            "class_mode": "auto",
            "num_classes": k,
            "requested_classes": requested_k,
            "names": names,
            "assignment": assignment,
            "counts": counts,
            "samples_used": int(z_features.shape[0]),
            "invalid_bboxes": invalid_bboxes,
            "skipped_videos": skipped_videos,
        }

    def stage_integrity_scan(self) -> Dict:
        videos = self._video_dirs()
        if not videos:
            raise RuntimeError(f"No Video_* folders found under {self.data_root}")

        report: Dict = {
            "created_at": utc_now(),
            "data_root": str(self.data_root),
            "video_count": len(videos),
            "videos": [],
            "totals": {
                "images": 0,
                "gt_lines": 0,
                "mismatch_videos": 0,
                "missing_imgs_dir": 0,
                "missing_gt": 0,
            },
        }

        for video_dir in videos:
            imgs_dir = video_dir / "imgs"
            gt_path = video_dir / "groundtruth_rect.txt"

            has_imgs = imgs_dir.exists()
            has_gt = gt_path.exists()
            image_files = []
            gt_lines = []
            if has_imgs:
                image_files = sorted(
                    [
                        *imgs_dir.glob("*.jpg"),
                        *imgs_dir.glob("*.jpeg"),
                        *imgs_dir.glob("*.png"),
                    ]
                )
            if has_gt:
                gt_lines = read_lines(gt_path)

            delta = len(image_files) - len(gt_lines)
            if delta != 0:
                report["totals"]["mismatch_videos"] += 1
            if not has_imgs:
                report["totals"]["missing_imgs_dir"] += 1
            if not has_gt:
                report["totals"]["missing_gt"] += 1

            report["totals"]["images"] += len(image_files)
            report["totals"]["gt_lines"] += len(gt_lines)

            report["videos"].append(
                {
                    "video": video_dir.name,
                    "path": str(video_dir),
                    "has_imgs": has_imgs,
                    "has_gt": has_gt,
                    "images": len(image_files),
                    "gt_lines": len(gt_lines),
                    "delta": delta,
                }
            )

        atomic_write_json(self.integrity_report, report)
        print(f"Integrity report saved: {self.integrity_report}")
        return {
            "video_count": report["video_count"],
            "total_images": report["totals"]["images"],
            "total_gt_lines": report["totals"]["gt_lines"],
            "mismatch_videos": report["totals"]["mismatch_videos"],
        }

    def stage_prepare_dataset(self) -> Dict:
        random.seed(self.args.seed)
        videos = self._video_dirs()
        class_info = self._discover_classes(videos)

        # Split by sequence to avoid leakage.
        shuffled = list(videos)
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * self.args.train_split)
        train_videos = sorted(shuffled[:split_idx])
        val_videos = sorted(shuffled[split_idx:])

        for split in ("train", "val"):
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        stats = {
            "train_videos": len(train_videos),
            "val_videos": len(val_videos),
            "images_linked_or_copied": 0,
            "labels_written": 0,
            "missing_gt_line_for_frame": 0,
            "invalid_bbox": 0,
            "empty_labels": 0,
            "hardlinks": 0,
            "copies": 0,
            "already_exists": 0,
            "class_mode": class_info["class_mode"],
            "num_classes": class_info["num_classes"],
        }

        def process_video(video_dir: Path, split: str) -> None:
            imgs_dir = video_dir / "imgs"
            gt_path = video_dir / "groundtruth_rect.txt"
            if not imgs_dir.exists() or not gt_path.exists():
                return

            gt_rows = [parse_bbox(line) for line in read_lines(gt_path)]
            image_files = sorted(
                [*imgs_dir.glob("*.jpg"), *imgs_dir.glob("*.jpeg"), *imgs_dir.glob("*.png")]
            )
            if not image_files:
                return

            first_img = cv2.imread(str(image_files[0]))
            if first_img is None:
                return
            img_h, img_w = first_img.shape[:2]

            for seq_idx, img in enumerate(image_files, start=1):
                try:
                    frame_idx = int(img.stem)
                except ValueError:
                    # Fallback to sequence index when stem is not numeric.
                    frame_idx = seq_idx

                gt_idx = frame_idx - 1
                out_base = f"{video_dir.name}_{img.stem}"
                out_img = self.dataset_dir / "images" / split / f"{out_base}{img.suffix.lower()}"
                out_lbl = self.dataset_dir / "labels" / split / f"{out_base}.txt"

                method = safe_link_or_copy(img, out_img)
                stats["images_linked_or_copied"] += 1
                if method == "hardlink":
                    stats["hardlinks"] += 1
                elif method == "copy":
                    stats["copies"] += 1
                else:
                    stats["already_exists"] += 1

                if gt_idx < 0 or gt_idx >= len(gt_rows) or gt_rows[gt_idx] is None:
                    stats["missing_gt_line_for_frame"] += 1
                    if not out_lbl.exists():
                        out_lbl.write_text("", encoding="utf-8")
                        stats["labels_written"] += 1
                        stats["empty_labels"] += 1
                    continue

                bbox = gt_rows[gt_idx]
                normalized = normalize_xywh(
                    x=bbox[0],
                    y=bbox[1],
                    w=bbox[2],
                    h=bbox[3],
                    img_w=img_w,
                    img_h=img_h,
                )
                if normalized is None:
                    stats["invalid_bbox"] += 1
                    if not out_lbl.exists():
                        out_lbl.write_text("", encoding="utf-8")
                        stats["labels_written"] += 1
                        stats["empty_labels"] += 1
                    continue

                class_id = 0
                if class_info["class_mode"] == "auto":
                    class_id = class_info["assignment"].get(f"{video_dir.name}:{gt_idx}", 0)

                if not out_lbl.exists():
                    out_lbl.write_text(
                        f"{class_id} {normalized[0]:.6f} {normalized[1]:.6f} {normalized[2]:.6f} {normalized[3]:.6f}\n",
                        encoding="utf-8",
                    )
                    stats["labels_written"] += 1

        for v in train_videos:
            process_video(v, "train")
        for v in val_videos:
            process_video(v, "val")

        data_yaml = self.dataset_dir / "data.yaml"
        class_names = class_info["names"]
        names_lines = [f"  {idx}: {class_names[idx]}" for idx in sorted(class_names.keys())]
        data_yaml.write_text(
            "\n".join(
                [
                    "# YOLOv8 data config generated from dataa tracking dataset",
                    f"path: {self.dataset_dir.as_posix()}",
                    "train: images/train",
                    "val: images/val",
                    f"nc: {class_info['num_classes']}",
                    "names:",
                    *names_lines,
                    "",
                ]
            ),
            encoding="utf-8",
        )

        prepare_report = {
            "created_at": utc_now(),
            "dataset_dir": str(self.dataset_dir),
            "data_yaml": str(data_yaml),
            "stats": stats,
            "split": {
                "train_videos": [v.name for v in train_videos],
                "val_videos": [v.name for v in val_videos],
            },
            "class_discovery": {
                "class_mode": class_info["class_mode"],
                "num_classes": class_info["num_classes"],
                "requested_classes": class_info.get("requested_classes", 1),
                "samples_used": class_info.get("samples_used", 0),
                "counts": class_info.get("counts", {}),
                "names": class_info.get("names", {}),
                "invalid_bboxes": class_info.get("invalid_bboxes", 0),
                "skipped_videos": class_info.get("skipped_videos", 0),
            },
        }
        atomic_write_json(self.prepare_report, prepare_report)
        print(f"Prepared dataset saved at: {self.dataset_dir}")
        return {
            "dataset_dir": str(self.dataset_dir),
            "data_yaml": str(data_yaml),
            "images": stats["images_linked_or_copied"],
            "labels": stats["labels_written"],
            "class_mode": class_info["class_mode"],
            "num_classes": class_info["num_classes"],
        }

    def stage_train(self) -> Dict:
        cfg = self.state["config"]
        model_name = cfg["model"]
        epochs = int(cfg["epochs"])
        batch = int(cfg["batch"])
        imgsz = int(cfg["imgsz"])
        patience = int(cfg["patience"])

        resolved_device = self._resolve_device(cfg["device"])

        train_dir = self.run_dir / "train"
        weights_dir = train_dir / "weights"
        last_pt = weights_dir / "last.pt"

        resume_mode = bool(self.args.resume and last_pt.exists())
        model = YOLO(str(last_pt if resume_mode else model_name))

        train_args = dict(
            data=str(self.dataset_dir / "data.yaml"),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=resolved_device,
            workers=int(cfg["workers"]),
            project=str(self.run_dir),
            name="train",
            exist_ok=True,
            patience=patience,
            amp=True,
            save_period=5,
            val=True,
            plots=True,
            optimizer="auto",
            lr0=0.01,
            seed=int(cfg["seed"]),
            verbose=True,
        )

        if resume_mode:
            train_args["resume"] = True

        model.train(**train_args)

        best_pt = weights_dir / "best.pt"
        if not best_pt.exists():
            raise RuntimeError(f"Training finished without best.pt at {best_pt}")

        return {
            "model": model_name,
            "resume": resume_mode,
            "device": resolved_device,
            "best": str(best_pt),
            "last": str(last_pt),
        }

    def stage_validate(self) -> Dict:
        cfg = self.state["config"]
        resolved_device = self._resolve_device(cfg["device"])
        best_pt = self.run_dir / "train" / "weights" / "best.pt"
        if not best_pt.exists():
            raise RuntimeError(f"Cannot validate. Missing model: {best_pt}")

        model = YOLO(str(best_pt))
        metrics = model.val(
            data=str(self.dataset_dir / "data.yaml"),
            imgsz=int(cfg["imgsz"]),
            device=resolved_device,
            workers=int(cfg["workers"]),
            plots=True,
        )

        out = {
            "created_at": utc_now(),
            "model": str(best_pt),
            "device": resolved_device,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }
        atomic_write_json(self.metrics_file, out)
        return out

    def stage_inference_smoke(self) -> Dict:
        cfg = self.state["config"]
        resolved_device = self._resolve_device(cfg["device"])
        best_pt = self.run_dir / "train" / "weights" / "best.pt"
        val_images = sorted((self.dataset_dir / "images" / "val").glob("*"))
        sample = val_images[: min(10, len(val_images))]
        if not sample:
            raise RuntimeError("No validation images available for inference smoke test")

        model = YOLO(str(best_pt))
        model.predict(
            source=[str(p) for p in sample],
            imgsz=int(cfg["imgsz"]),
            conf=0.25,
            device=resolved_device,
            save=True,
            project=str(self.run_dir),
            name="smoke_infer",
            exist_ok=True,
            verbose=True,
        )

        return {
            "samples_used": len(sample),
            "output_dir": str(self.run_dir / "smoke_infer"),
            "model": str(best_pt),
            "device": resolved_device,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resumable YOLOv8 pipeline for dataa")
    parser.add_argument("--repo-root", type=str, default=".", help="Repository root")
    parser.add_argument("--data-root", type=str, default="dataa", help="Path to dataa folder")
    parser.add_argument("--output-root", type=str, default="runs/dataa_yolov8", help="Output root")
    parser.add_argument("--run-name", type=str, default="", help="Stable run name for resume")

    parser.add_argument("--profile", type=str, default="fast", choices=["fast", "accurate"], help="Training profile")
    parser.add_argument("--model", type=str, default="", help="Override model checkpoint, e.g., yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=0, help="Override epoch count")
    parser.add_argument("--batch", type=int, default=0, help="Override batch size")
    parser.add_argument("--imgsz", type=int, default=0, help="Override image size")
    parser.add_argument("--patience", type=int, default=0, help="Override early stop patience")

    parser.add_argument("--device", type=str, default="", help="Training device, e.g., 0 or cpu")
    parser.add_argument("--workers", type=int, default=max(2, min(8, os.cpu_count() or 8)), help="Dataloader workers")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split by video sequences")
    parser.add_argument("--seed", type=int, default=42, help="Split and training seed")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos for quick smoke runs")
    parser.add_argument("--class-mode", type=str, default="auto", choices=["single", "auto"], help="Class assignment mode")
    parser.add_argument("--num-classes", type=int, default=12, help="Target classes when --class-mode auto (recommended 10-15)")

    parser.add_argument("--resume", action="store_true", help="Resume from existing progress file")
    parser.add_argument("--skip-train", action="store_true", help="Skip training stage")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.train_split <= 0.0 or args.train_split >= 1.0:
        raise ValueError("--train-split must be between 0 and 1 (exclusive)")

    pipeline = Pipeline(args)
    pipeline.run()
    print("Pipeline completed")
    print(f"Progress file: {pipeline.progress_file}")


if __name__ == "__main__":
    main()
