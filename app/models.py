"""
Model loading and inference logic for U-Net enhancement and YOLO detection.
"""

import base64
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from .utils import logger, settings, preprocess_for_enhancement, postprocess_enhanced_image


class UNet(nn.Module):
    """
    U-Net architecture for image enhancement.

    This is a standard U-Net implementation for underwater image enhancement.
    Adjust the architecture based on your trained model.
    """

    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output layer
        self.out = nn.Conv2d(64, out_channels, 1)

        # Max pooling
        self.pool = nn.MaxPool2d(2, 2)

    def conv_block(self, in_channels, out_channels):
        """Create a convolutional block with two conv layers."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the U-Net."""
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out(dec1)
        return torch.sigmoid(out)


class ModelManager:
    """
    Manager class for loading and running inference with enhancement + ensemble detection.

    This class supports:
    - U-Net enhancement (optional)
    - Loading configured YOLO detectors
    - Auto-discovering additional .pt detector models
    - Ensemble fusion via class-wise NMS
    - Optional Phi-4 multimodal verification for uncertain detections
    """

    def __init__(self):
        """Initialize the model manager."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.enhancer_model = None
        self.seaclear_model = None
        self.aquarium_model = None

        self.detector_models: List[Dict[str, Any]] = []
        self._loaded_detector_paths: set[str] = set()
        self.class_names: Dict[str, str] = {}

        self._load_models()

    def _load_models(self):
        """Load enhancement and detection models."""
        try:
            # Load U-Net enhancement model
            self._load_enhancer()

            # Load configured detectors
            if settings.USE_MULTI_MODEL:
                logger.info("Loading configured multi-model detection (Seaclear + Aquarium)")
                self._load_seaclear_model()
                self._load_aquarium_model()
            else:
                self._load_seaclear_model()

            # Auto-discover all candidate .pt detector weights if enabled
            if settings.AUTO_DISCOVER_YOLO_MODELS:
                self._auto_discover_models()

            logger.info(f"Detection models loaded: {len(self.detector_models)}")
            logger.info(f"Phi-4 verification enabled: {settings.PHI4_ENABLED}")

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def _parse_model_patterns(self) -> List[str]:
        """Parse model glob patterns from settings."""
        patterns = [p.strip() for p in settings.YOLO_MODEL_GLOB_PATTERNS.split(",") if p.strip()]
        if patterns:
            return patterns
        return ["runs/**/weights/*.pt", "models/*.pt", "yolo11*.pt"]

    def _resolve_model_path(self, configured_path: str, fallback_globs: Optional[List[str]] = None) -> Optional[Path]:
        """Resolve a model path, optionally trying fallback glob patterns when missing."""
        primary = Path(configured_path)
        if primary.exists() and primary.suffix.lower() == ".pt":
            return primary

        for pattern in (fallback_globs or []):
            matches = [p for p in Path(".").glob(pattern) if p.is_file() and p.suffix.lower() == ".pt"]
            matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
            if matches:
                logger.warning(
                    f"Configured model path not found: {primary}. Using fallback model: {matches[0]}"
                )
                return matches[0]

        return None

    def _model_tag_from_path(self, model_path: Path) -> str:
        """Build a readable model tag from path."""
        path = model_path.resolve()
        if path.stem in {"best", "last"} and len(path.parents) >= 2:
            raw_tag = f"{path.parents[1].name}_{path.stem}"
        else:
            raw_tag = path.stem
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw_tag).strip("_")
        return cleaned or "model"

    def _make_unique_tag(self, base_tag: str) -> str:
        """Ensure model tags are unique when loading many .pt files."""
        existing = {entry["tag"] for entry in self.detector_models}
        if base_tag not in existing:
            return base_tag

        idx = 2
        candidate = f"{base_tag}_{idx}"
        while candidate in existing:
            idx += 1
            candidate = f"{base_tag}_{idx}"
        return candidate

    def _load_detector_by_path(
        self,
        model_path: Path,
        forced_tag: Optional[str] = None,
        legacy_slot: Optional[str] = None,
    ) -> Optional[YOLO]:
        """Load and register a single YOLO detector from a .pt path."""
        if not model_path.exists() or model_path.suffix.lower() != ".pt":
            return None

        resolved = model_path.resolve()
        path_key = str(resolved)

        if path_key in self._loaded_detector_paths:
            logger.info(f"Skipping duplicate detector path: {resolved}")
            for entry in self.detector_models:
                if entry["path"] == path_key:
                    return entry["model"]
            return None

        try:
            model = YOLO(path_key)
            if self.device.type == 'cuda':
                model.to('cuda')

            tag = forced_tag or self._model_tag_from_path(resolved)
            tag = self._make_unique_tag(tag)

            self.detector_models.append(
                {
                    "tag": tag,
                    "path": path_key,
                    "model": model,
                }
            )
            self._loaded_detector_paths.add(path_key)

            if legacy_slot == "seaclear":
                self.seaclear_model = model
            elif legacy_slot == "aquarium":
                self.aquarium_model = model

            for class_id, class_name in model.names.items():
                self.class_names[f"{tag}_{class_id}"] = f"{tag}_{class_name}"

            logger.info(f"Loaded detector [{tag}] from {resolved} ({len(model.names)} classes)")
            return model

        except Exception as e:
            logger.error(f"Failed to load detector model from {resolved}: {e}")
            return None

    def _load_enhancer(self):
        """Load the U-Net image enhancement model."""
        enhancer_path = Path(settings.ENHANCER_MODEL_PATH)

        if not enhancer_path.exists():
            logger.warning(
                f"Enhancer model not found at {enhancer_path}. "
                "Image enhancement will be skipped."
            )
            self.enhancer_model = None
            return

        try:
            self.enhancer_model = UNet(in_channels=3, out_channels=3)
            checkpoint = torch.load(enhancer_path, map_location=self.device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.enhancer_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.enhancer_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.enhancer_model.load_state_dict(checkpoint)
            else:
                self.enhancer_model.load_state_dict(checkpoint)

            self.enhancer_model.to(self.device)
            self.enhancer_model.eval()

            logger.info(f"Enhancement model loaded from {enhancer_path}")

        except Exception as e:
            logger.error(f"Failed to load enhancement model: {e}")
            self.enhancer_model = None

    def _load_detector(self):
        """Load detector in legacy mode for backward compatibility."""
        self._load_seaclear_model()

    def _load_seaclear_model(self):
        """Load the configured Seaclear marine debris detector."""
        seaclear_path = self._resolve_model_path(settings.SEACLEAR_MODEL_PATH)

        if seaclear_path is None:
            logger.warning(f"Seaclear model not found at {settings.SEACLEAR_MODEL_PATH}")
            self.seaclear_model = None
            return

        loaded = self._load_detector_by_path(
            seaclear_path,
            forced_tag="seaclear",
            legacy_slot="seaclear",
        )
        if loaded is None:
            self.seaclear_model = None

    def _load_aquarium_model(self):
        """Load the configured Aquarium animals detector."""
        aquarium_path = self._resolve_model_path(
            settings.AQUARIUM_MODEL_PATH,
            fallback_globs=[
                "runs/dataa_yolov8/*/train/weights/best.pt",
                "runs/detect/*/weights/best.pt",
            ],
        )

        if aquarium_path is None:
            logger.warning(f"Aquarium model not found at {settings.AQUARIUM_MODEL_PATH}")
            self.aquarium_model = None
            return

        loaded = self._load_detector_by_path(
            aquarium_path,
            forced_tag="aquarium",
            legacy_slot="aquarium",
        )
        if loaded is None:
            self.aquarium_model = None

    def _auto_discover_models(self):
        """Auto-discover and load detector .pt files based on configured glob patterns."""
        patterns = self._parse_model_patterns()
        discovered: List[Path] = []
        seen: set[str] = set()

        for pattern in patterns:
            for candidate in Path(".").glob(pattern):
                if not candidate.is_file() or candidate.suffix.lower() != ".pt":
                    continue

                key = str(candidate.resolve())
                if key in seen:
                    continue

                seen.add(key)
                discovered.append(candidate)

        discovered.sort(key=lambda p: str(p))

        loaded_count = 0
        for candidate in discovered:
            before_count = len(self.detector_models)
            self._load_detector_by_path(candidate)
            if len(self.detector_models) > before_count:
                loaded_count += 1

        logger.info(f"Auto-discovered detector candidates: {len(discovered)}, newly loaded: {loaded_count}")

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance underwater image using U-Net model.

        Args:
            image: Input image in BGR format

        Returns:
            Enhanced image in BGR format
        """
        if self.enhancer_model is None:
            logger.warning("Enhancer model not available, returning original image")
            return image

        try:
            original_size = (image.shape[1], image.shape[0])

            input_tensor = preprocess_for_enhancement(image)
            input_tensor = input_tensor.to(self.device)

            with torch.no_grad():
                output_tensor = self.enhancer_model(input_tensor)

            enhanced_image = postprocess_enhanced_image(output_tensor, original_size)
            return enhanced_image

        except Exception as e:
            logger.error(f"Error during image enhancement: {e}")
            return image

    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = None,
        use_phi4: Optional[bool] = None,
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects by running all loaded YOLO detectors and fusing outputs.

        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for class-wise NMS fusion
            use_phi4: Enable Phi-4 verification for uncertain detections

        Returns:
            Tuple of (detections_list, annotated_image)
        """
        if confidence_threshold is None:
            confidence_threshold = settings.CONFIDENCE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = settings.NMS_THRESHOLD
        if use_phi4 is None:
            use_phi4 = settings.PHI4_ENABLED

        try:
            if not self.detector_models:
                logger.warning("No detector models loaded")
                return [], image.copy()

            all_detections: List[Dict[str, Any]] = []

            for entry in self.detector_models:
                model = entry["model"]
                model_tag = entry["tag"]
                logger.info(f"Running detector: {model_tag}")

                detections = self._run_single_model(
                    model,
                    image,
                    confidence_threshold,
                    nms_threshold,
                    model_tag=model_tag,
                )
                all_detections.extend(detections)

            fused_detections = self._apply_classwise_nms(all_detections, nms_threshold)

            if bool(use_phi4):
                fused_detections = self._verify_detections_with_phi4(image, fused_detections)

            annotated_image = self._draw_ensemble_detections(image, fused_detections)
            logger.info(
                f"Detections raw={len(all_detections)}, fused={len(fused_detections)}, "
                f"phi4={bool(use_phi4 and settings.PHI4_ENABLED)}"
            )

            return fused_detections, annotated_image

        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            raise

    def _run_single_model(
        self,
        model: YOLO,
        image: np.ndarray,
        confidence_threshold: float,
        nms_threshold: float,
        model_tag: str,
    ) -> List[Dict[str, Any]]:
        """Run inference for one model and normalize detections to a shared schema."""
        results = model.predict(
            image,
            conf=confidence_threshold,
            iou=nms_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            raw_class_name = str(model.names.get(class_id, f"Class_{class_id}"))
            detection = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": confidence,
                "class_id": class_id,
                "class_name": f"{model_tag}_{raw_class_name}",
                "raw_class_name": raw_class_name,
                "model": model_tag,
            }
            detections.append(detection)

        return detections

    def _apply_classwise_nms(self, detections: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
        """Fuse duplicate detections across models using class-wise NMS."""
        if not detections:
            return []

        sorted_detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept: List[Dict[str, Any]] = []

        for det in sorted_detections:
            det_class = str(det.get("raw_class_name", det.get("class_name", ""))).lower()
            should_suppress = False

            for kept_det in kept:
                kept_class = str(kept_det.get("raw_class_name", kept_det.get("class_name", ""))).lower()
                if det_class != kept_class:
                    continue

                iou = self._bbox_iou(det["bbox"], kept_det["bbox"])
                if iou >= iou_threshold:
                    should_suppress = True
                    break

            if not should_suppress:
                kept.append(det)

        return kept

    def _bbox_iou(self, box_a: List[int], box_b: List[int]) -> float:
        """Compute IoU between two [x1, y1, x2, y2] boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _draw_ensemble_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw fused detections with model-specific colors."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = self._get_model_color(det.get("model", "model"))

            class_text = det.get("raw_class_name", det.get("class_name", "object"))
            label = f"{class_text} {det['confidence']:.2f} [{det.get('model', 'model')}]"

            if det.get("phi4_checked"):
                phi4_text = "yes" if det.get("phi4_verified") else "no"
                label = f"{label} phi4:{phi4_text}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return annotated

    def _get_model_color(self, model_tag: str) -> Tuple[int, int, int]:
        """Generate a stable BGR color per model tag."""
        seed = abs(hash(model_tag))
        b = 80 + (seed % 176)
        g = 80 + ((seed // 176) % 176)
        r = 80 + ((seed // (176 * 176)) % 176)
        return int(b), int(g), int(r)

    def _should_run_phi4(self, confidence: float) -> bool:
        """Limit Phi-4 checks to uncertain detections only."""
        return (
            confidence >= settings.PHI4_VERIFY_MIN_CONFIDENCE
            and confidence <= settings.PHI4_VERIFY_MAX_CONFIDENCE
        )

    def _verify_detections_with_phi4(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run Phi-4 verification on uncertain detections and drop rejected boxes."""
        if not settings.PHI4_ENABLED or not detections:
            return detections

        candidates = [
            (idx, det)
            for idx, det in enumerate(detections)
            if self._should_run_phi4(float(det.get("confidence", 0.0)))
        ]

        if not candidates:
            return detections

        candidates.sort(key=lambda item: float(item[1].get("confidence", 0.0)))
        candidates = candidates[: max(1, int(settings.PHI4_MAX_CHECKS_PER_IMAGE))]

        keep_flags = [True] * len(detections)
        for idx, det in candidates:
            verdict = self._verify_single_detection_with_phi4(image, det)
            detections[idx]["phi4_checked"] = True
            detections[idx]["phi4_verified"] = bool(verdict)

            if not verdict:
                keep_flags[idx] = False

        filtered = [det for i, det in enumerate(detections) if keep_flags[i]]
        removed = len(detections) - len(filtered)
        if removed > 0:
            logger.info(f"Phi-4 removed {removed} uncertain detection(s)")

        return filtered

    def _verify_single_detection_with_phi4(self, image: np.ndarray, detection: Dict[str, Any]) -> bool:
        """Verify one detection crop with Phi-4 multimodal (fail-open on runtime errors)."""
        crop = self._crop_for_verification(image, detection["bbox"])
        if crop is None:
            return True

        ok, encoded = cv2.imencode(".jpg", crop)
        if not ok:
            return True

        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        provider = settings.PHI4_PROVIDER.strip().lower()
        try:
            if provider == "ollama":
                verdict = self._call_phi4_ollama(image_b64, detection)
            else:
                logger.warning(f"Unsupported PHI4_PROVIDER='{settings.PHI4_PROVIDER}', skipping verification")
                return True

            if verdict is None:
                return True
            return verdict

        except Exception as e:
            logger.warning(f"Phi-4 verification unavailable, keeping detection. Reason: {e}")
            return True

    def _crop_for_verification(self, image: np.ndarray, bbox: List[int]) -> Optional[np.ndarray]:
        """Crop a detection region with padding for multimodal verification."""
        x1, y1, x2, y2 = bbox
        h, w = image.shape[:2]
        pad = max(0, int(settings.PHI4_BOX_PADDING))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def _call_phi4_ollama(self, image_b64: str, detection: Dict[str, Any]) -> Optional[bool]:
        """Call local Ollama Phi-4 multimodal endpoint and parse YES/NO response."""
        class_name = detection.get("raw_class_name", detection.get("class_name", "object"))
        prompt = (
            "You are validating underwater object detections. "
            f"Candidate class: {class_name}. "
            "Reply with EXACTLY one token: YES or NO. "
            "Answer YES if the class appears in this crop, otherwise answer NO."
        )

        payload = {
            "model": settings.PHI4_MODEL_NAME,
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "options": {
                "temperature": 0,
            },
        }

        request_bytes = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            url=settings.PHI4_OLLAMA_URL,
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=float(settings.PHI4_TIMEOUT_SECONDS)) as response:
                response_data = response.read().decode("utf-8")
        except urllib_error.URLError as e:
            raise RuntimeError(f"Ollama request failed: {e}") from e

        parsed = json.loads(response_data)
        content = str(parsed.get("message", {}).get("content", "")).strip()
        return self._parse_phi4_verdict(content)

    def _parse_phi4_verdict(self, content: str) -> Optional[bool]:
        """Parse strict YES/NO style answers from Phi-4 output."""
        if not content:
            return None

        match = re.search(r"\b(YES|NO)\b", content.upper())
        if not match:
            return None
        return match.group(1) == "YES"

    def analyze_image(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = None,
        enhance: bool = True,
        use_phi4: Optional[bool] = None,
    ) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Complete image analysis pipeline: optional enhancement + ensemble detection.

        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            enhance: Whether to run image enhancement first
            use_phi4: Whether to apply Phi-4 verification on uncertain detections

        Returns:
            Tuple of (annotated_image, detections, metadata)
        """
        original_dims = {"width": image.shape[1], "height": image.shape[0]}

        logger.info("Starting image enhancement")
        enhanced_image = self.enhance_image(image) if enhance else image
        enhanced_dims = {"width": enhanced_image.shape[1], "height": enhanced_image.shape[0]}

        logger.info("Starting object detection")
        detections, annotated_image = self.detect_objects(
            enhanced_image,
            confidence_threshold,
            nms_threshold,
            use_phi4=use_phi4,
        )

        metadata = {
            "original_dimensions": original_dims,
            "enhanced_dimensions": enhanced_dims,
            "num_detections": len(detections),
            "enhancement_applied": bool(enhance and self.enhancer_model is not None),
            "detector_models_loaded": len(self.detector_models),
            "detector_model_tags": [entry["tag"] for entry in self.detector_models],
            "phi4_enabled": bool(settings.PHI4_ENABLED),
            "phi4_requested": bool(settings.PHI4_ENABLED if use_phi4 is None else use_phi4),
        }

        return annotated_image, detections, metadata

    def is_ready(self) -> Dict[str, Any]:
        """
        Check if models are loaded and ready.

        Returns:
            Dictionary with model status
        """
        status: Dict[str, Any] = {
            "enhancer": self.enhancer_model is not None,
            "detector": len(self.detector_models) > 0,
            "detector_count": len(self.detector_models),
            "seaclear": self.seaclear_model is not None,
            "aquarium": self.aquarium_model is not None,
            "phi4": bool(settings.PHI4_ENABLED),
        }
        return status

    def get_class_names(self) -> Dict[str, str]:
        """
        Get class names from all loaded models.

        Returns:
            Dictionary mapping prefixed class IDs to names
        """
        combined_classes: Dict[str, str] = {}

        for entry in self.detector_models:
            tag = entry["tag"]
            model = entry["model"]
            for class_id, class_name in model.names.items():
                combined_classes[f"{tag}_{class_id}"] = f"{tag}_{class_name}"

        return combined_classes
