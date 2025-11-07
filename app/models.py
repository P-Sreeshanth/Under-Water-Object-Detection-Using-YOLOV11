"""
Model loading and inference logic for U-Net enhancement and YOLOv11 detection.
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import cv2
import numpy as np
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
    Manager class for loading and running inference with both models.
    
    This class handles:
    - Loading the U-Net enhancement model
    - Loading multiple YOLOv11 detection models (Seaclear + Aquarium)
    - Running inference on images with ensemble detection
    - Managing GPU/CPU device selection
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.enhancer_model = None
        self.seaclear_model = None
        self.aquarium_model = None
        self.class_names = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load enhancement and detection models."""
        try:
            # Load U-Net enhancement model
            self._load_enhancer()
            
            # Load detection models
            if settings.USE_MULTI_MODEL:
                logger.info("Loading multi-model detection (Seaclear + Aquarium)")
                self._load_seaclear_model()
                self._load_aquarium_model()
            else:
                # Legacy single model support
                self._load_seaclear_model()
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")
    
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
            # Initialize model architecture
            self.enhancer_model = UNet(in_channels=3, out_channels=3)
            
            # Load weights
            checkpoint = torch.load(enhancer_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.enhancer_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.enhancer_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.enhancer_model.load_state_dict(checkpoint)
            else:
                self.enhancer_model.load_state_dict(checkpoint)
            
            # Move to device and set to eval mode
            self.enhancer_model.to(self.device)
            self.enhancer_model.eval()
            
            logger.info(f"Enhancement model loaded from {enhancer_path}")
            
        except Exception as e:
            logger.error(f"Failed to load enhancement model: {e}")
            self.enhancer_model = None
    
    def _load_detector(self):
        """Load the YOLOv11 object detection model (legacy - for backward compatibility)."""
        self._load_seaclear_model()
    
    def _load_seaclear_model(self):
        """Load the Seaclear marine debris detection model."""
        seaclear_path = Path(settings.SEACLEAR_MODEL_PATH)
        
        if not seaclear_path.exists():
            logger.warning(f"Seaclear model not found at {seaclear_path}")
            self.seaclear_model = None
            return
        
        try:
            self.seaclear_model = YOLO(str(seaclear_path))
            
            if self.device.type == 'cuda':
                self.seaclear_model.to('cuda')
            
            # Extract class names with seaclear prefix
            seaclear_names = self.seaclear_model.names
            for class_id, class_name in seaclear_names.items():
                self.class_names[f"seaclear_{class_id}"] = f"seaclear_{class_name}"
            
            logger.info(f"Seaclear model loaded from {seaclear_path}")
            logger.info(f"Seaclear classes: {len(seaclear_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load Seaclear model: {e}")
            self.seaclear_model = None
    
    def _load_aquarium_model(self):
        """Load the Aquarium animals detection model."""
        aquarium_path = Path(settings.AQUARIUM_MODEL_PATH)
        
        if not aquarium_path.exists():
            logger.warning(f"Aquarium model not found at {aquarium_path}")
            self.aquarium_model = None
            return
        
        try:
            self.aquarium_model = YOLO(str(aquarium_path))
            
            if self.device.type == 'cuda':
                self.aquarium_model.to('cuda')
            
            # Extract class names with aquarium prefix
            aquarium_names = self.aquarium_model.names
            offset = len(self.class_names)
            for class_id, class_name in aquarium_names.items():
                self.class_names[f"aquarium_{class_id}"] = f"aquarium_{class_name}"
            
            logger.info(f"Aquarium model loaded from {aquarium_path}")
            logger.info(f"Aquarium classes: {len(aquarium_names)}")
            
        except Exception as e:
            logger.error(f"Failed to load Aquarium model: {e}")
            self.aquarium_model = None
    
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
            # Store original size
            original_size = (image.shape[1], image.shape[0])  # (width, height)
            
            # Preprocess
            input_tensor = preprocess_for_enhancement(image)
            input_tensor = input_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output_tensor = self.enhancer_model(input_tensor)
            
            # Postprocess
            enhanced_image = postprocess_enhanced_image(output_tensor, original_size)
            
            return enhanced_image
            
        except Exception as e:
            logger.error(f"Error during image enhancement: {e}")
            return image
    
    def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = None
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect objects in image using YOLOv11 model(s).
        
        If multi-model mode is enabled, runs both Seaclear and Aquarium models
        and combines the results.
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for Non-Maximum Suppression
            
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        # Use default thresholds if not provided
        if confidence_threshold is None:
            confidence_threshold = settings.CONFIDENCE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = settings.NMS_THRESHOLD
        
        try:
            all_detections = []
            annotated_image = image.copy()
            
            # Run Seaclear model
            if self.seaclear_model is not None:
                logger.info("Running Seaclear model detection")
                seaclear_detections, seaclear_annotated = self._run_single_model(
                    self.seaclear_model,
                    image,
                    confidence_threshold,
                    nms_threshold,
                    model_prefix="seaclear"
                )
                all_detections.extend(seaclear_detections)
                annotated_image = seaclear_annotated
            
            # Run Aquarium model (if multi-model mode)
            if settings.USE_MULTI_MODEL and self.aquarium_model is not None:
                logger.info("Running Aquarium model detection")
                aquarium_detections, aquarium_annotated = self._run_single_model(
                    self.aquarium_model,
                    image,
                    confidence_threshold,
                    nms_threshold,
                    model_prefix="aquarium"
                )
                
                # Combine detections
                all_detections.extend(aquarium_detections)
                
                # Merge annotations (draw aquarium detections on existing image)
                for det in aquarium_detections:
                    x1, y1, x2, y2 = det['bbox']
                    label = f"{det['class_name']} {det['confidence']:.2f}"
                    
                    # Draw box (different color for aquarium)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated_image, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            logger.info(f"Total detections: {len(all_detections)} (Seaclear + Aquarium)")
            return all_detections, annotated_image
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            raise
    
    def _run_single_model(
        self,
        model: YOLO,
        image: np.ndarray,
        confidence_threshold: float,
        nms_threshold: float,
        model_prefix: str = ""
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Run inference on a single YOLO model.
        
        Args:
            model: YOLO model instance
            image: Input image
            confidence_threshold: Confidence threshold
            nms_threshold: NMS threshold
            model_prefix: Prefix for class names
            
        Returns:
            Tuple of (detections, annotated_image)
        """
        # Run YOLOv11 inference
        results = model.predict(
            image,
            conf=confidence_threshold,
            iou=nms_threshold,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    class_name = model.names.get(class_id, f"Class_{class_id}")
                    if model_prefix:
                        class_name = f"{model_prefix}_{class_name}"
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name,
                        'model': model_prefix
                    }
                    
                    detections.append(detection)
            
            # Get annotated image
            annotated_image = result.plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        else:
            annotated_image = image.copy()
        
        return detections, annotated_image
    
    def analyze_image(
        self,
        image: np.ndarray,
        confidence_threshold: float = None,
        nms_threshold: float = None
    ) -> Tuple[np.ndarray, List[Dict], Dict]:
        """
        Complete image analysis pipeline: enhance and detect.
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for NMS
            
        Returns:
            Tuple of (annotated_image, detections, metadata)
        """
        # Store original dimensions
        original_dims = {"width": image.shape[1], "height": image.shape[0]}
        
        # Step 1: Enhance image
        logger.info("Starting image enhancement")
        enhanced_image = self.enhance_image(image)
        enhanced_dims = {"width": enhanced_image.shape[1], "height": enhanced_image.shape[0]}
        
        # Step 2: Detect objects
        logger.info("Starting object detection")
        detections, annotated_image = self.detect_objects(
            enhanced_image,
            confidence_threshold,
            nms_threshold
        )
        
        # Prepare metadata
        metadata = {
            "original_dimensions": original_dims,
            "enhanced_dimensions": enhanced_dims,
            "num_detections": len(detections),
            "enhancement_applied": self.enhancer_model is not None
        }
        
        return annotated_image, detections, metadata
    
    def is_ready(self) -> Dict[str, bool]:
        """
        Check if models are loaded and ready.
        
        Returns:
            Dictionary with model status
        """
        status = {
            "enhancer": self.enhancer_model is not None,
            "seaclear": self.seaclear_model is not None
        }
        
        if settings.USE_MULTI_MODEL:
            status["aquarium"] = self.aquarium_model is not None
        
        return status
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get combined class names from all loaded models.
        
        Returns:
            Dictionary mapping class IDs to names with model prefixes
        """
        combined_classes = {}
        
        # Add Seaclear classes
        if self.seaclear_model is not None:
            for class_id, class_name in self.seaclear_model.names.items():
                combined_classes[f"seaclear_{class_id}"] = f"seaclear_{class_name}"
        
        # Add Aquarium classes if multi-model
        if settings.USE_MULTI_MODEL and self.aquarium_model is not None:
            for class_id, class_name in self.aquarium_model.names.items():
                combined_classes[f"aquarium_{class_id}"] = f"aquarium_{class_name}"
        
        return combined_classes
