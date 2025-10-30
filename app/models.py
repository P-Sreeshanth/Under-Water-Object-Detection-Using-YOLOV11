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
    - Loading the YOLOv11 detection model
    - Running inference on images
    - Managing GPU/CPU device selection
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.enhancer_model = None
        self.detector_model = None
        self.class_names = {}
        
        self._load_models()
    
    def _load_models(self):
        """Load both enhancement and detection models."""
        try:
            # Load U-Net enhancement model
            self._load_enhancer()
            
            # Load YOLOv11 detection model
            self._load_detector()
            
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
        """Load the YOLOv11 object detection model."""
        detector_path = Path(settings.DETECTOR_MODEL_PATH)
        
        if not detector_path.exists():
            raise FileNotFoundError(
                f"Detector model not found at {detector_path}. "
                "Please ensure the YOLOv11 model file exists."
            )
        
        try:
            # Load YOLOv11 model using ultralytics
            self.detector_model = YOLO(str(detector_path))
            
            # Move to appropriate device
            if self.device.type == 'cuda':
                self.detector_model.to('cuda')
            
            # Extract class names
            self.class_names = self.detector_model.names
            
            logger.info(f"YOLOv11 detector loaded from {detector_path}")
            logger.info(f"Detected classes: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to load detector model: {e}")
            raise
    
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
        Detect objects in image using YOLOv11 model.
        
        Args:
            image: Input image in BGR format
            confidence_threshold: Minimum confidence for detections
            nms_threshold: IoU threshold for Non-Maximum Suppression
            
        Returns:
            Tuple of (detections_list, annotated_image)
        """
        if self.detector_model is None:
            raise RuntimeError("Detector model not loaded")
        
        # Use default thresholds if not provided
        if confidence_threshold is None:
            confidence_threshold = settings.CONFIDENCE_THRESHOLD
        if nms_threshold is None:
            nms_threshold = settings.NMS_THRESHOLD
        
        try:
            # Run YOLOv11 inference
            results = self.detector_model.predict(
                image,
                conf=confidence_threshold,
                iou=nms_threshold,
                verbose=False
            )
            
            # Process results
            detections = []
            
            if len(results) > 0:
                result = results[0]  # Get first result
                
                # Extract boxes
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, f"Class_{class_id}")
                        }
                        
                        detections.append(detection)
                
                # Get annotated image from YOLO
                annotated_image = result.plot()
                # Convert RGB to BGR for consistency
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            else:
                annotated_image = image.copy()
            
            logger.info(f"Detected {len(detections)} objects")
            return detections, annotated_image
            
        except Exception as e:
            logger.error(f"Error during object detection: {e}")
            raise
    
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
        return {
            "enhancer": self.enhancer_model is not None,
            "detector": self.detector_model is not None
        }
    
    def get_class_names(self) -> Dict[int, str]:
        """
        Get the class names dictionary.
        
        Returns:
            Dictionary mapping class IDs to names
        """
        return self.class_names
