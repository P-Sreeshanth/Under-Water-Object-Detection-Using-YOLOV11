"""
Utility functions for image processing, logging, and configuration management.
"""

import os
import uuid
import logging
from typing import Tuple, Optional
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Model paths
    ENHANCER_MODEL_PATH: str = "models/enhancer_model.pth"
    SEACLEAR_MODEL_PATH: str = "runs/seaclear/yolov11n_seaclear/weights/best.pt"
    AQUARIUM_MODEL_PATH: str = "runs/detect/aquarium_yolov11/weights/best.pt"  # Your aquarium trained model
    
    # Multi-model mode
    USE_MULTI_MODEL: bool = True  # Set to True to use both models
    
    # Detection settings
    CONFIDENCE_THRESHOLD: float = 0.25  # Lowered for better recall on Seaclear dataset
    NMS_THRESHOLD: float = 0.45
    
    # File upload settings
    MAX_FILE_SIZE_MB: float = 10.0
    ALLOWED_EXTENSIONS: list = ["jpg", "jpeg", "png", "bmp"]
    
    # Static files
    STATIC_DIR: str = "static"
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Image processing
    ENHANCE_INPUT_SIZE: Tuple[int, int] = (256, 256)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def setup_logging():
    """Configure structured logging for the application."""
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('underwater_analysis.log')
        ]
    )
    
    return logging.getLogger(__name__)


logger = setup_logging()


def generate_request_id() -> str:
    """
    Generate a unique request ID for tracking.
    
    Returns:
        str: UUID4 string for request tracking
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format.
    
    Returns:
        str: Current timestamp
    """
    return datetime.utcnow().isoformat() + "Z"


def validate_image_file(filename: str, file_size: int) -> Tuple[bool, Optional[str]]:
    """
    Validate uploaded image file.
    
    Args:
        filename: Name of the uploaded file
        file_size: Size of the file in bytes
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check file extension
    extension = filename.lower().split('.')[-1]
    if extension not in settings.ALLOWED_EXTENSIONS:
        return False, f"Invalid file format. Allowed formats: {', '.join(settings.ALLOWED_EXTENSIONS)}"
    
    # Check file size
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size > max_size_bytes:
        return False, f"File size exceeds maximum limit of {settings.MAX_FILE_SIZE_MB}MB"
    
    return True, None


def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load image from bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        numpy array in BGR format or None if failed
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        logger.error(f"Failed to load image from bytes: {e}")
        return None


def preprocess_for_enhancement(image: np.ndarray, target_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    Preprocess image for U-Net enhancement model.
    
    Args:
        image: Input image in BGR format
        target_size: Target size for resizing (height, width)
        
    Returns:
        Preprocessed tensor ready for model input
    """
    if target_size is None:
        target_size = settings.ENHANCE_INPUT_SIZE
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    resized = cv2.resize(image_rgb, target_size)
    
    # Convert to float and normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Convert to tensor and add batch dimension
    tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def postprocess_enhanced_image(tensor: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
    """
    Postprocess enhanced image tensor to BGR format.
    
    Args:
        tensor: Output tensor from enhancement model
        original_size: Original image size (width, height)
        
    Returns:
        Enhanced image in BGR format
    """
    # Remove batch dimension and move to CPU
    image = tensor.squeeze(0).cpu().detach().numpy()
    
    # Permute from CHW to HWC
    image = np.transpose(image, (1, 2, 0))
    
    # Denormalize from [0, 1] to [0, 255]
    image = (image * 255).clip(0, 255).astype(np.uint8)
    
    # Resize back to original size
    image = cv2.resize(image, original_size)
    
    # Convert RGB to BGR
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image_bgr


def draw_detections(
    image: np.ndarray,
    detections: list,
    class_names: dict,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """
    Draw bounding boxes and labels on image.
    
    Args:
        image: Input image in BGR format
        detections: List of detection results
        class_names: Dictionary mapping class IDs to names
        confidence_threshold: Minimum confidence to display
        
    Returns:
        Annotated image
    """
    annotated = image.copy()
    
    # Define colors for different classes (BGR format)
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    for detection in detections:
        if detection['confidence'] < confidence_threshold:
            continue
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = detection['bbox']
        class_id = detection['class_id']
        confidence = detection['confidence']
        
        # Get class name and color
        class_name = class_names.get(class_id, f"Class_{class_id}")
        color = colors[class_id % len(colors)]
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_name}: {confidence:.2f}"
        
        # Get label size for background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    return annotated


def add_legend(image: np.ndarray, class_counts: dict, class_names: dict) -> np.ndarray:
    """
    Add a legend to the annotated image showing detected classes.
    
    Args:
        image: Annotated image
        class_counts: Dictionary of class ID to count
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Image with legend
    """
    if not class_counts:
        return image
    
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Create legend area
    legend_height = 30 + (len(class_counts) * 25)
    legend = np.ones((legend_height, 250, 3), dtype=np.uint8) * 240
    
    # Add title
    cv2.putText(
        legend,
        "Detected Objects:",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )
    
    # Add each class
    y_offset = 45
    for class_id, count in sorted(class_counts.items()):
        class_name = class_names.get(class_id, f"Class_{class_id}")
        color = colors[class_id % len(colors)]
        
        # Draw color box
        cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
        
        # Draw text
        text = f"{class_name}: {count}"
        cv2.putText(
            legend,
            text,
            (40, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
        
        y_offset += 25
    
    # Append legend to image
    result = np.vstack([image, legend])
    
    return result


def save_image(image: np.ndarray, request_id: str) -> str:
    """
    Save annotated image to static directory.
    
    Args:
        image: Image to save in BGR format
        request_id: Unique request identifier
        
    Returns:
        Relative path to saved image
    """
    # Ensure static directory exists
    static_dir = Path(settings.STATIC_DIR)
    static_dir.mkdir(exist_ok=True)
    
    # Generate filename
    filename = f"annotated_{request_id}.jpg"
    filepath = static_dir / filename
    
    # Save image
    cv2.imwrite(str(filepath), image)
    
    # Return relative URL path
    return f"/static/{filename}"


def cleanup_old_images(max_age_hours: int = 24):
    """
    Remove old images from static directory.
    
    Args:
        max_age_hours: Maximum age of files to keep in hours
    """
    try:
        static_dir = Path(settings.STATIC_DIR)
        if not static_dir.exists():
            return
        
        current_time = datetime.now().timestamp()
        max_age_seconds = max_age_hours * 3600
        
        for file_path in static_dir.glob("annotated_*.jpg"):
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_seconds:
                file_path.unlink()
                logger.info(f"Cleaned up old image: {file_path.name}")
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


def get_image_dimensions(image: np.ndarray) -> dict:
    """
    Get image dimensions.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with width and height
    """
    height, width = image.shape[:2]
    return {"width": int(width), "height": int(height)}
