"""
Pydantic schemas for request and response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class DetectionResult(BaseModel):
    """Schema for a single detection result."""
    
    class_name: str = Field(..., description="Name of the detected class")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: List[int] = Field(..., description="Bounding box coordinates [x_min, y_min, x_max, y_max]")
    
    @validator('bbox')
    def validate_bbox(cls, v):
        """Validate bounding box has 4 coordinates."""
        if len(v) != 4:
            raise ValueError('Bounding box must have 4 coordinates [x_min, y_min, x_max, y_max]')
        if v[0] >= v[2] or v[1] >= v[3]:
            raise ValueError('Invalid bounding box: max values must be greater than min values')
        return v

    class Config:
        schema_extra = {
            "example": {
                "class_name": "fish",
                "confidence": 0.95,
                "bbox": [100, 150, 300, 400]
            }
        }


class AnalysisResponse(BaseModel):
    """Schema for the complete analysis response."""
    
    success: bool = Field(..., description="Whether the analysis was successful")
    message: str = Field(..., description="Human-readable status message")
    request_id: str = Field(..., description="Unique request identifier for tracking")
    detections: List[DetectionResult] = Field(default_factory=list, description="List of detected objects")
    annotated_image_url: str = Field(..., description="URL path to the annotated image")
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    image_dimensions: Optional[dict] = Field(None, description="Original and enhanced image dimensions")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Analysis completed successfully",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "detections": [
                    {
                        "class_name": "fish",
                        "confidence": 0.95,
                        "bbox": [100, 150, 300, 400]
                    }
                ],
                "annotated_image_url": "/static/annotated_550e8400-e29b-41d4-a716-446655440000.jpg",
                "processing_time": 2.34,
                "image_dimensions": {
                    "original": {"width": 1920, "height": 1080},
                    "enhanced": {"width": 1920, "height": 1080}
                }
            }
        }


class ErrorResponse(BaseModel):
    """Schema for error responses."""
    
    success: bool = Field(default=False, description="Always False for errors")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request identifier if available")
    error_type: str = Field(..., description="Type of error")
    details: Optional[str] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "message": "Invalid image format",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "error_type": "ValidationError",
                "details": "Only JPEG, PNG, and BMP formats are supported"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    models_loaded: dict = Field(..., description="Status of loaded models")
    timestamp: str = Field(..., description="Current server timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "models_loaded": {
                    "enhancer": True,
                    "detector": True
                },
                "timestamp": "2025-10-30T12:00:00Z"
            }
        }


class ConfigResponse(BaseModel):
    """Schema for configuration response."""
    
    max_file_size_mb: float = Field(..., description="Maximum allowed file size in MB")
    confidence_threshold: float = Field(..., description="Detection confidence threshold")
    nms_threshold: float = Field(..., description="Non-Maximum Suppression IoU threshold")
    allowed_formats: List[str] = Field(..., description="Allowed image formats")
    
    class Config:
        schema_extra = {
            "example": {
                "max_file_size_mb": 10.0,
                "confidence_threshold": 0.5,
                "nms_threshold": 0.45,
                "allowed_formats": ["jpg", "jpeg", "png", "bmp"]
            }
        }
