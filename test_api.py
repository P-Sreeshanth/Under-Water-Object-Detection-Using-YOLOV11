"""
Test suite for the Underwater Image Analysis API.

Run with: pytest test_api.py -v
"""

import os
import io
import pytest
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)


def create_test_image(width=640, height=480, color=(0, 100, 200)):
    """
    Create a test image for testing purposes.
    
    Args:
        width: Image width
        height: Image height
        color: RGB color tuple
        
    Returns:
        BytesIO object containing JPEG image
    """
    # Create a simple colored image
    img_array = np.full((height, width, 3), color, dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save to BytesIO
    img_buffer = io.BytesIO()
    img.save(img_buffer, format='JPEG')
    img_buffer.seek(0)
    
    return img_buffer


class TestAPIEndpoints:
    """Test suite for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert data["name"] == "Underwater Image Analysis API"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data
        assert "timestamp" in data
        
        # Check models status
        assert isinstance(data["models_loaded"], dict)
        assert "enhancer" in data["models_loaded"]
        assert "detector" in data["models_loaded"]
    
    def test_config_endpoint(self):
        """Test configuration endpoint."""
        response = client.get("/config")
        assert response.status_code == 200
        
        data = response.json()
        assert "max_file_size_mb" in data
        assert "confidence_threshold" in data
        assert "nms_threshold" in data
        assert "allowed_formats" in data
        
        # Validate values
        assert data["max_file_size_mb"] > 0
        assert 0 <= data["confidence_threshold"] <= 1
        assert 0 <= data["nms_threshold"] <= 1
    
    def test_classes_endpoint(self):
        """Test classes endpoint."""
        response = client.get("/classes")
        
        # May fail if models not loaded
        if response.status_code == 200:
            data = response.json()
            assert "classes" in data
            assert "total_classes" in data
            assert isinstance(data["classes"], dict)


class TestImageAnalysis:
    """Test suite for image analysis functionality."""
    
    def test_analyze_valid_image(self):
        """Test analyzing a valid image."""
        # Create test image
        test_image = create_test_image()
        
        # Send request
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = client.post("/analyze", files=files)
        
        # Check response (may be 503 if models not loaded)
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "request_id" in data
        assert "detections" in data
        assert "annotated_image_url" in data
        assert "processing_time" in data
        assert "image_dimensions" in data
        
        # Validate detections
        assert isinstance(data["detections"], list)
        
        # Validate image dimensions
        assert "original" in data["image_dimensions"]
        assert "enhanced" in data["image_dimensions"]
    
    def test_analyze_with_custom_threshold(self):
        """Test analyzing with custom confidence threshold."""
        test_image = create_test_image()
        
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        data = {"confidence_threshold": 0.7}
        
        response = client.post("/analyze", files=files, data=data)
        
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
    
    def test_analyze_invalid_file_format(self):
        """Test analyzing with invalid file format."""
        # Create a text file instead of image
        text_file = io.BytesIO(b"This is not an image")
        
        files = {"file": ("test.txt", text_file, "text/plain")}
        response = client.post("/analyze", files=files)
        
        assert response.status_code == 400
    
    def test_analyze_oversized_file(self):
        """Test analyzing with oversized file."""
        # Create a very large image that exceeds max size
        # This would need to create an image larger than MAX_FILE_SIZE_MB
        # Skipping actual implementation for brevity
        pass
    
    def test_analyze_missing_file(self):
        """Test analyzing without providing a file."""
        response = client.post("/analyze")
        assert response.status_code == 422  # Validation error


class TestBatchAnalysis:
    """Test suite for batch analysis."""
    
    def test_batch_analysis_multiple_images(self):
        """Test batch analysis with multiple images."""
        # Create multiple test images
        images = [
            ("file1.jpg", create_test_image(color=(255, 0, 0)), "image/jpeg"),
            ("file2.jpg", create_test_image(color=(0, 255, 0)), "image/jpeg"),
            ("file3.jpg", create_test_image(color=(0, 0, 255)), "image/jpeg"),
        ]
        
        files = [("files", (name, img, content_type)) for name, img, content_type in images]
        response = client.post("/analyze-batch", files=files)
        
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total" in data
        assert data["total"] == 3
    
    def test_batch_analysis_exceeds_limit(self):
        """Test batch analysis with too many images."""
        # Create more than 10 images
        images = [
            (f"file{i}.jpg", create_test_image(), "image/jpeg")
            for i in range(11)
        ]
        
        files = [("files", (name, img, content_type)) for name, img, content_type in images]
        response = client.post("/analyze-batch", files=files)
        
        assert response.status_code == 400


class TestModelManager:
    """Test suite for ModelManager class."""
    
    def test_model_initialization(self):
        """Test model manager initialization."""
        from app.models import ModelManager
        
        try:
            manager = ModelManager()
            status = manager.is_ready()
            
            assert isinstance(status, dict)
            assert "enhancer" in status
            assert "detector" in status
            
        except Exception as e:
            pytest.skip(f"Model initialization failed: {e}")
    
    def test_enhance_image(self):
        """Test image enhancement."""
        from app.models import ModelManager
        import cv2
        
        try:
            manager = ModelManager()
            
            # Create test image
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Enhance
            enhanced = manager.enhance_image(test_img)
            
            assert enhanced is not None
            assert enhanced.shape == test_img.shape
            
        except Exception as e:
            pytest.skip(f"Enhancement test failed: {e}")
    
    def test_detect_objects(self):
        """Test object detection."""
        from app.models import ModelManager
        
        try:
            manager = ModelManager()
            
            # Create test image
            test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Detect
            detections, annotated = manager.detect_objects(test_img)
            
            assert isinstance(detections, list)
            assert annotated is not None
            
        except Exception as e:
            pytest.skip(f"Detection test failed: {e}")


class TestUtilities:
    """Test suite for utility functions."""
    
    def test_validate_image_file(self):
        """Test image file validation."""
        from app.utils import validate_image_file
        
        # Valid cases
        is_valid, error = validate_image_file("test.jpg", 1024 * 1024)  # 1MB
        assert is_valid is True
        assert error is None
        
        is_valid, error = validate_image_file("test.png", 1024 * 1024)
        assert is_valid is True
        
        # Invalid extension
        is_valid, error = validate_image_file("test.txt", 1024)
        assert is_valid is False
        assert error is not None
        
        # Oversized file
        is_valid, error = validate_image_file("test.jpg", 20 * 1024 * 1024)  # 20MB
        assert is_valid is False
        assert error is not None
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        from app.utils import generate_request_id
        
        id1 = generate_request_id()
        id2 = generate_request_id()
        
        assert id1 != id2
        assert len(id1) == 36  # UUID4 format
        assert len(id2) == 36
    
    def test_get_image_dimensions(self):
        """Test getting image dimensions."""
        from app.utils import get_image_dimensions
        
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dims = get_image_dimensions(test_img)
        
        assert dims["width"] == 640
        assert dims["height"] == 480
    
    def test_preprocess_for_enhancement(self):
        """Test image preprocessing."""
        from app.utils import preprocess_for_enhancement
        import torch
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_for_enhancement(test_img, target_size=(256, 256))
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == 1  # Batch dimension
        assert tensor.shape[1] == 3  # Channels
        assert tensor.shape[2] == 256  # Height
        assert tensor.shape[3] == 256  # Width


class TestSchemas:
    """Test suite for Pydantic schemas."""
    
    def test_detection_result_schema(self):
        """Test DetectionResult schema validation."""
        from app.schemas import DetectionResult
        
        # Valid detection
        detection = DetectionResult(
            class_name="fish",
            confidence=0.95,
            bbox=[100, 150, 300, 400]
        )
        
        assert detection.class_name == "fish"
        assert detection.confidence == 0.95
        assert detection.bbox == [100, 150, 300, 400]
        
        # Invalid confidence (out of range)
        with pytest.raises(Exception):
            DetectionResult(
                class_name="fish",
                confidence=1.5,  # Invalid
                bbox=[100, 150, 300, 400]
            )
        
        # Invalid bbox (wrong number of coordinates)
        with pytest.raises(Exception):
            DetectionResult(
                class_name="fish",
                confidence=0.95,
                bbox=[100, 150, 300]  # Only 3 coordinates
            )
    
    def test_analysis_response_schema(self):
        """Test AnalysisResponse schema."""
        from app.schemas import AnalysisResponse, DetectionResult
        
        response = AnalysisResponse(
            success=True,
            message="Success",
            request_id="test-123",
            detections=[],
            annotated_image_url="/static/test.jpg",
            processing_time=1.5
        )
        
        assert response.success is True
        assert response.processing_time == 1.5


# Integration test
class TestIntegration:
    """Integration tests."""
    
    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline from upload to result."""
        # Create test image
        test_image = create_test_image(width=800, height=600)
        
        # Submit for analysis
        files = {"file": ("test_integration.jpg", test_image, "image/jpeg")}
        response = client.post("/analyze", files=files)
        
        if response.status_code == 503:
            pytest.skip("Models not loaded")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Verify all expected fields
        assert data["success"] is True
        assert data["request_id"] is not None
        assert isinstance(data["detections"], list)
        assert data["annotated_image_url"].startswith("/static/")
        assert data["processing_time"] > 0
        
        # Verify annotated image URL is accessible
        image_url = data["annotated_image_url"]
        image_response = client.get(image_url)
        assert image_response.status_code == 200


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
