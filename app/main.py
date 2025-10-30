"""
FastAPI application for underwater image analysis.

This module implements the main API endpoints for image enhancement
and object detection using U-Net and YOLOv11 models.
"""

import time
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .models import ModelManager
from .schemas import (
    AnalysisResponse,
    ErrorResponse,
    HealthResponse,
    ConfigResponse,
    DetectionResult
)
from .utils import (
    logger,
    settings,
    generate_request_id,
    get_current_timestamp,
    validate_image_file,
    load_image_from_bytes,
    save_image,
    get_image_dimensions,
    cleanup_old_images
)

# Global model manager instance
model_manager: Optional[ModelManager] = None

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    
    This handles:
    - Loading models on startup
    - Cleaning up resources on shutdown
    """
    global model_manager
    
    # Startup
    logger.info("Starting Underwater Image Analysis API")
    logger.info(f"Version: 1.0.0")
    
    try:
        # Initialize model manager
        model_manager = ModelManager()
        logger.info("Models loaded successfully")
        
        # Clean up old images
        cleanup_old_images(max_age_hours=24)
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Underwater Image Analysis API")
    # Clean up resources if needed


# Initialize FastAPI app
app = FastAPI(
    title="Underwater Image Analysis API",
    description="Production-ready API for underwater image enhancement and object detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Mount static files for web interface
static_path = Path(__file__).parent.parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            message=exc.detail,
            error_type="HTTPException",
            details=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            message="Internal server error",
            error_type=type(exc).__name__,
            details=str(exc)
        ).dict()
    )


@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint - serve web interface"""
    from fastapi.responses import FileResponse
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    if static_path.exists():
        return FileResponse(static_path)
    return {
        "message": "Underwater Image Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "web_ui": "/",
            "docs": "/docs",
            "health": "/health",
            "analyze": "/analyze",
            "config": "/config"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status of the API and loaded models.
    """
    global model_manager
    
    models_loaded = {"enhancer": False, "detector": False}
    
    if model_manager is not None:
        models_loaded = model_manager.is_ready()
    
    status = "healthy" if all(models_loaded.values()) or models_loaded["detector"] else "degraded"
    
    return HealthResponse(
        status=status,
        version="1.0.0",
        models_loaded=models_loaded,
        timestamp=get_current_timestamp()
    )


@app.get("/config", response_model=ConfigResponse, tags=["Configuration"])
async def get_config():
    """
    Get current API configuration.
    
    Returns the current settings for file upload and detection.
    """
    return ConfigResponse(
        max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        confidence_threshold=settings.CONFIDENCE_THRESHOLD,
        nms_threshold=settings.NMS_THRESHOLD,
        allowed_formats=settings.ALLOWED_EXTENSIONS
    )


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None
):
    """
    Analyze underwater image: enhance and detect objects.
    
    This endpoint:
    1. Validates the uploaded image
    2. Enhances the image using U-Net model
    3. Detects objects using YOLOv11 model
    4. Returns annotated image and detection results
    
    Args:
        file: Uploaded image file (JPEG, PNG, BMP)
        confidence_threshold: Optional confidence threshold (default: 0.5)
        nms_threshold: Optional NMS IoU threshold (default: 0.45)
        
    Returns:
        AnalysisResponse with detections and annotated image URL
    """
    global model_manager
    
    # Generate request ID
    request_id = generate_request_id()
    start_time = time.time()
    
    logger.info(f"[{request_id}] Starting image analysis")
    
    try:
        # Check if models are loaded
        if model_manager is None:
            raise HTTPException(
                status_code=503,
                detail="Models not loaded. Service unavailable."
            )
        
        # Validate file
        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        
        is_valid, error_message = validate_image_file(file.filename, file_size)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_message)
        
        logger.info(f"[{request_id}] File validated: {file.filename} ({file_size} bytes)")
        
        # Load image
        image = load_image_from_bytes(contents)
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to decode image. Please upload a valid image file."
            )
        
        original_dims = get_image_dimensions(image)
        logger.info(f"[{request_id}] Image loaded: {original_dims}")
        
        # Run analysis
        annotated_image, detections, metadata = model_manager.analyze_image(
            image,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )
        
        # Save annotated image
        image_url = save_image(annotated_image, request_id)
        logger.info(f"[{request_id}] Annotated image saved: {image_url}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Format detections for response
        detection_results = [
            DetectionResult(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox']
            )
            for det in detections
        ]
        
        # Prepare response
        response = AnalysisResponse(
            success=True,
            message=f"Analysis completed successfully. Found {len(detections)} object(s).",
            request_id=request_id,
            detections=detection_results,
            annotated_image_url=image_url,
            processing_time=round(processing_time, 2),
            image_dimensions={
                "original": metadata["original_dimensions"],
                "enhanced": metadata["enhanced_dimensions"]
            }
        )
        
        logger.info(
            f"[{request_id}] Analysis completed in {processing_time:.2f}s. "
            f"Detected {len(detections)} objects."
        )
        
        return response
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"[{request_id}] Error during analysis: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/analyze-batch", tags=["Analysis"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def analyze_batch(
    request: Request,
    files: list[UploadFile] = File(..., description="Multiple image files to analyze"),
    confidence_threshold: Optional[float] = None,
    nms_threshold: Optional[float] = None
):
    """
    Analyze multiple underwater images in batch.
    
    This endpoint processes multiple images and returns results for each.
    
    Args:
        files: List of uploaded image files
        confidence_threshold: Optional confidence threshold
        nms_threshold: Optional NMS IoU threshold
        
    Returns:
        List of AnalysisResponse objects
    """
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )
    
    results = []
    
    for file in files:
        try:
            result = await analyze_image(
                request=request,
                file=file,
                confidence_threshold=confidence_threshold,
                nms_threshold=nms_threshold
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "success": False,
                "message": f"Failed to process {file.filename}",
                "error": str(e)
            })
    
    return {"results": results, "total": len(files), "processed": len(results)}


@app.get("/classes", tags=["Information"])
async def get_classes():
    """
    Get list of detectable object classes.
    
    Returns the classes that the YOLOv11 model can detect.
    """
    global model_manager
    
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded"
        )
    
    class_names = model_manager.get_class_names()
    
    return {
        "classes": class_names,
        "total_classes": len(class_names)
    }


@app.delete("/cleanup", tags=["Maintenance"])
async def cleanup_images(max_age_hours: int = 24):
    """
    Clean up old annotated images.
    
    Args:
        max_age_hours: Maximum age of images to keep (default: 24 hours)
        
    Returns:
        Status message
    """
    try:
        cleanup_old_images(max_age_hours=max_age_hours)
        return {
            "success": True,
            "message": f"Cleaned up images older than {max_age_hours} hours"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
