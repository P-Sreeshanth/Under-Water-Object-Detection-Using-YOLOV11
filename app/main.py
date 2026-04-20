"""
FastAPI application for underwater image analysis.

This module implements the main API endpoints for image enhancement
and object detection using U-Net and YOLOv11 models.
"""

import time
import json
import base64
import queue
import asyncio
import threading
from dataclasses import dataclass
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form, WebSocket, WebSocketDisconnect, Query
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


@dataclass
class StreamStats:
    """Runtime counters for a single WebSocket stream session."""

    captured: int = 0
    processed: int = 0
    sent: int = 0
    dropped_capture: int = 0
    dropped_output: int = 0


class RealtimeVideoSession:
    """Producer-consumer streaming session with bounded queues.

    Architecture:
    - Producer thread: capture frames continuously via OpenCV
    - Consumer thread: run model inference, encode JPEG, push to output queue
    - Async sender: reads output queue and writes to WebSocket
    """

    def __init__(
        self,
        source,
        confidence_threshold: float,
        nms_threshold: float,
        enhance: bool,
        use_phi4: bool,
        jpeg_quality: int = 80,
        queue_size: int = 2,
    ):
        self.source = source
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.enhance = enhance
        self.use_phi4 = use_phi4
        self.jpeg_quality = int(jpeg_quality)
        self.stop_event = threading.Event()

        # Keep queues tiny to cap memory and prioritize newest frames.
        self.capture_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.output_queue: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self.stats = StreamStats()
        self.last_error: Optional[str] = None

        self.producer_thread: Optional[threading.Thread] = None
        self.consumer_thread: Optional[threading.Thread] = None

    @staticmethod
    def parse_source(source_text: str):
        source_text = (source_text or "0").strip()
        if source_text.isdigit():
            return int(source_text)
        return source_text

    @staticmethod
    def _drop_oldest_put(q: queue.Queue, item) -> bool:
        dropped = False
        if q.full():
            try:
                q.get_nowait()
                dropped = True
            except queue.Empty:
                pass
        q.put_nowait(item)
        return dropped

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.source)

        # Hardware acceleration note:
        # If your OpenCV build supports accelerated decode, configure it here.
        # For low-latency sources, forcing small capture buffers may help:
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.last_error = f"Failed to open video source: {self.source}"
            logger.error(self.last_error)
            self.stop_event.set()
            return

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # File source: loop back to start. Stream source: small backoff.
                    if isinstance(self.source, str) and Path(self.source).exists():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        time.sleep(0.01)
                    continue

                self.stats.captured += 1
                if self._drop_oldest_put(self.capture_queue, frame):
                    self.stats.dropped_capture += 1
        except Exception as e:
            self.last_error = f"Capture error: {e}"
            logger.error(self.last_error, exc_info=True)
            self.stop_event.set()
        finally:
            cap.release()

    def _inference_loop(self):
        global model_manager

        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.capture_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if model_manager is None:
                    self.last_error = "Models not loaded. Service unavailable."
                    self.stop_event.set()
                    break

                # YOLO acceleration note:
                # model_manager automatically uses CUDA when torch.cuda.is_available().
                annotated_image, detections, _ = model_manager.analyze_image(
                    frame,
                    confidence_threshold=self.confidence_threshold,
                    nms_threshold=self.nms_threshold,
                    enhance=self.enhance,
                    use_phi4=self.use_phi4,
                )

                self.stats.processed += 1
                ok, encoded = cv2.imencode(
                    ".jpg",
                    annotated_image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if not ok:
                    continue

                frame_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
                payload = json.dumps(
                    {
                        "type": "frame",
                        "image": frame_b64,
                        "detections": detections,
                        "stats": {
                            "captured": self.stats.captured,
                            "processed": self.stats.processed,
                            "sent": self.stats.sent,
                            "dropped_capture": self.stats.dropped_capture,
                            "dropped_output": self.stats.dropped_output,
                        },
                    }
                )

                if self._drop_oldest_put(self.output_queue, payload):
                    self.stats.dropped_output += 1
        except Exception as e:
            self.last_error = f"Inference error: {e}"
            logger.error(self.last_error, exc_info=True)
            self.stop_event.set()

    def start(self):
        self.producer_thread = threading.Thread(target=self._capture_loop, daemon=True, name="stream-producer")
        self.consumer_thread = threading.Thread(target=self._inference_loop, daemon=True, name="stream-consumer")
        self.producer_thread.start()
        self.consumer_thread.start()

    def stop(self):
        self.stop_event.set()
        for t in (self.producer_thread, self.consumer_thread):
            if t and t.is_alive():
                t.join(timeout=1.0)

    def get_payload_nowait(self) -> Optional[str]:
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None


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

# Mount static files directory once
if settings.STATIC_DIR and Path(settings.STATIC_DIR).exists():
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
    
    models_loaded = {"enhancer": False, "detector": False, "seaclear": False}
    
    if model_manager is not None:
        models_loaded = model_manager.is_ready()
    
    # Determine health status
    # System is healthy if at least seaclear model is loaded
    has_detection_model = (
        models_loaded.get("detector", False)
        or models_loaded.get("seaclear", False)
        or models_loaded.get("aquarium", False)
    )
    status = "healthy" if has_detection_model else "degraded"
    
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
        allowed_formats=settings.ALLOWED_EXTENSIONS,
        use_multi_model=settings.USE_MULTI_MODEL,
        auto_discover_yolo_models=settings.AUTO_DISCOVER_YOLO_MODELS,
        phi4_enabled=settings.PHI4_ENABLED,
        phi4_model_name=settings.PHI4_MODEL_NAME,
    )


@app.websocket("/ws/stream")
async def websocket_stream(
    websocket: WebSocket,
    source: str = Query(default="0", description="Camera index, local file path, or RTSP URL"),
    confidence_threshold: float = Query(default=0.25, ge=0.01, le=1.0),
    nms_threshold: float = Query(default=0.45, ge=0.01, le=1.0),
    enhance: bool = Query(default=False),
    use_phi4: bool = Query(default=False),
    jpeg_quality: int = Query(default=80, ge=50, le=95),
):
    """Real-time frame streaming endpoint with low-latency producer-consumer pipeline."""
    await websocket.accept()

    parsed_source = RealtimeVideoSession.parse_source(source)
    session = RealtimeVideoSession(
        source=parsed_source,
        confidence_threshold=confidence_threshold,
        nms_threshold=nms_threshold,
        enhance=enhance,
        use_phi4=use_phi4,
        jpeg_quality=jpeg_quality,
        queue_size=2,
    )
    session.start()

    async def sender_loop():
        while not session.stop_event.is_set():
            if session.last_error:
                await websocket.send_text(json.dumps({"type": "error", "message": session.last_error}))
                break

            payload = await asyncio.to_thread(session.get_payload_nowait)
            if payload is None:
                await asyncio.sleep(0.01)
                continue

            await websocket.send_text(payload)
            session.stats.sent += 1

    async def receiver_loop():
        while not session.stop_event.is_set():
            msg = await websocket.receive_text()
            if msg.strip().lower() in {"disconnect", "stop", "close"}:
                session.stop_event.set()
                break

    sender_task = None
    receiver_task = None
    try:
        sender_task = asyncio.create_task(sender_loop())
        receiver_task = asyncio.create_task(receiver_loop())

        done, pending = await asyncio.wait(
            {sender_task, receiver_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

        for task in done:
            err = task.exception()
            if err is not None:
                raise err

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket stream error: {e}", exc_info=True)
    finally:
        session.stop()
        if sender_task:
            sender_task.cancel()
        if receiver_task:
            receiver_task.cancel()


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
@limiter.limit(f"{settings.RATE_LIMIT_PER_MINUTE}/minute")
async def analyze_image(
    request: Request,
    file: UploadFile = File(..., description="Image file to analyze"),
    confidence_threshold: Optional[float] = Form(default=None),
    nms_threshold: Optional[float] = Form(default=None),
    enhance: Optional[bool] = Form(default=True),
    use_phi4: Optional[bool] = Form(default=None),
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
            nms_threshold=nms_threshold,
            enhance=bool(enhance),
            use_phi4=use_phi4,
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
                bbox=det['bbox'],
                model=det.get('model'),
                phi4_checked=det.get('phi4_checked'),
                phi4_verified=det.get('phi4_verified'),
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
    nms_threshold: Optional[float] = None,
    enhance: Optional[bool] = True,
    use_phi4: Optional[bool] = None,
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
                nms_threshold=nms_threshold,
                enhance=enhance,
                use_phi4=use_phi4,
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
