"""
FastAPI backend for Football Highlights Generator
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os

# ===== ADD THIS SECTION FOR LARGE FILE UPLOADS (800 MB) =====
from starlette.formparsers import MultiPartParser

# Increase upload limits to 800 MB (800 * 1024 * 1024 = 838,860,800 bytes)
MultiPartParser.max_file_size = 800 * 1024 * 1024  # 800 MB
MultiPartParser.max_part_size = 800 * 1024 * 1024  # 800 MB
# ============================================================

os.environ["IMAGEMAGICK_BINARY"] = r"E:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
from pathlib import Path
from typing import List, Optional
import json
import asyncio
from datetime import datetime
import logging
import traceback

from src.main_pipeline import FootballHighlightsPipeline
from config.config import config
from src.utils.logger import setup_logger

from pydantic import BaseModel
from typing import Optional

# Add this near the top of your file
class ProcessRequest(BaseModel):
    video_path: Optional[str] = None
    filename: Optional[str] = None
    output_dir: Optional[str] = None
    generate_highlights: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "20260208_012611_videoplayback.mp4",
                "generate_highlights": True
            }
        }

# Setup logging
logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Football Highlights Generator API",
    description="API for real-time football match highlights generation",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global pipeline instance
pipeline = None
processing_jobs = {}

# Job status tracker
class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global pipeline
    
    try:
        pipeline = FootballHighlightsPipeline()
        logger.info("🚀 Football Highlights Generator API started")
        logger.info(f"📁 Data directory: {config.PATH.RAW_VIDEO_DIR}")
        logger.info(f"🎯 Event keywords: {config.EVENT.KEYWORDS[:5]}...")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Football Highlights Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "api_docs": "/api/docs",
            "upload_video": "/api/upload",
            "process_video": "/api/process",
            "get_status": "/api/status/{job_id}",
            "download": "/api/download/{filename}"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_ready": pipeline is not None
    }

@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    metadata: Optional[str] = None
):
    """Upload a video file for processing - Supports up to 800 MB"""
    
    # 800 MB limit
    MAX_FILE_SIZE = 800 * 1024 * 1024  # 838,860,800 bytes
    
    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "File must be a video")
    
    # Create upload directory
    upload_dir = Path(config.PATH.RAW_VIDEO_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = Path(file.filename).stem
    # Clean filename for Windows
    safe_original = "".join(c for c in original_filename if c.isalnum() or c in '._- ')[:50]
    safe_filename = f"{timestamp}_{safe_original}.mp4"
    file_path = upload_dir / safe_filename
    
    try:
        # Read and write in chunks for large files
        file_size = 0
        chunk_size = 5 * 1024 * 1024  # 5 MB chunks for better performance
        
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                
                # Check size limit
                if file_size > MAX_FILE_SIZE:
                    buffer.close()
                    file_path.unlink()  # Delete partial file
                    raise HTTPException(
                        413, 
                        f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB. "
                        f"Your file is {file_size // (1024*1024)} MB"
                    )
                
                buffer.write(chunk)
        
        # Parse metadata if provided
        file_metadata = {}
        if metadata:
            try:
                file_metadata = json.loads(metadata)
            except:
                file_metadata = {"raw_metadata": metadata}
        
        # Log successful upload
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ Video uploaded: {safe_filename} ({file_size_mb:.2f} MB)")
        
        # Create response
        response = {
            "status": "success",
            "filename": safe_filename,
            "original_filename": file.filename,
            "path": str(file_path),
            "size_bytes": file_path.stat().st_size,
            "size_mb": round(file_size_mb, 2),
            "metadata": file_metadata,
            "message": f"Video uploaded successfully ({file_size_mb:.1f} MB)"
        }
        
        return JSONResponse(content=response, status_code=201)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        # Clean up partial file
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(500, f"Error saving file: {str(e)}")

@app.post("/api/process")
async def process_video(
    background_tasks: BackgroundTasks,
    request: ProcessRequest  # Use Pydantic model for JSON body
):
    """Start processing a video for highlights generation"""
    
    # Extract parameters from request model
    video_path = request.video_path
    filename = request.filename
    output_dir = request.output_dir
    generate_highlights = request.generate_highlights
    
    # DEBUG: Log what we receive
    logger.info(f"🔍 /api/process endpoint called with:")
    logger.info(f"   video_path = '{video_path}'")
    logger.info(f"   filename = '{filename}'")
    logger.info(f"   filename type = {type(filename)}")
    logger.info(f"   filename is None = {filename is None}")
    logger.info(f"   filename bool = {bool(filename)}")
    
    # Determine video path - Check for empty strings too
    if video_path and str(video_path).strip():
        video_path = Path(video_path)
        logger.info(f"📁 Using video_path: {video_path}")
    elif filename and str(filename).strip():
        # Construct full path
        raw_video_dir = Path(config.PATH.RAW_VIDEO_DIR)
        video_path = raw_video_dir / filename
        logger.info(f"📁 Constructed video_path from filename: {video_path}")
        logger.info(f"📁 Raw video directory exists: {raw_video_dir.exists()}")
        logger.info(f"📁 Full path exists: {video_path.exists()}")
    else:
        error_msg = f"Either video_path or filename must be provided. Got video_path='{video_path}', filename='{filename}'"
        logger.error(error_msg)
        raise HTTPException(400, error_msg)
    
    # Check if video exists
    if not video_path.exists():
        # List files in directory for debugging
        raw_video_dir = Path(config.PATH.RAW_VIDEO_DIR)
        files_in_dir = list(raw_video_dir.glob("*"))
        error_msg = f"Video file not found: {video_path}. Files in directory: {[f.name for f in files_in_dir]}"
        logger.error(error_msg)
        raise HTTPException(404, error_msg)
    
    logger.info(f"✅ Video file found: {video_path}")
    
    # Generate job ID
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{video_path.stem}"
    
    # Create output directory
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path(config.PATH.HIGHLIGHTS_DIR) / job_id
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize job status
    processing_jobs[job_id] = {
        "status": JobStatus.PENDING,
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "start_time": None,
        "end_time": None,
        "progress": 0,
        "error": None,
        "result": None,
        "status_message": "Job created"
    }
    
    # Start processing in background
    background_tasks.add_task(
        process_video_background,
        job_id,
        str(video_path),
        str(output_dir),
        generate_highlights
    )
    
    response = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "message": "Processing started in background"
    }
    
    logger.info(f"✅ Started processing job {job_id} for {video_path}")
    logger.info(f"✅ Response: {response}")
    
    return response

def run_pipeline_with_progress(process_video_func, video_path: str, output_dir: str, job: dict):
    """Run pipeline with progress tracking"""
    try:
        # Update progress at different stages
        job["progress"] = 10
        job["status_message"] = "Initializing video processing..."
        
        # Run the pipeline
        result = process_video_func(video_path, output_dir)
        
        # Update progress for completion
        job["progress"] = 90
        job["status_message"] = "Finalizing results..."
        
        return result
        
    except Exception as e:
        logger.error(f"Pipeline execution error: {e}")
        logger.error(traceback.format_exc())
        raise

async def process_video_background(job_id: str, video_path: str, output_dir: str, generate_highlights: bool):
    """Background task for video processing"""
    
    job = processing_jobs[job_id]
    job["status"] = JobStatus.PROCESSING
    job["start_time"] = datetime.now().isoformat()
    job["progress"] = 5
    job["status_message"] = "Starting processing..."
    
    try:
        # Check if pipeline is available
        if pipeline is None:
            raise Exception("Pipeline not initialized")
        
        # Process video
        logger.info(f"Starting processing job {job_id}: {video_path}")
        
        # Update progress
        job["progress"] = 10
        job["status_message"] = "Processing video clips and audio..."
        
        # Run pipeline with progress tracking
        result = await asyncio.to_thread(
            run_pipeline_with_progress,
            pipeline.process_video,
            video_path,
            output_dir,
            job
        )
        
        # Update job status
        job["status"] = JobStatus.COMPLETED
        job["end_time"] = datetime.now().isoformat()
        job["progress"] = 100
        job["status_message"] = "Processing completed successfully!"
        job["result"] = result
        
        logger.info(f"✅ Completed processing job {job_id}")
        
    except Exception as e:
        # Update job status with error
        job["status"] = JobStatus.FAILED
        job["end_time"] = datetime.now().isoformat()
        job["error"] = str(e)
        job["status_message"] = f"Failed: {str(e)[:100]}"
        
        logger.error(f"❌ Failed processing job {job_id}: {e}")
        logger.error(traceback.format_exc())

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a processing job"""
    
    if job_id not in processing_jobs:
        raise HTTPException(404, f"Job not found: {job_id}")
    
    job = processing_jobs[job_id]
    
    # Calculate processing time if completed
    processing_time = None
    if job["start_time"] and job["end_time"]:
        start = datetime.fromisoformat(job["start_time"])
        end = datetime.fromisoformat(job["end_time"])
        processing_time = (end - start).total_seconds()
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "video_path": job["video_path"],
        "output_dir": job["output_dir"],
        "progress": job["progress"],
        "start_time": job["start_time"],
        "end_time": job["end_time"],
        "processing_time": processing_time,
        "error": job["error"],
        "status_message": job.get("status_message", "")
    }
    
    # Add result if available
    if job["status"] == JobStatus.COMPLETED and job["result"]:
        response["result"] = job["result"]
    
    return response

@app.get("/api/jobs")
async def list_jobs(limit: int = 10, status: Optional[str] = None):
    """List all processing jobs"""
    
    jobs = list(processing_jobs.items())
    
    # Filter by status if specified
    if status:
        jobs = [(jid, job) for jid, job in jobs if job["status"] == status]
    
    # Sort by start time (most recent first)
    jobs.sort(key=lambda x: x[1].get("start_time") or "", reverse=True)
    
    # Limit results
    jobs = jobs[:limit]
    
    response = []
    for job_id, job in jobs:
        response.append({
            "job_id": job_id,
            "status": job["status"],
            "video_path": job["video_path"],
            "progress": job["progress"],
            "start_time": job["start_time"],
            "status_message": job.get("status_message", "")
        })
    
    return {
        "total_jobs": len(processing_jobs),
        "jobs": response
    }

@app.get("/api/download/{filename:path}")
async def download_file(filename: str):
    """Download a file from the highlights directory"""
    
    # Security check: prevent directory traversal
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "Invalid filename")
    
    # Look for file in highlights directory
    highlights_dir = Path(config.PATH.HIGHLIGHTS_DIR)
    file_path = None
    
    # Check direct path
    potential_path = highlights_dir / filename
    if potential_path.exists() and potential_path.is_file():
        file_path = potential_path
    else:
        # Search recursively
        for file in highlights_dir.rglob(filename):
            if file.is_file():
                file_path = file
                break
    
    if not file_path or not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    
    return FileResponse(
        path=file_path,
        filename=file_path.name,
        media_type="application/octet-stream"
    )

@app.get("/api/videos")
async def list_videos(limit: int = 20):
    """List available videos in the raw videos directory"""
    
    videos_dir = Path(config.PATH.RAW_VIDEO_DIR)
    
    if not videos_dir.exists():
        return {"videos": []}
    
    videos = []
    for video_file in videos_dir.glob("*.mp4"):
        videos.append({
            "filename": video_file.name,
            "path": str(video_file),
            "size": video_file.stat().st_size,
            "size_mb": round(video_file.stat().st_size / (1024 * 1024), 2),
            "modified": video_file.stat().st_mtime
        })
    
    # Sort by modified time (newest first)
    videos.sort(key=lambda x: x["modified"], reverse=True)
    
    return {
        "total_videos": len(videos),
        "videos": videos[:limit]
    }

@app.get("/api/highlights")
async def list_highlights(limit: int = 10):
    """List generated highlights"""
    
    highlights_dir = Path(config.PATH.HIGHLIGHTS_DIR)
    
    if not highlights_dir.exists():
        return {"highlights": []}
    
    highlights = []
    for job_dir in highlights_dir.iterdir():
        if job_dir.is_dir():
            # Look for highlights file
            highlights_files = list(job_dir.glob("*.mp4"))
            annotation_files = list(job_dir.glob("*.json"))
            
            if highlights_files:
                highlight_info = {
                    "job_id": job_dir.name,
                    "highlight_file": highlights_files[0].name,
                    "annotation_file": annotation_files[0].name if annotation_files else None,
                    "path": str(job_dir),
                    "created": job_dir.stat().st_ctime
                }
                highlights.append(highlight_info)
    
    # Sort by creation time (newest first)
    highlights.sort(key=lambda x: x["created"], reverse=True)
    
    return {
        "total_highlights": len(highlights),
        "highlights": highlights[:limit]
    }

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    
    return {
        "video_config": config.VIDEO.__dict__,
        "audio_config": config.AUDIO.__dict__,
        "model_config": config.MODEL.__dict__,
        "event_config": config.EVENT.__dict__,
        "path_config": {
            k: str(v) for k, v in config.PATH.__dict__.items() if isinstance(v, Path)
        }
    }

@app.post("/api/test")
async def test_processing():
    """Test endpoint for processing"""
    
    # Find a sample video
    sample_video = Path(config.PATH.RAW_VIDEO_DIR) / "sample_match.mp4"
    
    if not sample_video.exists():
        return {
            "status": "error",
            "message": "Sample video not found. Please upload a video first."
        }
    
    # Start processing
    response = await process_video(
        BackgroundTasks(),
        video_path=str(sample_video),
        output_dir=str(Path(config.PATH.HIGHLIGHTS_DIR) / "test_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )
    
    return {
        "status": "success",
        "message": "Test processing started",
        "job_id": response["job_id"]
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config.APP.API_HOST,
        port=config.APP.API_PORT,
        reload=config.APP.API_RELOAD,
        workers=config.APP.API_WORKERS,
        timeout_keep_alive=600,  # 10 minutes for long uploads/processing
        limit_max_requests=1000
    )