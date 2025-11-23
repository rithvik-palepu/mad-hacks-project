"""
EvidenceCheck MVP - FastAPI REST API

REST API for video-text consistency checking.
Processes videos and incident reports to determine consistency.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import tempfile
import os
import cv2
import base64
from io import BytesIO
import re

from video_keyframe_processor import VideoKeyframeProcessor
from text_parser import ReportProcessor
from video_analyzer import analyze_video  # Keep for backward compatibility
from scoring import score_consistency

# Initialize FastAPI app
app = FastAPI(
    title="EvidenceCheck MVP API",
    description="Video ↔ Text Consistency Checker REST API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
video_processor = VideoKeyframeProcessor(
    collision_model_path="yolov8n.pt",
    keyframe_interval=5,
    collision_threshold=0.5
)

report_processor = ReportProcessor(use_handwriting_model=False)


# Pydantic models for request/response
class AnalysisRequest(BaseModel):
    text_description: str
    use_keyframe_processor: bool = True  # Use new VideoKeyframeProcessor by default


class AnalysisResponse(BaseModel):
    success: bool
    consistency_score: int
    details: list
    video_analysis: Dict[str, Any]
    text_claims: Dict[str, Any]
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    message: str


def extract_claims_from_text(text: str) -> Dict[str, Any]:
    """
    Extract claims (people, cars, weapons) from text description.
    This is a compatibility function that works with the old extract_claims API
    but uses the new ReportProcessor internally.
    
    Args:
        text: Text description to parse
        
    Returns:
        Dictionary with people, cars, weapon_present (each can be None)
    """
    claims = {
        "people": None,
        "cars": None,
        "weapon_present": None
    }
    
    text_lower = text.lower()
    
    # Extract people count
    # Patterns: "three people", "2 persons", "one person", etc.
    people_patterns = [
        r'\b(?:one|single|a|an)\s+(?:person|people|individual|man|woman|pedestrian)\b',
        r'\b(two|three|four|five|six|seven|eight|nine|ten)\s+(?:people|persons|individuals|men|women|pedestrians)\b',
        r'\b(\d+)\s+(?:people|persons|individuals|men|women|pedestrians)\b'
    ]
    
    for pattern in people_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if match.group(0).startswith(('one', 'single', 'a ', 'an ')):
                claims["people"] = 1
                break
            else:
                # Try to extract number
                number_words = {
                    'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                if len(match.groups()) > 0:
                    num_str = match.group(1)
                    if num_str.isdigit():
                        claims["people"] = int(num_str)
                    elif num_str in number_words:
                        claims["people"] = number_words[num_str]
                    break
    
    # Extract cars/vehicles count
    # Patterns: "two cars", "1 vehicle", "three vehicles", etc.
    vehicle_patterns = [
        r'\b(?:one|single|a|an)\s+(?:car|vehicle|auto|sedan|suv|truck|van)\b',
        r'\b(two|three|four|five|six|seven|eight|nine|ten)\s+(?:cars|vehicles|autos|sedans|suvs|trucks|vans)\b',
        r'\b(\d+)\s+(?:cars|vehicles|autos|sedans|suvs|trucks|vans)\b'
    ]
    
    for pattern in vehicle_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if match.group(0).startswith(('one', 'single', 'a ', 'an ')):
                claims["cars"] = 1
                break
            else:
                number_words = {
                    'two': 2, 'three': 3, 'four': 4, 'five': 5,
                    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                if len(match.groups()) > 0:
                    num_str = match.group(1)
                    if num_str.isdigit():
                        claims["cars"] = int(num_str)
                    elif num_str in number_words:
                        claims["cars"] = number_words[num_str]
                    break
    
    # Extract weapon presence
    # Patterns: "gun present", "no weapon", "weapon visible", etc.
    weapon_present_patterns = [
        r'\b(?:gun|knife|weapon|firearm|pistol|rifle)\s+(?:present|visible|seen|detected|shown)\b',
        r'\b(?:a|an|the)\s+(?:gun|knife|weapon|firearm|pistol|rifle)\b'
    ]
    
    weapon_absent_patterns = [
        r'\bno\s+(?:gun|knife|weapon|firearm|pistol|rifle)\b',
        r'\b(?:no|without)\s+weapons?\b'
    ]
    
    # Check for weapon presence
    for pattern in weapon_present_patterns:
        if re.search(pattern, text_lower):
            claims["weapon_present"] = True
            break
    
    # Check for weapon absence (only if not already set to True)
    if claims["weapon_present"] is None:
        for pattern in weapon_absent_patterns:
            if re.search(pattern, text_lower):
                claims["weapon_present"] = False
                break
    
    return claims


def convert_keyframe_results_to_video_stats(keyframe_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert VideoKeyframeProcessor results to format compatible with scoring.py.
    Uses video_analyzer for people/cars/weapons detection.
    
    Args:
        keyframe_results: Results from VideoKeyframeProcessor.process_video()
        
    Returns:
        Dictionary with people, cars, weapon_present, frames
    """
    # We still need to use video_analyzer for people/cars/weapons counts
    # VideoKeyframeProcessor focuses on collision detection and severity
    # So we'll combine both approaches
    
    # For now, use video_analyzer to get people/cars/weapons
    # In the future, we could enhance VideoKeyframeProcessor to also count these
    video_path = keyframe_results.get('video_path')
    
    if video_path and os.path.exists(video_path):
        # Use existing video_analyzer for object detection
        video_stats = analyze_video(video_path)
        
        # Add keyframe processor results
        video_stats['collision_detected'] = keyframe_results.get('collision_detected', False)
        video_stats['collision_timestamp'] = keyframe_results.get('T_actual')
        video_stats['collision_confidence'] = keyframe_results.get('collision_confidence', 0.0)
        video_stats['severity'] = keyframe_results.get('severity_actual')
        video_stats['severity_confidence'] = keyframe_results.get('severity_confidence', 0.0)
        
        return video_stats
    else:
        # Fallback if video path not available
        return {
            "people": 0,
            "cars": 0,
            "weapon_present": False,
            "frames": [],
            "collision_detected": keyframe_results.get('collision_detected', False),
            "collision_timestamp": keyframe_results.get('T_actual'),
            "severity": keyframe_results.get('severity_actual')
        }


def frame_to_base64(frame: Any) -> Optional[str]:
    """Convert OpenCV frame to base64 encoded string."""
    try:
        if frame is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception:
        return None


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "message": "EvidenceCheck MVP API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "API is operational"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_consistency(
    video: UploadFile = File(..., description="Video file to analyze"),
    text_description: str = Form(..., description="Text description of the incident"),
    use_keyframe_processor: bool = Form(True, description="Use VideoKeyframeProcessor (new) or video_analyzer (legacy)")
):
    """
    Analyze video and text description for consistency.
    
    Args:
        video: Video file (MP4, MOV, AVI, MKV)
        text_description: Text description of the incident
        use_keyframe_processor: Whether to use new VideoKeyframeProcessor or legacy video_analyzer
        
    Returns:
        AnalysisResponse with consistency score and details
    """
    tmp_path = None
    
    try:
        # Validate video file
        if not video.filename:
            raise HTTPException(status_code=400, detail="No video file provided")
        
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv'}
        file_ext = os.path.splitext(video.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Validate text description
        if not text_description or not text_description.strip():
            raise HTTPException(status_code=400, detail="Text description is required")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await video.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract claims from text
        claims = extract_claims_from_text(text_description)
        
        # Process video
        if use_keyframe_processor:
            # Use new VideoKeyframeProcessor
            keyframe_results = video_processor.process_video(
                video_path=tmp_path,
                output_dir=None  # Don't save keyframes for API
            )
            
            # Convert to format compatible with scoring
            video_stats = convert_keyframe_results_to_video_stats(keyframe_results)
        else:
            # Use legacy video_analyzer
            video_stats = analyze_video(tmp_path)
        
        # Calculate consistency score
        result = score_consistency(claims, video_stats)
        
        # Convert frames to base64 for JSON response
        frames_base64 = []
        frames = video_stats.get("frames", [])
        for frame in frames[:3]:  # Limit to 3 frames
            frame_b64 = frame_to_base64(frame)
            if frame_b64:
                frames_base64.append(frame_b64)
        
        # Prepare video analysis summary
        video_analysis_summary = {
            "people": video_stats.get("people", 0),
            "cars": video_stats.get("cars", 0),
            "weapon_present": video_stats.get("weapon_present", False),
            "frames_count": len(frames_base64),
            "frames": frames_base64  # Base64 encoded frames
        }
        
        # Add keyframe processor results if available
        if use_keyframe_processor:
            video_analysis_summary.update({
                "collision_detected": video_stats.get("collision_detected", False),
                "collision_timestamp": video_stats.get("collision_timestamp"),
                "collision_confidence": video_stats.get("collision_confidence", 0.0),
                "severity": video_stats.get("severity"),
                "severity_confidence": video_stats.get("severity_confidence", 0.0)
            })
        
        # Prepare text claims summary
        text_claims_summary = {
            "people": claims.get("people"),
            "cars": claims.get("cars"),
            "weapon_present": claims.get("weapon_present"),
            "raw_text_snippet": text_description[:100] + "..." if len(text_description) > 100 else text_description
        }
        
        # Process report using ReportProcessor for time/severity extraction
        report_data = report_processor.process_report(text_description, is_image_path=False)
        if report_data and "error" not in report_data:
            text_claims_summary.update({
                "time_seconds": report_data.get("TReport"),
                "severity_report": report_data.get("SeverityReport")
            })
        
        return AnalysisResponse(
            success=True,
            consistency_score=result["score"],
            details=result["details"],
            video_analysis=video_analysis_summary,
            text_claims=text_claims_summary,
            error=None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        return AnalysisResponse(
            success=False,
            consistency_score=0,
            details=[],
            video_analysis={},
            text_claims={},
            error=str(e)
        )
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@app.post("/analyze-text-only")
async def analyze_text_only(
    text_description: str = Form(..., description="Text description or report to analyze")
):
    """
    Analyze text/report only (no video).
    Extracts time, severity, and claims from text.
    
    Returns:
        Dictionary with extracted information
    """
    try:
        if not text_description or not text_description.strip():
            raise HTTPException(status_code=400, detail="Text description is required")
        
        # Extract claims using regex patterns
        claims = extract_claims_from_text(text_description)
        
        # Process report using ReportProcessor
        report_data = report_processor.process_report(text_description, is_image_path=False)
        
        return {
            "success": True,
            "claims": claims,
            "report_data": report_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
