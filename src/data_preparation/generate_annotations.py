# scripts/generate_annotations.py
import json
import cv2
import numpy as np
from pathlib import Path
import subprocess
import tempfile
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_video_duration(video_path):
    """Get video duration using OpenCV"""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    return duration

def detect_scene_changes(video_path, threshold=30.0):
    """Detect scene changes in video"""
    cap = cv2.VideoCapture(str(video_path))
    prev_frame = None
    scene_changes = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(gray, prev_frame)
            diff_value = np.mean(diff)
            
            if diff_value > threshold:
                time_sec = frame_idx / fps
                scene_changes.append(time_sec)
        
        prev_frame = gray
        frame_idx += 1
    
    cap.release()
    return scene_changes

def extract_audio_features(video_path):
    """Extract audio features using ffmpeg"""
    try:
        import librosa
        import tempfile
        
        # Extract audio to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use ffmpeg to extract audio
        cmd = ['ffmpeg', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
               '-ar', '16000', '-ac', '1', tmp_path, '-y']
        subprocess.run(cmd, capture_output=True)
        
        # Load audio
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # Calculate energy
        energy = librosa.feature.rms(y=y)[0]
        
        # Find peaks in energy (potential excitement moments)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(energy, height=np.mean(energy) * 1.5)
        
        # Convert to timestamps
        hop_length = 512
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        
        # Clean up
        Path(tmp_path).unlink()
        
        return times.tolist()
        
    except ImportError:
        logger.warning("Librosa not installed, skipping audio analysis")
        return []

def generate_annotations_for_video(video_path, output_dir):
    """Generate automatic annotations for a video"""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing: {video_path.name}")
    
    # Get video duration
    duration = get_video_duration(video_path)
    logger.info(f"Video duration: {duration:.2f} seconds")
    
    # Generate events based on different strategies
    events = []
    
    # Strategy 1: Equal intervals (fallback)
    if duration > 0:
        num_segments = max(3, int(duration / 60))  # One segment per minute, min 3
        interval = duration / num_segments
        
        for i in range(num_segments):
            start = i * interval
            end = min(start + 15, duration)  # 15-second segments
            
            events.append({
                "start": float(start),
                "end": float(end),
                "label": "potential_highlight",
                "description": f"Segment {i+1}",
                "confidence": 0.5,
                "source": "auto_generated"
            })
    
    # Strategy 2: Try to detect scene changes
    try:
        scene_changes = detect_scene_changes(video_path)
        logger.info(f"Detected {len(scene_changes)} scene changes")
        
        for i, change_time in enumerate(scene_changes[:10]):  # Limit to first 10
            start = max(0, change_time - 7.5)
            end = min(duration, change_time + 7.5)
            
            events.append({
                "start": float(start),
                "end": float(end),
                "label": "scene_change",
                "description": f"Scene change {i+1}",
                "confidence": 0.7,
                "source": "scene_detection"
            })
    except Exception as e:
        logger.warning(f"Scene detection failed: {e}")
    
    # Strategy 3: Audio analysis
    try:
        audio_events = extract_audio_features(video_path)
        for i, audio_time in enumerate(audio_events[:10]):  # Limit to first 10
            start = max(0, audio_time - 5)
            end = min(duration, audio_time + 10)
            
            events.append({
                "start": float(start),
                "end": float(end),
                "label": "audio_peak",
                "description": f"Audio peak {i+1}",
                "confidence": 0.6,
                "source": "audio_analysis"
            })
    except Exception as e:
        logger.warning(f"Audio analysis failed: {e}")
    
    # Create annotation structure
    annotation = {
        "video_path": video_path.name,
        "duration": float(duration),
        "resolution": "1920x1080",
        "events": events,
        "generated_at": datetime.now().isoformat(),
        "generator": "auto_annotation_generator"
    }
    
    # Save annotation
    output_path = output_dir / f"{video_path.stem}.json"
    with open(output_path, 'w') as f:
        json.dump(annotation, f, indent=2)
    
    logger.info(f"Generated annotation with {len(events)} events: {output_path}")
    return output_path

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate automatic annotations for videos')
    parser.add_argument('--input-dir', default='data/raw_videos', help='Input video directory')
    parser.add_argument('--output-dir', default='data/annotations', help='Output annotation directory')
    parser.add_argument('--video', help='Process specific video file')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if args.video:
        video_files = [Path(args.video)]
    else:
        video_files = list(input_dir.glob('*.mp4')) + list(input_dir.glob('*.avi')) + list(input_dir.glob('*.mkv'))
    
    logger.info(f"Found {len(video_files)} video files")
    
    for video_file in video_files:
        if video_file.exists():
            generate_annotations_for_video(video_file, output_dir)
        else:
            logger.error(f"Video file not found: {video_file}")

if __name__ == "__main__":
    main()