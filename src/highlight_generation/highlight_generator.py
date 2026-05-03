"""
Highlight video generation
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import logging
from datetime import datetime
import json
import subprocess

# Configure MoviePy to use ImageMagick BEFORE importing
import os
# Set ImageMagick path explicitly
imagemagick_path = r"E:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
if os.path.exists(imagemagick_path):
    os.environ['IMAGEMAGICK_BINARY'] = imagemagick_path
    os.environ['MAGICK_HOME'] = r"E:\Program Files\ImageMagick-7.1.2-Q16-HDRI"
    print(f"✅ ImageMagick configured at: {imagemagick_path}")
else:
    print(f"⚠️ ImageMagick not found at {imagemagick_path}, trying PATH...")


# Try to import moviepy, but don't fail if not available
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip
    from moviepy.video.fx.all import resize, fadein, fadeout
    from moviepy.config import change_settings
    
    # Configure MoviePy to use ImageMagick
    if os.path.exists(imagemagick_path):
        change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
    
    # Test if TextClip works
    try:
        # Create a test text clip to verify ImageMagick works
        test_clip = TextClip("Test", fontsize=10, color='white', font='Arial')
        test_clip.close()
        MOVIEPY_AVAILABLE = True
        print("✅ MoviePy with TextClip working correctly")
    except Exception as e:
        print(f"⚠️ MoviePy TextClip failed: {e}. Will use OpenCV fallback.")
        MOVIEPY_AVAILABLE = False
        
except ImportError as e:
    MOVIEPY_AVAILABLE = False
    print(f"⚠️ MoviePy not available: {e}")

from config.config import config
from src.event_detection.event_detector import DetectedEvent
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HighlightGenerator:
    """Generate highlight videos from detected events"""
    
    def __init__(self, config):
        self.config = config
        self.video_config = config.VIDEO
        self.event_config = config.EVENT
        
        # Set ImageMagick path in instance as well
        self.imagemagick_path = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"
        if os.path.exists(self.imagemagick_path):
            os.environ['IMAGEMAGICK_BINARY'] = self.imagemagick_path
        
        if not MOVIEPY_AVAILABLE:
            logger.warning("MoviePy not fully available. Using OpenCV for video operations (no text overlays).")
    
    def generate_highlights(
        self,
        video_path: str,
        events: List[DetectedEvent],
        output_path: str,
        max_duration: int = 300,  # 5 minutes max
        min_clip_duration: int = 3,
        max_clips: int = 20
    ) -> Optional[str]:
        """
        Generate highlights video from detected events
        
        Args:
            video_path: Path to original video
            events: List of detected events
            output_path: Path to save highlights video
            max_duration: Maximum highlights duration in seconds
            min_clip_duration: Minimum clip duration in seconds
            max_clips: Maximum number of clips to include
            
        Returns:
            Path to generated highlights video, or None if failed
        """
        logger.info(f"Generating highlights from {len(events)} events")
        
        # Prioritize events
        prioritized_events = self._prioritize_events(events)
        
        # Select top events based on constraints
        selected_events = self._select_events(
            prioritized_events,
            max_duration=max_duration,
            min_clip_duration=min_clip_duration,
            max_clips=max_clips
        )
        
        if not selected_events:
            logger.warning("No events selected for highlights")
            return None
        
        logger.info(f"Selected {len(selected_events)} events, total duration: {sum(e.end_time - e.start_time for e in selected_events):.1f}s")
        
        # Generate highlights
        try:
            # Try MoviePy first, fallback to OpenCV
            if MOVIEPY_AVAILABLE:
                highlights_path = self._generate_with_moviepy(
                    video_path, selected_events, output_path
                )
            else:
                logger.info("MoviePy not available, using OpenCV fallback")
                highlights_path = self._generate_with_opencv(
                    video_path, selected_events, output_path
                )
            
            if highlights_path and Path(highlights_path).exists():
                logger.info(f"Highlights generated successfully: {highlights_path}")
                
                # Save highlights metadata
                self._save_highlights_metadata(
                    video_path, selected_events, highlights_path, output_path
                )
                
                return highlights_path
            else:
                logger.error("Failed to generate highlights")
                return None
                
        except Exception as e:
            logger.error(f"Error generating highlights: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prioritize_events(self, events: List[DetectedEvent]) -> List[Dict]:
        """
        Prioritize events based on weights and confidence
        
        Args:
            events: List of detected events
            
        Returns:
            List of prioritized events with scores
        """
        prioritized = []
        
        for event in events:
            # Calculate event score
            weight = self.event_config.EVENT_WEIGHTS.get(event.event_type, 0.5)
            confidence = event.confidence
            
            # Base score
            score = weight * confidence
            
            # Bonus for events from multiple sources
            if event.source == 'fusion':
                score *= 1.2
            
            # Bonus for high confidence
            if confidence > 0.8:
                score *= 1.1
            
            # Create event dict with score
            event_dict = {
                'event': event,
                'score': score,
                'duration': event.end_time - event.start_time
            }
            
            prioritized.append(event_dict)
        
        # Sort by score (descending)
        prioritized.sort(key=lambda x: x['score'], reverse=True)
        
        return prioritized
    
    def _select_events(
        self,
        prioritized_events: List[Dict],
        max_duration: int = 300,
        min_clip_duration: int = 3,
        max_clips: int = 20
    ) -> List[DetectedEvent]:
        """
        Select events for highlights based on constraints
        
        Args:
            prioritized_events: Prioritized events with scores
            max_duration: Maximum total duration
            min_clip_duration: Minimum clip duration
            max_clips: Maximum number of clips
            
        Returns:
            Selected events
        """
        selected_events = []
        total_duration = 0
        
        for event_info in prioritized_events:
            if len(selected_events) >= max_clips:
                break
            
            event = event_info['event']
            event_duration = event_info['duration']
            
            # Ensure minimum clip duration
            if event_duration < min_clip_duration:
                # Extend clip to minimum duration
                extension = (min_clip_duration - event_duration) / 2
                event.start_time = max(0, event.start_time - extension)
                event.end_time = event.end_time + extension
                event_duration = min_clip_duration
            
            # Check if we can add this event
            if total_duration + event_duration <= max_duration:
                selected_events.append(event)
                total_duration += event_duration
        
        logger.info(f"Selected {len(selected_events)} events, total duration: {total_duration:.1f}s")
        
        return selected_events
    
    def _generate_with_moviepy(
        self,
        video_path: str,
        events: List[DetectedEvent],
        output_path: str
    ) -> Optional[str]:
        """Generate highlights using MoviePy"""
        original_clip = None
        clips = []
        
        try:
            # Verify ImageMagick is available
            if not self._check_imagemagick():
                logger.warning("ImageMagick not properly configured, falling back to OpenCV")
                return self._generate_with_opencv(video_path, events, output_path)
            
            # Load original video
            logger.info(f"Loading video: {video_path}")
            original_clip = VideoFileClip(video_path)
            
            # Extract clips for each event
            for i, event in enumerate(events):
                try:
                    # Calculate clip times with buffers from config
                    start_time = max(0, event.start_time - getattr(self.event_config, 'EVENT_BUFFER_BEFORE', 2))
                    end_time = min(original_clip.duration, event.end_time + getattr(self.event_config, 'EVENT_BUFFER_AFTER', 3))
                    
                    logger.info(f"Extracting clip {i+1}/{len(events)}: {event.event_type} ({start_time:.1f}s - {end_time:.1f}s)")
                    
                    # Extract subclip
                    subclip = original_clip.subclip(start_time, end_time)
                    
                    # Add fade in/out
                    subclip = subclip.fx(fadein, 0.5).fx(fadeout, 0.5)
                    
                    # Add event label (simpler approach to avoid ImageMagick issues)
                    try:
                        # Try with simple settings
                        label = TextClip(
                            f"{event.event_type.upper()}",
                            fontsize=24,
                            color='white',
                            font='Arial',
                            stroke_color='black',
                            stroke_width=1,
                            method='label'  # Use 'label' instead of 'caption'
                        ).set_position(('center', 'top')).set_duration(subclip.duration)
                        
                        # Composite clip with label
                        composite_clip = CompositeVideoClip([subclip, label])
                        clips.append(composite_clip)
                        
                    except Exception as text_error:
                        logger.warning(f"Could not add text label for event {i}: {text_error}")
                        # Fall back to just the video clip without text
                        clips.append(subclip)
                        
                except Exception as clip_error:
                    logger.error(f"Error processing clip {i}: {clip_error}")
                    continue
            
            if not clips:
                logger.error("No clips were successfully processed")
                return None
            
            # Concatenate all clips
            logger.info(f"Concatenating {len(clips)} clips")
            if len(clips) > 1:
                final_clip = concatenate_videoclips(clips)
            else:
                final_clip = clips[0]
            
            # Write highlights video
            logger.info(f"Writing highlights to: {output_path}")
            final_clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                fps=self.video_config.FRAME_RATE,
                preset='medium',
                threads=4,
                logger=None  # Disable verbose logging
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"MoviePy highlight generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Clean up resources
            if original_clip:
                original_clip.close()
            for clip in clips:
                try:
                    clip.close()
                except:
                    pass
    
    def _check_imagemagick(self) -> bool:
        """Check if ImageMagick is properly configured"""
        try:
            # Try to find ImageMagick using moviepy's config
            from moviepy.config import get_setting
            try:
                binary = get_setting("IMAGEMAGICK_BINARY")
                if binary and os.path.exists(binary):
                    return True
            except:
                pass
            
            # Try environment variable
            binary = os.environ.get('IMAGEMAGICK_BINARY')
            if binary and os.path.exists(binary):
                return True
            
            # Try direct path
            if os.path.exists(self.imagemagick_path):
                return True
            
            # Try running magick command
            result = subprocess.run(['magick', '--version'], 
                                   capture_output=True, text=True)
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _generate_with_opencv(
        self,
        video_path: str,
        events: List[DetectedEvent],
        output_path: str
    ) -> Optional[str]:
        """Generate highlights using OpenCV (basic version with text support)"""
        try:
            # Get video properties
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return None
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps}fps, {total_frames} frames")
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                logger.error("Could not create output video writer")
                cap.release()
                return None
            
            # Extract and write clips for each event
            total_frames_written = 0
            
            for i, event in enumerate(events):
                start_time = max(0, event.start_time - getattr(self.event_config, 'EVENT_BUFFER_BEFORE', 2))
                end_time = event.end_time + getattr(self.event_config, 'EVENT_BUFFER_AFTER', 3)
                
                start_frame = int(start_time * fps)
                end_frame = int(end_time * fps)
                
                logger.info(f"Processing event {i+1}/{len(events)}: {event.event_type} (frames {start_frame}-{end_frame})")
                
                # Set start position
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                # Read and write frames
                frame_count = 0
                for frame_num in range(start_frame, min(end_frame, total_frames)):
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Could not read frame {frame_num}")
                        break
                    
                    # Add event label (OpenCV version)
                    label = f"{event.event_type.upper()} ({event.confidence:.0%})"
                    
                    # Add background rectangle for text
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    cv2.rectangle(frame, 
                                (10, 10), 
                                (10 + text_size[0] + 20, 10 + text_size[1] + 20), 
                                (0, 0, 0), 
                                -1)
                    
                    # Add text
                    cv2.putText(
                        frame, label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2
                    )
                    
                    out.write(frame)
                    frame_count += 1
                    total_frames_written += 1
                
                logger.info(f"  Wrote {frame_count} frames for event {i+1}")
            
            # Release resources
            cap.release()
            out.release()
            
            duration = total_frames_written / fps if fps > 0 else 0
            logger.info(f"OpenCV highlights generated: {total_frames_written} frames, {duration:.1f}s")
            
            if total_frames_written > 0:
                return output_path
            else:
                logger.error("No frames were written")
                return None
            
        except Exception as e:
            logger.error(f"OpenCV highlight generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _save_highlights_metadata(
        self,
        video_path: str,
        events: List[DetectedEvent],
        highlights_path: str,
        output_dir: str
    ):
        """
        Save metadata about generated highlights
        
        Args:
            video_path: Path to original video
            events: List of detected events
            highlights_path: Path to generated highlights video
            output_dir: Output directory (can be a file path or directory)
        """
        # Ensure output_dir is a directory, not a file
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        # If output_dir is actually a file path, get its parent directory
        if output_dir.suffix:  # If it has an extension like .mp4
            output_dir = output_dir.parent
        
        # Make sure the directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'original_video': str(video_path),
            'highlights_video': str(highlights_path),
            'generated_at': datetime.now().isoformat(),
            'total_events': len(events),
            'events': [],
            'total_duration': 0
        }
        
        total_duration = 0
        
        for i, event in enumerate(events):
            event_duration = event.end_time - event.start_time
            total_duration += event_duration
            
            event_info = {
                'index': i,
                'event_type': event.event_type,
                'confidence': event.confidence,
                'start_time': event.start_time,
                'end_time': event.end_time,
                'duration': event_duration,
                'source': event.source
            }
            
            metadata['events'].append(event_info)
        
        metadata['total_duration'] = total_duration
        
        # Save to file in the output directory (not inside the video file!)
        metadata_path = output_dir / "highlights_metadata.json"
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved highlights metadata to {metadata_path}")
        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_path}: {e}")
            # Try saving in a different location as fallback
            try:
                fallback_path = Path(highlights_path).parent / "highlights_metadata.json"
                with open(fallback_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata to fallback location: {fallback_path}")
            except Exception as fallback_error:
                logger.error(f"Even fallback save failed: {fallback_error}")
    
    def create_highlights_summary(
        self,
        video_path: str,
        events: List[DetectedEvent],
        output_dir: str
    ) -> Dict:
        """
        Create a summary of highlights
        
        Args:
            video_path: Path to original video
            events: List of events
            output_dir: Output directory
            
        Returns:
            Summary dictionary
        """
        # Group events by type
        events_by_type = {}
        for event in events:
            event_type = event.event_type
            if event_type not in events_by_type:
                events_by_type[event_type] = []
            events_by_type[event_type].append(event)
        
        # Create summary
        summary = {
            'video': Path(video_path).name,
            'total_events': len(events),
            'events_by_type': {},
            'timeline': []
        }
        
        # Add event type counts
        for event_type, event_list in events_by_type.items():
            summary['events_by_type'][event_type] = len(event_list)
        
        # Add timeline
        for event in sorted(events, key=lambda x: x.start_time):
            timeline_entry = {
                'time': f"{event.start_time:.1f}s - {event.end_time:.1f}s",
                'event': event.event_type,
                'confidence': f"{event.confidence:.0%}",
                'duration': f"{event.end_time - event.start_time:.1f}s"
            }
            summary['timeline'].append(timeline_entry)
        
        # Save summary
        summary_path = Path(output_dir) / "highlights_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also create a text summary
        text_summary = self._create_text_summary(summary)
        text_path = Path(output_dir) / "highlights_summary.txt"
        with open(text_path, 'w') as f:
            f.write(text_summary)
        
        logger.info(f"Created highlights summary")
        
        return summary
    
    def _create_text_summary(self, summary: Dict) -> str:
        """Create a human-readable text summary"""
        text = f"FOOTBALL HIGHLIGHTS SUMMARY\n"
        text += "=" * 50 + "\n\n"
        
        text += f"Video: {summary['video']}\n"
        text += f"Total Events: {summary['total_events']}\n\n"
        
        text += "Event Breakdown:\n"
        for event_type, count in summary['events_by_type'].items():
            text += f"  {event_type.upper()}: {count}\n"
        
        text += "\nTimeline:\n"
        for entry in summary['timeline']:
            text += f"  {entry['time']}: {entry['event']} "
            text += f"(Confidence: {entry['confidence']}, Duration: {entry['duration']})\n"
        
        return text


def test_highlight_generator():
    """Test highlight generator functions"""
    print("Testing HighlightGenerator...")
    
    from config.config import config
    from src.event_detection.event_detector import DetectedEvent
    
    # Create dummy highlight generator
    generator = HighlightGenerator(config)
    
    # Create dummy events
    dummy_events = [
        DetectedEvent(
            event_type='goal',
            confidence=0.9,
            start_time=30.0,
            end_time=35.0,
            source='model'
        ),
        DetectedEvent(
            event_type='foul',
            confidence=0.7,
            start_time=120.0,
            end_time=125.0,
            source='commentary',
            keywords=['foul']
        ),
        DetectedEvent(
            event_type='corner',
            confidence=0.8,
            start_time=240.0,
            end_time=245.0,
            source='fusion'
        ),
    ]
    
    # Test event prioritization
    prioritized = generator._prioritize_events(dummy_events)
    print(f"✅ Prioritized {len(prioritized)} events")
    
    # Test event selection
    selected = generator._select_events(
        prioritized,
        max_duration=60,
        max_clips=2
    )
    print(f"✅ Selected {len(selected)} events for highlights")
    
    # Test summary creation
    summary = generator.create_highlights_summary(
        'test_video.mp4',
        dummy_events,
        '.'
    )
    
    print(f"✅ Created summary with {summary['total_events']} events")
    
    print("✅ All highlight generator tests passed!")


if __name__ == "__main__":
    test_highlight_generator()