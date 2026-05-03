"""
Main pipeline for football highlights generation
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import json
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import config
from src.video_processing.preprocessor import VideoProcessor
from src.audio_processing.audio_processor import AudioProcessor
from src.feature_extraction.multimodal_fusion import create_multimodal_model, load_model
from src.event_detection.event_detector import EventDetector
from src.highlight_generation.highlight_generator import HighlightGenerator
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class FootballHighlightsPipeline:
    """Main pipeline for football highlights generation"""
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the pipeline
        
        Args:
            model_path: Path to trained model
            device: Device to use (cuda/cpu)
        """
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.video_processor = VideoProcessor(config)
        self.audio_processor = AudioProcessor(config)
        self.highlight_generator = HighlightGenerator(config)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Initialize event detector
        self.event_detector = EventDetector(config, self.model)
        
        logger.info(f"Initialized FootballHighlightsPipeline on {self.device}")
        
    def _load_model(self, model_path: str = None):
        """Load trained model"""
        if model_path is None:
            # Try to find latest model
            deployment_dir = Path("models/deployment")
            if deployment_dir.exists():
                model_files = list(deployment_dir.glob("*.pth"))
                if model_files:
                    model_path = str(sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])
        
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            model = load_model(model_path, device=self.device)
        else:
            logger.warning("No model found, creating default model")
            model = create_multimodal_model(
                model_type="multimodal",
                num_classes=len(config.EVENT.KEYWORDS),
                pretrained=False
            )
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def process_video(self, video_path: str, output_dir: str = None) -> dict:
        """
        Process a video and generate highlights
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing video: {video_path}")
        
        # Setup output directory
        if output_dir is None:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(config.PATH.HIGHLIGHTS_DIR) / f"{video_name}_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Process video
        logger.info("Step 1: Processing video...")
        video_clips = self.video_processor.create_clips(video_path)
        logger.info(f"Created {len(video_clips)} video clips")
        
        # Step 2: Process audio
        logger.info("Step 2: Processing audio...")
        audio_path = output_dir / "extracted_audio.wav"
        self.audio_processor.extract_audio_from_video(video_path, str(audio_path))
        
        # Transcribe commentary
        logger.info("Transcribing commentary...")
        timed_words = self.audio_processor.transcribe_with_whisper(str(audio_path))
        keywords_found = self.audio_processor.detect_keywords(timed_words)
        logger.info(f"Found {len(keywords_found)} keywords in commentary")
        
        # Extract audio features
        audio_features = self.audio_processor.extract_audio_features(str(audio_path))
        
        # Step 3: Detect events
        logger.info("Step 3: Detecting events...")
        all_events = []
        
        for i, clip in enumerate(video_clips):
            if i % 10 == 0:  # Log every 10 clips
                logger.info(f"Processing clip {i+1}/{len(video_clips)}")
            
            # Filter keywords for this clip
            clip_keywords = [
                kw for kw in keywords_found
                if clip.start_time <= kw['start_time'] <= clip.end_time
            ]
            
            # Detect events in clip
            clip_events = self.event_detector.detect_events_from_clip(
                clip, audio_features, clip_keywords
            )
            all_events.extend(clip_events)
        
        logger.info(f"Detected {len(all_events)} total events")
        
        # Step 4: Save annotations
        logger.info("Step 4: Saving annotations...")
        annotation_path = output_dir / "annotations.json"
        annotations = self.event_detector.save_annotations(
            all_events, video_path, str(annotation_path)
        )
        
        # Step 5: Generate highlights
        logger.info("Step 5: Generating highlights...")
        highlight_path = output_dir / "highlights.mp4"
        
        final_path = self.highlight_generator.generate_highlights(
            video_path, all_events, str(highlight_path)
        )
        
        if final_path:
            logger.info(f"Highlights generated successfully: {final_path}")
        else:
            logger.error("Failed to generate highlights")
        
        # Step 6: Generate summary
        logger.info("Step 6: Generating summary...")
        summary = self._generate_summary(video_path, annotations, final_path, output_dir)
        
        # Save processing metadata
        metadata = {
            'video_path': video_path,
            'processing_date': datetime.now().isoformat(),
            'output_dir': str(output_dir),
            'total_clips': len(video_clips),
            'total_events': len(all_events),
            'keywords_found': len(keywords_found),
            'summary': summary
        }
        
        metadata_path = output_dir / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Processing complete. Results saved to {output_dir}")
        
        return {
            'video_path': video_path,
            'annotations_path': str(annotation_path),
            'highlights_path': final_path,
            'total_events': len(all_events),
            'output_dir': str(output_dir),
            'metadata': metadata
        }
    
    def _generate_summary(self, video_path: str, annotations: dict, 
                         highlights_path: str, output_dir: Path) -> dict:
        """Generate processing summary"""
        
        summary = {
            'video_file': Path(video_path).name,
            'processing_date': datetime.now().isoformat(),
            'total_events': len(annotations.get('events', [])),
            'events_by_type': {},
            'highlight_file': Path(highlights_path).name if highlights_path else None,
            'output_directory': str(output_dir)
        }
        
        # Count events by type
        for event in annotations.get('events', []):
            event_type = event['event_type']
            summary['events_by_type'][event_type] = summary['events_by_type'].get(event_type, 0) + 1
        
        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def process_live_stream(self, stream_url: str, duration: int = 300):
        """
        Process live stream (simplified version)
        
        Args:
            stream_url: URL of live stream
            duration: Duration to process in seconds
            
        Returns:
            Processing results
        """
        logger.info(f"Processing live stream: {stream_url}")
        
        # This is a simplified version
        # In production, you would use OpenCV to capture the stream
        # and process it in real-time
        
        raise NotImplementedError("Live stream processing requires additional implementation")


def run_pipeline(video_path: str, output_dir: str = None, model_path: str = None) -> dict:
    """
    Run the complete highlights generation pipeline
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save outputs
        model_path: Path to trained model
        
    Returns:
        Processing results
    """
    try:
        pipeline = FootballHighlightsPipeline(model_path=model_path)
        result = pipeline.process_video(video_path, output_dir)
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Output directory: {result['output_dir']}")
        print(f"🎥 Highlights: {result.get('highlights_path', 'Not generated')}")
        print(f"📊 Total events detected: {result['total_events']}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return None


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Run football highlights pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--model", type=str, help="Path to model file")
    
    args = parser.parse_args()
    
    run_pipeline(args.video, args.output, args.model)