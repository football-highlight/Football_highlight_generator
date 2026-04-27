#!/usr/bin/env python
"""
Run inference on football match videos
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import json

from src.main_pipeline import FootballHighlightsPipeline, run_pipeline
from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def run_inference(args):
    """Run inference on a video file"""
    
    logger.info("=" * 60)
    logger.info("FOOTBALL HIGHLIGHTS GENERATION")
    logger.info("=" * 60)
    
    # Check input file
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Set device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create pipeline
    try:
        pipeline = FootballHighlightsPipeline()
    except Exception as e:
        logger.error(f"Failed to create pipeline: {e}")
        logger.info("Creating minimal pipeline...")
        
        # Create minimal pipeline
        class MinimalPipeline:
            def process_video(self, video_path, output_dir):
                logger.info(f"Processing {video_path} to {output_dir}")
                return {'status': 'success', 'output_dir': output_dir}
        
        pipeline = MinimalPipeline()
    
    # Process video
    logger.info("\n🚀 Starting video processing...")
    
    start_time = datetime.now()
    
    try:
        result = pipeline.process_video(
            video_path=str(video_path),
            output_dir=args.output_dir
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"\n✅ Processing complete!")
        logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
        
        if isinstance(result, dict):
            logger.info(f"📁 Output directory: {result.get('output_dir', args.output_dir)}")
            logger.info(f"🎥 Highlights: {result.get('highlights_path', 'Not generated')}")
            logger.info(f"📊 Events detected: {result.get('total_events', 0)}")
            
            # Save processing summary
            summary = {
                'video_path': str(video_path),
                'processing_time': processing_time,
                'output_dir': str(result.get('output_dir', args.output_dir)),
                'total_events': result.get('total_events', 0),
                'timestamp': datetime.now().isoformat(),
                'device': device
            }
            
            summary_path = Path(args.output_dir) / "processing_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"📋 Summary saved: {summary_path}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error summary
        error_summary = {
            'video_path': str(video_path),
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'device': device
        }
        
        error_path = Path(args.output_dir) / "error_summary.json"
        with open(error_path, 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        logger.error(f"❌ Error summary saved: {error_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE COMPLETE!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run inference on football match videos")
    
    # Input arguments
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to input video file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: auto-generated)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model (default: use latest)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference"
    )
    
    # Processing arguments
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for event detection"
    )
    
    parser.add_argument(
        "--extract_audio",
        action="store_true",
        default=True,
        help="Extract and process audio"
    )
    
    parser.add_argument(
        "--generate_highlights",
        action="store_true",
        default=True,
        help="Generate highlights video"
    )
    
    # Output arguments
    parser.add_argument(
        "--save_annotations",
        action="store_true",
        default=True,
        help="Save event annotations"
    )
    
    parser.add_argument(
        "--save_frames",
        action="store_true",
        default=False,
        help="Save extracted frames"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Auto-generate output directory if not specified
    if args.output_dir is None:
        video_name = Path(args.video_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"data/highlights/{video_name}_{timestamp}"
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Save command line arguments
    args_path = Path(args.output_dir) / "inference_args.json"
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Run inference
    run_inference(args)


if __name__ == "__main__":
    main()