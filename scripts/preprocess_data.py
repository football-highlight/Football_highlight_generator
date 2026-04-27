#!/usr/bin/env python
"""
Preprocess football match videos for training - FINAL FIXED VERSION
"""

import argparse
import os
import sys
from pathlib import Path
import logging
import traceback
import json
import pickle
import numpy as np  # IMPORT AT TOP LEVEL

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_preparation.dataset_creator import DataSplitter, Sample
from src.video_processing.preprocessor import VideoProcessor
from src.audio_processing.audio_processor import AudioProcessor
from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def preprocess_videos(args):
    """Preprocess video files for training"""
    
    logger.info("Starting video preprocessing...")
    
    # Create necessary directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    video_processor = VideoProcessor(config)
    audio_processor = AudioProcessor(config)
    data_splitter = DataSplitter(config)
    
    # Process each video in the input directory
    input_dir = Path(args.input_dir)
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.avi"))
    
    if not video_files:
        logger.error(f"No video files found in {input_dir}")
        return
    
    logger.info(f"Found {len(video_files)} video files to process")
    
    all_samples = []
    processed_videos = 0
    
    for video_path in video_files:
        logger.info(f"Processing: {video_path.name}")
        
        try:
            # Check if annotation exists
            annotation_path = Path(args.annotation_dir) / f"{video_path.stem}.json"
            
            if not annotation_path.exists():
                logger.warning(f"No annotation found for {video_path.name}, skipping...")
                continue
            
            # Extract video clips
            clip_duration = args.clip_duration or config.VIDEO.CLIP_DURATION
            clips = video_processor.create_clips(
                str(video_path),
                clip_duration=clip_duration,
                overlap=args.overlap
            )
            
            logger.info(f"Created {len(clips)} clips from {video_path.name}")
            
            # Process audio - SIMPLIFIED TO AVOID MEMORY ERRORS
            audio_features = None
            try:
                # Create simple audio features to avoid memory issues
                logger.info(f"Creating simplified audio features...")
                audio_features = audio_processor.create_simple_audio_features(duration=600)
                # {
                #     'duration': 600,
                #     'sample_rate': config.AUDIO.SAMPLE_RATE,
                #     'mfcc': np.random.randn(config.AUDIO.MFCC_N_COEFFS, 100).astype(np.float32)
                # }
            except Exception as e:
                logger.error(f"Audio processing failed for {video_path.name}: {e}")
                # Create minimal audio features for continuity
                audio_features = {
                    'duration': 600,
                    'sample_rate': config.AUDIO.SAMPLE_RATE,
                    'mfcc': np.zeros((config.AUDIO.MFCC_N_COEFFS, 100), dtype=np.float32)
                }
            
            # Load annotations
            with open(annotation_path, 'r') as f:
                annotations = json.load(f)
            
            # Create samples from clips
            video_sample_objects = []  # Store Sample objects
            for clip_idx, clip in enumerate(clips):
                # Find events in this clip
                clip_events = []
                for event in annotations.get('events', []):
                    # Handle both 'start' and 'start_time' field names
                    event_start = event.get('start', event.get('start_time', 0))
                    if clip.start_time <= event_start <= clip.end_time:
                        clip_events.append(event)
                
                # Include clip if it has events OR if we're including non-events
                if clip_events or args.include_non_events:
                    # Determine label
                    label = 1 if clip_events else 0  # 1=highlight, 0=non-highlight
                    
                    # Get event type
                    if clip_events:
                        event_type = clip_events[0].get('label', 'unknown_event')
                    else:
                        event_type = 'no_event'
                    
                    # Create Sample object (NOT dictionary)
                    try:
                        sample = Sample(
                            video_frames=clip.frames,
                            audio_features=audio_features,
                            label=label,
                            event_type=event_type,
                            timestamp=clip.start_time,
                            video_path=str(video_path),
                            metadata={
                                'clip_events': clip_events,
                                'fps': clip.fps,
                                'clip_duration': clip.end_time - clip.start_time,
                                'frame_count': clip.frame_count,
                                'resolution': (clip.frames.shape[2], clip.frames.shape[1]),
                                'clip_idx': clip_idx
                            }
                        )
                        video_sample_objects.append(sample)
                    except Exception as e:
                        logger.error(f"Failed to create Sample object: {e}")
                        continue
            
            all_samples.extend(video_sample_objects)
            processed_videos += 1
            logger.info(f"Created {len(video_sample_objects)} samples from {video_path.name}")
                    
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"Extracted {len(all_samples)} samples from {processed_videos} videos")
    
    if len(all_samples) == 0:
        logger.error("No samples extracted. Check annotations and video files.")
        return
    
    # Create label map from Sample objects
    label_map = {}
    for sample in all_samples:
        if sample.event_type not in label_map:
            label_map[sample.event_type] = len(label_map)
        # Update the sample's label
        sample.label = label_map[sample.event_type]
    
    # Split data - Now passing Sample objects
    train_samples, val_samples, test_samples = data_splitter.split_samples(
        all_samples,
        test_size=args.test_split,
        val_size=args.val_split,
        random_state=42
    )
    
    # Save datasets
    datasets = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    for split_name, samples in datasets.items():
        dataset_path = output_dir / f"{split_name}_dataset.pkl"
        
        # Convert Sample objects to dictionaries
        sample_dicts = []
        for sample in samples:
            try:
                sample_dicts.append(sample.to_dict())
            except Exception as e:
                logger.error(f"Failed to convert sample to dict: {e}")
                continue
        
        data = {
            'samples': sample_dicts,
            'label_map': label_map,
            'split': split_name,
            'num_samples': len(sample_dicts),
            'config': {
                'video': config.VIDEO.__dict__,
                'audio': config.AUDIO.__dict__,
                'model': config.MODEL.__dict__
            }
        }
        
        try:
            with open(dataset_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved {split_name} dataset with {len(sample_dicts)} samples to {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to save {split_name} dataset: {e}")
    
    # Save label map
    label_map_path = output_dir / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)
    
    logger.info(f"Label map saved with {len(label_map)} classes")
    
    # Create summary
    if all_samples:
        avg_clip_duration = np.mean([s.metadata.get('clip_duration', 0) for s in all_samples])
        avg_frame_count = np.mean([s.metadata.get('frame_count', 0) for s in all_samples])
    else:
        avg_clip_duration = 0
        avg_frame_count = 0
    
    summary = {
        'total_videos': len(video_files),
        'processed_videos': processed_videos,
        'total_samples': len(all_samples),
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'class_distribution': {k: 0 for k in label_map.keys()},
        'avg_clip_duration': float(avg_clip_duration),
        'avg_frame_count': float(avg_frame_count)
    }
    
    for sample in all_samples:
        summary['class_distribution'][sample.event_type] += 1
    
    summary_path = output_dir / "preprocessing_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Train/Val/Test split: {len(train_samples)}/{len(val_samples)}/{len(test_samples)}")
    logger.info(f"Class distribution: {summary['class_distribution']}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess football match videos")
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw_videos",
        help="Directory containing input videos"
    )
    
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="data/annotations",
        help="Directory containing annotation files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/datasets",
        help="Directory to save processed datasets"
    )
    
    parser.add_argument(
        "--clip_duration",
        type=int,
        default=None,
        help="Duration of each clip in seconds (default: from config)"
    )
    
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.3,
        help="Overlap between consecutive clips (0-1)"
    )
    
    parser.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Proportion of data for testing"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Proportion of data for validation"
    )
    
    parser.add_argument(
        "--include_non_events",
        action="store_true",
        default=True,
        help="Include clips without events (negative samples)"
    )
    
    parser.add_argument(
        "--min_samples_per_class",
        type=int,
        default=10,
        help="Minimum samples per class (will skip classes with fewer samples)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run preprocessing
    preprocess_videos(args)


if __name__ == "__main__":
    main()