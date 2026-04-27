"""
Dataset creation and management for football event detection
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm

from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class Sample:
    """Data sample for training"""
    video_frames: np.ndarray  # Shape: (num_frames, height, width, channels)
    audio_features: Dict[str, np.ndarray]
    label: int
    event_type: str
    timestamp: float
    video_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert sample to dictionary"""
        return {
            'video_frames': self.video_frames,
            'audio_features': self.audio_features,
            'label': self.label,
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'video_path': self.video_path,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Sample':
        """Create sample from dictionary"""
        return cls(
            video_frames=data['video_frames'],
            audio_features=data['audio_features'],
            label=data['label'],
            event_type=data['event_type'],
            timestamp=data['timestamp'],
            video_path=data['video_path'],
            metadata=data.get('metadata', {})
        )

class FootballDataset(Dataset):
    """Custom PyTorch dataset for football event detection"""
    
    def __init__(
        self, 
        data_dir: str, 
        split: str = 'train',
        transform=None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing dataset files
            split: Dataset split (train, val, test)
            transform: Optional transformations to apply
            max_samples: Maximum number of samples to load
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_samples = max_samples
        
        # Load dataset
        self.samples = []
        self.label_map = {}
        self.reverse_label_map = {}
        
        self._load_dataset()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split")
        
    def _load_dataset(self):
        """Load dataset from processed files"""
        dataset_path = self.data_dir / f"{self.split}_dataset.pkl"
        
        if dataset_path.exists():
            logger.info(f"Loading dataset from {dataset_path}")
            with open(dataset_path, 'rb') as f:
                data = pickle.load(f)
                
            self.samples = [Sample.from_dict(sample_dict) for sample_dict in data['samples']]
            self.label_map = data['label_map']
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
            
            # Limit samples if specified
            if self.max_samples and len(self.samples) > self.max_samples:
                self.samples = self.samples[:self.max_samples]
                logger.info(f"Limited to {self.max_samples} samples")
                
        else:
            logger.warning(f"Dataset file not found: {dataset_path}")
            logger.info("Creating empty dataset")
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset - FIXED for variable audio sizes"""
        sample = self.samples[idx]
        
        # Convert to tensors
        video_frames = sample.video_frames.astype(np.float32) / 255.0
        
        # Convert to PyTorch format: (C, T, H, W)
        video_tensor = torch.from_numpy(video_frames).permute(3, 0, 1, 2)
        
        # Audio features (MFCC) - HANDLE VARIABLE SIZES
        audio_features = sample.audio_features.get('mfcc', np.zeros((13, 100)))
        
        # Ensure audio is 2D (coeffs, time)
        if audio_features.ndim == 1:
            audio_features = audio_features.reshape(13, -1)
        elif audio_features.ndim == 3:
            audio_features = audio_features.squeeze(0)  # Remove channel dim if present
        
        # Target shape from config: (13, 100)
        target_coeffs = 13
        target_time = 100
        
        # Fix coefficient dimension
        if audio_features.shape[0] != target_coeffs:
            if audio_features.shape[0] < target_coeffs:
                # Pad with zeros
                pad_rows = target_coeffs - audio_features.shape[0]
                audio_features = np.pad(audio_features, ((0, pad_rows), (0, 0)), mode='constant')
            else:
                # Truncate
                audio_features = audio_features[:target_coeffs, :]
        
        # Fix time dimension
        if audio_features.shape[1] != target_time:
            if audio_features.shape[1] < target_time:
                # Pad with zeros
                pad_cols = target_time - audio_features.shape[1]
                audio_features = np.pad(audio_features, ((0, 0), (0, pad_cols)), mode='constant')
            else:
                # Truncate
                audio_features = audio_features[:, :target_time]
        
        # Convert to tensor: (1, 13, 100) - Add channel dimension
        audio_tensor = torch.from_numpy(audio_features.astype(np.float32)).unsqueeze(0)
        
        # Label
        label = torch.tensor(sample.label, dtype=torch.long)
        
        # Apply transformations if any
        if self.transform:
            video_tensor = self.transform(video_tensor)
            
        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'label': label,
            'event_type': sample.event_type,
            'timestamp': sample.timestamp,
            'video_path': sample.video_path,
            'metadata': sample.metadata
        }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in dataset"""
        class_counts = {}
        for sample in self.samples:
            class_counts[sample.event_type] = class_counts.get(sample.event_type, 0) + 1
        return class_counts
    
    def get_sample_by_event_type(self, event_type: str) -> List[Sample]:
        """Get all samples of a specific event type"""
        return [s for s in self.samples if s.event_type == event_type]
    
    def save_dataset(self, output_path: Optional[str] = None):
        """Save dataset to file"""
        if output_path is None:
            output_path = self.data_dir / f"{self.split}_dataset.pkl"
            
        data = {
            'samples': [s.to_dict() for s in self.samples],
            'label_map': self.label_map,
            'split': self.split,
            'num_samples': len(self.samples)
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        logger.info(f"Saved dataset to {output_path}")
        
    def create_label_map(self, samples: List[Sample]) -> Dict[str, int]:
        """Create label map from samples"""
        unique_events = sorted(set(s.event_type for s in samples))
        label_map = {event: idx for idx, event in enumerate(unique_events)}
        return label_map

class DataSplitter:
    """Split data into train/validation/test sets"""
    
    def __init__(self, config):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
    def split_samples(
        self, 
        all_samples: List[Sample], 
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[List[Sample], List[Sample], List[Sample]]:
        """
        Split samples into train, validation, and test sets
        
        Args:
            all_samples: List of all samples
            test_size: Proportion of test samples
            val_size: Proportion of validation samples (from train set)
            random_state: Random seed for reproducibility
            
        Returns:
            train_samples, val_samples, test_samples
        """
        # Extract labels for stratification
        labels = [s.label for s in all_samples]
        
        # First split: train+val vs test
        train_val_samples, test_samples = train_test_split(
            all_samples,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        
        # Second split: train vs val
        train_labels = [s.label for s in train_val_samples]
        train_samples, val_samples = train_test_split(
            train_val_samples,
            test_size=val_size/(1-test_size),
            random_state=random_state,
            stratify=train_labels
        )
        
        self.logger.info(f"Split completed:")
        self.logger.info(f"  Train samples: {len(train_samples)}")
        self.logger.info(f"  Validation samples: {len(val_samples)}")
        self.logger.info(f"  Test samples: {len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def create_dataset_from_videos(
        self,
        video_dir: str,
        annotation_dir: str,
        output_dir: str,
        clip_duration: int = 5,
        overlap: float = 0.5
    ) -> Tuple[FootballDataset, FootballDataset, FootballDataset]:
        """
        Create dataset from raw videos and annotations
        
        Args:
            video_dir: Directory containing video files
            annotation_dir: Directory containing annotation files
            output_dir: Directory to save processed dataset
            clip_duration: Duration of each clip in seconds
            overlap: Overlap between consecutive clips (0-1)
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        self.logger.info(f"Creating dataset from videos in {video_dir}")
        
        all_samples = []
        video_dir = Path(video_dir)
        annotation_dir = Path(annotation_dir)
        
        # Process each video
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        
        for video_path in tqdm(video_files, desc="Processing videos"):
            annotation_path = annotation_dir / f"{video_path.stem}.json"
            
            if annotation_path.exists():
                samples = self._extract_samples_from_video(
                    str(video_path),
                    str(annotation_path),
                    clip_duration,
                    overlap
                )
                all_samples.extend(samples)
            else:
                self.logger.warning(f"No annotation found for {video_path}")
        
        self.logger.info(f"Extracted {len(all_samples)} samples from {len(video_files)} videos")
        
        # Create label map
        label_map = {}
        for samples in [all_samples]:
            for sample in samples:
                if sample.event_type not in label_map:
                    label_map[sample.event_type] = len(label_map)
                    
        # Assign labels
        for sample in all_samples:
            sample.label = label_map[sample.event_type]
        
        # Split samples
        train_samples, val_samples, test_samples = self.split_samples(all_samples)
        
        # Create datasets
        train_dataset = self._create_dataset_from_samples(train_samples, label_map, "train")
        val_dataset = self._create_dataset_from_samples(val_samples, label_map, "val")
        test_dataset = self._create_dataset_from_samples(test_samples, label_map, "test")
        
        # Save datasets
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_dataset.save_dataset(output_dir / "train_dataset.pkl")
        val_dataset.save_dataset(output_dir / "val_dataset.pkl")
        test_dataset.save_dataset(output_dir / "test_dataset.pkl")
        
        # Save label map
        with open(output_dir / "label_map.json", 'w') as f:
            json.dump(label_map, f, indent=2)
        
        self.logger.info(f"Dataset saved to {output_dir}")
        
        return train_dataset, val_dataset, test_dataset
    
    def _extract_samples_from_video(
        self,
        video_path: str,
        annotation_path: str,
        clip_duration: int,
        overlap: float
    ) -> List[Sample]:
        """Extract samples from a single video"""
        samples = []
        
        # Load annotations
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate clip parameters
        frames_per_clip = int(fps * clip_duration)
        overlap_frames = int(frames_per_clip * overlap)
        stride = frames_per_clip - overlap_frames
        
        # Extract clips
        for start_frame in range(0, total_frames - frames_per_clip, stride):
            end_frame = start_frame + frames_per_clip
            clip_start_time = start_frame / fps
            clip_end_time = end_frame / fps
            
            # Find events in this clip
            clip_events = []
            for event in annotations.get('events', []):
                event_time = event.get('start_time', 0)
                if clip_start_time <= event_time <= clip_end_time:
                    clip_events.append(event)
            
            # Create sample if there are events
            if clip_events:
                # Read frames
                frames = []
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for _ in range(frames_per_clip):
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (224, 224))
                        frames.append(frame)
                    else:
                        break
                
                if len(frames) == frames_per_clip:
                    # Create sample
                    sample = Sample(
                        video_frames=np.array(frames),
                        audio_features={},  # Will be filled later
                        label=0,  # Temporary
                        event_type=clip_events[0]['event_type'],
                        timestamp=clip_start_time,
                        video_path=video_path,
                        metadata={
                            'clip_events': clip_events,
                            'fps': fps,
                            'clip_duration': clip_duration
                        }
                    )
                    samples.append(sample)
        
        cap.release()
        return samples
    
    def _create_dataset_from_samples(
        self,
        samples: List[Sample],
        label_map: Dict[str, int],
        split: str
    ) -> FootballDataset:
        """Create FootballDataset from samples"""
        
        # Create dataset instance
        dataset = FootballDataset(
            data_dir="",  # Will be set manually
            split=split,
            transform=None,
            max_samples=None
        )
        
        # Manually set attributes
        dataset.samples = samples
        dataset.label_map = label_map
        dataset.reverse_label_map = {v: k for k, v in label_map.items()}
        
        return dataset


# ADD CUSTOM COLLATE FUNCTION HERE
def custom_collate_fn(batch):
    """
    Custom collate function to ensure all samples have consistent shapes
    This acts as a safety net in case any samples slipped through
    """
    if not batch:
        return batch
    
    # Check first sample for structure
    first_sample = batch[0]
    
    if isinstance(first_sample, dict):
        collated = {}
        
        for key in first_sample.keys():
            values = [sample[key] for sample in batch]
            
            # Handle tensors
            if torch.is_tensor(values[0]):
                try:
                    # Try to stack normally
                    collated[key] = torch.stack(values)
                except RuntimeError as e:
                    if "size" in str(e).lower():
                        # Size mismatch - handle padding
                        if key == 'audio':
                            # Find max dimensions for audio
                            max_dims = max(v.shape for v in values)
                            padded_values = []
                            
                            for v in values:
                                if v.shape != max_dims:
                                    # Pad with zeros
                                    padding = []
                                    for current, target in zip(v.shape, max_dims):
                                        padding.extend([0, target - current])
                                    
                                    # Reverse padding tuple for F.pad
                                    padding = padding[::-1]
                                    v_padded = torch.nn.functional.pad(v, padding, mode='constant', value=0)
                                    padded_values.append(v_padded)
                                else:
                                    padded_values.append(v)
                            
                            collated[key] = torch.stack(padded_values)
                        else:
                            # For other tensors, just use list
                            collated[key] = values
                    else:
                        raise e
            else:
                # Non-tensor values
                collated[key] = values
                
        return collated
    
    # Default fallback
    return batch


# Utility functions
def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test sets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # ADDED: Custom collate function
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # ADDED: Custom collate function
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn  # ADDED: Custom collate function
    )
    
    logger.info(f"Created data loaders:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader

def load_dataset(
    data_dir: str,
    split: str = 'train',
    **kwargs
) -> FootballDataset:
    """
    Load dataset from directory
    
    Args:
        data_dir: Directory containing dataset files
        split: Dataset split (train, val, test)
        **kwargs: Additional arguments for FootballDataset
        
    Returns:
        FootballDataset instance
    """
    return FootballDataset(data_dir, split, **kwargs)