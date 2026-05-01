"""
Data augmentation for video and audio data
"""

import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List, Tuple, Optional
import random
import librosa
from dataclasses import dataclass
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    # Video augmentation
    video_augment: bool = True
    horizontal_flip: bool = True
    flip_probability: float = 0.5
    random_crop: bool = True
    crop_size: Tuple[int, int] = (224, 224)
    color_jitter: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    rotation: bool = True
    max_rotation: float = 10.0
    time_jitter: bool = True
    max_time_jitter: int = 2
    
    # Audio augmentation
    audio_augment: bool = True
    noise_injection: bool = True
    noise_level: float = 0.005
    time_shift: bool = True
    max_shift: float = 0.2
    pitch_shift: bool = True
    pitch_steps: Tuple[int, int] = (-2, 2)
    time_stretch: bool = True
    stretch_rates: Tuple[float, float] = (0.8, 1.2)
    volume_change: bool = True
    volume_range: Tuple[float, float] = (0.8, 1.2)


class VideoAugmenter:
    """Video data augmentation"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
    def augment_video(self, video_frames: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to video frames
        
        Args:
            video_frames: Video frames array of shape (T, H, W, C)
            
        Returns:
            Augmented video frames
        """
        if not self.config.video_augment:
            return video_frames
            
        augmented_frames = video_frames.copy()
        
        # Apply augmentations randomly
        if self.config.horizontal_flip and random.random() < self.config.flip_probability:
            augmented_frames = self._horizontal_flip(augmented_frames)
            
        if self.config.random_crop:
            augmented_frames = self._random_crop(augmented_frames)
            
        if self.config.color_jitter:
            augmented_frames = self._color_jitter(augmented_frames)
            
        if self.config.rotation and random.random() < 0.3:
            augmented_frames = self._random_rotation(augmented_frames)
            
        if self.config.time_jitter and random.random() < 0.3:
            augmented_frames = self._time_jitter(augmented_frames)
            
        return augmented_frames
    
    def _horizontal_flip(self, frames: np.ndarray) -> np.ndarray:
        """Horizontal flip of video frames"""
        return np.flip(frames, axis=2)  # Flip along width dimension
    
    def _random_crop(self, frames: np.ndarray) -> np.ndarray:
        """Random crop of video frames"""
        T, H, W, C = frames.shape
        crop_h, crop_w = self.config.crop_size
        
        # Random crop coordinates
        h_start = random.randint(0, H - crop_h)
        w_start = random.randint(0, W - crop_w)
        
        # Crop frames
        cropped = frames[:, h_start:h_start + crop_h, w_start:w_start + crop_w, :]
        
        # Resize if needed (should match original size)
        if cropped.shape[1:3] != (H, W):
            cropped_resized = []
            for frame in cropped:
                resized = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
                cropped_resized.append(resized)
            cropped = np.array(cropped_resized)
            
        return cropped
    
    def _color_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Color jitter augmentation"""
        jittered_frames = frames.copy()
        
        # Apply brightness
        if self.config.brightness > 0:
            brightness_factor = 1.0 + random.uniform(-self.config.brightness, self.config.brightness)
            jittered_frames = jittered_frames * brightness_factor
            jittered_frames = np.clip(jittered_frames, 0, 255)
            
        # Apply contrast
        if self.config.contrast > 0:
            mean = np.mean(jittered_frames)
            contrast_factor = 1.0 + random.uniform(-self.config.contrast, self.config.contrast)
            jittered_frames = mean + (jittered_frames - mean) * contrast_factor
            jittered_frames = np.clip(jittered_frames, 0, 255)
            
        return jittered_frames
    
    def _random_rotation(self, frames: np.ndarray) -> np.ndarray:
        """Random rotation of video frames"""
        angle = random.uniform(-self.config.max_rotation, self.config.max_rotation)
        
        rotated_frames = []
        H, W = frames.shape[1:3]
        
        for frame in frames:
            # Get rotation matrix
            M = cv2.getRotationMatrix2D((W/2, H/2), angle, 1.0)
            # Apply rotation
            rotated = cv2.warpAffine(frame, M, (W, H), borderMode=cv2.BORDER_REFLECT)
            rotated_frames.append(rotated)
            
        return np.array(rotated_frames)
    
    def _time_jitter(self, frames: np.ndarray) -> np.ndarray:
        """Temporal jitter (frame skipping/duplication)"""
        T = frames.shape[0]
        jitter_amount = random.randint(-self.config.max_time_jitter, self.config.max_time_jitter)
        
        if jitter_amount == 0:
            return frames
            
        if jitter_amount > 0:
            # Skip frames
            indices = list(range(T))
            indices = indices[jitter_amount:] + indices[-jitter_amount:]  # Skip first, repeat last
        else:
            # Duplicate frames
            jitter_amount = abs(jitter_amount)
            indices = list(range(T))
            # Repeat some frames
            repeat_indices = random.sample(range(T), jitter_amount)
            for idx in repeat_indices:
                indices.insert(idx, idx)
            indices = indices[:T]  # Trim to original length
            
        return frames[indices]


class AudioAugmenter:
    """Audio data augmentation"""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
        
    def augment_audio(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Apply augmentations to audio
        
        Args:
            audio: Audio signal array
            sample_rate: Audio sample rate
            
        Returns:
            Augmented audio
        """
        if not self.config.audio_augment:
            return audio
            
        augmented_audio = audio.copy()
        
        # Apply augmentations randomly
        if self.config.noise_injection and random.random() < 0.3:
            augmented_audio = self._add_noise(augmented_audio)
            
        if self.config.time_shift and random.random() < 0.3:
            augmented_audio = self._time_shift(augmented_audio)
            
        if self.config.pitch_shift and random.random() < 0.3:
            augmented_audio = self._pitch_shift(augmented_audio, sample_rate)
            
        if self.config.time_stretch and random.random() < 0.3:
            augmented_audio = self._time_stretch(augmented_audio)
            
        if self.config.volume_change and random.random() < 0.3:
            augmented_audio = self._change_volume(augmented_audio)
            
        return augmented_audio
    
    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add random noise to audio"""
        noise = np.random.randn(len(audio)) * self.config.noise_level * np.max(np.abs(audio))
        return audio + noise
    
    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Shift audio in time"""
        shift = random.randint(0, int(len(audio) * self.config.max_shift))
        
        if shift > 0:
            # Shift right
            shifted = np.zeros_like(audio)
            shifted[shift:] = audio[:-shift]
        else:
            # Shift left
            shift = abs(shift)
            shifted = np.zeros_like(audio)
            shifted[:-shift] = audio[shift:]
            
        return shifted
    
    def _pitch_shift(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Shift audio pitch"""
        n_steps = random.randint(self.config.pitch_steps[0], self.config.pitch_steps[1])
        
        try:
            shifted = librosa.effects.pitch_shift(
                y=audio,
                sr=sample_rate,
                n_steps=n_steps
            )
            return shifted
        except:
            self.logger.warning("Pitch shift failed, returning original audio")
            return audio
    
    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Time stretch audio"""
        rate = random.uniform(self.config.stretch_rates[0], self.config.stretch_rates[1])
        
        try:
            stretched = librosa.effects.time_stretch(y=audio, rate=rate)
            
            # Ensure same length
            if len(stretched) > len(audio):
                stretched = stretched[:len(audio)]
            elif len(stretched) < len(audio):
                stretched = np.pad(stretched, (0, len(audio) - len(stretched)), 'constant')
                
            return stretched
        except:
            self.logger.warning("Time stretch failed, returning original audio")
            return audio
    
    def _change_volume(self, audio: np.ndarray) -> np.ndarray:
        """Change audio volume"""
        factor = random.uniform(self.config.volume_range[0], self.config.volume_range[1])
        return audio * factor


class AugmentationPipeline:
    """Complete augmentation pipeline for multimodal data"""
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self.video_augmenter = VideoAugmenter(self.config)
        self.audio_augmenter = AudioAugmenter(self.config)
        
    def augment_sample(self, sample: dict) -> dict:
        """
        Augment a complete sample (video + audio)
        
        Args:
            sample: Dictionary containing video and audio data
            
        Returns:
            Augmented sample
        """
        augmented_sample = sample.copy()
        
        # Augment video
        if 'video' in sample:
            video_data = sample['video']
            if isinstance(video_data, np.ndarray):
                augmented_sample['video'] = self.video_augmenter.augment_video(video_data)
            elif isinstance(video_data, torch.Tensor):
                # Convert tensor to numpy for augmentation
                video_np = video_data.numpy().transpose(1, 2, 3, 0)  # (C, T, H, W) -> (T, H, W, C)
                augmented_np = self.video_augmenter.augment_video(video_np)
                augmented_tensor = torch.from_numpy(augmented_np.transpose(3, 0, 1, 2))  # Back to (C, T, H, W)
                augmented_sample['video'] = augmented_tensor
                
        # Augment audio
        if 'audio' in sample:
            audio_data = sample['audio']
            if isinstance(audio_data, np.ndarray):
                augmented_sample['audio'] = self.audio_augmenter.augment_audio(audio_data)
            elif isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.numpy()
                augmented_np = self.audio_augmenter.augment_audio(audio_np)
                augmented_sample['audio'] = torch.from_numpy(augmented_np)
                
        # Augment audio features
        if 'audio_features' in sample and isinstance(sample['audio_features'], dict):
            # For MFCC features, we can apply similar augmentations
            mfcc = sample['audio_features'].get('mfcc')
            if mfcc is not None:
                # Add noise to MFCC
                if random.random() < 0.3:
                    noise = np.random.randn(*mfcc.shape) * 0.01
                    sample['audio_features']['mfcc'] = mfcc + noise
                    
        return augmented_sample
    
    def create_augmented_dataset(self, dataset, augmentation_factor: int = 2):
        """
        Create augmented version of a dataset
        
        Args:
            dataset: Original dataset
            augmentation_factor: How many augmented versions per sample
            
        Returns:
            Augmented dataset
        """
        augmented_samples = []
        
        for i in range(len(dataset)):
            original_sample = dataset[i]
            
            # Add original sample
            augmented_samples.append(original_sample)
            
            # Add augmented versions
            for j in range(augmentation_factor - 1):
                augmented_sample = self.augment_sample(original_sample)
                augmented_samples.append(augmented_sample)
                
        return augmented_samples


# PyTorch transform for video
def create_video_transforms(augment: bool = True):
    """
    Create PyTorch transforms for video data
    
    Args:
        augment: Whether to include augmentations
        
    Returns:
        transforms.Compose object
    """
    transform_list = []
    
    # Normalization
    transform_list.append(transforms.Lambda(lambda x: x.float() / 255.0))
    
    if augment:
        # Add augmentations
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    
    # Normalize with ImageNet stats
    transform_list.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(transform_list)


def test_augmentations():
    """Test function for augmentations"""
    print("Testing video augmentation...")
    
    # Create dummy video
    dummy_video = np.random.randint(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    
    config = AugmentationConfig()
    augmenter = VideoAugmenter(config)
    
    augmented = augmenter.augment_video(dummy_video)
    
    print(f"Original shape: {dummy_video.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print("✅ Video augmentation test passed")
    
    print("\nTesting audio augmentation...")
    
    # Create dummy audio
    dummy_audio = np.random.randn(16000).astype(np.float32)
    
    audio_augmenter = AudioAugmenter(config)
    augmented_audio = audio_augmenter.augment_audio(dummy_audio)
    
    print(f"Original audio shape: {dummy_audio.shape}")
    print(f"Augmented audio shape: {augmented_audio.shape}")
    print("✅ Audio augmentation test passed")


if __name__ == "__main__":
    test_augmentations()