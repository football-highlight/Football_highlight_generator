"""
Video preprocessing and frame extraction
"""

import cv2
import numpy as np
import torch
from typing import Generator, Tuple, List, Optional
import os
from dataclasses import dataclass
from pathlib import Path
import logging
from tqdm import tqdm

from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class VideoClip:
    """Video clip data structure"""
    frames: np.ndarray  # Shape: (num_frames, height, width, channels)
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    fps: float  # Frames per second
    frame_count: int  # Number of frames
    
    def __post_init__(self):
        """Validate clip data"""
        if self.frames.ndim != 4:
            raise ValueError(f"Frames must be 4D array, got {self.frames.ndim}D")
        
        if self.frames.shape[0] != self.frame_count:
            raise ValueError(f"Frame count mismatch: {self.frames.shape[0]} != {self.frame_count}")


class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, config):
        self.config = config.VIDEO
        self.fps = None
        self.total_frames = None
        self.duration = None
    
    def extract_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            
        Yields:
            Frames one by one
        """
        # FIX: Use absolute path
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Cannot find video file: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # FIX: Handle invalid FPS
        if self.fps <= 0:
            logger.warning(f"Invalid FPS ({self.fps}). Using default FPS of 30.")
            self.fps = 30.0
        
        # FIX: Handle invalid frame count
        if self.total_frames <= 0:
            logger.warning("Invalid frame count. Trying manual count...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                self.total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"FPS: {self.fps}, Total frames: {self.total_frames}, Duration: {self.duration:.2f}s")
        
        if self.total_frames == 0:
            logger.error("Video has 0 frames. Cannot process.")
            cap.release()
            return
        
        frame_count = 0
        
        with tqdm(total=self.total_frames, desc="Extracting frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame if needed
                if frame is not None and frame.shape[:2] != (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH):
                    frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
                
                yield frame
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        logger.info(f"Extracted {frame_count} frames")
    
    def create_clips(self, video_path: str, clip_duration: int = None, 
                    overlap: float = 0.0) -> List[VideoClip]:
        """
        Create video clips of specified duration
        
        Args:
            video_path: Path to video file
            clip_duration: Duration of each clip in seconds
            overlap: Overlap between clips (0-1)
            
        Returns:
            List of VideoClip objects
        """
        # FIX: Use absolute path
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        clip_duration = clip_duration or self.config.CLIP_DURATION

        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # FIX: Handle invalid FPS
        if fps <= 0:
            logger.warning(f"Invalid FPS ({fps}). Using default FPS of 30.")
            fps = 30.0
        
        # FIX: Handle invalid frame count
        if total_frames <= 0:
            logger.warning("Invalid frame count. Trying manual count...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info(f"Manually counted {total_frames} frames")
        
        # FIX: Handle case where we still have 0 frames
        if total_frames <= 0:
            logger.error("Video has 0 frames. Cannot create clips.")
            cap.release()
            return []

        # Calculate clip parameters
        frames_per_clip = int(fps * clip_duration)
        
        # FIX: Ensure frames_per_clip is at least 1
        if frames_per_clip < 1:
            frames_per_clip = max(1, int(fps))  # Use at least 1 frame
            
        overlap_frames = int(frames_per_clip * overlap)
        
        # FIX: Ensure stride is at least 1
        stride = frames_per_clip - overlap_frames
        if stride <= 0:
            stride = 1  # Minimum stride of 1 frame

        # FIX: Also check if we have enough frames
        if total_frames <= frames_per_clip:
            # If video is shorter than clip duration, use the whole video
            frames_per_clip = total_frames
            stride = total_frames  # Only one clip

        clips = []

        logger.info(f"Creating clips: {clip_duration}s each, {overlap*100:.1f}% overlap")
        logger.info(f"FPS: {fps}, Total frames: {total_frames}, Frames per clip: {frames_per_clip}, Stride: {stride}")

        # FIX: Handle the range calculation
        if stride > 0 and total_frames > frames_per_clip and frames_per_clip > 0:
            # Calculate maximum start frame
            max_start_frame = total_frames - frames_per_clip
            if max_start_frame > 0:
                for start_frame in range(0, max_start_frame, stride):
                    end_frame = start_frame + frames_per_clip
                    clip_start_time = start_frame / fps
                    clip_end_time = end_frame / fps

                    # Read frames for this clip
                    clip_frames = []
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    
                    for _ in range(frames_per_clip):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Resize if needed
                        if frame is not None and frame.shape[:2] != (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH):
                            frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
                        clip_frames.append(frame)
                    
                    if len(clip_frames) == frames_per_clip:
                        clip = VideoClip(
                            frames=np.array(clip_frames),
                            start_time=clip_start_time,
                            end_time=clip_end_time,
                            fps=fps,
                            frame_count=frames_per_clip
                        )
                        clips.append(clip)
        else:
            # If stride is 0 or video is too short, create one clip from the whole video
            logger.warning(f"Video too short or stride is 0. Creating single clip from entire video.")
            clip_frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            frames_to_read = min(frames_per_clip, total_frames) if total_frames > 0 else frames_per_clip
            
            for _ in range(frames_to_read):
                ret, frame = cap.read()
                if not ret:
                    break
                if frame is not None and frame.shape[:2] != (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH):
                    frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
                clip_frames.append(frame)
            
            if clip_frames:
                clip = VideoClip(
                    frames=np.array(clip_frames),
                    start_time=0,
                    end_time=len(clip_frames) / fps,
                    fps=fps,
                    frame_count=len(clip_frames)
                )
                clips.append(clip)

        cap.release()
        logger.info(f"Created {len(clips)} clips")
        
        return clips
    
    def save_clip(self, clip: VideoClip, output_path: str, codec: str = 'mp4v'):
        """
        Save video clip to file
        
        Args:
            clip: VideoClip object
            output_path: Output file path
            codec: Video codec
        """
        if len(clip.frames) == 0:
            logger.error("Cannot save empty clip")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            output_path,
            fourcc,
            clip.fps,
            (clip.frames.shape[2], clip.frames.shape[1])
        )
        
        for frame in clip.frames:
            out.write(frame)
        
        out.release()
        logger.info(f"Saved clip to {output_path}")
    
    def extract_optical_flow(self, frames: np.ndarray) -> np.ndarray:
        """
        Extract optical flow between consecutive frames
        
        Args:
            frames: Array of frames
            
        Returns:
            Optical flow array
        """
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames for optical flow")
        
        # Convert to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        
        optical_flows = []
        
        for i in range(len(gray_frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], gray_frames[i + 1],
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Split into x and y components
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
            
            # Normalize to [0, 255]
            flow_x_norm = cv2.normalize(flow_x, None, 0, 255, cv2.NORM_MINMAX)
            flow_y_norm = cv2.normalize(flow_y, None, 0, 255, cv2.NORM_MINMAX)
            
            # Stack into 2-channel image
            flow_img = np.stack([flow_x_norm, flow_y_norm], axis=2).astype(np.uint8)
            optical_flows.append(flow_img)
        
        return np.array(optical_flows)
    
    def extract_frames_at_timestamps(self, video_path: str, timestamps: List[float], 
                                    buffer: float = 2.0) -> List[np.ndarray]:
        """
        Extract frames at specific timestamps
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            buffer: Buffer around timestamp to extract frames
            
        Returns:
            List of frame sequences
        """
        video_path = os.path.abspath(video_path)
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # FIX: Handle invalid FPS
        if fps <= 0:
            fps = 30.0
        
        frame_sequences = []
        
        for timestamp in timestamps:
            start_frame = max(0, int((timestamp - buffer) * fps))
            end_frame = int((timestamp + buffer) * fps)
            frames_to_extract = end_frame - start_frame
            
            frames = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for _ in range(frames_to_extract):
                ret, frame = cap.read()
                if ret:
                    if frame.shape[:2] != (self.config.FRAME_HEIGHT, self.config.FRAME_WIDTH):
                        frame = cv2.resize(frame, (self.config.FRAME_WIDTH, self.config.FRAME_HEIGHT))
                    frames.append(frame)
                else:
                    break
            
            if frames:
                frame_sequences.append(np.array(frames))
        
        cap.release()
        
        return frame_sequences


# Utility functions
def frames_to_tensor(frames: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """
    Convert frames array to PyTorch tensor
    
    Args:
        frames: Frames array (T, H, W, C)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        PyTorch tensor (C, T, H, W)
    """
    # Convert to tensor
    tensor = torch.from_numpy(frames).float()
    
    # Rearrange dimensions: (T, H, W, C) -> (C, T, H, W)
    tensor = tensor.permute(3, 0, 1, 2)
    
    # Normalize
    if normalize:
        tensor = tensor / 255.0
    
    return tensor


def tensor_to_frames(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    """
    Convert PyTorch tensor to frames array
    
    Args:
        tensor: PyTorch tensor (C, T, H, W)
        denormalize: Whether to denormalize from [0, 1]
        
    Returns:
        Frames array (T, H, W, C)
    """
    # Denormalize
    if denormalize:
        tensor = tensor * 255.0
    
    # Convert to numpy
    frames = tensor.detach().cpu().numpy()
    
    # Rearrange dimensions: (C, T, H, W) -> (T, H, W, C)
    frames = np.transpose(frames, (1, 2, 3, 0))
    
    # Clip to valid range
    frames = np.clip(frames, 0, 255).astype(np.uint8)
    
    return frames


def test_video_processing():
    """Test video processing functions"""
    print("Testing video processing...")
    
    # Create a dummy video processor
    processor = VideoProcessor(config)
    
    # Create dummy frames
    dummy_frames = np.random.randint(0, 256, (16, 224, 224, 3), dtype=np.uint8)
    
    # Test optical flow
    try:
        optical_flow = processor.extract_optical_flow(dummy_frames)
        print(f"✅ Optical flow shape: {optical_flow.shape}")
    except Exception as e:
        print(f"❌ Optical flow failed: {e}")
    
    # Test tensor conversion
    tensor = frames_to_tensor(dummy_frames)
    frames_back = tensor_to_frames(tensor)
    
    print(f"✅ Tensor shape: {tensor.shape}")
    print(f"✅ Frames shape after conversion: {frames_back.shape}")
    
    # Test clip creation
    dummy_clip = VideoClip(
        frames=dummy_frames,
        start_time=0.0,
        end_time=5.0,
        fps=30.0,
        frame_count=16
    )
    
    print(f"✅ Clip created: {dummy_clip}")
    
    print("✅ All video processing tests passed!")


if __name__ == "__main__":
    test_video_processing()